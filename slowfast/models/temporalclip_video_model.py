# 主模型
import torch
import torch.nn as nn
from . import clip
import random
from .build import MODEL_REGISTRY
import os
import numpy as np
import json
import heapq
from typing import Tuple, Union
from .clip.model import CLIP,LayerNorm,Transformer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .clip.model import convert_weights
from .clip.clip import _MODELS, _download

from . import customize_visiontransformer
from .customize_visiontransformer import TemporalVisionTransformer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import sys
sys.path.append('..')
from .videomae.modeling_pretrain import pretrain_videomae_base_patch16_224
from .videomae.masking_generator import TubeMaskingGenerator

import slowfast.utils.logging as logging
import pickle
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
logger = logging.get_logger(__name__)

class Video:
    def __init__(self, similarity, feature):
        self.similarity = similarity
        self.feature = feature
      

    # 定义比较运算符，基于 similarity 进行排序
    def __lt__(self, other):
        return self.similarity < other.similarity  # 降序排列

    def __repr__(self):
        return f"VideoSimilarity(similarity={self.similarity})"

class Attention_Layer(nn.Module):
    def __init__(self, dim, dropout_prob=0.2, reduction=4):
        """
        SE 层实现，不使用全局平均池化
        Args:
            dim: 特征通道维度 (dim)
            time_steps: 时间维度长度 (t)

            reduction: 压缩比
        """
        super(Attention_Layer, self).__init__()

        # 全连接层，保持时间维度 t 不变
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(dim // reduction, dim, bias=False),  # 恢复维度
            nn.Sigmoid()  # 激活生成缩放权重
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状 (batchsize, t, dim)
        Returns:
            缩放后的张量，形状 (batchsize, t, dim)
        """
        # b, t, dim = x.size()
        # 计算每个时间步上通道维度的权重
        y = self.fc(x)  # 输入 (batchsize, t, dim)，输出 (batchsize, t, dim)
        return x * y  # 缩放输入张量





@MODEL_REGISTRY.register()
class TemporalClipVideo(nn.Module):
    """
    Clip visual encoder for space feature extraction. Adding various temporal fusion type.
    """
    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
            comments of the config file.
        """
        super(TemporalClipVideo, self).__init__()
        self.cfg = cfg
        self.num_pathways = 1
        
        self._construct_network(cfg)
        self.model.eval()

        for k, v in self.model.named_parameters():
            v.requires_grad = True
        
        if not cfg.TEST.OPENSET:
            self.text_dict = self.text_prompt(os.path.join(cfg.DATA.INDEX_LABEL_MAPPING_FILE))
        else:
            self.text_dict = self.text_prompt(os.path.join(cfg.DATA.INDEX_LABEL_MAPPING_FILE))
        
        self.prompt_type_num = len(self.text_dict)
        self.cls_num = self.text_dict[0].shape[0]
        self.tune_head = cfg.TUNE_HEAD
        self.text_prompting = cfg.MODEL.TEXT_PROMPT
        self.context_length = cfg.MODEL.CONTEXT_LENGTH
        self.record_routing = cfg.MODEL.RECORD_ROUTING
        self.keep_raw_model = cfg.MODEL.KEEP_RAW_MODEL
        self.ensemble_pred = cfg.MODEL.ENSEMBLE_PRED
        self.distillation = cfg.MODEL.RAW_MODEL_DISTILLATION

        self.projector_v = nn.Sequential(
            nn.Linear(self.model.embed_dim, self.model.embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.model.embed_dim, self.model.embed_dim, bias=False)
        )
        nn.init.zeros_(self.projector_v[2].weight)
        nn.init.kaiming_normal_(self.projector_v[0].weight)

        self.projector_t = nn.Sequential(
            nn.Linear(self.model.embed_dim, self.model.embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.model.embed_dim, self.model.embed_dim, bias=False)
        )
        self.mlp = nn.Sequential(
                nn.Linear(self.model.embed_dim, self.model.embed_dim),  # 提升维度 + 显式变换
                nn.ReLU(),                      # 引入非线性
                nn.Dropout(p=0.1),              # 在激活后随机丢弃部分特征
                nn.Linear(self.model.embed_dim, self.model.embed_dim)  # 最终投影输出
        )
        nn.init.zeros_(self.projector_t[2].weight)
        nn.init.kaiming_normal_(self.projector_t[0].weight)

        nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[3].weight, mean=0.0, std=1e-4)  # 非零但极小扰动
        nn.init.zeros_(self.mlp[3].bias)

        if self.distillation and (not self.keep_raw_model):
            print("not support distillation if not keeping the raw model")
            exit()

        # check
        if (self.keep_raw_model and self.ensemble_pred) and self.record_routing:
            print("ensemble pred should not exists together with record-routing")
            exit()
        
        if self.tune_head:
            self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)
            self.head = torch.nn.Parameter(self.dynamic_classifier, requires_grad=True)
        elif self.text_prompting:
            self.prompt_num = int(cfg.MODEL.PROMPT_NUM)
            embedding_dim = self.model.ln_final.weight.shape[0]
            
            self.prompt_embed = torch.nn.Parameter(
                        torch.rand(int(self.prompt_num), embedding_dim).cuda(), requires_grad=True
                    )
            torch.nn.init.normal_(self.prompt_embed, std=0.01)
            
            id2cls = {}
            for idx, cls in  json.load(open(cfg.DATA.INDEX_LABEL_MAPPING_FILE, 'r')).items():
                id2cls[int(idx)] = cls
            self.classnames = [id2cls[i] for i in range(len(id2cls))]
            prompts = [" ".join(["X"] * self.prompt_num) + " " + name + "." for name in self.classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p, context_length=self.context_length) for p in prompts])
            tokenized_prompts = tokenized_prompts.cuda()
            
            with torch.no_grad():
                embedding = self.model.token_embedding(tokenized_prompts)
            self.token_prefix = embedding[:, :1, :]  # SOT
            self.token_suffix = embedding[:, 1 + self.prompt_num:, :]  # CLS, EOT
            self.tokenized_prompts = tokenized_prompts  # for localizing EOT
            
            for name, param in self.model.transformer.named_parameters():
                param.requires_grad = False
        else:
            self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)
        
        # self.prompt_embed. -> token_prefix + prompt_embed + token_suffix

        # learning factor
        # if self.cfg and self.cfg.MODEL.FINETUNE_FACTOR != 1.0:
        # Indicate parameters for finetuning.
        self.lr_factor = {
            "message": cfg.MODEL.FINETUNE_FACTOR,
            "stadapt": cfg.MODEL.ADAPT_FINETUNE_FACTOR,
            "mlp": cfg.MODEL.MLP_FINETUNE_FACTOR,
            "experts": cfg.MODEL.EXPERT_FINETUNE_FACTOR,
            "routing": cfg.MODEL.ROUTING_FINETUNE_FACTOR,
        } 
        self.top_k_heaps=None
        self.attibute_feature_cache=None
        #base类别对应的行为属性词
        self.adapterlayer=Attention_Layer(512)

        
      
        # self.sentence_model=SentenceTransformer('all-MiniLM-L6-v2').eval()
        self.sentence_model=SentenceTransformer('/root/tmp/all-mpnet-base-v2').eval()
        for param in self.sentence_model.parameters():
            param.requires_grad = False
        self.action_attribute_matrix=None
        self.action_embeddings=None
        self.attribute_embeddings=None
        # self.action_embeddings=action_embeddings
        # self.attribute_embedings=attribute_embeddings
        self.cur_epoch=0

        # self.selayer=SELayer()
       

    def _construct_network(self, cfg):

        context_length = cfg.MODEL.CONTEXT_LENGTH
        if cfg.MODEL.ARCH == 'vitb32':
            self.model, self.preprocess = load("ViT-B/32", jit=False, 
                    T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=cfg.MODEL.TEMPORAL_MODELING_TYPE,
                    use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                    num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                    record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                    )
            if cfg.MODEL.KEEP_RAW_MODEL:   
                self.raw_model, self.preprocess = load("ViT-B/32", jit=False, 
                        T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=None,
                        use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                        num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                        record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                        )
                for name, p in self.raw_model.named_parameters():
                    p.requires_grad = False

        elif cfg.MODEL.ARCH == 'vitb16':
            self.model, self.preprocess = load("ViT-B/16", jit=False, 
                    T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=cfg.MODEL.TEMPORAL_MODELING_TYPE,
                    use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                    num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                    record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                    )
            if cfg.MODEL.KEEP_RAW_MODEL:   
                self.raw_model, self.preprocess = load("ViT-B/16", jit=False, 
                        T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=None,
                        use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                        num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                        record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                        )
                for name, p in self.raw_model.named_parameters():
                    p.requires_grad = False
                
        elif cfg.MODEL.ARCH == 'vitl14':
            self.model, self.preprocess = load("ViT-L/14", jit=False, 
                    T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=cfg.MODEL.TEMPORAL_MODELING_TYPE,
                    use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                    num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                    record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                    )
            if cfg.MODEL.KEEP_RAW_MODEL:   
                self.raw_model, self.preprocess = load("ViT-L/14", jit=False, 
                        T=cfg.DATA.NUM_FRAMES, temporal_modeling_type=None,
                        use_checkpoint=cfg.MODEL.USE_CHECKPOINT, context_length=context_length,
                        num_experts=cfg.MODEL.NUM_EXPERTS, expert_insert_layers=cfg.MODEL.EXPERT_INSERT_LAYERS,
                        record_routing=cfg.MODEL.RECORD_ROUTING, routing_type=cfg.MODEL.ROUTING_TYPE
                )

                for name, p in self.raw_model.named_parameters():
                    p.requires_grad = False
        else:
            print("error loading arch")
            exit()

        self.model.float() 
        if cfg.MODEL.KEEP_RAW_MODEL:
            self.raw_model.float()
    
    def update_state(self):
        self.dynamic_classifier = self.achieve_csf_matrix(self.text_dict, self.model)

    def forward(self, x=None, attribute_ids=None, attribute_similarities=None,update=False):
        # shape of x(input) is (bz, channel, clip_len, h, w)
        
        # print(attribute_ids)
        # [tensor([ 18,  85,  87, 159, 289, 291, 294, 299], device='cuda:0'), tensor([229
        # bz
        # print(len(attribute_similarities))
        # print(attribute_similarities[0].shape)
        # torch.Size([8])


        assert len(x) == self.num_pathways
        # 修复输入格式不匹配的问题
        # 如果x不是列表或元组，则将其包装成列表
        if not isinstance(x, (list, tuple)):
            x = [x]
        # 确保x的长度至少为1
        if len(x) < 1:
            raise ValueError("Input x must have at least one element")
        # 如果x的长度与self.num_pathways不匹配，使用第一个元素
        x = x[0]
        if len(x.shape) == 4:
            # image input
            x = x.unsqueeze(2)
        
        # ensure eval state all the time, cost time ?
        if self.keep_raw_model:
            self.raw_model.eval()

        bz, channel_dim, clip_len, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bz*clip_len, channel_dim, h, w)
        
        if self.record_routing:
            img_encode, routing_state = self.model.encode_image(x)
        else:
            img_encode = self.model.encode_image(x)
            
        feature = None
        if isinstance(img_encode, list):
            img_encode, feature = img_encode
            c = feature.shape[-1]

        # print(img_encode.shape)
        # a=8/0
        #
        img_encode=self.adapterlayer(img_encode)



        if self.training:
            # img encode [bz, feat_size]
            # text_dict  {id: [400, feat_size]},
            # pre_img_encode = img_encode
           

            img_encode = img_encode / img_encode.norm(dim=-1, keepdim=True)
            # print("img_encode.shape:")
            # torch.Size([64, 512])
          
            
            if self.tune_head:
                norm_head = self.head / self.head.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ norm_head.T
            elif self.text_prompting:
                # encode head.
                text_embedding = torch.cat((self.token_prefix, 
                            self.prompt_embed.unsqueeze(0).expand(len(self.classnames), -1, -1), 
                            self.token_suffix
                            ), 1)
                norm_head = self.model.prompt_encode_text(text_embedding, self.tokenized_prompts,)
                norm_head /= norm_head.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ norm_head.T
            else:
                # csf_matrix = self.dynamic_classifier / self.dynamic_classifier.norm(dim=-1, keepdim=True)
                text_dict = self.text_prompt(os.path.join(self.cfg.DATA.INDEX_LABEL_MAPPING_FILE))
                dynamic_classifier_new = self.achieve_csf_matrix(text_dict, self.model, trainable=True)
                pred = self.model.logit_scale.exp() * img_encode @ dynamic_classifier_new.T

            pred = pred.reshape(bz, clip_len, -1).mean(1)
            
            
            
            #在此处保存每个样本的特征
           
            
            #从保存的行为属性词对应的视觉特征中提取
            # print(attribute_ids)
            #attribute_ids中的每一项表示与一个proxy类别最相关的N个行为属性词的下标
            # [tensor([ 18,  85,  87, 159, 289, 291, 294, 299], device='cuda:0'), tensor([229
            # print("model cur_epoch")
            # print(self.cur_epoch)
            
            self.attributes_text_dict=self.attributes_text_prompt(os.path.join(self.cfg.DATA.ACTION_ATTRIBUTE_MAPPING_FILE))
            self.attributes_classifier=self.achieve_csf_matrix(self.attributes_text_dict, self.model, trainable=False)
            # print(img_encode.shape)
            # print(self.attributes_classifier.shape)
            # torch.Size([64, 512])
            # torch.Size([312, 512])
           
            #这里的k表示在属性特征缓存中，每个视觉属性描述选择多少个对应的视觉特征
            #self.select_most_relevant_attributes(img_encode,self.attributes_classifier,bz,clip_len,k=5)原代码
            if isinstance(attribute_similarities, torch.Tensor):
            # 已经是单个张量，无需stack，直接转CPU
                attribute_similarities = attribute_similarities.cpu()
            else:
            # 是张量的列表/元组，stack合并后转CPU
                attribute_similarities = torch.stack(attribute_similarities, dim=0).cpu()
            num_attributes = attribute_similarities.size(1)
            k = min(5, num_attributes)
            self.select_most_relevant_attributes(img_encode, self.attributes_classifier, bz, clip_len, k=k)

            if self.cur_epoch>0:
                #计算增广类别损失L_ACL
                #ids 是什么？
                # select_attribute_num=bz  #对一个增广类别文本特征，使用多少个对应的
                attribute_ids_list=[ids.cpu().tolist() for ids in attribute_ids]
                
                #attribute_similarities=torch.stack(attribute_similarities,dim=0).cpu()改
                tensors = [attribute_similarities] if isinstance(attribute_similarities, torch.Tensor) else attribute_similarities
                attribute_similarities = torch.stack(tensors, dim=0).cpu()
                # print(attribute_similarities.shape)
                # torch.Size([8, 8])
                # 每个proxy类别文本对应的 bz个行为属性词的相似分数
                # raise FileExistsError
                proxy_features=[]
                for index in range(bz):
                    attribute_ids_per_instance=attribute_ids_list[index]
                    # 关键：去掉维度0的冗余维度（如果大小是1），恢复形状为 (batch_size, num_attributes)
                    attribute_similarities = attribute_similarities.squeeze(0)
                    attribute_similarities_per_instance=attribute_similarities[index]
                    # print(attribute_similarities_per_instance.shape)
                    # raise FileExistsError
                    #一个proxy类别样本对应的相关特征
                    # print(len(self.top_k_heaps))
                    related_features_per_instance=[random.choice(self.top_k_heaps[id]).feature for id in attribute_ids_per_instance]
                    # print(len(related_features_per_instance))
                    # attribute_num_perInstance 个 torch.Size([8, 512]) 维度的特征向量
                    # print(related_features_per_instance[0].shape)
                    # a=8/0
                    # (8,512)
                    # print(bz)
                    #选择一个增广类别文本特征对应的attribute_num_perInstance个视觉属性特征
                    attribute_num_perInstance=len(related_features_per_instance)
                    related_features_per_instance=torch.stack(related_features_per_instance, dim=0).view(attribute_num_perInstance,-1)
                    # print(related_features_per_instance.shape)
                    #torch.Size([8, 4096]) (attribute_num_perInstance,clip_len*dim)
                    
                    # raise FileExistsError
                    #[1,8]*[8,4096]=[1,4096]
                    # print(attribute_similarities_per_instance.shape)
                    # raise FileExistsError
                    proxy_feature_per_instance=F.softmax(attribute_similarities_per_instance,dim=0).unsqueeze(0)@related_features_per_instance
                    
                    proxy_feature_per_instance=proxy_feature_per_instance.view(clip_len,-1)
                    # print(proxy_feature_per_instance.shape)
                    #torch.Size([8, 512])
                    proxy_features.append(proxy_feature_per_instance)
                    
                proxy_features=torch.stack(proxy_features,dim=0).to(img_encode.device)
                # proxy_features=proxy_features.detach()
                proxy_features=self.mlp(proxy_features)
                proxy_features = proxy_features / proxy_features.norm(dim=-1, keepdim=True)
                # proxy_features=self.selayer(proxy_features)
                proxy_text_dict = self.text_prompt(os.path.join(self.cfg.DATA.PROXY_LABEL_MAPPING_FILE))
                proxy_dynamic_classifier = self.achieve_csf_matrix(proxy_text_dict, self.model, trainable=True)
                proxy_pred=self.model.logit_scale.exp() * proxy_features @proxy_dynamic_classifier.T
                proxy_pred=proxy_pred.reshape(bz, clip_len, -1).mean(1)
                
                #计算属性联系损失 L_ARL
                # print(self.top_k_heaps)
                avg_feature_list = []
                index_list = []
                for idx, videos in self.top_k_heaps.items():
                    features = [v.feature for v in videos]              # 提取所有 feature
                    stacked = torch.stack(features, dim=0)              # 堆叠成 2D tensor: (K, D)
                    avg_feature = stacked.mean(dim=0)                   # 对 K 取平均，得到 (D,)
                    avg_feature_list.append(avg_feature)
                    index_list.append(idx)
                # 构建最终的 N × D tensor
                avg_feature_tensor = torch.stack(avg_feature_list, dim=0).to(next(self.parameters()).device) # (N, D)
                avg_feature_tensor = F.normalize(avg_feature_tensor, p=2, dim=-1) # (N, D)

                # print(avg_feature_tensor.shape)
                # (属性特征数量,clip_len,dim)

                
                #对于每个视频特征，找到与它最相关的N个视觉属性信息对应的序号。
                N=4
                # print(img_encode.shape)
                # print(avg_feature_tensor.shape)
                # torch.Size([64, 512])
                # torch.Size([312, 8, 512])

                # print(img_encode.shape)
                # raise FileExistsError
                
                img_encode_video=img_encode.clone().reshape(bz, clip_len, -1).mean(1)
                avg_feature_tensor=avg_feature_tensor.mean(1)

                img_encode_video = F.normalize(img_encode_video, p=2, dim=-1)
                avg_feature_tensor = F.normalize(avg_feature_tensor, p=2, dim=-1)
                
                # print(avg_feature_tensor.shape)
                # print(img_encode_video.shape)
                # raise FileExistsError
                
                #计算视频特征和属性特征的相似度

                # print(img_encode_video.shape)
                # torch.Size([8, 512]) batchsize, dim
                # print(avg_feature_tensor.shape)
                # torch.Size([104, 512])  attribute_num, dim

                # raise FileExistsError

                sim_matrix = img_encode_video @ avg_feature_tensor.T
                # batchsize, attribute_num 一个batch内每个视频与多个属性的相似度
                _, topn_indices = torch.topk(sim_matrix, k=N, dim=1)
                # 每个视频相似度前N的属性对应的下标

                # 修复设备不匹配：先将 attribute_embeddings 移到 GPU，然后索引
                attribute_embeddings_gpu = self.attribute_embeddings.to(img_encode.device)
                selected_attr = attribute_embeddings_gpu[topn_indices]  # 使用索引选择 (B, N, D)

               # 计算每个样本的属性与所有行为类别之间的相似度：(B, N, D) @ (D, num_attributes) → (B, N, num_attributes)

                sim = torch.matmul(selected_attr, self.action_embeddings.T.to(img_encode.device))



               # 对 N 个属性求和，得到 (B, M)
                similarity_sum = sim.sum(dim=1)
                similarity_distribution=F.softmax(similarity_sum, dim=-1)

 



                
                # # self.attibute_feature_cache=
                # raise FileExistsError
                   
                   
                    
                    
                    
                    
                    
                
                
            
            # add distillation here
            if self.keep_raw_model and (self.ensemble_pred or self.distillation):
                # pass
                with torch.no_grad():
                    raw_img_encode = self.raw_model.encode_image(x)
                    if isinstance(raw_img_encode, list):
                        raw_img_encode = raw_img_encode[0]
                    raw_img_encode /= raw_img_encode.norm(dim=-1, keepdim=True)
                    # raw_pred = self.raw_model.logit_scale.exp() * raw_img_encode @ self.dynamic_classifier_raw.T
                    # raw_pred = raw_pred.reshape(bz, clip_len, -1).mean(1)

                dynamic_classifier_raw = self.achieve_csf_matrix(text_dict, self.raw_model, trainable=True)
                
                alpha = 0.1
                img_encode = img_encode + alpha * self.projector_v(img_encode)
                
                dynamic_classifier_new = dynamic_classifier_new + alpha * self.projector_t(dynamic_classifier_new)
                
                if self.cur_epoch>1:
                    return [proxy_pred, pred,similarity_distribution, img_encode, dynamic_classifier_new], [None, raw_img_encode, dynamic_classifier_raw]
                else:
                    return [pred, img_encode, dynamic_classifier_new], [None, raw_img_encode, dynamic_classifier_raw]
                # return [pred, dynamic_classifier_new], [None, dynamic_classifier_raw]
            
            if self.record_routing:
                return pred, routing_state
            return pred
        else:
            # img_encode [bz, feat_size]
            # dynamic_clf shape [type_num * cls_num, feat_size]
            # pre_img_encode = img_encode

            img_encode /= img_encode.norm(dim=-1, keepdim=True)
            
            
            
           

            if self.tune_head:
                norm_head = self.head / self.head.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ norm_head.T

            elif self.text_prompting:
                # encode head.
                text_embedding = torch.cat((self.token_prefix, 
                            self.prompt_embed.unsqueeze(0).expand(len(self.classnames), -1, -1), 
                            self.token_suffix
                            ), 1)
                
                norm_head = self.model.prompt_encode_text(text_embedding, self.tokenized_prompts,)
                norm_head /= norm_head.norm(dim=-1, keepdim=True)
                pred = self.model.logit_scale.exp() * img_encode @ norm_head.T
            else:
                text_dict = self.text_prompt(os.path.join(self.cfg.DATA.INDEX_LABEL_MAPPING_FILE))
                dynamic_classifier_new = self.achieve_csf_matrix(text_dict, self.model, trainable=False)
                pred = self.model.logit_scale.exp() * img_encode @ dynamic_classifier_new.T

            pred = pred.reshape(bz, clip_len, -1).mean(1)
            
            if self.keep_raw_model and (self.ensemble_pred or self.distillation):
                pass

            if self.record_routing:
                return pred, routing_state
            
            if self.keep_raw_model and (self.ensemble_pred or self.distillation):
                return [pred, None], [None, None]
            
            return pred
            # if feature is not None:
            #     return [pred, feature.view(bz, -1, c)]
            # else:
            #     return pred
    
    def text_prompt(self, data_file):
        text_aug = [
                f'a photo of {{}}.',
                f'a photo of a person {{}}.',
                f'a video of {{}}.',
                f'a video of a person {{}}.',
                f'{{}}'
            ]
        text_dict = {}
        
        id2cls = {}
        temp_mapping = json.load(open(data_file, 'r'))
        for key in temp_mapping:
            id2cls[int(key)] = temp_mapping[key]
        
        """
        # parse datafile
        lines = open(data_file, 'r').readlines()
        for line in lines:
            cls_name, cls_id = line.strip().split(',')
            cls_name = cls_name.split('/')[1]
            cls_name = cls_name.replace('_', ' ')
            if cls_name not in id2cls:
                id2cls[int(cls_id)] = cls_name
        """

        cls_num = len(id2cls)
        # construct the source of dynamic classifier
        if self.training:
            index = random.randint(0, len(text_aug)-2)
            text_aug = [text_aug[index], text_aug[-1]]

        for idx, txt in enumerate(text_aug):
            # text_dict[idx] = torch.cat([clip.tokenize(txt.format(id2cls[id])) for id in range(cls_num)])
            # text_dict[idx] = torch.cat([clip.tokenize(txt.format(id2cls[id].split(':')[0]) + ' ' + id2cls[id]) for id in range(cls_num)])
            if idx == len(text_aug)-1:
                text_dict[idx] = torch.cat([clip.tokenize(txt.format(id2cls[id])) for id in range(cls_num)])
            else:
                text_dict[idx] = torch.cat([clip.tokenize(txt.format(id2cls[id].split(':')[0]) + ' ' + id2cls[id]) for id in range(cls_num)])

        return text_dict
        
    def achieve_csf_matrix(self, text_dict, model, trainable=False):
        if not trainable:
            with torch.no_grad():
                csf_matrix_list = [model.encode_text(text_dict[i].cuda()).detach() for i in range(len(text_dict))]
                for csf_matrix in csf_matrix_list:
                    csf_matrix /= csf_matrix.norm(dim=-1, keepdim=True)
        else:
            csf_matrix_list = [model.encode_text(text_dict[i].cuda()) for i in range(len(text_dict))]
            for csf_matrix in csf_matrix_list:
                csf_matrix /= csf_matrix.norm(dim=-1, keepdim=True)
        
        csf_matrix = torch.stack(csf_matrix_list, 0).mean(0)
        csf_matrix /= csf_matrix.norm(dim=-1, keepdim=True)
        
        return csf_matrix
    
    
    #产生词汇对应的代码模板，
    def attributes_text_prompt(self, data_file):
        text_aug = [
                f'a photo of {{}}.',
                f'a photo of a person {{}}.',
                f'a video of {{}}.',
                f'a video of a person {{}}.',
                f'{{}}'
            ]
        text_dict = {}
        
        id2cls = {}
        temp_mapping = json.load(open(data_file, 'r'))
        for key in temp_mapping:
            id2cls[int(key)] = temp_mapping[key]
            
        self.top_k_heaps = {i: [] for i in range(len(temp_mapping))}
        
        """
        # parse datafile
        lines = open(data_file, 'r').readlines()
        for line in lines:
            cls_name, cls_id = line.strip().split(',')
            cls_name = cls_name.split('/')[1]
            cls_name = cls_name.replace('_', ' ')
            if cls_name not in id2cls:
                id2cls[int(cls_id)] = cls_name
        """

        cls_num = len(id2cls)
        # construct the source of dynamic classifier
        if self.training:
            index = random.randint(0, len(text_aug)-2)
            text_aug = [text_aug[index], text_aug[-1]]

        for idx, txt in enumerate(text_aug):
            text_dict[idx] = torch.cat([clip.tokenize(txt.format(id2cls[id])) for id in range(cls_num)])
        return text_dict

    # k是属性特征缓存中每个属性对应的保存的最相似特征的个数
    def select_most_relevant_attributes(self,video_features,attributes_features,bz,clip_len,k=3):
        
        #计算一个batch内每个视频和属性文本特征的相似度
        affinity = self.model.logit_scale.exp() * video_features @ attributes_features.T
        #bz*T,attribute_num
        affinity=affinity.reshape(bz, clip_len, -1).mean(1)
        #bz,attribute_num 的相似度矩阵
        #堆操作需要再cpu()上进行
        video_features = video_features.cpu()#将视频特征也移动到堆上。
        # print(video_features.shape)
        # torch.Size([64, 512])
        # raise FileExistsError
        video_features=video_features.reshape(bz,clip_len,-1)
        
        num_attributes = attributes_features.size(0)

        #对属性特征缓存里面的每个视觉属性词，进行更新
        for attr_idx in range(num_attributes):
            #一个batch里每个视频与特定文本标签的相似度
            similarities = affinity[:, attr_idx]  # (batch_size,)
            # 使用 torch.topk 获取每个文本类别与视频的相似度排名，返回前 k 个
            # print(similarities.shape)
            #选择与每个属性最相关的k个视觉特征
            top_similarities, top_indices = torch.topk(similarities, k=k, largest=True, sorted=True)
            # 获取最相似的视频特征
            # print(top_indices.shape)
            # print(top_indices)
            top_similarities=top_similarities.cpu()
            top_indices=top_indices.cpu()
            
            top_video_features = video_features[top_indices]
            # top_indices_list = top_indices.tolist()  # 转换为普通 Python 列表
           
           
            
            
            # if attr_idx not in self.top_k_heaps:
            #     self.top_k_heaps[attr_idx] = []
            #某个类别的文本信息对应的小根堆
            if attr_idx not in self.top_k_heaps:
                self.top_k_heaps[attr_idx] = []
            heap = self.top_k_heaps[attr_idx]
            heapq.heapify(heap)
            # print("attr_idx: "+str(attr_idx))
            # print("before heap:")
            # print(heap)
            
            for similarity, feature in zip(top_similarities, top_video_features):
                              
                similarity = similarity.item()  # 转为标量
                # print(feature.shape)
                
                feature = feature.detach()
                # print(similarity)
                # print(path)
                if len(heap) < k:
                    heapq.heappush(heap, Video(similarity, feature))
                else:
                    if similarity > heap[0].similarity:  # 如果新相似度大于堆中最小相似度
                        heapq.heappushpop(heap, Video(similarity, feature))
                        
            # print("after heap:")
            # print(heap)  
    
    def save_features(self,path):
        print("Saving attributes videos and features to file...")
        # print(self.top_k_heaps)
        with open(path, "wb") as f:
            pickle.dump(self.top_k_heaps, f)

      #作用：计算与每个增广类别文本名称最相似的视觉属性信息。
    def retrieval_most_action_attributes(self,proxy_labels,action_label_file,attribute_label_file,bz,attribute_num_perInstance=4):
        
        #计算了在sentenceBert模型的文本编码中，行为类别文本和行为属性词文本的关系
        if self.action_attribute_matrix==None:
            text_prefix=f'a video of {{}}.'
            attribute_temp_mapping = json.load(open(attribute_label_file, 'r'))
            attribute_sentence_list=[]
            
            for key in attribute_temp_mapping:
                attribute_sentence_list.append(text_prefix.format(attribute_temp_mapping[key]))
            action_temp_mapping = json.load(open(action_label_file, 'r'))
            action_sentence_list=[]
            for key in action_temp_mapping:
                action_sentence_list.append(text_prefix.format(action_temp_mapping[key]))
                
            with torch.no_grad():
                attribute_embeddings=self.sentence_model.encode(attribute_sentence_list)
                action_embeddings=self.sentence_model.encode(action_sentence_list)
            
            attribute_embeddings=torch.from_numpy(attribute_embeddings)
            action_embeddings=torch.from_numpy(action_embeddings)
            
            attribute_embeddings = F.normalize(attribute_embeddings, p=2, dim=-1)
            action_embeddings = F.normalize(action_embeddings, p=2, dim=-1)
            
            self.action_embeddings=action_embeddings
            self.attribute_embeddings=attribute_embeddings

            self.action_attribute_matrix= (action_embeddings@attribute_embeddings.T).to(next(self.parameters()).device).detach()
            
        # print(self.action_attribute_matrix.shape)
        # torch.Size([25, 312])
        # proxy_class_nums, attribute_nums
        #与一个batch内的每个增广类别和对应的属性相似度
        batch_class_attribute_matrix=self.action_attribute_matrix[proxy_labels]
        # print(batch_class.shape)
        # torch.Size([8, 312])
        # bz ,attribute_nums
        attribute_ids=[]
        attribute_similarities=[]
        #选了与每个行为类别最相似的bz个视觉属性
        for index in range(bz):
            top_similarities, top_indices = torch.topk(batch_class_attribute_matrix[index], k=attribute_num_perInstance, largest=True, sorted=True)
            attribute_ids.append(top_indices)
            attribute_similarities.append(top_similarities)

        #与每个增广类别最相似的attribute_num_perInstance个视觉属性id,对应的attribute_num_perInstance个特征相似度
        return attribute_ids, attribute_similarities
        
        
        
            
            
        
        
        
        
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.Conv3d)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        
        if isinstance(l, (nn.Parameter)):
            l.data = l.data.half()
        
        if isinstance(l, (nn.LayerNorm)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    if isinstance(attr, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.Conv3d)):
                        attr.weight.data = attr.weight.data.half()
                        if attr.bias is not None:
                            attr.bias.data = attr.bias.data.half()
                    else:
                        attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

class WCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # video
                 T=8,
                 temporal_modeling_type=None,
                 # other
                 use_checkpoint=False,
                 num_experts=0,
                 expert_insert_layers=[],
                 record_routing=False,
                 routing_type = 'patch-level'
                ):
        super().__init__(
                embed_dim,
                image_resolution, vision_layers, vision_width, vision_patch_size,
                context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
            )

        self.vision_width = vision_width
        vision_heads = vision_width // 64
        self.visual = TemporalVisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            T=T,
            temporal_modeling_type=temporal_modeling_type,
            use_checkpoint=use_checkpoint,
            num_experts=num_experts,
            expert_insert_layers=expert_insert_layers,
            record_routing = record_routing,
            routing_type = routing_type,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(max(self.context_length, 77), transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.embed_dim = embed_dim
        self.initialize_parameters()
        self.temporal_modeling_type = temporal_modeling_type
        
        
    # ignore. copy from videoX
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}
    
    def encode_image(self, image, maeout=None):
        if maeout is not None:
            maskf = maeout[0]
            mask = maeout[1]
        else:
            maskf, mask = None, None
        return self.visual([image.type(self.dtype), [maskf, mask]])

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def prompt_encode_text(self, prompts, tokenized_prompts,):
        prompts = prompts.type(self.dtype)
        x = prompts + self.positional_embedding.type(self.dtype)[:self.context_length, :]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x 


def build_model(state_dict: dict, T=8, temporal_modeling_type=None, use_checkpoint=False,
                context_length=None, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level'):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    
    else:
        raise NotImplementedError
    
    embed_dim = state_dict["text_projection"].shape[1]
    if context_length:
        context_length = context_length
    else:
        context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64

    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    model = WCLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            T=T, temporal_modeling_type=temporal_modeling_type,
            use_checkpoint=use_checkpoint, num_experts=num_experts,
            expert_insert_layers=expert_insert_layers,
            record_routing=record_routing,
            routing_type=routing_type,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    # convert_weights(model)
    if num_experts > 0:
        for key in list(state_dict.keys()):
            if 'mlp' in key and key.startswith('visual'):
                for expert_id in range(num_experts):
                    if 'c_fc' in key or 'gelu' in key:
                        new_key = key.replace('mlp', 'experts_head.%d'%expert_id)
                    else:
                        new_key = key.replace('mlp', 'experts_tail.%d'%expert_id)
                    state_dict[new_key] = state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    logger.info("load pretrained CLIP:{}".format(msg))

    return model.eval()



def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        jit:bool = False, download_root: str = None, T=8, temporal_modeling_type=False, use_checkpoint=False, context_length = 77, num_experts=0, expert_insert_layers=[], record_routing=False, routing_type='patch-level'):
    
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
     
    model = build_model(state_dict or model.state_dict(), 
            T=T, temporal_modeling_type=temporal_modeling_type, 
            use_checkpoint=use_checkpoint, context_length = context_length,
            num_experts=num_experts, expert_insert_layers=expert_insert_layers,
            record_routing=record_routing, routing_type=routing_type
            ).to(device)
    if str(device) == "cpu":
        model.float()

    return model, _transform(model.visual.input_resolution)

if __name__ == '__main__':
    model, preprocess = clip.load("/share/home/jia/.cache/clip/ViT-B-32.pt", jit=False, )
    
    # model: text and vision