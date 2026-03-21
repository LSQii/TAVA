export CUDA_VISIBLE_DEVICES=0,1,2,3
ROOT=/root/TAVA/
CKPT=/root/autodl-tmp

cd $ROOT

# HMDB51 数据集配置
TRAIN_FILE=train.csv
VAL_FILE=val.csv
TEST_FILE=test.csv

TORCH_DISTRIBUTED_DEBUG=INFO python -W ignore -u tools/run_net.py \
  --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter_HMDB51.yaml \
  --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/hmdb_split1 \
  DATA.PATH_PREFIX $ROOT/data/hmdb51/videos \
  TRAIN_FILE $TRAIN_FILE \
  VAL_FILE $VAL_FILE \
  TEST_FILE $TEST_FILE \
  DATA.PATH_LABEL_SEPARATOR , \
  DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/hmdb-index2cls.json \
  DATA.PROXY_LABEL_MAPPING_FILE  $ROOT/zs_label_db/B2N_hmdb/test_rephrased.json \
  DATA.ACTION_ATTRIBUTE_MAPPING_FILE $ROOT/attribute_files/final_visual_attributes_hmdb.json \
  TRAIN.ENABLE True \
  OUTPUT_DIR $CKPT/basetraining/hmdb51_froster \
  TRAIN.BATCH_SIZE 32 \
  TEST.BATCH_SIZE 48 \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  TEST.NUM_SPATIAL_CROPS 1 \
  NUM_GPUS 4 \
  SOLVER.MAX_EPOCH 22 \
  SOLVER.WARMUP_EPOCHS 2.0 \
  SOLVER.BASE_LR 3.33e-6 \
  SOLVER.WARMUP_START_LR 3.33e-8 \
  SOLVER.COSINE_END_LR 3.33e-8 \
  TRAIN.MIXED_PRECISION True \
  DATA.DECODING_BACKEND "pyav" \
  MODEL.NUM_CLASSES 51 \
  MIXUP.ENABLE False \
  AUG.ENABLE False \
  AUG.NUM_SAMPLE 1 \
  TRAIN.EVAL_PERIOD 3 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  MODEL.LOSS_FUNC soft_cross_entropy \
  TRAIN.LINEAR_CONNECT_CLIMB False \
  TRAIN.CLIP_ORI_PATH /root/.cache/clip/ViT-B-16.pt \
  TRAIN.LINEAR_CONNECT_LOSS_RATIO 0.0 \
  MODEL.RAW_MODEL_DISTILLATION True \
  MODEL.KEEP_RAW_MODEL True \
  MODEL.DISTILLATION_RATIO 2.0


if [ -d "$CKPT/basetraining/hmdb51_froster/checkpoints" ]; then
  echo "训练完成，开始权重平均..."
  python weight_average_tool.py \
    --source_dir $CKPT/basetraining/hmdb51_froster/checkpoints \
    --output_dir $CKPT/basetraining/hmdb51_froster/wa_checkpoints \
    --raw_clip /root/.cache/clip/ViT-B-16.pt \
    --wa_start 2 \
    --wa_end 22
else
  echo "训练尚未完成，跳过权重平均步骤"
fi

