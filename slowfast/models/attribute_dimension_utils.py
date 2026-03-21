import json
import os
import numpy as np
import torch

class AttributeDimensionClassifier:
    """
    视觉属性维度分类器，将属性映射到高级抽象维度
    """
    def __init__(self, mapping_file_path):
        """
        初始化属性维度分类器
        
        参数:
            mapping_file_path: 属性维度映射JSON文件路径
        """
        if not os.path.exists(mapping_file_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")
            
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
            
        self.dimensions = self.mapping_data['dimensions']
        self.dimension_description = self.mapping_data['dimension_description']
        
        # 创建反向映射：属性ID -> 维度列表
        self.attribute_to_dimensions = self._create_attribute_to_dimensions_mapping()
        
    def _create_attribute_to_dimensions_mapping(self):
        """
        创建属性ID到维度列表的反向映射
        """
        attribute_to_dimensions = {}
        for dim, attrs in self.dimensions.items():
            for attr_id in attrs:
                if attr_id not in attribute_to_dimensions:
                    attribute_to_dimensions[attr_id] = []
                attribute_to_dimensions[attr_id].append(dim)
        return attribute_to_dimensions
    
    def get_dimensions_for_attribute(self, attribute_id):
        """
        获取特定属性ID所属的维度
        
        参数:
            attribute_id: 属性ID (支持整数或字符串)
            
        返回:
            维度列表，如果属性ID不存在则返回空列表
        """
        # 统一转换为整数处理
        if isinstance(attribute_id, str):
            attribute_id = int(attribute_id)
        return self.attribute_to_dimensions.get(attribute_id, [])
    
    def get_attribute_dimension(self, attribute_id):
        """
        获取特定属性ID所属的第一个维度（兼容旧版本API）
        
        参数:
            attribute_id: 属性ID (支持整数或字符串)
            
        返回:
            维度名称，如果属性ID不存在则返回None
        """
        dimensions = self.get_dimensions_for_attribute(attribute_id)
        return dimensions[0] if dimensions else None
    
    def get_attributes_dimensions(self, attribute_ids):
        """
        批量获取属性ID所属的维度列表（兼容旧版本API）
        
        参数:
            attribute_ids: 属性ID列表 (支持整数或字符串)
            
        返回:
            维度列表，与输入的attribute_ids一一对应
        """
        return [self.get_attribute_dimension(attr_id) for attr_id in attribute_ids]
    
    def is_attribute_in_dimension(self, attribute_id, dimension):
        """
        检查属性ID是否属于特定维度
        
        参数:
            attribute_id: 属性ID (支持整数或字符串)
            dimension: 维度名称
            
        返回:
            如果属性ID属于该维度则返回True，否则返回False
        """
        dimensions = self.get_dimensions_for_attribute(attribute_id)
        return dimension in dimensions
    
    def get_attributes_in_dimension(self, dimension):
        """
        获取特定维度下的所有属性ID
        
        参数:
            dimension: 维度名称
            
        返回:
            属性ID列表，如果维度不存在则返回空列表
        """
        return self.dimensions.get(dimension, [])
    
    def get_all_dimensions(self):
        """
        获取所有高级抽象维度
        
        返回:
            维度名称列表
        """
        return list(self.dimensions.keys())
    
    def get_dimension_description(self, dimension):
        """
        获取特定维度的描述
        
        参数:
            dimension: 维度名称
            
        返回:
            维度描述字符串，如果维度不存在则返回空字符串
        """
        return self.dimension_description.get(dimension, "")
    
    def classify_attributes(self, attribute_ids):
        """
        对一批属性ID进行维度分类
        
        参数:
            attribute_ids: 属性ID列表
            
        返回:
            字典，键为维度名称，值为该维度下的属性ID列表
        """
        classified = {dim: [] for dim in self.dimensions}
        
        for attr_id in attribute_ids:
            dims = self.get_dimensions_for_attribute(attr_id)
            for dim in dims:
                classified[dim].append(attr_id)
        
        return classified
    
    def get_dimension_embedding(self, attribute_ids, dimension, attribute_embeddings):
        """
        获取特定维度下的属性嵌入
        
        参数:
            attribute_ids: 属性ID列表
            dimension: 维度名称
            attribute_embeddings: 所有属性的嵌入矩阵 (shape: [num_attributes, embedding_dim])
            
        返回:
            该维度下的属性嵌入矩阵 (shape: [num_dimension_attributes, embedding_dim])
        """
        # 获取该维度下的所有属性ID
        dim_attr_ids = self.get_attributes_in_dimension(dimension)
        
        # 过滤出attribute_ids中属于该维度的属性
        filtered_attr_ids = [attr_id for attr_id in attribute_ids if attr_id in dim_attr_ids]
        
        if not filtered_attr_ids:
            return None
            
        # 获取对应的嵌入
        return attribute_embeddings[filtered_attr_ids]
    
    def group_attributes_by_dimension(self, attribute_ids):
        """
        将属性ID按维度分组
        
        参数:
            attribute_ids: 属性ID列表
            
        返回:
            字典，键为维度名称，值为该维度下的属性ID列表
        """
        grouped = {dim: [] for dim in self.dimensions}
        
        for attr_id in attribute_ids:
            dims = self.get_dimensions_for_attribute(attr_id)
            for dim in dims:
                grouped[dim].append(attr_id)
        
        # 移除空列表
        return {dim: attrs for dim, attrs in grouped.items() if attrs}
