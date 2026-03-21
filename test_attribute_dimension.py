#!/usr/bin/env python3
"""
测试属性维度分类功能的脚本
"""

import sys
import os
import json
import numpy as np

# 指向你本地的pytorchvideo（已验证成功）
PYTORCHVIDEO_PATH = "/root/TAVA/pytorchvideo-main"
sys.path.insert(0, PYTORCHVIDEO_PATH)

from slowfast.models.attribute_dimension_utils import AttributeDimensionClassifier


def test_attribute_dimension_classifier():
    """
    测试属性维度分类器的功能
    """
    print("=== 测试属性维度分类功能 ===")
    
    # 1. 测试属性维度分类器初始化
    mapping_file_path = "attribute_files/attribute_dimension_mapping.json"
    
    if not os.path.exists(mapping_file_path):
        print(f"错误: 属性维度映射文件不存在于 {mapping_file_path}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"当前目录文件: {os.listdir('.')}")
        return False
    
    try:
        classifier = AttributeDimensionClassifier(mapping_file_path)
        print("✓ 成功初始化属性维度分类器")
    except Exception as e:
        print(f"✗ 属性维度分类器初始化失败: {e}")
        return False
    
    # 2. 测试获取所有维度
    try:
        dimensions = classifier.get_all_dimensions()
        print(f"✓ 获取到所有维度: {dimensions}")
    except Exception as e:
        print(f"✗ 获取所有维度失败: {e}")
        return False
    
    # 3. 测试获取维度描述
    try:
        for dim in dimensions:
            description = classifier.get_dimension_description(dim)
            print(f"✓ 维度 '{dim}' 描述: {description}")
    except Exception as e:
        print(f"✗ 获取维度描述失败: {e}")
        return False
    
    # 4. 测试按维度分组属性
    try:
        # 随机选择一些属性ID进行测试
        test_attribute_ids = list(range(20))  # 测试前20个属性
        grouped_attributes = classifier.group_attributes_by_dimension(test_attribute_ids)
        print("\n✓ 按维度分组属性成功:")
        for dim, attrs in grouped_attributes.items():
            print(f"  维度 '{dim}' ({classifier.get_dimension_description(dim)}): {attrs}")
    except Exception as e:
        print(f"✗ 按维度分组属性失败: {e}")
        return False
    
    # 5. 测试获取维度下的所有属性
    try:
        for dim in dimensions:
            attrs = classifier.get_attributes_in_dimension(dim)
            print(f"✓ 维度 '{dim}' 包含 {len(attrs)} 个属性")
    except Exception as e:
        print(f"✗ 获取维度下的属性失败: {e}")
        return False
    
    # 6. 测试获取属性维度
    try:
        for attr_id in [0, 10, 20, 30, 40]:
            dim = classifier.get_attribute_dimension(attr_id)
            print(f"✓ 属性ID {attr_id} 属于维度: {dim}")
    except Exception as e:
        print(f"✗ 获取属性维度失败: {e}")
        return False
    
    # 7. 测试批量获取属性维度
    try:
        attr_ids = [5, 15, 25, 35, 45]
        dims = classifier.get_attributes_dimensions(attr_ids)
        print(f"✓ 批量获取属性维度: {list(zip(attr_ids, dims))}")
    except Exception as e:
        print(f"✗ 批量获取属性维度失败: {e}")
        return False
    
    # 8. 测试是否是某个维度的属性
    try:
        attr_id = 0
        dim = "motion"
        is_in_dim = classifier.is_attribute_in_dimension(attr_id, dim)
        print(f"✓ 属性ID {attr_id} {'属于' if is_in_dim else '不属于'} 维度 '{dim}'")
    except Exception as e:
        print(f"✗ 测试属性是否在维度失败: {e}")
        return False
    
    print("\n=== 所有测试通过! ===")
    return True


def test_hmdb_attributes():
    """
    测试HMDB51属性文件与维度映射的匹配
    """
    print("\n=== 测试HMDB51属性与维度映射匹配 ===")
    
    # 加载HMDB51属性文件
    hmdb_attribute_file = "attribute_files/final_visual_attributes_hmdb.json"
    
    if not os.path.exists(hmdb_attribute_file):
        print(f"错误: HMDB51属性文件不存在于 {hmdb_attribute_file}")
        return False
    
    try:
        with open(hmdb_attribute_file, 'r', encoding='utf-8') as f:
            hmdb_attributes = json.load(f)
        print(f"✓ 成功加载HMDB51属性文件，包含 {len(hmdb_attributes)} 个属性")
    except Exception as e:
        print(f"✗ 加载HMDB51属性文件失败: {e}")
        return False
    
    # 加载维度映射
    mapping_file_path = "attribute_files/attribute_dimension_mapping.json"
    try:
        classifier = AttributeDimensionClassifier(mapping_file_path)
    except Exception as e:
        print(f"✗ 加载维度映射失败: {e}")
        return False
    
    # 验证所有属性都已映射到维度
        # HMDB属性文件是字典格式，键为字符串类型的ID
        all_attr_ids = [int(key) for key in sorted(hmdb_attributes.keys(), key=int)]
        total_attributes = len(all_attr_ids)
        
        mapped_attr_ids = []
        for dim in classifier.get_all_dimensions():
            mapped_attr_ids.extend(classifier.get_attributes_in_dimension(dim))
        
        mapped_attr_ids = sorted(list(set(mapped_attr_ids)))
        
        if len(mapped_attr_ids) == total_attributes:
            print(f"✓ 所有 {total_attributes} 个属性都已映射到维度")
        else:
            unmapped_ids = [id for id in all_attr_ids if id not in mapped_attr_ids]
            print(f"✗ 有 {len(unmapped_ids)} 个属性未映射到任何维度: {unmapped_ids}")
            
            # 显示未映射的属性内容
            if unmapped_ids:
                print("未映射的属性内容:")
                for idx in unmapped_ids[:5]:  # 只显示前5个
                    print(f"  ID {idx}: {hmdb_attributes[str(idx)]}")
            return False
    
    # 显示各维度的属性数量和示例
        print("\n各维度属性统计:")
        for dim in classifier.get_all_dimensions():
            attrs = classifier.get_attributes_in_dimension(dim)
            print(f"\n维度 '{dim}' ({classifier.get_dimension_description(dim)}): {len(attrs)} 个属性")
            print("示例属性:")
            for idx in attrs[:3]:  # 显示前3个示例
                print(f"  ID {idx}: {hmdb_attributes[str(idx)]}")
    
    return True


def test_dimension_usage():
    """
    测试维度分类功能在实际场景中的使用
    """
    print("\n=== 测试维度分类功能实际使用场景 ===")
    
    mapping_file_path = "attribute_files/attribute_dimension_mapping.json"
    
    try:
        classifier = AttributeDimensionClassifier(mapping_file_path)
    except Exception as e:
        print(f"✗ 初始化分类器失败: {e}")
        return False
    
    # 模拟使用场景：将相似属性按维度分组
    print("\n模拟场景：将相似属性按维度分组")
    
    # 假设这些是与某个动作最相似的属性ID
    similar_attribute_ids = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    
    print(f"相似属性ID: {similar_attribute_ids}")
    
    # 按维度分组
    grouped_attributes = classifier.group_attributes_by_dimension(similar_attribute_ids)
    
    print("按高级抽象维度分组结果:")
    for dim, attrs in grouped_attributes.items():
        print(f"\n{dim} ({classifier.get_dimension_description(dim)}):")
        print(f"  属性ID: {attrs}")
    
    return True


if __name__ == "__main__":
    print("开始测试属性维度分类功能...")
    print("当前工作目录:", os.getcwd())
    
    test1 = test_attribute_dimension_classifier()
    test2 = test_hmdb_attributes()
    test3 = test_dimension_usage()
    
    print("\n=== 测试结果总结 ===")
    print(f"属性维度分类器功能测试: {'通过' if test1 else '失败'}")
    print(f"HMDB51属性与维度映射匹配测试: {'通过' if test2 else '失败'}")
    print(f"维度分类功能实际使用场景测试: {'通过' if test3 else '失败'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 所有测试通过! 维度分类功能正常工作。")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败，需要检查修复。")
        sys.exit(1)

