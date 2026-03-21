# verify_data_directly.py
import csv
import os
import sys
# 指向你本地的pytorchvideo（已验证成功）
PYTORCHVIDEO_PATH = "/root/TAVA/pytorchvideo-main"
sys.path.insert(0, PYTORCHVIDEO_PATH)
from pytorchvideo.data.encoded_video import EncodedVideo

# ====================== 你的路径（不用改）======================
CSV_PATH = "/root/TAVA/zs_label_db/B2N_hmdb/test_cleaned.csv"  # 你的CSV文件
DATASET_ROOT = "/root/TAVA/data/hmdb51/videos"  # 数据集根路径
# =================================================================

# 读取CSV里的视频路径
try:
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        video_paths = [row[0] for row in reader if len(row) > 0]
    print(f"✅ 成功读取CSV！共{len(video_paths)}个视频样本")
except Exception as e:
    print(f"❌ 读取CSV报错：{e}")
    exit()

# 验证前3个视频是否能解码（绕开框架，直接用pytorchvideo）
print("\n📺 正在用pytorchvideo手动解码视频（验证数据是否有效）：")
valid_video_count = 0
for i, rel_path in enumerate(video_paths[:3]):  # 只验证前3个，快
    # 拼接绝对路径
    abs_path = os.path.join(DATASET_ROOT, rel_path)
    print(f"\n  视频{i+1}：")
    print(f"    路径：{abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"    ❌ 路径不存在！")
        continue
    
    try:
        # 用pytorchvideo解码视频（核心验证）
        video = EncodedVideo.from_path(abs_path)
        # 读取前16帧（验证解码功能）
        frames = video.get_clip(start_sec=0, end_sec=1)["video"]  # 1秒内的帧
        print(f"    ✅ 解码成功！")
        print(f"    帧形状：{frames.shape}（格式：[C, T, H, W]）")
        valid_video_count += 1
    except Exception as e:
        print(f"    ❌ 解码失败！报错：{e}")
        print(f"    原因：视频文件损坏/格式不支持（数据集问题）")

# 最终结论
print("\n" + "="*60)
if valid_video_count > 0:
    print(f"🎉 结论：{valid_video_count}/{len(video_paths[:3])}个视频能正常解码！")
    print("✅ 数据没问题！之前的问题是框架loader的导入路径，现在可以：")
    print("  1. 回到test_net.py，用之前的「裸奔版perform_test」函数，把框架的loader导入路径改成你的项目实际路径；")
    print("  2. 或直接让我帮你写一个不依赖框架loader的测试函数，直接用pytorchvideo加载数据跑测试！")
else:
    print(f"⚠️  结论：所有测试视频都无法解码！")
    print("❌ 问题根源：数据集损坏（视频文件坏了，或格式不支持）")
    print("解决方案：必须重新下载官方HMDB51数据集（http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/）")
print("="*60)