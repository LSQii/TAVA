# 创建脚本：verify_pyav_load.py（用pyav读取，和模型逻辑一致）
import csv
import os
import av

TEST_CSV = "/root/TAVA/zs_label_db/B2N_hmdb/test_cleaned.csv"
VIDEO_ROOT = "/root/TAVA/data/hmdb51/videos/"
SEPARATOR = ","
REQUIRED_FRAMES = 32  # 模型要求至少32帧

# 校验单个视频是否能被pyav读取，且帧数足够
def pyav_can_load(row):
    video_path = os.path.join(VIDEO_ROOT, row[0].strip())
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        frame_count = stream.frames  # 实际能读取的帧数
        container.close()
        
        # 两个条件：能打开 + 帧数≥32（满足模型预处理要求）
        if frame_count >= REQUIRED_FRAMES:
            return row, True, "正常"
        else:
            return row, False, f"帧数不足（实际{frame_count}帧）"
    except Exception as e:
        return row, False, f"pyav加载失败：{str(e)[:30]}"

# 校验前10个样本（快速定位问题）
print("用pyav（模型同款解码器）校验前10个样本：")
with open(TEST_CSV, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=SEPARATOR)
    for idx, row in enumerate(reader):
        if idx >= 10:
            break
        row, is_valid, msg = pyav_can_load(row)
        status = "✅" if is_valid else "❌"
        print(f"  行号{idx}：{row[0]} → {status} {msg}")