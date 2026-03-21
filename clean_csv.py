import csv

# 输入文件路径
input_file = "/data/usr/OV-HAR/attribute_end_to_end/FROSTER-main/zs_label_db/B2N_k400/train_3.csv"
# 输出文件路径
output_file = "/data/usr/OV-HAR/attribute_end_to_end/FROSTER-main/zs_label_db/B2N_k400/train_3_name_only.csv"

# 打开输入文件并处理
with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # 拆分路径，保留第一部分和最后一部分
        parts = row[0].split('/')
        modified_path = f"{parts[0]}/{parts[-1]}"  # 拼接第一部分和最后一部分
        # 写入新的文件
        writer.writerow([modified_path, row[1]])

print(f"处理完成，结果已保存到 {output_file}")

