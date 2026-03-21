import torch
import os
import argparse

def average_checkpoint(checkpoint_list, raw_clip, wa_start, wa_end):
    ckpt_list = []

    # raw clip
    raw_clip_weight = {}
    clip_ori_state = torch.jit.load(raw_clip, map_location='cpu').state_dict()
    _ = [clip_ori_state.pop(i) for i in ['input_resolution', 'context_length', 'vocab_size']]
    for key in clip_ori_state:
        raw_clip_weight['model.' + key] = clip_ori_state[key]

    ckpt_list.append((0, raw_clip_weight))
    for name, ckpt_id in checkpoint_list:
        ckpt_list.append((ckpt_id, torch.load(name, map_location='cpu')['model_state']))

    # 查找需要平均的 key
    linear_proj_keys = []
    for k in ckpt_list[-1][1].keys():
        if any(x in k for x in ['projector', 'adapter', 'post_prompt']):
            linear_proj_keys.append(k)
    print("Keys to average:", linear_proj_keys)

    # 筛选要参与平均的模型
    new_ckpt_list = []
    ckpt_id_list = []
    for i in ckpt_list:
        if wa_start <= int(i[0]) <= wa_end:
            new_ckpt_list.append(i)
            ckpt_id_list.append(int(i[0]))

    print("Files with the following checkpoints will participate in parameter averaging:")
    print(ckpt_id_list)

    # 执行平均操作
    state_dict = {}
    for key in raw_clip_weight:
        state_dict[key] = [ckpt[1][key] for ckpt in new_ckpt_list]

    for key in linear_proj_keys:
        state_dict[key] = [ckpt[1][key] for ckpt in new_ckpt_list]

    for key in state_dict:
        try:
            state_dict[key] = torch.mean(torch.stack(state_dict[key], 0), 0)
        except Exception as e:
            print(f"Skipping key due to error: {key} -> {e}")

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Average model checkpoints with raw CLIP base.")
    parser.add_argument('--source_dir', type=str, required=True, help='Directory of model checkpoints')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the averaged model')
    parser.add_argument('--raw_clip', type=str, required=True, help='Path to raw ViT-B-16 CLIP weights (.pt)')
    parser.add_argument('--wa_start', type=int, default=2, help='Start epoch of weight averaging')
    parser.add_argument('--wa_end', type=int, default=12, help='End epoch of weight averaging')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_list = [
        (os.path.join(args.source_dir, i), int(i.split('.')[0].split('_')[-1]))
        for i in os.listdir(args.source_dir)
    ]
    checkpoint_list = sorted(checkpoint_list, key=lambda d: d[1])

    swa_state_dict = average_checkpoint(checkpoint_list, args.raw_clip, args.wa_start, args.wa_end)

    output_path = os.path.join(args.output_dir, f'swa_{args.wa_start}_{args.wa_end}.pth')
    torch.save({'model_state': swa_state_dict}, output_path)
    print(f"Averaged model saved to: {output_path}")


if __name__ == '__main__':
    main()
