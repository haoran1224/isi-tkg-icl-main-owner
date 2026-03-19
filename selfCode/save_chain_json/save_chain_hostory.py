import argparse
import json
import os

# 写成jsonl文件
def save_generated_chains_jsonl(x, round_num, generated_chains, args, output_file=None):
    if output_file is None:
        base_filename = get_chain_filename(args)
        output_file = base_filename.replace(".json", "_generated_chains.jsonl")
    else:
        output_file = output_file.replace(".json", ".jsonl")

    prediction_target = f"{x[0]}_{x[1]}_{x[2]}_{x[3]}"
    existing_data = {}

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # 只有文件不为空才加载
                    existing_data = json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: File {output_file} is corrupted or empty. Starting new.")
            existing_data = {}  # 解析失败则重置，避免程序崩溃
        except Exception as e:
            print(f"Error reading file: {e}")
            return  # 读取IO错误直接返回

    if prediction_target not in existing_data:
        existing_data[prediction_target] = {}
    if str(round_num) not in existing_data[prediction_target]:
        existing_data[prediction_target][str(round_num)] = []

    for chain_item in generated_chains:
        chain, combined_score, chain_id, quad_score, time_score, rel_score, tlogic_score = chain_item
        chain_entry = {
            "new_chain": chain,
            "score": combined_score,
            "chain_id": chain_id,
            "quad_score": quad_score,
            "time_score": time_score,
            "rel_score": rel_score,
            "tlogic_score": tlogic_score
        }
        existing_data[prediction_target][str(round_num)].append(chain_entry)

    try:
        with open(output_file, 'w', encoding='utf-8') as writer:
            json.dump(existing_data, writer, indent=4)  # 使用 json.dump 直接写入流
        # print(f"Saved...")
    except Exception as e:
        print(f"Error writing: {e}")


def get_chain_filename(args: argparse.Namespace):
    model_name = args.model.split("/")[-1]
    filename_args = "_".join(
        [
            model_name,
            args.dataset,
            "llm_evaluation_back_"
            f"history_len_{args.history_len}",
        ]
    )
    filename = f"outputs/{filename_args}_chains.json"
    print(f"output chain file: {filename}")
    return filename