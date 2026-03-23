"""
基于大模型的时序图谱预测方法 V2 - 启动入口
使用新的 prepare_history_chain_v2 函数，融合局部分支和全局分支
"""

import torch
from tqdm import tqdm

from selfCode.LLMAPI.qwen_utils import predict_k_chatGLM
from utils import (
    HitsMetric,
    adjust_top_k,
    get_args,
    get_filename,
    load_data,
    update_history,
    update_metric,
    write_results,
    get_chain_filename,
    print_round_statistics,
    save_round2_samples_to_file,
    retrieve_global_history_facts,
)

from prepare_history_chain_v2 import prepare_history_chain_v2


if __name__ == "__main__":
    args = get_args()

    test_data, head_search_space, tail_search_space = load_data(args)

    adjust_top_k(test_data, args)

    metric = HitsMetric()
    filename = get_filename(args)
    fileChainName = get_chain_filename(args)

    with torch.no_grad(), open(filename, "w", encoding="utf-8") as writer, tqdm(test_data) as pbar:
        for i, (x, direction) in enumerate(pbar):
            if i % args.world_size != args.rank:
                continue

            if direction == "tail":
                search_space = head_search_space
            elif direction == "head":
                search_space = tail_search_space
                continue
            else:
                raise ValueError


            global_history_quadruples = retrieve_global_history_facts(x, search_space, args)
            model_input, candidates = prepare_history_chain_v2(
                x, search_space, args, fileChainName, global_history_quadruples
            )

            if args.model == "chatGLM":
                predictions = predict_k_chatGLM(model_input)

                for cand in candidates:
                    if cand not in predictions:
                        predictions.append(cand)
                        if len(predictions) >= 10:  # 保证列表至少有10个以提高Hits@10
                            break
            else:
                # predictions = predict(model_input, args)
                continue

            update_history(x, search_space, predictions, candidates, args)

            example = write_results(x, predictions, candidates, direction, writer, args)

            update_metric(example, metric, args)
            pbar.set_postfix(metric.dump())

    # 处理完所有查询后，打印统计信息
    print("\n" + "="*60)
    print("所有查询处理完成，开始统计...")
    print("="*60)

    # 打印统计信息
    print_round_statistics()
    save_round2_samples_to_file()

    print("="*60)
    print("处理完成！")
    print("="*60 + "\n")