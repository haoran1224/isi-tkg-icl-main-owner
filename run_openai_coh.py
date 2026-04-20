"""
基于大模型的时序图谱预测方法 CoH (Chain-of-History) - 启动入口
使用 prepare_history_chain_coh 函数，实现多步逐步推理
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
)

from prepare_history_chain_coh import prepare_history_chain_coh


if __name__ == "__main__":
    args = get_args()

    test_data, head_search_space, tail_search_space = load_data(args)

    adjust_top_k(test_data, args)

    metric = HitsMetric()
    filename = get_filename(args)

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

            model_input, candidates = prepare_history_chain_coh(
                x, search_space, args
            )

            if args.model == "chatGLM":
                predictions = predict_k_chatGLM(model_input)
            else:
                # predictions = predict(model_input, args)
                continue

            update_history(x, search_space, predictions, candidates, args)

            example = write_results(x, predictions, candidates, direction, writer, args)

            update_metric(example, metric, args)
            pbar.set_postfix(metric.dump())

    print("\n" + "=" * 60)
    print("CoH processing complete!")
    print("=" * 60 + "\n")
