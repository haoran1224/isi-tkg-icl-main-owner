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
    write_results, prepare_history_chain, get_chain_filename,
    retrieve_global_history_facts,
    print_round_statistics,
    save_round2_samples_to_file,
)

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

            # 大模型自主控制的历史检索模块
            # 先检索全局历史事实：在所有历史上发生的s-r都一致的四元组
            global_history_quadruples = retrieve_global_history_facts(x, search_space, args)
            # 准备历史事件链，同时传入全局历史四元组，将在内部正确整合局部-全局历史
            model_input, candidates = prepare_history_chain(x, search_space, args, fileChainName, global_history_quadruples)

            if args.model == "chatGLM":
                predictions = predict_k_chatGLM(model_input)
            else:
                # predictions = predict(model_input, args)
                continue

            update_history(x, search_space, predictions, candidates, args)

            example = write_results(x, predictions, candidates, direction, writer, args)

            update_metric(example, metric, args)
            pbar.set_postfix(metric.dump())

    # 处理完所有查询后，自动进行轮次统计
    print("\n" + "="*60)
    print("所有查询处理完成，开始轮次统计...")
    print("="*60)

    # 打印轮次统计信息
    print_round_statistics()

    # 保存结束轮次为2的样本到文件
    save_round2_samples_to_file()

    print("="*60)
    print("轮次统计完成！")
    print("="*60 + "\n")
