import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import random
import json

from selfCode.LLM_util.score_LLM_chain import prune_relation_set, prune_quadruples_score_set, evaluate_chain_sufficiency
from typing import Any, Dict, List, Optional

from selfCode.save_chain_json.save_chain_hostory import save_generated_chains_jsonl

MAX_HITS = 10

# 轮次统计全局变量
round_statistics = {
    "total_queries": 0,
    "round_distribution": {},  # {round_num: count}
    "round2_samples": []  # 存储结束轮次为2的样本详情
}


@dataclass
class HitsMetric:
    total: int = 0
    hit1: int = 0
    hit3: int = 0
    hit10: int = 0
    mrr_sum: float = 0.0  # 累积倒数排名和

    def update(self, rank):
        if rank <= 1:
            self.hit1 += 1
        if rank <= 3:
            self.hit3 += 1
        if rank <= 10:
            self.hit10 += 1
        # MRR: 1/rank
        self.mrr_sum += 1.0 / rank

    def dump(self):
        return {
            "total": self.total,
            "hit1": self.hit1 / self.total,
            "hit3": self.hit3 / self.total,
            "hit10": self.hit10 / self.total,
            "mrr": self.mrr_sum / self.total,
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chatGLM", type=str)
    parser.add_argument(
        "--dataset",
        choices=["ICEWS14", "ICEWS18", "WIKI", "YAGO"],
        default="ICEWS18",
        type=str,
    )
    parser.add_argument(
        "--multi_step", default=False, action="store_true"
    )  # inference in multi_step
    # History Modeling
    parser.add_argument(
        "--history_type", choices=["entity", "pair"], default="entity", type=str
    )  # history type
    parser.add_argument(
        "--history_direction", choices=["uni", "bi"], default="uni", type=str
    )  # history type
    parser.add_argument("--history_len", default=0, type=int)  # length of history
    parser.add_argument("--history_top_k", default=1, type=int)  # length of targets from history
    # Prompt Construction
    parser.add_argument("--label", default=False, action="store_true")  # express prompt with label
    parser.add_argument(
        "--text_style", default=False, action="store_true"
    )  # express prompt in text
    parser.add_argument(
        "--no_entity", default=False, action="store_true"
    )  # express prompt without entity
    parser.add_argument("--sys_instruction", default="", type=str)  # system instcution for ChatGPT
    parser.add_argument(
        "--no_time", default=False, action="store_true"
    )  # express prompt without time
    parser.add_argument("--shuffle_history", default=False, action="store_true")  # shuffle history
    # Hyperparameter
    parser.add_argument("--top_k", default=20, type=int)  # number of predictions to store
    parser.add_argument(
        "--dec_cand", default=5, type=int
    )  # number of candidates to decode at each step
    parser.add_argument("--max_length", default=1, type=int)  # max decoding length
    parser.add_argument("--world_size", default=1, type=int)  # number of chunks
    parser.add_argument("--rank", default=0, type=int)  # rankd of the executor
    parser.add_argument(
        "--tokenizer_revision", default="main", type=str
    )  # change tokenizer revision (for llama)
    parser.add_argument(
        "--fp16", default=False, action="store_true"
    )  # use float16 instead of float32
    parser.add_argument("--verbose", default=False, action="store_true")  # print extra information
    # Evaluation
    parser.add_argument(
        "--eval_filter",
        choices=["none", "static", "time-aware"],
        type=str,
        default="none",
    )
    parser.add_argument("--max_rounds", default=2, type=int)  # 多轮迭代的轮次
    parser.add_argument("--use_llm_global", default=False, action="store_true")  # V2: 使用LLM生成全局分支描述

    args = parser.parse_args()
    assert args.label or not args.no_entity

    return args


# Read entity2id, relation2id
def load_dictionary(in_path: str, file_name: str) -> Dict[int, str]:
    _dict = {}
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split("\t")
            node = line_split[0]
            index = int(line_split[1])

            _dict[index] = node
    return _dict


# Read train, valid data to construct search space
def load_quadruples(
        search_dictionary: Dict[Any, Dict[Any, Dict[Any, List[Any]]]],
        in_path: str,
        file_name: str,
        entity_dictionary: Optional[Dict[int, str]] = None,
        relation_dictionary: Optional[Dict[int, str]] = None,
        query: str = "head",
):
    discard_line, total_line = 0, 0
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            total_line += 1
            line_split = line.split()
            if entity_dictionary and relation_dictionary:
                if (
                        int(line_split[0]) not in entity_dictionary
                        or int(line_split[2]) not in entity_dictionary
                        or int(line_split[1]) not in relation_dictionary
                ):
                    print(line)
                    discard_line += 1
                    continue
                head = entity_dictionary[int(line_split[0])]
                tail = entity_dictionary[int(line_split[2])]
                rel = relation_dictionary[int(line_split[1])]
            else:
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])

            time = int(line_split[3])

            if query == "head":
                if head not in search_dictionary:
                    search_dictionary[head] = {}
                if time not in search_dictionary[head]:
                    search_dictionary[head][time] = {}
                if rel not in search_dictionary[head][time]:
                    search_dictionary[head][time][rel] = []
                search_dictionary[head][time][rel].append(tail)
            elif query == "tail":
                if tail not in search_dictionary:
                    search_dictionary[tail] = {}
                if time not in search_dictionary[tail]:
                    search_dictionary[tail][time] = {}
                if rel not in search_dictionary[tail][time]:
                    search_dictionary[tail][time][rel] = []
                search_dictionary[tail][time][rel].append(head)

    print(f"# line discarded due to index issue: {discard_line} / {total_line}")


# Read test data to inference
def load_quadruples_for_test(
        in_path: str,
        file_name: str,
        entity_dictionary: Optional[Dict[int, str]] = None,
        relation_dictionary: Optional[Dict[int, str]] = None,
) -> List[List[Any]]:
    test_instances = []
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split()
            if entity_dictionary and relation_dictionary:
                if (
                        int(line_split[0]) not in entity_dictionary
                        or int(line_split[2]) not in entity_dictionary
                        or int(line_split[1]) not in relation_dictionary
                ):
                    print(line)
                    continue
                head = entity_dictionary[int(line_split[0])]
                tail = entity_dictionary[int(line_split[2])]
                rel = relation_dictionary[int(line_split[1])]
            else:
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
            time = int(line_split[3])
            test_instances.append((head, rel, tail, time))
    return test_instances


def format_data(data):
    tail_prediction, head_prediction = {}, {}
    for head, rel, tail, time in data:
        tail_key = (head, rel, time)
        if tail_key not in tail_prediction:
            tail_prediction[tail_key] = []
        tail_prediction[tail_key].append(tail)

        head_key = (tail, rel, time)
        if head_key not in head_prediction:
            head_prediction[head_key] = []
        head_prediction[head_key].append(head)

    formatted_data = list(
        # sorted(
        #     [([k[0], k[1], list(set(v)), k[2]], "tail") for k, v in tail_prediction.items()]
        #     + [([k[0], k[1], list(set(v)), k[2]], "head") for k, v in head_prediction.items()],
        #     key=lambda x: x[0][3],
        # )
        sorted(
            [([k[0], k[1], list(set(v)), k[2]], "tail") for k, v in tail_prediction.items()],
            key=lambda x: x[0][3],
        )
    )
    return formatted_data


def load_data(args: argparse.Namespace):
    entity_dictionary, relation_dictionary = None, None
    if args.text_style:
        entity_dictionary = load_dictionary("data", os.path.join(args.dataset, "entity2id.txt"))
        relation_dictionary = load_dictionary("data", os.path.join(args.dataset, "relation2id.txt"))

    head_search_space = {}
    load_quadruples(
        head_search_space,
        "data",
        os.path.join(args.dataset, "train.txt"),
        entity_dictionary,
        relation_dictionary,
        query="head",
    )
    load_quadruples(
        head_search_space,
        "data",
        os.path.join(args.dataset, "valid.txt"),
        entity_dictionary,
        relation_dictionary,
        query="head",
    )

    tail_search_space = {}
    load_quadruples(
        tail_search_space,
        "data",
        os.path.join(args.dataset, "train.txt"),
        entity_dictionary,
        relation_dictionary,
        query="tail",
    )
    load_quadruples(
        tail_search_space,
        "data",
        os.path.join(args.dataset, "valid.txt"),
        entity_dictionary,
        relation_dictionary,
        query="tail",
    )

    if args.history_direction == "bi":
        head_search_space.update(tail_search_space)
        tail_search_space = head_search_space

    test_data = load_quadruples_for_test(
        "data",
        os.path.join(args.dataset, "test.txt"),
        entity_dictionary,
        relation_dictionary,
    )
    # 这一块是有解释的，避免同一时间连接多个实体，其实最后的格式是(head, rel, [tail1, tail2, ...], time)
    formatted_test_data = format_data(test_data)

    return formatted_test_data, head_search_space, tail_search_space


def adjust_top_k(test_data, args):
    max_targets_len = max([len(x[0][2]) for x in test_data])
    args.top_k = max(args.top_k, MAX_HITS, max_targets_len + MAX_HITS)
    if args.verbose:
        print(f"max targets len: {max_targets_len}")
        print(f"adjusted top k: {args.top_k}")


def get_filename(args: argparse.Namespace, is_eval: bool = False):
    model_name = args.model.split("/")[-1]
    filename_args = "_".join(
        [
            model_name,
            args.dataset,
            f"multi_step_{args.multi_step}",
            f"history_len_{args.history_len}",
            f"history_type_{args.history_type}",
            f"history_direction_{args.history_direction}",
            f"no_time_{args.no_time}",
            f"shuffle_history_{args.shuffle_history}",
            f"label_{args.label}",
            f"text_style_{args.text_style}",
            f"no_entity_{args.no_entity}",
            f'world_size_{"*" if is_eval else args.world_size}',
            f'rank_{"*" if is_eval else args.rank}',
        ]
    )
    filename = f"outputs/{filename_args}.jsonl"
    print(f"output file: {filename}")
    return filename


# 根据entity或者entity-realtion获取所有的历史
def construct_history_by_search(
        search_space: Dict[str, Any], entity: str, relation: str, history_type: str
):
    if entity not in search_space:
        return {}

    search_graph = {entity: {}}

    if history_type == "entity":
        search_graph[entity] = search_space[entity]
    elif history_type == "pair":
        search_graph[entity] = {
            k: {relation: v[relation]} for k, v in search_space[entity].items() if relation in v
        }

    return search_graph


def filter_time(history_graph: Dict[str, Any],
                question: [int, str, str]
                ):
    quadruples = []
    for entity in history_graph:
        for time in history_graph[entity]:
            if time >= question[0]:
                continue
            for relation in history_graph[entity][time]:
                for target in history_graph[entity][time][relation]:
                    quadruples.append([entity, relation, target, time])

    return quadruples


def format_history(
        history_graph: Dict[str, Any],
        history_len: int,
        question: List[str],
        args: argparse.Namespace,
        return_prompt: bool = True,
):
    # 遍历历史图，提取所有早于当前查询时间的四元组
    quadruples = filter_time(history_graph=history_graph, question=question)

    candidates_stats = {}
    if args.model == "recency":
        for x in quadruples[-history_len:]:
            if x[2] not in candidates_stats:
                candidates_stats[x[2]] = -1
            candidates_stats[x[2]] = max(candidates_stats[x[2]], x[3])
    else:
        for x in quadruples[-history_len:]:
            if x[2] not in candidates_stats:
                candidates_stats[x[2]] = 0
            candidates_stats[x[2]] += 1

    candidates_stats_sorted = list(
        sorted(candidates_stats.items(), key=lambda item: item[1], reverse=True)
    )

    candidates_mapping = {}
    for i, (entity, _) in enumerate(candidates_stats_sorted):
        candidates_mapping[entity] = i

    if (args.label or args.no_entity) and args.model not in ["recency", "frequency"]:
        candidates = {v: k for k, v in candidates_mapping.items()}  # label --> entity
    else:
        candidates = {k: k for k, _ in candidates_mapping.items()}  # entity --> entity

    if return_prompt:
        prompt = ""
        history = quadruples[-history_len:]
        if args.shuffle_history:
            random.shuffle(history)
        for x in history:
            entity, relation, target, time = x[0], x[1], x[2], x[3]
            if not args.no_time:
                prompt += f"{time}:"
            if args.no_entity:
                prompt += f"[{entity},{relation},{candidates_mapping[target]}]\n"
            elif args.label:
                prompt += f"[{entity},{relation},{candidates_mapping[target]}.{target}]\n"
            else:
                prompt += f"[{entity},{relation},{target}]\n"
        if not args.no_time:
            prompt += f"{question[0]}:"
        prompt += f"[{question[1]},{question[2]},"

        return prompt, candidates
    else:
        return candidates_stats_sorted, candidates


def get_entity_edges_before_time(entity_search_space, entity, time, length):
    """
    获取在指定时间前与实体相连的边（四元组）
    参数:
    entity_search_space: 实体搜索空间字典
    entity: 目标实体
    time: 当前时间
    length: 要获取的边数
    返回:
    四元组列表 [entity, relation, target, time]，按时间从新到旧排序，最多返回 length 条边
    返回的四元组中存在的关系列表
    """
    if entity not in entity_search_space or length <= 0:
        return [], []

    quadruples = []
    # 遍历实体的所有时间戳
    for t in entity_search_space[entity]:
        if t >= time:
            continue
        # 遍历所有关系
        for relation in entity_search_space[entity][t]:
            # 遍历所有目标实体
            for target in entity_search_space[entity][t][relation]:
                quadruples.append([entity, relation, target, t])

    # 按时间从新到旧排序
    quadruples.sort(key=lambda x: x[3], reverse=True)

    # 截取前 length 条边
    selected_quadruples = quadruples[:length]

    # 从截取后的四元组中提取关系集合
    relations = set()
    for quad in selected_quadruples:
        relations.add(quad[1])

    # 返回前 length 条边和这些边中存在的关系列表
    return selected_quadruples, list(relations)


def prepare_history_chain(x, entity_search_space, args, fileChainName, global_history_quadruples=None):
    """
    准备历史事件链，集成关系和实体剪枝功能（统一处理第一轮和多轮扩展）

    参数:
    x: 查询四元组，格式为 [subject, relation, target, time]
    entity_search_space: 实体搜索空间字典
    args: 配置参数
    global_history_quadruples: 全局历史四元组列表，可选

    返回:
    last_query_prompt: 最终查询的prompt
    []
    """
    entity, relation, query_time = x[0], x[1], x[3]
    evidence_chains = {}
    total_rounds = args.max_rounds + 1

    end_round = 0  # 记录实际结束轮次
    for round_idx in range(total_rounds):
        if round_idx == 0:
            # 第一轮：从查询实体开始初始化事件链
            evidence_chains = _expand_chains_from_entity(
                [], entity_search_space, x, round_idx, args, fileChainName
            )
        else:
            # 多轮扩展：从现有事件链扩展
            evidence_chains = _expand_chains_from_existing(
                evidence_chains, entity_search_space, x, round_idx, args, fileChainName
            )

        # 更新结束轮次（当前轮次）
        end_round = round_idx

        # 如果没有生成新的链路，提前结束
        if not evidence_chains:
            break

        if evaluate_chain_sufficiency(evidence_chains, x):
            print(f"Early stopping at round {round_idx} as LLM judged evidence is sufficient.")
            break

    # 根据evidence_chains，构建最后查询的prompt（包含局部历史和全局历史）
    last_query_prompt = get_last_query_prompt(evidence_chains, x, global_history_quadruples)

    # 轮次统计：记录每个查询实例的结束轮次
    track_round_statistics(end_round, x, evidence_chains, global_history_quadruples, args)


    if entity not in entity_search_space:
        entity_search_space[entity] = {}
    if query_time not in entity_search_space[entity]:
        entity_search_space[entity][query_time] = {}
    if relation not in entity_search_space[entity][query_time]:
        entity_search_space[entity][query_time][relation] = []

    return last_query_prompt, []


def _expand_chains_from_entity(existing_chain, entity_search_space, x, round, args, fileChainName):
    # 获取实体历史四元组和关系集合
    start_entity, relation, query_time = x[0], x[1], x[3]

    length = args.history_len if round == 0 else 50
    quadruples, relation_set = get_entity_edges_before_time(entity_search_space, start_entity, query_time, length)

    if not relation_set:
        return {}

    # 使用LLM对关系集合进行剪枝，获取关系评分
    top_relation = 5 if round == 0 else 3
    pruned_relations, relation_scores_dict, _ = prune_relation_set(relation_set, x, existing_chain, top_relation)

    if not pruned_relations:
        return {}

    # 并行处理每个关系
    all_generated_chains = []
    with ThreadPoolExecutor(max_workers=min(max(len(pruned_relations), 1), 10)) as executor:
        future_to_relation = {
            executor.submit(_process_single_relation,rel, quadruples, existing_chain, x, query_time,
                relation_scores_dict, round
            ): rel
            for rel in pruned_relations
        }
        for future in as_completed(future_to_relation):
            try:
                relation_chains = future.result()
                all_generated_chains.extend(relation_chains)
            except Exception as e:
                print(f"Error processing relation in round {round}: {e}")

    # 保存当前轮次的所有评分结果到JSON文件
    # save_generated_chains_jsonl(x, round, all_generated_chains, args, output_file=fileChainName)

    # 根据综合评分排序，筛选出top_k条链路
    all_generated_chains.sort(key=lambda x: x[1], reverse=True)
    top_k = 10 if round == 0 else args.top_k
    top_chains = all_generated_chains[:top_k]

    # 构建事件链字典
    evidence_chains = {}
    for chain_item in top_chains:
        chain, combined_score, chain_id, quad_score, time_score, rel_score, step_score = chain_item
        evidence_chains[chain_id] = {
            "chain": chain,
            "score": combined_score,
            "time_score": time_score,
            "rel_score": rel_score,
            "quad_score": quad_score,
            "step_score": step_score
        }

    return evidence_chains


def _expand_chains_from_existing(evidence_chains, entity_search_space, x, round, args, fileChainName):
    # 复制当前的事件链字典
    current_chains = evidence_chains.copy()

    # 存储所有生成的链路及其综合评分
    all_generated_chains = []

    # 并行处理每条事件链
    with ThreadPoolExecutor(max_workers=min(max(len(current_chains), 1), 10)) as executor:
        future_to_chain = {
            executor.submit(
                _process_single_chain,
                chain_id, chain_data, entity_search_space, x, round
            ): chain_id
            for chain_id, chain_data in current_chains.items()
        }
        for future in as_completed(future_to_chain):
            try:
                chain_chains = future.result()
                all_generated_chains.extend(chain_chains)
            except Exception as e:
                print(f"Error processing chain {future_to_chain[future]}: {e}")

    # 保存当前轮次的所有评分结果到JSON文件
    # save_generated_chains_jsonl(x, round, all_generated_chains, args, output_file=fileChainName)

    # 根据综合评分排序，筛选出top_k条链路
    all_generated_chains.sort(key=lambda x: x[1], reverse=True)
    top_chains = all_generated_chains[:args.top_k]

    # 将筛选后的链路添加到事件链字典中
    evidence_chains.clear()
    for chain_item in top_chains:
        new_chain, combined_score, new_chain_id, quad_score, time_score, rel_score, step_score = chain_item
        evidence_chains[new_chain_id] = {
            "chain": new_chain,
            "score": combined_score,
            "time_score": time_score,
            "rel_score": rel_score,
            "quad_score": quad_score,
            "step_score": step_score
        }

    return evidence_chains


def _process_single_relation(pruned_rel, quadruples, existing_chain, x, query_time, relation_scores_dict, round):
    """
    处理单个关系，生成链路（第一轮）
    """
    relation_chains = []
    rel_score = relation_scores_dict.get(pruned_rel, 0.5)

    # 获取该关系下的所有实体边
    rel_quadruples = [quad for quad in quadruples if quad[1] == pruned_rel]

    # 预筛选：保留离查询时间最近的10个四元组，避免LLM上下文过大
    if len(rel_quadruples) > 10:
        prev_time = query_time if not existing_chain else existing_chain[-1][3]
        rel_quadruples = sorted(rel_quadruples, key=lambda q: abs(q[3] - prev_time))[:10]

    # 使用LLM进行四元组剪枝（保留最靠近查询的几个四元组）
    pruned_quadruples, quadruple_scores_dict = prune_quadruples_score_set(
        rel_quadruples, x, existing_chain, pruned_rel, top_quadruples=3
    )

    # 处理每个剪枝后的四元组
    for quad_idx, quad in enumerate(pruned_quadruples):
        quad_score = quadruple_scores_dict.get(tuple(quad), 0.5)

        # 计算时序评分
        prev_time = query_time if not existing_chain else existing_chain[-1][3]
        time_score = calculate_step_time_score(quad[3], prev_time)

        # 计算综合评分：使用加权平均，保证分数为正数
        # 关系权重40%，四元组权重40%，时序权重20%
        w_rel, w_quad, w_time = 0.4, 0.4, 0.2
        step_score = w_rel * rel_score + w_quad * quad_score + w_time * time_score

        if existing_chain:
            # 多轮：当前分数与父链分数的加权平均（父链权重更高）
            parent_score = existing_chain[-1] if isinstance(existing_chain[-1], (int, float)) else 0.5
            w_parent, w_current = 0.6, 0.4
            combined_score = w_parent * parent_score + w_current * step_score
        else:
            # 第一轮：直接使用当前分数
            combined_score = step_score

        # 生成链路ID
        if round == 0:
            chain_id = f"round0_{pruned_rel}_{quad[2]}"
        else:
            chain_id = f"round{round}_{pruned_rel}_quad{quad_idx}"

        # 生成新链
        new_chain = existing_chain.copy() if existing_chain else []
        new_chain.append(quad)

        relation_chains.append(
            (new_chain, combined_score, chain_id, quad_score, time_score, rel_score, step_score)
        )

    return relation_chains


def _process_single_chain(chain_id, chain_data, entity_search_space, x, round):
    """
    处理单条事件链，生成新的扩展链路，用于2-n轮
    """
    chain = chain_data["chain"]

    # 获取链的最后节点
    last_quad = chain[-1]
    last_entity, last_relation, last_target, last_time = last_quad

    # 获取最后节点的历史四元组和关系集合
    last_quadruples, last_relation_set = get_entity_edges_before_time(
        entity_search_space, last_target, last_time, length=50
    )

    if not last_relation_set:
        return []

    # 使用LLM进行关系剪枝
    pruned_relations, relation_scores_dict, _ = prune_relation_set(last_relation_set, x, chain, top_relation=3)

    if not pruned_relations:
        return []

    # 处理每个关系
    chain_generated_chains = []
    for pruned_rel in pruned_relations:
        rel_score = relation_scores_dict.get(pruned_rel, 0.5)

        # 获取该关系下的四元组
        rel_quadruples = [quad for quad in last_quadruples if quad[1] == pruned_rel]

        # 预筛选：保留离查询时间最近的10个四元组，避免LLM上下文过大
        if len(rel_quadruples) > 10:
            rel_quadruples = sorted(rel_quadruples, key=lambda q: abs(q[3] - last_time))[:10]

        # 使用LLM进行四元组剪枝（保留最靠近查询的几个四元组）
        pruned_quadruples, quadruple_scores_dict = prune_quadruples_score_set(
            rel_quadruples, x, chain, pruned_rel, top_quadruples=3
        )

        # 处理每个四元组
        for quad_idx, quad in enumerate(pruned_quadruples):
            quad_score = quadruple_scores_dict.get(tuple(quad), 0.5)

            # 计算时序评分
            time_score = calculate_step_time_score(quad[3], last_time)

            # 计算综合评分：使用加权平均，保证分数为正数
            # 关系权重40%，四元组权重40%，时序权重20%
            w_rel, w_quad, w_time = 0.4, 0.4, 0.2
            step_score = w_rel * rel_score + w_quad * quad_score + w_time * time_score

            # 多轮：当前分数与父链分数的加权平均（父链权重更高）
            parent_score = chain_data.get("score", 0.5)
            w_parent, w_current = 0.6, 0.4
            combined_score = w_parent * parent_score + w_current * step_score

            # 生成新链路ID
            new_chain_id = f"{chain_id}_round{round}_rel{pruned_rel}_quad{quad_idx}"

            # 生成新链
            new_chain = chain.copy()
            new_chain.append(quad)

            chain_generated_chains.append(
                (new_chain, combined_score, new_chain_id, quad_score, time_score, rel_score, step_score)
            )

    return chain_generated_chains


def update_history(x, entity_search_space, predictions, candidates, args):
    entity, relation, targets, time = x[0], x[1], x[2], x[3]
    if args.verbose:
        print(
            f"search space:\n{entity},{relation},{time} --> {entity_search_space[entity][time][relation]}"
        )
    if args.multi_step:
        filtered_predictions = [candidates[x[0]] for x in predictions if x[0] in candidates]
        targets = filtered_predictions[: args.history_top_k]
    entity_search_space[entity][time][relation] += targets
    if args.verbose:
        print(f"history:\n{entity},{relation},{time} --> {targets}")
        print(
            f"search space:\n{entity},{relation},{time} --> {entity_search_space[entity][time][relation]}"
        )


def write_results(x, predictions, candidates, direction, writer, args):
    entity, relation, targets, time = x[0], x[1], x[2], x[3]
    example = {
        "timestamp": time,
        "entity": entity,
        "relation": relation,
        "targets": targets,
        "direction": direction,
        # "predictions": [candidates[x[0]] for x in predictions if x[0] in candidates],
        "predictions": [x for x in predictions],
    }
    writer.write(json.dumps(example) + "\n")

    if args.verbose:
        print(f"example:\n{json.dumps(example, indent=2)}")

    return example


def update_metric(example, metric, args):
    if args.verbose:
        print(f'predictions: {example["predictions"]}')
    for target in example["targets"]:
        metric.total += 1
        index = example["predictions"].index(target) if target in example["predictions"] else -1
        if index >= 0:
            _predictions = [
                x for x in example["predictions"][:index] if x not in example["targets"]
            ]
            rank = len(_predictions) + 1
            if args.verbose:
                print(f"target: {target} --> rank: {rank}")
            metric.update(rank)

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


def track_round_statistics(end_round, x, evidence_chains, global_history_quadruples, args):
    """
    轮次统计函数：记录每个查询实例的结束轮次，统计各轮次分布，
    对于结束轮次为2的样本保存局部历史和全局历史到文件

    参数:
    end_round: 实际结束的轮次
    x: 查询四元组，格式为 [subject, relation, target, time]
    evidence_chains: 局部历史事件链字典
    global_history_quadruples: 全局历史四元组列表
    args: 配置参数
    """
    global round_statistics

    # 更新总查询数
    round_statistics["total_queries"] += 1

    # 更新轮次分布
    if end_round not in round_statistics["round_distribution"]:
        round_statistics["round_distribution"][end_round] = 0
    round_statistics["round_distribution"][end_round] += 1

    # 如果结束轮次为2，保存局部历史和全局历史详情
    if end_round == 2:
        sample_data = {
            "query": {
                "subject": x[0],
                "relation": x[1],
                "targets": x[2],
                "time": x[3]
            },
            "local_history_chains": {},
            "global_history_quadruples": global_history_quadruples if global_history_quadruples else []
        }

        # 保存局部历史（按评分排序）
        sorted_chains = sorted(evidence_chains.items(), key=lambda item: item[1]["score"], reverse=True)
        for chain_id, chain_data in sorted_chains:
            sample_data["local_history_chains"][chain_id] = {
                "chain": chain_data["chain"],
                "score": chain_data["score"],
                "time_score": chain_data["time_score"],
                "rel_score": chain_data["rel_score"],
                "quad_score": chain_data["quad_score"]
            }

        round_statistics["round2_samples"].append(sample_data)


def print_round_statistics():
    """
    打印轮次统计信息
    """
    global round_statistics

    print("\n" + "="*50)
    print("轮次统计信息 (Round Statistics)")
    print("="*50)
    print(f"总查询数 (Total Queries): {round_statistics['total_queries']}")
    print("\n各轮次分布 (Round Distribution):")
    for round_num in sorted(round_statistics["round_distribution"].keys()):
        count = round_statistics["round_distribution"][round_num]
        percentage = (count / round_statistics["total_queries"] * 100) if round_statistics["total_queries"] > 0 else 0
        print(f"  轮次 {round_num}: {count} 次 ({percentage:.2f}%)")
    print("="*50 + "\n")


def save_round2_samples_to_file(output_dir="outputs"):
    """
    将结束轮次为2的所有样本保存到文件中

    参数:
    output_dir: 输出目录
    """
    global round_statistics

    if not round_statistics["round2_samples"]:
        print("没有结束轮次为2的样本需要保存")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存所有结束轮次为2的样本
    filename = os.path.join(output_dir, "round2_samples.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(round_statistics["round2_samples"], f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(round_statistics['round2_samples'])} 个结束轮次为2的样本到 {filename}")

    # 单独保存轮次统计摘要
    summary_filename = os.path.join(output_dir, "round_statistics_summary.json")
    summary_data = {
        "total_queries": round_statistics["total_queries"],
        "round_distribution": round_statistics["round_distribution"],
        "round2_samples_count": len(round_statistics["round2_samples"])
    }
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"已保存轮次统计摘要到 {summary_filename}")


def retrieve_global_history_facts(x, entity_search_space, args):
    subject, relation, query_time = x[0], x[1], x[3]

    # 存储所有匹配的全局历史四元组
    global_quadruples = []

    # 只检索 query 中 subject 的历史，确保 s-r 一致
    if subject in entity_search_space:
        # 遍历该实体的所有时间戳
        for time in entity_search_space[subject]:
            # 只考虑早于查询时间的历史
            if time >= query_time:
                continue

            # 检查是否存在与查询关系相同的关系
            if relation in entity_search_space[subject][time]:
                # 获取所有目标实体
                for target in entity_search_space[subject][time][relation]:
                    global_quadruples.append([subject, relation, target, time])

    # 按时间从新到旧排序
    global_quadruples.sort(key=lambda quad: quad[3], reverse=True)

    # 限制全局历史的数量，避免prompt过长
    max_global_facts = getattr(args, 'global_history_len', 10)
    global_quadruples = global_quadruples[:max_global_facts]

    return global_quadruples


def calculate_step_time_score(current_time, prev_time, time_scale=100.0):
    time_diff = abs(prev_time - current_time)
    if time_diff <= 0:
        return 1.0
    # 设定 0.4 的基础保底分，剩余 0.6 随时间差平滑衰减
    return 0.4 + 0.6 * math.exp(-time_diff / time_scale)

def get_last_query_prompt(evidence_chains, x, global_history_quadruples=None):
    # 提取查询信息
    query_subject = x[0]
    query_relation = x[1]
    query_time = x[3]

    # 构建prompt的各个部分
    parts = []

    # 开头部分
    parts.append(
        "You must be able to correctly predict the {whom} of the given query from a given text consisting of multiple historical events in the form of \"{subject} {relation} {object} {time}\" and the query in the form of \"{subject} {relation} {whom} {time}?\" You must output several {object} that you think may be the answer to the given query based on the given historical events. Please list all possible {object} which may be answers to the query. Please assign each answer a serial number to represent its probability of being the correct answer. Note that answers with a high probability of being correct should be listed first.\n\n")

    # 历史事件部分
    parts.append("Here are the given historical events:\n")

    # 遍历所有历史发展链路中的四元组（局部历史）
    # 按评分排序事件链，优先使用评分高的事件链
    sorted_chains = sorted(evidence_chains.items(), key=lambda item: item[1]["score"], reverse=True)

    for chain_idx, (chain_id, chain_data) in enumerate(sorted_chains, start=1):
        chain = chain_data["chain"]
        parts.append(f"{chain_idx}. ")
        for link in chain:
            subject, relation, obj, time = link
            parts.append(f"{subject}, {relation}, {obj}, on the {time}th day; ")
        parts.append("\n")

    # 添加全局历史事实（如果有）
    if global_history_quadruples:
        parts.append("\nGlobal Historical Facts (consistent subject-relation patterns):\n")
        for quad in global_history_quadruples:
            s, r, o, t = quad
            parts.append(f"{s}, {r}, {o}, on the {t}th day; ")
        parts.append("\n")

    # 移除末尾多余的空格和分号
    if parts[-1].strip():
        parts[-1] = parts[-1].rstrip(' ;') + "\n\n"

    # 查询部分
    parts.append("Here is the query:\n")
    parts.append(f"{query_subject}, {query_relation} to, whom, on the {query_time}th day?\n\n")

    # 输出要求部分
    parts.append(
        "Please list all possible {object} which may be answers (one per line) without explanations. Note that answers with high probability should be listed first.\n")
    parts.append("For example:\n")
    parts.append("Possible answers:\n")
    parts.append("1. XXX\n")
    parts.append("2. XXX\n")
    parts.append("3. XXX\n")
    parts.append("... ...\n")
    parts.append("Please strictly follow the above demands for output.")

    # 组合所有部分
    return ''.join(parts)

