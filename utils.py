import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import random
import json

from .selfCode.LLM_util.score_LLM_chain import prune_relation_set, prune_quadruples_score_set
from typing import Any, Dict, List, Optional

from .selfCode.save_chain_json.save_chain_hostory import save_generated_chains_jsonl

MAX_HITS = 10


@dataclass
class HitsMetric:
    total: int = 0
    hit1: int = 0
    hit3: int = 0
    hit10: int = 0

    def update(self, rank):
        if rank <= 1:
            self.hit1 += 1
        if rank <= 3:
            self.hit3 += 1
        if rank <= 10:
            self.hit10 += 1

    def dump(self):
        return {
            "total": self.total,
            "hit1": self.hit1 / self.total,
            "hit3": self.hit3 / self.total,
            "hit10": self.hit10 / self.total,
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
    parser.add_argument("--max_rounds", default=1, type=int)  # 多轮迭代的轮次

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


def prepare_history_chain(x, entity_search_space, args, fileChainName):
    """
    准备历史事件链，集成关系和实体剪枝功能
    参数:
    x: 查询四元组，格式为 [subject, relation, target, time]
    entity_search_space: 实体搜索空间字典
    args: 配置参数
    """
    entity, relation, time = x[0], x[1], x[3]
    # 1.进行第一轮问询，使用关系+实体剪枝初始化事件链
    evidence_chains = initialize_evidence_chains(x, entity_search_space, args, fileChainName)

    # 2.进行多轮问询，根据evidence_chains，进行事件发展链的构建
    # 调用外部的扩展事件链函数，支持多轮问询
    if args.max_rounds >= 1:
        evidence_chains = expand_event_chain(evidence_chains, entity_search_space, x, args)

    # 3.根据evidence_chains，构建最后查询的prompt
    last_query_prompt = get_last_query_prompt(evidence_chains, x)

    if entity not in entity_search_space:
        entity_search_space[entity] = {}
    if time not in entity_search_space[entity]:
        entity_search_space[entity][time] = {}
    if relation not in entity_search_space[entity][time]:
        entity_search_space[entity][time][relation] = []

    return last_query_prompt, []


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


def calculate_first_chain_time_score(quad, query_time):
    # 计算初始四元组与查询时间的时间差评分
    # 时间差越小，评分越高，范围0-1
    time_diff = abs(quad[3] - query_time)
    # 使用与calculate_chain_score相同的衰减函数
    initial_score = 1.0 / (1.0 + time_diff) if time_diff > 0 else 1.0

    return initial_score

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



def _process_relation_for_init(pruned_rel, quadruples, x, query_time, relation_scores_dict):
    relation_chains = []
    # 获取关系评分
    rel_score = relation_scores_dict.get(pruned_rel, 0.5)

    # 获取该关系下的所有实体边
    rel_quadruples = [quad for quad in quadruples if quad[1] == pruned_rel]

    # 对 quadruples 进行去重，当 subject-relation-object 一致时只保留时间最大的
    unique_quad_dict = {}
    for quad in rel_quadruples:
        s, r, o, t = quad
        key = (s, r, o)
        if key not in unique_quad_dict or t > unique_quad_dict[key][3]:
            unique_quad_dict[key] = quad

    # 使用LLM进行四元组剪枝，第一轮问询没有历史发展链路，使用空列表作为chain参数
    pruned_quadruples, quadruple_scores_dict = prune_quadruples_score_set(unique_quad_dict.values(), x, [],
                                                                          pruned_rel, top_quadruples=3)

    # 直接使用剪枝后的四元组，不需要额外筛选
    for quad_idx, quad in enumerate(pruned_quadruples):
        # 获取四元组评分
        quad_score = quadruple_scores_dict.get(tuple(quad), 0.5)

        # 计算时序评分（使用现有的calculate_chain_score函数，仅考虑时序维度）
        time_score = calculate_first_chain_time_score(quad, query_time)

        # # 计算逻辑规则分数
        # Logic_score = get_chain_TLogic_score([quad], x)

        # 计算综合评分：关系分数 * 四元组分数 * 时序评分
        # combined_score = rel_score * quad_score * time_score * Logic_score
        combined_score = rel_score * quad_score * time_score

        # 存储生成的链路及其综合评分
        relation_chains.append(
            ([quad], combined_score, f"round0_{pruned_rel}_{quad[2]}", quad_score, time_score, rel_score, 0))

    return relation_chains


# 初始化事件链，使用关系+四元组剪枝
def initialize_evidence_chains(x, entity_search_space, args, fileChainName):
    entity, relation, query_time = x[0], x[1], x[3]
    # 获取初始四元组和关系集合
    quadruples, relation_set = get_entity_edges_before_time(entity_search_space, entity, query_time, args.history_len)

    # 使用LLM对关系集合进行剪枝，获取关系评分
    pruned_relations, relation_scores_dict = prune_relation_set(relation_set, x, [], 5)

    # 存储所有生成的链路及其综合评分
    all_generated_chains = []

    # 并行处理每个关系
    with ThreadPoolExecutor(max_workers=min(max(len(pruned_relations), 1), 10)) as executor:
        future_to_relation = {
            executor.submit(_process_relation_for_init, rel, quadruples, x, query_time, relation_scores_dict): rel
            for rel in pruned_relations
        }
        for future in as_completed(future_to_relation):
            try:
                relation_chains = future.result()
                all_generated_chains.extend(relation_chains)
            except Exception as e:
                print(f"Error processing relation in initialize_evidence_chains: {e}")
    # 保存当前轮次的所有评分结果到JSON文件
    save_generated_chains_jsonl(x, 0, all_generated_chains, args, output_file=fileChainName)

    # 根据综合评分排序，筛选出topn条链路
    all_generated_chains.sort(key=lambda x: x[1], reverse=True)
    # 筛选前args.top_k条链路
    top_chains = all_generated_chains[:10]

    # 构建初始事件发展链，添加评分属性
    evidence_chains = {}
    for idx, chain_item in enumerate(top_chains):
        chain, combined_score, chain_id, quad_score, time_score, rel_score, logic_score = chain_item
        # 修改evidence_chains数据结构，添加评分属性
        evidence_chains[chain_id] = {
            "chain": chain,
            "score": combined_score,  # 使用综合评分作为初始评分
            "time_score": time_score,
            "rel_score": rel_score,
            "quad_score": quad_score,
            "logic_score": logic_score
        }

    return evidence_chains

def _process_chain_item(chain_id, chain_data, round, entity_search_space, x):
    """
    处理单条事件链，生成新的扩展链路

    参数:
        chain_id: 事件链ID
        chain_data: 事件链数据，包含chain和score属性
        round: 当前轮次
        entity_search_space: 实体搜索空间
        x: 查询四元组

    返回:
        生成的新链路列表
    """
    chain = chain_data["chain"]
    # 获取该条链子的最后节点，获取相应连接的实体关系边
    last_quad = chain[-1]
    last_entity, last_relation, last_target, last_time = last_quad
    last_quadruples, last_relation_set = get_entity_edges_before_time(entity_search_space, last_target,
                                                                      last_time, length=50)

    # 使用LLM进行关系剪枝，获取关系评分
    pruned_relations, relation_scores_dict = prune_relation_set(last_relation_set, x, chain, top_relation=3)

    # 存储当前事件链生成的所有新链路
    chain_generated_chains = []

    # 顺序处理每个关系
    for pruned_rel in pruned_relations:
        # 获取关系评分
        rel_score = relation_scores_dict.get(pruned_rel, 0.5)

        # 直接从last_quadruples中筛选出该关系的四元组，避免重复遍历entity_search_space
        rel_quadruples = [quad for quad in last_quadruples if quad[1] == pruned_rel]

        # 对四元组进行去重，当 subject-relation-object 一致时只保留时间最大的
        unique_quad_dict = {}
        for quad in rel_quadruples:
            s, r, o, t = quad
            key = (s, r, o)
            if key not in unique_quad_dict or t > unique_quad_dict[key][3]:
                unique_quad_dict[key] = quad

        # 使用LLM进行四元组剪枝，获取四元组评分
        pruned_quadruples, quadruple_scores_dict = prune_quadruples_score_set(unique_quad_dict.values(), x, chain,
                                                                              pruned_rel, top_quadruples=3)

        # 直接使用剪枝后的四元组，不需要额外筛选
        for quad_idx, quad in enumerate(pruned_quadruples):
            # 获取四元组评分
            quad_score = quadruple_scores_dict.get(tuple(quad), 0.5)

            # 计算时序评分（使用现有的calculate_chain_score函数）
            temp_chain = chain.copy()
            temp_chain.append(quad)
            time_score = calculate_chain_score(temp_chain)
            # TODO:可以考虑在这里加入对链子的评分
            # Logic_score = get_chain_TLogic_score(temp_chain, x)
            # 计算综合评分：关系分数 * 四元组分数 * 时序评分
            # combined_score = rel_score * quad_score * time_score * Logic_score
            combined_score = rel_score * quad_score * time_score

            # 创建唯一的事件链ID，避免冲突
            new_chain_id = f"{chain_id}_round{round}_rel{pruned_rel}_quad{quad_idx}"
            # 复制当前链并添加新边
            new_chain = chain.copy()
            new_chain.append(quad)

            # 存储生成的链路及其综合评分
            chain_generated_chains.append(
                (new_chain, combined_score, new_chain_id, quad_score, time_score, rel_score, 0))
    return chain_generated_chains


# 扩展事件链，支持多轮问询
def expand_event_chain(evidence_chains, entity_search_space, x, args):
    for round in range(args.max_rounds):
        # 复制当前的事件链字典，避免在迭代过程中修改
        current_chains = evidence_chains.copy()

        # 清空当前的事件链字典，只保留本轮扩展后的链子
        evidence_chains.clear()

        # 存储所有生成的链路及其综合评分
        all_generated_chains = []

        # 并行处理每条事件链
        with ThreadPoolExecutor(max_workers=min(max(len(current_chains), 1), 10)) as executor:
            future_to_chain = {
                executor.submit(_process_chain_item, chain_id, chain_data, round, entity_search_space, x): chain_id
                for chain_id, chain_data in current_chains.items()
            }
            for future in as_completed(future_to_chain):
                try:
                    chain_chains = future.result()
                    all_generated_chains.extend(chain_chains)
                except Exception as e:
                    print(f"Error processing chain {future_to_chain[future]}: {e}")
        # 保存当前轮次的所有评分结果到JSON文件
        save_generated_chains_jsonl(x, round + 1, all_generated_chains, args)

        # 根据综合评分排序，筛选出topn条链路
        all_generated_chains.sort(key=lambda x: x[1], reverse=True)
        # 筛选前args.top_k条链路
        top_chains = all_generated_chains[:args.top_k]

        # 将筛选后的链路添加到事件链字典中
        for new_chain, combined_score, new_chain_id, quad_score, time_score, rel_score, logic_score in top_chains:
            evidence_chains[new_chain_id] = {
                "chain": new_chain,
                "score": combined_score,
                "time_score": time_score,
                "rel_score": rel_score,
                "quad_score": quad_score,
                "logic_score": logic_score
            }

        # 如果当前轮次没有新的链，提前结束扩展
        if not evidence_chains:
            evidence_chains = current_chains
            break

    return evidence_chains


def calculate_chain_score(chain):
    if len(chain) <= 1:
        return 1.0

    total_score = 0.0
    for i in range(1, len(chain)):
        # 获取当前四元组和前一个四元组的时间
        current_time = chain[i][3]
        prev_time = chain[i - 1][3]

        # 计算时间差（注意：时间是从大到小排序的，所以前一个时间应该大于当前时间）
        time_diff = prev_time - current_time

        # 计算时间差评分，时间差越小评分越高
        # 使用指数衰减函数：score = e^(-time_diff/100)，可以根据需要调整衰减系数
        diff_score = 1.0 / (1.0 + time_diff) if time_diff > 0 else 1.0
        total_score += diff_score

    # 计算平均评分
    avg_score = total_score / (len(chain) - 1)
    return avg_score


def get_last_query_prompt(evidence_chains, x):
    """
    将历史发展链路转换为大语言模型的prompt，用于最终查询预测

    参数:
    evidence_chains: 历史发展链路字典，每个值包含chain和score属性
    x: 查询四元组，格式为 [subject, relation, target, time]

    返回:
    格式化的prompt字符串
    """
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

    # 遍历所有历史发展链路中的四元组
    # 按评分排序事件链，优先使用评分高的事件链
    sorted_chains = sorted(evidence_chains.items(), key=lambda item: item[1]["score"], reverse=True)

    for chain_idx, (chain_id, chain_data) in enumerate(sorted_chains, start=1):
        chain = chain_data["chain"]
        parts.append(f"{chain_idx}. ")
        for link in chain:
            subject, relation, obj, time = link
            parts.append(f"{subject}, {relation}, {obj}, on the {time}th day; ")
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

