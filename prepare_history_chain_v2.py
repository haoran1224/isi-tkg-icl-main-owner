"""
基于大模型的时序图谱预测方法 V2
新的 prepare_history_chain 实现，包含局部分支和全局分支的融合
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
import argparse

from selfCode.LLMAPI.qwen_utils import get_evaluation_results
from selfCode.LLM_util.score_LLM_chain import (
    prune_relation_set, prune_quadruples_score_set,
    prune_relation_set_v2  # 新增V2版本
)


def get_entity_edges_before_time(entity_search_space, entity, time, length):
    """
    获取在指定时间前与实体相连的边（四元组）
    参数:
    entity_search_space: 实体搜索空间字典
    entity: 目标实体
    time: 当前时间
    length: 要获取的边数
    返回:
    四元组列表 [entity, relation, target, time]，按时间从新到旧排序
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


def calculate_step_time_score(current_time, prev_time, time_scale=100.0):
    """计算时序评分"""
    time_diff = abs(prev_time - current_time)
    if time_diff <= 0:
        return 1.0
    # 设定 0.4 的基础保底分，剩余 0.6 随时间差平滑衰减
    return 0.4 + 0.6 * math.exp(-time_diff / time_scale)


# ============================================================
# 局部分支：按尾实体分组形成一阶候选线索
# ============================================================

def build_local_branch(quadruples: List[List[Any]], relation_scores: Dict[str, float] = None,
                       query_time: int = None) -> Dict[str, List[List[Any]]]:
    """
    局部分支：将四元组按尾实体分组，形成一阶候选线索

    参数:
    quadruples: 四元组列表 [[s, r, o, t], ...]
    relation_scores: 关系评分字典 {relation: score}
    query_time: 查询时间，用于计算时间差

    返回:
    按尾实体分组的字典 {tail_entity: [(s, r, o, t), ...]}
    """
    local_branches = defaultdict(list)

    for quad in quadruples:
        s, r, o, t = quad
        local_branches[o].append(quad)

    # 对每个尾实体的四元组排序：使用加权综合评分
    for tail_entity in local_branches:
        if relation_scores and query_time is not None:
            # 计算综合分数：关系评分(80%) + 时间分数(20%)
            def calculate_combined_score(quad):
                rel_score = relation_scores.get(quad[1], 0.5)
                time_diff = abs(quad[3] - query_time)
                # 时间分数：时间差越小分数越高，范围0-1
                time_score = 1.0 / (1.0 + time_diff / 100.0)
                # 加权平均：关系50% + 时间50%
                return 0.8 * rel_score + 0.2 * time_score

            # 按综合分数降序排序
            local_branches[tail_entity].sort(key=calculate_combined_score, reverse=True)
        else:
            # 回退到按时间降序（最新的在前）
            local_branches[tail_entity].sort(key=lambda x: -x[3])

    return dict(local_branches)


def format_local_branch_prompt(local_branches: Dict[str, List[List[Any]]],
                                max_branches: int = 15) -> str:
    """
    格式化局部分支为prompt文本

    参数:
    local_branches: 按尾实体分组的字典
    max_branches: 最多展示的分支数量

    返回:
    格式化的prompt文本
    """
    parts = []

    # 按每个尾实体的四元组数量排序
    sorted_branches = sorted(local_branches.items(),
                             key=lambda x: len(x[1]),
                             reverse=True)[:max_branches]

    parts.append("Local Historical Evidence (Candidate-Specific Patterns):\n")

    for idx, (tail_entity, quads) in enumerate(sorted_branches, start=1):
        parts.append(f"Candidate {idx} ({tail_entity}):\n")
        for quad in quads:
            s, r, o, t = quad
            parts.append(f"  {s}, {r}, {o}, on the {t}th day\n")
        parts.append("\n")

    return ''.join(parts)


# ============================================================
# 全局分支：统计头实体在近期时间窗口内的关系频率
# ============================================================

def build_global_branch(quadruples: List[List[Any]]) -> Dict[str, Any]:
    """
    全局分支：统计头实体在近期时间窗口内发起的各类关系频率

    参数:
    quadruples: 四元组列表 [[s, r, o, t], ...]

    返回:
    包含关系频率统计的字典
    """
    if not quadruples:
        return {"total_interactions": 0, "relation_frequency": {}, "summary": "No historical data available."}

    # 统计关系频率
    relation_counter = Counter()
    time_span = []

    for quad in quadruples:
        s, r, o, t = quad
        relation_counter[r] += 1
        time_span.append(t)

    total_interactions = len(quadruples)

    # 生成宏观状态描述文本
    summary_parts = []
    summary_parts.append(f"The subject entity has participated in {total_interactions} historical interactions.\n")
    summary_parts.append("Most frequent relation types:\n")

    for rel, count in relation_counter.most_common(5):
        percentage = (count / total_interactions) * 100
        summary_parts.append(f"  - {rel}: {count} times ({percentage:.1f}%)\n")

    if time_span:
        earliest = min(time_span)
        latest = max(time_span)
        summary_parts.append(f"\nTime span: from day {earliest} to day {latest}\n")

    return {
        "total_interactions": total_interactions,
        "relation_frequency": dict(relation_counter),
        "summary": ''.join(summary_parts)
    }


def format_global_branch_prompt(global_branch: Dict[str, Any]) -> str:
    """
    格式化全局分支为prompt文本

    参数:
    global_branch: 全局分支统计字典

    返回:
    格式化的prompt文本
    """
    parts = []

    parts.append("Global Historical Context (Subject Entity's Macro State):\n")
    # parts.append(global_branch["summary"])
    parts.append("\n")

    return ''.join(parts)


# ============================================================
# 全局分支（LLM增强版）：使用大模型生成宏观状态描述
# ============================================================

def build_global_branch_summary_prompt(quadruples: List[List[Any]], query_subject: str) -> str:
    """
    构建用于生成宏观状态描述的 Prompt

    参数:
    quadruples: 四元组列表
    query_subject: 查询的主体实体

    返回:
    LLM 使用的 prompt
    """
    # 先进行基础统计
    relation_counter = Counter()
    entity_counter = Counter()
    time_span = []

    for quad in quadruples:
        s, r, o, t = quad
        relation_counter[r] += 1
        entity_counter[o] += 1
        time_span.append(t)

    total_interactions = len(quadruples)

    # 构建 prompt
    parts = []

    parts.append("You are an expert in analyzing temporal knowledge graphs and diplomatic/international relations.\n\n")

    parts.append("Task: Generate a concise macro-level state description for a subject entity based on its historical interactions.\n\n")

    parts.append("Subject Entity: " + query_subject + "\n\n")

    parts.append("Historical Statistics:\n")
    parts.append(f"- Total interactions: {total_interactions}\n\n")

    parts.append("Most frequent relation types:\n")
    for rel, count in relation_counter.most_common(10):
        percentage = (count / total_interactions) * 100
        parts.append(f"  * {rel}: {count} times ({percentage:.1f}%)\n")

    parts.append("\nMost frequent interaction partners (top 5):\n")
    for entity, count in entity_counter.most_common(5):
        percentage = (count / total_interactions) * 100
        parts.append(f"  * {entity}: {count} times ({percentage:.1f}%)\n")

    if time_span:
        parts.append(f"\nTime span: from day {min(time_span)} to day {max(time_span)}\n")

    parts.append("\n" + "="*60 + "\n")
    parts.append("Please analyze the above statistics and generate a 2-3 sentence macro-level state description.\n\n")

    parts.append("Your description should:\n")
    parts.append("1. Identify the general diplomatic stance or behavior pattern (e.g., active diplomacy, isolationist, cooperative, assertive)\n")
    parts.append("2. Highlight the dominant interaction types and what they suggest\n")
    parts.append("3. Note any temporal trends or shifts in behavior if apparent\n\n")

    parts.append("Output ONLY the descriptive paragraph, no explanations or metadata.\n\n")

    parts.append("Example output format:\n")
    parts.append('"Recently, [Entity] has demonstrated a strong proactive diplomatic stance, frequently engaging\n')
    parts.append('in Sign_formal_agreement and Engage_in_diplomatic_cooperation with multiple entities, suggesting\n')
    parts.append('a period of active alliance-building and international cooperation."\n')

    return ''.join(parts)


def build_global_branch_with_llm(quadruples: List[List[Any]], query: List[Any],
                                  use_llm: bool = True) -> Dict[str, Any]:
    """
    使用大模型生成全局分支的宏观状态描述
    """
    if not quadruples:
        return {
            "total_interactions": 0,
            "relation_frequency": {},
            "summary": "No historical data available.",
            "llm_generated": False
        }

    query_subject = query[0]

    # 先进行基础统计（无论是否使用LLM都需要这些数据）
    relation_counter = Counter()
    entity_counter = Counter()
    time_span = []

    for quad in quadruples:
        s, r, o, t = quad
        relation_counter[r] += 1
        entity_counter[o] += 1
        time_span.append(t)

    total_interactions = len(quadruples)

    # 如果使用LLM生成描述
    if use_llm:
        try:
            # 构建LLM prompt
            llm_prompt = build_global_branch_summary_prompt(quadruples, query_subject)

            # 调用LLM生成宏观状态描述
            llm_summary = get_evaluation_results(llm_prompt)

            if llm_summary and len(llm_summary.strip()) > 20:
                # LLM成功生成描述
                return {
                    "total_interactions": total_interactions,
                    "relation_frequency": dict(relation_counter),
                    "entity_frequency": dict(entity_counter),
                    "time_span": (min(time_span), max(time_span)) if time_span else None,
                    "summary": llm_summary.strip(),
                    "llm_generated": True
                }
        except Exception as e:
            print(f"LLM generation failed, falling back to statistical summary: {e}")

    # 回退到统计方法
    summary_parts = []
    summary_parts.append(f"The subject entity has participated in {total_interactions} historical interactions.\n")
    summary_parts.append("Most frequent relation types:\n")

    for rel, count in relation_counter.most_common(5):
        percentage = (count / total_interactions) * 100
        summary_parts.append(f"  - {rel}: {count} times ({percentage:.1f}%)\n")

    if time_span:
        earliest = min(time_span)
        latest = max(time_span)
        summary_parts.append(f"\nTime span: from day {earliest} to day {latest}\n")

    return {
        "total_interactions": total_interactions,
        "relation_frequency": dict(relation_counter),
        "entity_frequency": dict(entity_counter),
        "time_span": (min(time_span), max(time_span)) if time_span else None,
        "summary": ''.join(summary_parts),
        "llm_generated": False
    }


# ============================================================
# 大模型预测：融合全局和局部分支，使用思维链推演
# ============================================================

def build_cot_prompt(local_branches: Dict[str, List[List[Any]]],
                     global_branch: Dict[str, Any],
                     query: List[Any],
                     candidates: List[str],global_history_quadruples) -> str:
    """
    构建思维链（Chain of Thought）推理prompt
    """
    query_subject = query[0]
    query_relation = query[1]
    query_time = query[3]

    parts = []

    # 系统指令部分
    # parts.append("=" * 60 + "\n")
    parts.append("TEMPORAL KNOWLEDGE GRAPH FORECASTING TASK\n")
    # parts.append("=" * 60 + "\n\n")

    parts.append("You are an expert reasoning model specialized in Temporal Knowledge Graph (TKG) forecasting.\n")
    parts.append("Your task is to predict the missing object entity for a given query based on historical evidence.\n\n")

    parts.append("You will be provided with:\n")
    parts.append("1. Global Historical Facts(consistent subject-relation patterns):\n")
    # parts.append("2. LOCAL CONTEXT: The subject entity's macro-level interaction patterns\n")
    parts.append("2. LOCAL EVIDENCE: Candidate-specific historical interaction chains\n")
    parts.append("3. QUERY: The prediction task to solve\n\n")

    # 全局通用历史
    if global_history_quadruples:
        parts.append("\nGlobal Historical Facts (consistent subject-relation patterns):\n")
        for quad in global_history_quadruples:
            s, r, o, t = quad
            parts.append(f"{s}, {r}, {o}, on the {t}th day; \n")
        parts.append("\n")

    # # 全局分支
    # parts.append(format_global_branch_prompt(global_branch))

    # 局部分支
    parts.append(format_local_branch_prompt(local_branches, max_branches=15))

    # 查询部分
    parts.append("-" * 60 + "\n")
    parts.append("QUERY TO ANSWER:\n")
    parts.append(f"{query_subject}, {query_relation}, to whom, on the {query_time}th day?\n")
    parts.append("-" * 60 + "\n\n")

    # 思维链推理要求
    parts.append("REASONING INSTRUCTIONS (Chain of Thought):\n")
    parts.append("Please follow these steps:\n\n")

    # parts.append("Step 1: Analyze the global context\n")
    # parts.append("  - What are the subject's dominant interaction patterns?\n")
    # parts.append("  - Which relation types are most frequent? What does this suggest?\n\n")
    parts.append("Step 1: Analyze the global Facts\n")
    parts.append("  - Understand the same events that have occurred in history\n")
    parts.append("  - And refer to detailed local history\n")

    parts.append("Step 2: Examine local evidence for each candidate\n")
    parts.append("  - Which candidates have strong historical connections?\n")
    parts.append("  - What is the temporal proximity of these connections?\n")
    # parts.append("  - Are there patterns suggesting future interaction?\n\n")

    parts.append("Step 3: Synthesize and predict\n")
    # parts.append("  - Combine global patterns and local evidence\n")
    parts.append("  - Combine global Facts and local evidence\n")
    parts.append("  - Apply causal reasoning: what is likely to happen next?\n")
    parts.append("  - Consider temporal trends and recency effects\n\n")
    parts.append("  - Consider global history Facts and recency effects\n\n")

    parts.append("OUTPUT FORMAT:\n")
    # parts.append("Please list all possible answers which may be answers to the query. ")
    # # parts.append("Please assign each answer a serial number to represent its probability of being the correct answer. ")
    # parts.append("Note that answers with a high probability of being correct should be listed first.\n\n")

    # parts.append("Output format (one candidate per line):\n")
    # parts.append("1. [Candidate_Name]\n")
    # parts.append("2. [Candidate_Name]\n")
    # parts.append("3. [Candidate_Name]\n")
    # parts.append("...\n\n")
    #
    # parts.append("Example:\n")
    # parts.append("1. China\n")
    # parts.append("2. Japan\n")
    # parts.append("3. South_Korea\n\n")
    # parts.append("Please strictly follow the above format for output.\n")
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

    return ''.join(parts)


def parse_llm_predictions(llm_output: str, candidates: List[str]) -> List[Tuple[str, float]]:
    """
    解析LLM输出，提取候选实体和置信度分数

    参数:
    llm_output: LLM返回的原始输出
    candidates: 候选实体列表

    返回:
    [(candidate, confidence), ...] 按置信度降序排列
    """
    predictions = []

    # 尝试匹配 "数字. 实体名: 分数" 格式
    pattern = r'\d+\.\s*([^:]+):\s*([\d.]+)'

    matches = re.findall(pattern, llm_output)

    for candidate_name, confidence in matches:
        candidate_name = candidate_name.strip()
        try:
            conf_score = float(confidence)
            # 检查候选是否在候选列表中
            if candidate_name in candidates:
                predictions.append((candidate_name, conf_score))
        except ValueError:
            continue

    # 如果没有匹配到任何结果，尝试其他格式
    if not predictions:
        # 尝试匹配 "实体名" 格式（假设出现在输出中）
        for candidate in candidates:
            if candidate in llm_output:
                # 没有明确分数，给一个默认分数
                predictions.append((candidate, 0.5))

    # 按置信度降序排列
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions


def get_candidates_from_local_branches(local_branches: Dict[str, List[List[Any]]]) -> List[str]:
    """
    从局部分支中获取候选实体列表

    参数:
    local_branches: 按尾实体分组的字典 {tail_entity: [(s, r, o, t), ...]}

    返回:
    候选实体列表（尾实体列表）
    """
    # 按每个尾实体的四元组数量排序，返回排序后的候选实体列表
    sorted_candidates = sorted(
        local_branches.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    return [entity for entity, _ in sorted_candidates]


# ============================================================
# 主函数：prepare_history_chain_v2
# ============================================================

def prepare_history_chain_v2(x, entity_search_space, args, fileChainName=None,
                             global_history_quadruples=None):
    """
    新的历史链准备函数，融合局部分支和全局分支
    步骤：
    1) 从头实体出发找到最近的n个时间步的四元组
    2) 通过关系筛选过滤高价值历史链路
    3) 根据筛选得到的关系链路筛选四元组
    4) 局部分支：按尾实体分组形成一阶候选线索
    5) 全局分支：统计头实体在近期时间窗口内的关系频率
    6) 大模型预测：融合全局和局部分支，使用思维链推演
    """
    entity, relation, query_time = x[0], x[1], x[3]

    # 首先初始化 entity_search_space 以确保 update_history 不会出错
    if entity not in entity_search_space:
        entity_search_space[entity] = {}
    if query_time not in entity_search_space[entity]:
        entity_search_space[entity][query_time] = {}
    if relation not in entity_search_space[entity][query_time]:
        entity_search_space[entity][query_time][relation] = []

    # ===== 步骤1：从头实体出发找到最近的n个时间步的四元组 =====
    history_len = getattr(args, 'history_len', 100)
    initial_quadruples, initial_relations = get_entity_edges_before_time(
        entity_search_space, entity, query_time, history_len
    )

    if not initial_quadruples:
        # 没有历史数据，返回简单prompt
        prompt = f"No historical data available for {entity}. Query: {entity}, {relation}, ?, {query_time}\n"
        return prompt, []

    # ===== 步骤2：通过关系筛选过滤高价值历史链路（使用V2版本）=====
    # pruned_relations, relation_scores = prune_relation_set_v2(
    #     initial_relations, x, [], top_relation=min(5, len(initial_relations))
    # )
    pruned_relations, relation_scores = prune_relation_set(
        initial_relations, x, [], top_relation=min(5, len(initial_relations))
    )

    if not pruned_relations:
        # 如果没有筛选出关系，使用初始关系
        pruned_relations = initial_relations[:5]

    # ===== 步骤3：根据筛选得到的关系链路筛选四元组 =====
    # 获取筛选后的关系对应的四元组
    final_quadruples = [
        quad for quad in initial_quadruples
        if quad[1] in pruned_relations
    ]

    # 对每个关系进行四元组筛选
    # final_quadruples = []
    # for rel in pruned_relations:
    #     rel_quads = [quad for quad in filtered_quadruples if quad[1] == rel]
    #     if rel_quads:
    #         pruned_quads, _ = prune_quadruples_score_set(
    #             rel_quads, x, [], rel, top_quadruples=min(3, len(rel_quads))
    #         )
    #         final_quadruples.extend(pruned_quads)

    # 如果没有筛选出任何四元组，使用初始四元组的前20个
    if not final_quadruples:
        final_quadruples = initial_quadruples[:20]

    # ===== 步骤4：局部分支 - 按尾实体分组形成一阶候选线索 =====
    local_branches = build_local_branch(final_quadruples, relation_scores, query_time)

    # ===== 步骤5：全局分支 - 使用LLM生成宏观状态描述 =====
    # 获取是否使用LLM生成全局分支的参数
    use_llm_global = getattr(args, 'use_llm_global', True)
    global_branch = build_global_branch_with_llm(final_quadruples, x, use_llm=use_llm_global)

    # ===== 步骤6：大模型预测 - 融合全局和局部分支，使用思维链推演 =====
    # 从局部分支中获取候选实体列表（尾实体）
    candidates = get_candidates_from_local_branches(local_branches)

    # 构建思维链prompt
    model_input = build_cot_prompt(local_branches, global_branch, x, candidates, global_history_quadruples)

    # 为了保持与现有代码的兼容性，返回空列表作为candidates
    # 实际的候选实体会在LLM输出中解析得到
    return model_input, []


# ============================================================
# 兼容性函数：parse_results_v2
# ============================================================

def parse_results_v2(llm_output: str, candidates: List[str] = None) -> List[str]:
    """
    解析V2版本的LLM输出，返回实体列表

    参数:
    llm_output: LLM输出文本
    candidates: 候选实体列表（可选，用于验证）

    返回:
    实体列表（按置信度降序排列）
    """
    parsed = parse_llm_predictions(llm_output, candidates or [])
    return [entity for entity, _ in parsed]