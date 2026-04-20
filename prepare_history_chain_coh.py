"""
CoH (Chain-of-History) 多步推理流水线
严格复现论文《Chain-of-History Reasoning for Temporal Knowledge Graph Forecasting》的核心机制
三阶段：一阶筛选 → 二阶链构建与筛选 → 最终预测
"""

from typing import List, Any, Tuple

from selfCode.LLMAPI.qwen_utils import get_evaluation_results
from selfCode.LLM_util.coh_utils import (
    get_coh_step1_prompt,
    get_coh_step2_prompt,
    get_coh_predict_prompt,
    parse_selected_ids,
)
from prepare_history_chain_v2 import get_entity_edges_before_time


def prepare_history_chain_coh(x, entity_search_space, args):
    """
    CoH 算法入口函数：三阶段多步推理流水线

    阶段 1 — 一阶历史筛选：获取 head 实体最近 100 条历史 → LLM 选出 Top-30
    阶段 2 — 二阶链构建与筛选：对选中的一阶事实扩展二阶历史 → LLM 选出 Top-15
    阶段 3 — 最终预测：基于筛选后的链路构建预测 Prompt

    参数:
        x: 查询四元组 [subject, relation, target, time]
        entity_search_space: 实体搜索空间字典
        args: 参数对象，需包含 history_len 属性

    返回:
        (model_input, candidates): 预测 prompt 字符串和候选实体列表
    """
    entity, relation, query_time = x[0], x[1], x[3]

    # 新增：初始化指标统计变量
    total_candidate_facts = 0
    total_tokens = 0

    # 确保 entity_search_space 初始化（与 V2 保持一致，防止 update_history 出错）
    if entity not in entity_search_space:
        entity_search_space[entity] = {}
    if query_time not in entity_search_space[entity]:
        entity_search_space[entity][query_time] = {}
    if relation not in entity_search_space[entity][query_time]:
        entity_search_space[entity][query_time][relation] = []

    # ============================================================
    # 阶段 1：一阶历史筛选
    # ============================================================
    history_len = getattr(args, 'history_len', 100)
    first_order_quads, _ = get_entity_edges_before_time(
        entity_search_space, entity, query_time, history_len
    )

    # 统计：累加一阶检索的候选事实数
    total_candidate_facts += len(first_order_quads)

    if not first_order_quads:
        prompt = f"No historical data available for {entity}. Query: {entity}, {relation}, ?, {query_time}\n"
        return prompt, []

    # 调用 LLM 进行一阶筛选
    top_n = 30  # 阶段1筛选数量
    step1_prompt = get_coh_step1_prompt(first_order_quads, x)

    # 统计：累加阶段1 Prompt 的 Token 消耗 (粗略估算)
    total_tokens += int(len(step1_prompt.split()) * 1.3)

    try:
        step1_result = get_evaluation_results(step1_prompt)
        selected_step1_ids = parse_selected_ids(step1_result, len(first_order_quads), top_n)
    except Exception as e:
        print(f"CoH Step 1 LLM call failed: {e}")
        selected_step1_ids = []

    # 降级策略：如果解析不到有效 ID，默认取前 top_n 项
    if not selected_step1_ids:
        selected_step1_ids = list(range(1, min(top_n, len(first_order_quads)) + 1))

    # 提取被选中的一阶事实（ID 是 1-indexed）
    selected_first_order = [first_order_quads[i - 1] for i in selected_step1_ids]

    # ============================================================
    # 阶段 2：二阶历史链构建与筛选
    # ============================================================
    second_order_limit = 5  # 每个一阶尾实体最多取 5 条二阶事实
    chains = []  # 二维列表：每条链 = [一阶事实, 二阶事实]  1对1关系

    for quad in selected_first_order:
        tail_entity = quad[2]  # 一阶事实的尾实体
        # 获取该尾实体的历史事实（二阶）
        second_order_quads, _ = get_entity_edges_before_time(
            entity_search_space, tail_entity, query_time, second_order_limit
        )

        # 统计：累加为当前尾实体检索的二阶候选事实数
        total_candidate_facts += len(second_order_quads)

        # 构建链：一阶事实与每一个二阶事实分别组成独立链路（1对1）
        if second_order_quads:
            for sq in second_order_quads:
                chains.append([quad, sq])
        else:
            # 没有二阶历史时，仅保留一阶事实作为单条链
            chains.append([quad])

    if not chains:
        # 极端情况：没有任何链，回退到简单 prompt
        prompt = f"No chains available for {entity}. Query: {entity}, {relation}, ?, {query_time}\n"
        return prompt, [], total_candidate_facts, total_tokens

    # 调用 LLM 进行二阶筛选
    top_m = 30  # 阶段2筛选数量
    step2_prompt = get_coh_step2_prompt(chains, x)

    # 🌟 统计：累加阶段2 Prompt 的 Token 消耗
    total_tokens += int(len(step2_prompt.split()) * 1.3)

    try:
        step2_result = get_evaluation_results(step2_prompt)
        selected_step2_ids = parse_selected_ids(step2_result, len(chains), top_m)
    except Exception as e:
        print(f"CoH Step 2 LLM call failed: {e}")
        selected_step2_ids = []

    # 降级策略：如果解析不到有效 ID，默认取前 top_m 项
    if not selected_step2_ids:
        selected_step2_ids = list(range(1, min(top_m, len(chains)) + 1))

    # 提取被选中的二阶历史链
    screened_chains = [chains[i - 1] for i in selected_step2_ids]

    # ============================================================
    # 阶段 3：准备最终预测
    # ============================================================
    # 从筛选后的链中提取候选实体（所有尾实体，去重，保持出现顺序）
    candidate_set = set()
    candidates = []
    for chain in screened_chains:
        for quad in chain:
            tail = quad[2]
            if tail not in candidate_set:
                candidate_set.add(tail)
                candidates.append(tail)

    # 构建最终预测 Prompt
    model_input = get_coh_predict_prompt(screened_chains, x, candidates)

    # 🌟 统计：累加阶段3 (最终预测) Prompt 的 Token 消耗
    total_tokens += int(len(model_input.split()) * 1.3)

    return model_input, candidates, total_candidate_facts, total_tokens
