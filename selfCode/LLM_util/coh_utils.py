"""
CoH (Chain-of-History) 工具模块
提供 CoH 算法三阶段的 Prompt 构建函数和通用 ID 解析函数
"""

import re
from typing import List, Any


# ============================================================
# 阶段 1：一阶历史筛选 Prompt
# ============================================================

def get_coh_step1_prompt(first_order_quads: List[List[Any]], query: List[Any]) -> str:
    """
    构建一阶筛选 Prompt，要求 LLM 输出最重要的 Top-30 历史事实 ID

    参数:
        first_order_quads: 一阶历史四元组列表 [[s, r, o, t], ...]
        query: 查询四元组 [subject, relation, target, time]

    返回:
        Prompt 字符串
    """
    entity, relation, query_time = query[0], query[1], query[3]

    parts = []

    # Role & Task
    parts.append("You are an expert in Temporal Knowledge Graph (TKG) reasoning.\n")
    parts.append("Your task is to select the most important historical facts that are relevant to predicting the answer of a target query.\n\n")

    # Query
    parts.append("Target Query:\n")
    parts.append(f"{entity}, {relation}, ?, on the {query_time}th day\n\n")

    # Historical Facts with IDs
    parts.append(f"Historical Facts (total: {len(first_order_quads)}):\n")
    for idx, quad in enumerate(first_order_quads, start=1):
        s, r, o, t = quad
        parts.append(f"ID {idx}: {s}, {r}, {o}, on the {t}th day\n")
    parts.append("\n")

    # Output instruction
    parts.append("Please select the most important historical facts from the list above that would help predict the answer to the target query.\n")
    parts.append("Consider:\n")
    parts.append("- Temporal proximity: More recent events are generally more predictive\n")
    parts.append("- Relation relevance: Facts with relations similar or complementary to the query relation are more important\n")
    parts.append("- Entity connections: Facts involving entities that frequently interact with the subject are more valuable\n\n")

    parts.append("Output ONLY the IDs of the top 30 most important facts, separated by commas.\n")
    parts.append("Example output: 1, 3, 5, 7, 12, 15, 20, 25, 30, 35, 40, 42, 45, 50, 55, 58, 60, 65, 70, 72, 75, 78, 80, 85, 88, 90, 92, 95, 98, 100\n")
    parts.append("Do NOT output any explanations or additional text.\n")

    return ''.join(parts)


# ============================================================
# 阶段 2：二阶历史链筛选 Prompt
# ============================================================

def get_coh_step2_prompt(chains: List[List[List[Any]]], query: List[Any]) -> str:
    """
    构建二阶筛选 Prompt，输入为带 ID 的二阶历史链，要求 LLM 输出最重要的 Top-30 链 ID

    参数:
        chains: 二阶历史链列表，每条链是一个二维列表
                [[first_order_quad, second_order_quad1, second_order_quad2, ...], ...]
        query: 查询四元组 [subject, relation, target, time]

    返回:
        Prompt 字符串
    """
    entity, relation, query_time = query[0], query[1], query[3]

    parts = []

    # Role & Task
    parts.append("You are an expert in Temporal Knowledge Graph (TKG) reasoning and chain-of-history analysis.\n")
    parts.append("Your task is to evaluate historical evidence chains and select the most important ones for predicting the answer to a target query.\n\n")

    # Query
    parts.append("Target Query:\n")
    parts.append(f"{entity}, {relation}, ?, on the {query_time}th day\n\n")

    # History Chains with IDs
    parts.append(f"History Chains (total: {len(chains)}):\n")
    for idx, chain in enumerate(chains, start=1):
        parts.append(f"ID {idx}:\n")
        for i, quad in enumerate(chain):
            s, r, o, t = quad
            if i == 0:
                # parts.append(f"  First-order fact: {s}, {r}, {o}, on the {t}th day\n")
                parts.append(f"{s}, {r}, {o}, on the {t}th day;")
            else:
                # parts.append(f"  Related fact: {s}, {r}, {o}, on the {t}th day\n")
                parts.append(f"{s}, {r}, {o}, on the {t}th day\n")
        parts.append("\n")

    # Output instruction
    parts.append("Please evaluate each history chain and select the most important ones for predicting the target query.\n")
    parts.append("Consider:\n")
    parts.append("- Logical coherence: Does the chain form a coherent logical path leading to the query?\n")
    parts.append("- Temporal consistency: Are the events in a reasonable temporal order?\n")
    parts.append("- Information value: Does the chain provide unique and valuable contextual information?\n")
    parts.append("- Predictive power: How likely is this chain to lead to the correct answer?\n\n")

    parts.append("Output ONLY the IDs of the top 30 most important chains, separated by commas.\n")
    parts.append("Example output: 1, 2, 3, 5, 7, 8, 10, 12, 14, 15\n")
    parts.append("Do NOT output any explanations or additional text.\n")

    return ''.join(parts)


# ============================================================
# 阶段 3：最终预测 Prompt
# ============================================================

def get_coh_predict_prompt(screened_chains: List[List[List[Any]]], query: List[Any],
                           candidates: List[str]) -> str:
    """
    构建最终预测 Prompt，基于筛选后的链路输出预测实体排序

    参数:
        screened_chains: 筛选后的二阶历史链列表
        query: 查询四元组 [subject, relation, target, time]
        candidates: 候选实体列表

    返回:
        Prompt 字符串
    """
    entity, relation, query_time = query[0], query[1], query[3]

    parts = []

    # System-level instruction
    parts.append("TEMPORAL KNOWLEDGE GRAPH FORECASTING TASK\n\n")
    parts.append("You are an expert reasoning model specialized in Temporal Knowledge Graph (TKG) forecasting.\n")
    parts.append("Your task is to predict the missing object entity for a given query based on historical evidence chains.\n\n")

    parts.append("You will be provided with:\n")
    parts.append("1. Screened History Chains: Multi-hop historical evidence that has been carefully selected for relevance\n")
    parts.append("2. Candidate Entities: A list of possible answer entities\n")
    parts.append("3. Query: The prediction task to solve\n\n")

    # History Chains
    parts.append("=" * 60 + "\n")
    parts.append("SCREENED HISTORY CHAINS:\n")
    parts.append("=" * 60 + "\n\n")

    for idx, chain in enumerate(screened_chains, start=1):
        parts.append(f"Chain {idx}:\n")
        for i, quad in enumerate(chain):
            s, r, o, t = quad
            if i == 0:
                parts.append(f"  Direct fact: {s}, {r}, {o}, on the {t}th day\n")
            else:
                parts.append(f"  Related fact: {s}, {r}, {o}, on the {t}th day\n")
        parts.append("\n")

    # Query
    parts.append("-" * 60 + "\n")
    parts.append("QUERY TO ANSWER:\n")
    parts.append(f"{entity}, {relation}, to whom, on the {query_time}th day?\n")
    parts.append("-" * 60 + "\n\n")

    # Candidates
    parts.append("Available Candidates:\n")
    parts.append(", ".join(candidates[:30]) + "\n\n")

    # Reasoning instructions
    parts.append("REASONING INSTRUCTIONS:\n")
    parts.append("Step 1: Analyze each history chain for patterns and connections\n")
    parts.append("Step 2: Identify which candidate entities appear in or can be inferred from the chains\n")
    parts.append("Step 3: Consider temporal proximity and logical coherence\n")
    parts.append("Step 4: Rank candidates by predictive confidence\n\n")

    # Output format
    parts.append("OUTPUT FORMAT:\n")
    parts.append("Please list the top 10 most likely entities that could answer the query, ranked by probability.\n")
    parts.append("For example:\n")
    parts.append("1. XXX\n")
    parts.append("2. XXX\n")
    parts.append("3. XXX\n")
    parts.append("... ...\n")
    parts.append("Please strictly follow the above format for output. Output entity names exactly as they appear in the candidates list.\n")

    return ''.join(parts)


# ============================================================
# 通用 ID 解析函数
# ============================================================

def parse_selected_ids(llm_output: str, max_id: int, top_k: int) -> List[int]:
    """
    通用的 ID 解析函数，从 LLM 返回文本中提取数字 ID

    参数:
        llm_output: LLM 返回的原始文本
        max_id: 合法的最大 ID 值
        top_k: 需要返回的最大 ID 数量

    返回:
        解析出的合法 ID 列表（1-indexed），最多 top_k 个
        如果解析失败返回空列表，由调用方执行降级策略
    """
    if not llm_output or not isinstance(llm_output, str):
        return []

    # 尝试提取所有数字
    numbers = re.findall(r'\d+', llm_output)

    valid_ids = []
    seen = set()

    for num_str in numbers:
        num = int(num_str)
        # 合法性校验：ID 在 [1, max_id] 范围内且不重复
        if 1 <= num <= max_id and num not in seen:
            valid_ids.append(num)
            seen.add(num)
            if len(valid_ids) >= top_k:
                break

    return valid_ids
