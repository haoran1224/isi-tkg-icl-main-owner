import re

from selfCode.LLMAPI.qwen_utils import get_evaluation_results


# 使用LLM对关系集合进行剪枝
def prune_relation_set(relation_set, query, chain, top_relation=3):
    # 构建关系剪枝prompt
    prompt = get_prune_relation_prompt(relation_set, query, chain)

    try:
        # 调用LLM获取关系评分
        result = get_evaluation_results(prompt)

        relation_scores = parse_relation_scores(result)

        # 如果评分结果为空，返回原始关系集合的前几个
        if not relation_scores:
            pruned_relations = relation_set[:top_relation] if len(relation_set) > top_relation else relation_set
            return pruned_relations, {}

        # 关联关系和评分
        relation_score_pairs = list(zip(relation_set, relation_scores))

        # 按评分降序排序
        relation_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 筛选前topk个关系（默认top3）
        pruned_relations_with_scores = relation_score_pairs[:top_relation]

        # 返回剪枝后的关系和对应的评分
        pruned_relations = [rel for rel, score in pruned_relations_with_scores]
        relation_scores_dict = {rel: score for rel, score in pruned_relations_with_scores}

        return pruned_relations, relation_scores_dict
    except Exception as e:
        # 如果发生错误，返回原始关系集合的前几个关系
        print(f"Error in prune_relation_set: {e}")
        pruned_relations = relation_set[:top_relation] if len(relation_set) > top_relation else relation_set
        return pruned_relations, {}

# 构建关系剪枝时使用的prompt
def get_prune_relation_prompt(relation_set, query, chain):
    entity, relation, time = query[0], query[1], query[3]

    # 构建prompt的各个部分
    parts = []

    # 开头部分：描述任务
    parts.append(
        "You are given a set of relations, a query, and an event history chain.\n")
    parts.append(
        "If you must infer {object} that you think may be the answer to the given query based on the given historical events,\n")
    parts.append(
        "what important history chains do you base your predictions on?\n")
    parts.append(
        "Please rate the importance of each chain between 0~1, Higher scores indicate higher importance\n\n")

    # 查询部分
    parts.append("Query:\n")
    parts.append(f"{entity} {relation} {{object}} on {time} time\n\n")

    # Available chains部分：chain事件 + relation->query格式
    parts.append("Available chains:\n")

    # 构建chain事件链部分（只使用relation部分）
    chain_relations = [link[1] for link in chain]

    # 如果chain不为空，生成带chain前缀的格式
    if chain:
        chain_prefix = ",".join(chain_relations)
        for idx, rel in enumerate(relation_set):
            parts.append(f"{idx + 1}. {chain_prefix},{rel}->Query\n")
    else:
        # chain为空时，直接使用relation->Query格式
        for idx, rel in enumerate(relation_set):
            parts.append(f"{idx + 1}. {rel}->Query\n")

    parts.append("\n")

    # 输出要求部分
    parts.append("Output format:\n")
    parts.append("Example:\n1:0.3\n2:0.4\n3:0.2\n")
    parts.append("Only output the number and score, output in the order of 1, 2, 3, 4, 5, no additional explanation.")

    # 组合所有部分
    return ''.join(parts)


# 使用LLM对四元组集合进行剪枝
def prune_quadruples_score_set(quadruples, query, chain, relation, top_quadruples):
    # 构建四元组剪枝prompt
    prompt = get_prune_quadruples_prompt(quadruples, query, chain, relation)

    try:
        # 调用LLM获取四元组评分
        result = get_evaluation_results(prompt)
        scores = parse_relation_scores(result)

        # 验证评分结果
        if not scores or len(scores) != len(quadruples):
            # 如果评分结果无效，返回原始四元组的前几个
            pruned_quadruples = list(quadruples)[:top_quadruples] if len(quadruples) > top_quadruples else list(
                quadruples)
            return pruned_quadruples, {}

        # 将四元组从列表转换为元组，以便作为字典键
        quadruples_tuples = [tuple(quad) for quad in quadruples]

        # 关联四元组和评分,按评分降序排序
        quadruple_score_pairs = list(zip(quadruples_tuples, scores))
        quadruple_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 筛选前topk个四元组（默认top3）
        pruned_quadruples_with_scores = quadruple_score_pairs[:top_quadruples]

        # 返回剪枝后的四元组和对应的评分
        # 将元组转换回列表格式返回，保持与原始数据格式一致
        pruned_quadruples = [list(quad) for quad, score in pruned_quadruples_with_scores]
        # 字典键使用元组格式，值使用评分
        quadruple_scores_dict = {quad: score for quad, score in pruned_quadruples_with_scores}

        return pruned_quadruples, quadruple_scores_dict
    except Exception as e:
        # 如果发生错误，返回原始四元组的前几个
        print(f"Error in prune_quadruples_score_set: {e}")
        pruned_quadruples = list(quadruples)[:top_quadruples] if len(quadruples) > top_quadruples else list(quadruples)
        return pruned_quadruples, {}


# 构建四元组剪枝时使用的prompt
def get_prune_quadruples_prompt(quadruples, query, chain, relation):
    entity, query_rel, time = query[0], query[1], query[3]

    # 构建prompt的各个部分
    parts = []

    # 开头部分：描述任务
    parts.append(
        "You are given a set of quadruples, a query, an event history chain, and a relation.\n")
    parts.append(
        "If you must infer {object} that you think may be the answer to the given query based on the given historical events and relation,\n")
    parts.append(
        "what important history chains do you base your predictions on?\n")
    parts.append(
        "Please rate the importance of each chain between 0~1, Higher scores indicate higher importance\n\n")

    # 查询和关系部分
    parts.append("Query:\n")
    parts.append(f"{entity} {query_rel} {{object}} on {time} time\n\n")
    parts.append(f"Current relation to consider: {relation}\n\n")

    # Available chains部分：chain事件（relation + entity） + entity->query格式
    parts.append("Available chains:\n")

    # 构建chain事件链部分（完整的历史事实，包含时间）
    chain_events = []
    for link in chain:
        subject, rel, obj, t = link
        # 构建完整的事件描述，包含时间
        event_desc = f"{subject}, {rel}, {obj}, on the {t}th day"
        chain_events.append(event_desc)

    # 如果chain不为空，生成带chain前缀的格式
    if chain:
        # 事件之间用分号分隔
        chain_prefix = "; ".join(chain_events)
        for idx, quad in enumerate(quadruples):
            s, r, o, t = quad
            parts.append(f"{idx + 1}. {chain_prefix};{s}, {r}, {o}, on the {t}th day->Query\n")
    else:
        # chain为空时，直接使用entity->Query格式
        for idx, ent in enumerate(quadruples):
            s, r, o, t = ent
            parts.append(f"{idx + 1}. {s}, {r}, {o}, on the {t}th day->Query\n")

    parts.append("\n")

    # 输出要求部分
    parts.append("Output format:\n")
    parts.append("Example:\n1:0.3\n2:0.4\n3:0.2\n")
    parts.append("Only output the number and score, output in the order of 1, 2, 3, 4, 5, no additional explanation.")

    # 组合所有部分
    return ''.join(parts)


# 解析 "序号:分数" 格式的字符串
def parse_relation_scores(result_str):
    if not result_str or not isinstance(result_str, str):
        return []

    scores_dict = {}

    # 按换行符或空格分割
    lines = re.split(r'[\n\s]+', result_str.strip())

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 匹配 "序号:分数" 格式
        match = re.match(r'^(\d+)[:：](\d*\.?\d+)$', line)
        if match:
            index = int(match.group(1))
            score = float(match.group(2))
            scores_dict[index] = score

    if not scores_dict:
        return []

    # 按序号排序并返回分数列表
    max_index = max(scores_dict.keys())
    scores_list = [scores_dict.get(i, 0.0) for i in range(1, max_index + 1)]

    return scores_list


def get_sufficiency_prompt(evidence_chains, query):
    """
    生成评估历史链路是否充分的 Prompt

    参数:
    evidence_chains: 事件链字典，格式为 {chain_id: {"chain": [[s, r, o, t], ...], "score": float, ...}}
    query: 查询四元组，格式为 [subject, relation, target, time]

    返回:
    格式化的 prompt 字符串
    """
    entity, relation, time = query[0], query[1], query[3]

    parts = []

    # 任务描述
    parts.append("You are given multiple historical event chains and a query.\n\n")

    # 按评分排序事件链
    sorted_chains = sorted(evidence_chains.items(), key=lambda x: x[1]["score"], reverse=True)

    # 历史事件链部分
    parts.append("Historical event chains (sorted by relevance score):\n")
    for idx, (chain_id, chain_data) in enumerate(sorted_chains, start=1):
        chain = chain_data["chain"]
        score = chain_data["score"]
        parts.append(f"Chain {idx} (score: {score:.4f}):\n")
        for link in chain:
            subject, rel, obj, t = link
            parts.append(f"  {subject}, {rel}, {obj}, on the {t}th day\n")
        parts.append("\n")

    # 查询部分
    parts.append("Query:\n")
    parts.append(f"{entity}, {relation}, to whom, on the {time}th day?\n\n")

    # 评估要求
    parts.append("Based on the above historical event chains, is there sufficient and direct evidence to answer the query?\n")
    parts.append("Please strictly output only 'YES' or 'NO' (case-insensitive).\n")
    parts.append("- 'YES': The historical event chains contain sufficient and direct evidence to answer the query.\n")
    parts.append("- 'NO': The historical event chains do NOT contain sufficient evidence, need more exploration.\n\n")
    parts.append("Output only 'YES' or 'NO' without any explanation.")

    return ''.join(parts)


def evaluate_chain_sufficiency(chains, query):
    """
    评估历史链路是否包含足够信息来回答查询
    参数:
    chains: 事件链字典，格式为 {chain_id: {"chain": [[s, r, o, t], ...], "score": float, ...}}
    query: 查询四元组，格式为 [subject, relation, target, time]
    返回:
    True: 如果 LLM 认为信息充足 False: 如果 LLM 认为信息不足
    """
    try:
        # 生成评估 prompt
        prompt = get_sufficiency_prompt(chains, query)

        # 调用 LLM 获取评估结果
        result = get_evaluation_results(prompt)

        # 检查结果中是否包含 "YES"（忽略大小写）
        if result and isinstance(result, str):
            return "YES" in result.upper()

        return False
    except Exception as e:
        print(f"Error in evaluate_chain_sufficiency: {e}")
        return False


