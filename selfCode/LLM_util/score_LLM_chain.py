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
    parts.append("Based on the above historical event chains, are there enough relevant patterns or sufficient contextual clues to make a reasonable prediction for the query?\n")
    parts.append("First, briefly explain your reasoning in 1-2 sentences. Then, explicitly output your final decision as exactly 'VERDICT: YES' or 'VERDICT: NO' on a new line.\n")
    parts.append("- 'VERDICT: YES': The history provides enough helpful context or behavioral patterns to guess the target entity.\n")
    parts.append("- 'VERDICT: NO': The history is too disconnected, irrelevant, or lacks necessary clues for a reasonable guess.\n")

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
            return "VERDICT: YES" in result.upper()

        return False
    except Exception as e:
        print(f"Error in evaluate_chain_sufficiency: {e}")
        return False


# ============================================================
# V2 版本：增强的关系剪枝函数（基于详细评估维度）
# ============================================================

def get_prune_relation_prompt_v2(relation_set, query, chain):
    """
    构建V2版本的关系剪枝 prompt（基于 fixPrompt.txt 的设计）
    """
    head_entity = query[0]
    target_relation = query[1]
    target_time = query[3]

    parts = []

    # Role
    parts.append("Role\n\n")
    parts.append("You are a senior international relations analyst and an expert in temporal knowledge graph (TKG) feature engineering. ")
    parts.append("Your task is to evaluate the information value of a set of recently observed relations under a specific historical context for predicting a future target event.\n\n")

    # Task Definitions
    parts.append("Task Definitions\n")
    parts.append(f"Target Query: Predict which tail entity will form the relation [{target_relation}] with the head entity [{head_entity}] at time [{target_time}].\n")
    parts.append("Contextual Relation Set: The set of all relation types that [Head_Entity] has participated in within a recent historical time window prior to [Target_Time].\n\n")

    # Current Context
    parts.append("Current Context\n")
    parts.append(f"Query to Predict: ({head_entity}, {target_relation}, ?, {target_time})\n")
    parts.append("Recently Observed Relation Set: ")
    parts.append(", ".join(relation_set))
    parts.append("\n\n")

    # Evaluation Criteria
    parts.append("Evaluation Criteria\n\n")
    parts.append("Carefully evaluate each relation in the Contextual Relation Set. You need to determine:\n\n")
    parts.append(f"If the head entity has such a relation with a candidate country, how much predictive value does it provide for inferring the [{target_relation}]?\n\n")
    parts.append("Please conduct a comprehensive assessment based on the following four dimensions, and assign:\n")
    parts.append("an Information Value Score (0–10)\n")
    # parts.append("a Polarity (Positive / Negative / Neutral)\n\n")

    parts.append("1. Strong Logical Path (Local Path Relevance) [8–10 points]\n")
    parts.append(f"Positive Correlation: The relation is a direct precursor, necessary condition, or key catalyst of the target relation [{target_relation}]\n")
    parts.append("(e.g., for predicting \"Sign Agreement\", relations like \"Consult\" or \"Mutual Visits\" should receive very high scores).\n")
    parts.append("Negative Correlation: The relation is strongly contradictory or destructive with respect to the target relation\n")
    parts.append("(e.g., for predicting \"Cooperation\", relations like \"Expel\" or \"Military Conflict\" carry strong exclusionary value—high score but negative polarity).\n\n")

    parts.append("2. Macro Contextual Setup (Global Evolution Context) [5–7 points]\n")
    parts.append("The relation does not directly lead to the target event but reflects the overall diplomatic tone or strategic tendency of the head entity in the recent period\n")
    parts.append("(e.g., \"Provide Aid\", \"Express Optimism\" help establish a cooperative atmosphere).\n\n")

    parts.append("3. Weak or Ambiguous Signals [2–4 points]\n")
    parts.append("One-sided actions without strong constraints on future developments, or relations that may lead to multiple possible outcomes in the current context\n")
    parts.append("(e.g., \"Make Speech\", \"Symbolic Actions\").\n\n")

    parts.append("4. Noise [0–1 point]\n")
    parts.append("Overly generic, high-frequency daily interactions with low information entropy that do not help distinguish specific target entities.\n\n")

    # Output Format - 简化版本：只输出关系名和分数
    parts.append("Output Format\n\n")
    parts.append("For each relation in the set, output ONLY the relation name and its information value score.\n\n")
    parts.append("Format: [Relation Name]: [Score]\n\n")
    parts.append("Example:\n")
    parts.append("Consult: 9\n")
    parts.append("Express_intent_to_cooperate: 7\n")
    parts.append("Make_statement: 3\n\n")
    parts.append("Please strictly follow the above format. Do NOT output any explanations, justifications, or additional text.\n")

    return ''.join(parts)


def parse_relation_scores_v2(result_str):
    """
    解析V2版本的关系评分输出（简化版：只包含关系名和分数）

    输入格式：
    Relation: Score

    返回:
    字典: {relation_name: normalized_score (0-1)}
    """
    if not result_str or not isinstance(result_str, str):
        return {}

    relation_scores = {}
    lines = result_str.split('\n')

    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue

        # 分割关系名和分数
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        relation = parts[0].strip()
        score_str = parts[1].strip()

        # 尝试解析分数
        try:
            score = float(score_str)
            # 归一化到 0-1 范围（假设输入是 0-10）
            normalized_score = score / 10.0
            relation_scores[relation] = normalized_score
        except ValueError:
            # 如果解析失败，跳过这一行
            continue

    return relation_scores


def prune_relation_set_v2(relation_set, query, chain, top_relation=3):
    """
    V2 版本的关系剪枝函数（使用详细的评估维度，简化输出）
    """
    # 构建V2版本的 prompt
    prompt = get_prune_relation_prompt_v2(relation_set, query, chain)

    try:
        # 调用LLM获取关系评分
        result = get_evaluation_results(prompt)

        # 解析V2格式的评分（简化版：关系名:分数）
        relation_scores = parse_relation_scores_v2(result)

        # 如果评分结果为空，返回原始关系集合的前几个
        if not relation_scores:
            pruned_relations = relation_set[:top_relation] if len(relation_set) > top_relation else relation_set
            return pruned_relations, {}

        # 按评分降序排序
        relation_score_pairs = sorted(
            relation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 筛选前 top_k 个关系
        # TODO：这一块可以改一下字根据阈值也可以
        pruned_relations_with_scores = relation_score_pairs[:top_relation]

        # 返回剪枝后的关系和对应的评分
        pruned_relations = [rel for rel, score in pruned_relations_with_scores]
        relation_scores_dict = {rel: score for rel, score in pruned_relations_with_scores}

        return pruned_relations, relation_scores_dict

    except Exception as e:
        # 如果发生错误，返回原始关系集合的前几个关系
        print(f"Error in prune_relation_set_v2: {e}")
        pruned_relations = relation_set[:top_relation] if len(relation_set) > top_relation else relation_set
        return pruned_relations, {}
