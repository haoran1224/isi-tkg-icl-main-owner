from typing import Dict, List, Any, Tuple

from selfCode.LLMAPI.qwen_utils import get_evaluation_results
from utils import get_entity_edges_before_time


def evaluate_candidate_entities_globally(
    local_branches: Dict[str, List[List[Any]]],
    query: List[Any],
    top_n: int = 10
) -> Tuple[Dict[str, List[List[Any]]], Dict[str, List[List[Any]]], List[str]]:
    """
    主调度与数据路由函数 - 高阶历史路径推演核心

    该函数执行以下流程：
    1. 提取 local_branches 中的所有候选实体
    2. 构建 LLM 联合决策 prompt
    3. 调用 LLM 获取全局评估结果
    4. 解析每个实体的动作（HALT/DISCARD/EXPAND）
    5. 根据动作进行数据路由切分

    参数:
        local_branches: 按尾实体分组的字典 {tail_entity: [(s, r, o, t), ...]}
        query: 查询信息 [subject, relation, target, time]
        top_n: 评估前 N 个候选实体（默认10）

    返回:
        Tuple 包含三个元素:
        - halt_branches: 直接进入最终 CoT 预测池的实体及证据簇
        - expand_branches: 需要二阶历史检索的实体及证据簇
        - discard_entities: 被丢弃的实体名称列表（用于日志记录）
    """
    # 初始化返回结果
    halt_branches: Dict[str, List[List[Any]]] = {}
    expand_branches: Dict[str, List[List[Any]]] = {}
    discard_entities: List[str] = []

    # 如果没有候选实体，直接返回空结果
    if not local_branches:
        return halt_branches, expand_branches, discard_entities, 0

    # 提取所有候选实体名称（按四元组数量排序，取前 top_n）
    sorted_entities = sorted(local_branches.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]
    candidate_names = [entity for entity, _ in sorted_entities]

    try:
        # 步骤1: 构建全局评估 prompt
        # 传递已排序的实体，避免在函数内部重复排序
        eval_prompt = get_global_eval_prompt(local_branches, query, top_n, sorted_entities)
        tokens_count = int(len(eval_prompt.split()) * 1.3)

        # 步骤2: 调用 LLM 获取评估结果
        llm_output = get_evaluation_results(eval_prompt)

        # 步骤3: 解析 LLM 输出，获取每个实体的动作
        entity_actions = parse_global_eval_actions(llm_output, candidate_names)

        # 步骤4: 数据路由 - 根据动作分发实体及证据簇
        for entity_name, action in entity_actions.items():
            if entity_name not in local_branches:
                continue

            evidence_cluster = local_branches[entity_name]

            if action == "HALT":
                # 终止并预测：划入 halt_branches
                halt_branches[entity_name] = evidence_cluster
            elif action == "EXPAND":
                # 高阶扩展：划入 expand_branches
                expand_branches[entity_name] = evidence_cluster
            elif action == "DISCARD":
                # 剪枝丢弃：仅记录名称，不保留证据簇
                discard_entities.append(entity_name)

    except Exception as e:
        # 如果发生任何错误，采用保守策略：所有实体进入 halt_branches
        print(f"Error in evaluate_candidate_entities_globally: {e}")
        print("Adopting conservative strategy: routing all entities to halt_branches")
        halt_branches = local_branches.copy()
        tokens_count = 0

    return halt_branches, expand_branches, discard_entities, tokens_count




# ============================================================
# 高阶历史路径推演 - 扩展实体判断器模块
# ============================================================

def get_global_eval_prompt(local_branches: Dict[str, List[List[Any]]],
                           query: List[Any],
                           top_n: int = 10,
                           sorted_branches: List[Tuple[str, List[List[Any]]]] = None) -> str:
    """
    构建用于 LLM 联合决策的 Prompt（全局对比模式）

    该函数将所有候选实体的证据簇同时放入一个 Prompt 中，
    让 LLM 在横向对比中自主决定每个候选实体的归属。

    参数:
        local_branches: 按尾实体分组的字典 {tail_entity: [(s, r, o, t), ...]}
        query: 查询信息 [subject, relation, target, time]
        top_n: 展示前 N 个候选实体（默认10）
        sorted_branches: 已排序的候选实体列表（可选，避免重复排序）

    返回:
        格式化的 prompt 字符串
    """
    query_subject = query[0]
    query_relation = query[1]
    query_time = query[3]

    # 如果没有提供已排序的分支，则进行排序（兼容其他调用场景）
    if sorted_branches is None:
        sorted_branches = sorted(
            local_branches.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:top_n]

    parts = []

    # ==================== 系统角色与任务定义 ====================
    parts.append("ROLE AND TASK\n\n")
    parts.append("You are a senior expert in Temporal Knowledge Graph (TKG) reasoning and high-order path inference. ")
    parts.append("Your task is to perform a GLOBAL COMPARATIVE EVALUATION of multiple candidate entities based on their historical evidence clusters.\n\n")

    # ==================== 查询上下文 ====================
    parts.append("=" * 70 + "\n")
    parts.append("TARGET QUERY\n")
    parts.append("=" * 70 + "\n")
    parts.append(f"Predict: ({query_subject}, {query_relation}, ?, on the {query_time}th day)\n")
    parts.append("Task: Determine which candidate entities should advance to final prediction, ")
    parts.append("which should trigger multi-hop expansion, and which should be discarded.\n\n")

    # ==================== 候选实体证据簇展示 ====================
    parts.append("=" * 70 + "\n")
    parts.append("CANDIDATE EVIDENCE CLUSTERS (FOR GLOBAL COMPARISON)\n")
    parts.append("=" * 70 + "\n\n")

    for idx, (tail_entity, quads) in enumerate(sorted_branches, start=1):
        parts.append(f"[Candidate {idx}: {tail_entity}]\n")
        parts.append(f"Historical Evidence Chain:\n")
        for quad in quads:
            s, r, o, t = quad
            parts.append(f"  - {s}, {r}, {o}, on the {t}th day\n")
        parts.append("\n")

    # ==================== 三分类判定标准 ====================
    parts.append("=" * 70 + "\n")
    parts.append("CLASSIFICATION CRITERIA (STRICT THREE-WAY SPLIT)\n")
    parts.append("=" * 70 + "\n\n")

    parts.append("You must assign ONE of the following THREE actions to EACH candidate entity:\n\n")

    parts.append("1. HALT (Terminate and Predict)\n")
    parts.append("   - The entity's historical facts already constitute SUFFICIENT preconditions ")
    parts.append("for the target relation to occur.\n")
    parts.append("   - It is the MOST COMPETITIVE prediction target in global comparison.\n")
    parts.append("   - Direct evidence: clear repeated patterns, strong logical paths, or high temporal proximity.\n\n")

    parts.append("2. DISCARD (Prune and Discard)\n")
    parts.append("   - The candidate's event chain is full of NOISE with low relevance.\n")
    parts.append("   - Probability of target relation occurring is EXTREMELY LOW.\n")
    parts.append("   - Clear interference: contradictory patterns or completely unrelated contexts.\n\n")

    parts.append("3. EXPAND (High-Order Multi-Hop Expansion)\n")
    parts.append("   - Important 'bridge entity' (e.g., intermediary, facilitator) with LOGICAL GAPS.\n")
    parts.append("   - Current evidence is insufficient but shows potential as a connector.\n")
    parts.append("   - Needs multi-hop expansion to uncover deeper causal chains.\n\n")

    # ==================== 输出格式约束 ====================
    parts.append("=" * 70 + "\n")
    parts.append("OUTPUT FORMAT (STRICT)\n")
    parts.append("=" * 70 + "\n\n")

    parts.append("You must output EXACTLY ONE LINE PER CANDIDATE in the following format:\n\n")
    parts.append("[Entity_Name] | [ACTION]\n\n")
    parts.append("Where [ACTION] must be exactly one of: HALT, DISCARD, EXPAND\n\n")

    parts.append("Example output:\n")
    parts.append("China | HALT\n")
    parts.append("Japan | EXPAND\n")
    parts.append("Germany | DISCARD\n")
    parts.append("South_Korea | HALT\n\n")

    parts.append("=" * 70 + "\n")
    parts.append("IMPORTANT INSTRUCTIONS:\n")
    parts.append("=" * 70 + "\n")
    parts.append("1. Output ONE LINE for EACH of the " + str(len(sorted_branches)) + " candidates listed above.\n")
    parts.append("2. Use the EXACT entity names as shown in the candidate list.\n")
    parts.append("3. Use EXACTLY 'HALT', 'DISCARD', or 'EXPAND' (uppercase, no variations).\n")
    parts.append("4. Use ' | ' (space pipe space) as separator.\n")
    parts.append("5. NO explanations, NO numbering, NO extra text.\n")
    parts.append("6. Perform GLOBAL comparison—evaluate entities RELATIVE to each other.\n\n")

    parts.append("Begin your evaluation now:\n")

    return ''.join(parts)


def parse_global_eval_actions(llm_output: str,
                               candidate_names: List[str]) -> Dict[str, str]:
    """
    解析 LLM 批量输出的判定结果

    从 LLM 输出中提取每个 candidate_name 对应的 HALT、DISCARD 或 EXPAND 动作。

    参数:
        llm_output: LLM 返回的原始输出文本
        candidate_names: 候选实体名称列表

    返回:
        字典 {entity_name: action}，action 为 'HALT', 'DISCARD', 或 'EXPAND'
        未匹配到的实体默认赋予 'HALT' 动作（保守策略，不丢失潜在线索）
    """
    actions = {}

    # 默认所有候选实体为 HALT（保守策略）
    for name in candidate_names:
        actions[name] = "HALT"

    if not llm_output or not isinstance(llm_output, str):
        return actions

    lines = llm_output.split('\n')

    # 定义有效的动作类型
    valid_actions = {'HALT', 'DISCARD', 'EXPAND'}

    for line in lines:
        line = line.strip()

        # 尝试匹配 "Entity_Name | ACTION" 格式
        if '|' in line:
            parts = line.split('|')
            if len(parts) == 2:
                entity = parts[0].strip()
                action = parts[1].strip().upper()

                # 验证实体名称和动作有效性
                if entity in candidate_names and action in valid_actions:
                    actions[entity] = action
                    continue

        # 尝试匹配 "Entity_Name: ACTION" 或 "Entity_Name ACTION" 格式（容错）
        for name in candidate_names:
            if name in line:
                for action in valid_actions:
                    if action in line.upper():
                        actions[name] = action
                        break
                break

    return actions


def apply_second_order_expansion(
    halt_branches: Dict[str, List[List[Any]]],
    expand_branches: Dict[str, List[List[Any]]],
    entity_search_space: Dict[str, Dict[int, Dict[str, List[str]]]],
    query: List[Any],
    args,
    relation_scores: Dict[str, float] = None
) -> Dict[str, List[List[Any]]]:
    """
    应用二阶扩展：对expand_branches中的桥梁实体进行高阶历史挖掘，
    并将扩展后的结果合并到halt_branches中

    参数:
        halt_branches: 暂停分支（包含高置信度的候选实体）
        expand_branches: 需要扩展的桥梁实体及其一阶证据簇
        entity_search_space: 实体搜索空间
        query: 查询信息
        args: 参数配置
        relation_scores: 关系评分字典（可选）

    返回:
        合并了二阶扩展结果后的halt_branches
    """
    # 限制扩展实体数量，避免过多的大模型调用
    max_expand_entities = getattr(args, 'max_expand_entities', 3)

    if not expand_branches:
        return halt_branches

    print("Performing second-order expansion with relation-entity dual pruning...")
    expanded_clusters = perform_second_order_expansion(
        expand_branches, entity_search_space, query, args, relation_scores
    )

    # 将高阶扩展后的证据簇合并到halt_branches中
    # 因为它们最终都要作为局部历史视图输入给混合预测模块
    for entity, cluster in expanded_clusters.items():
        if entity in halt_branches:
            # 如果已存在，合并证据簇
            halt_branches[entity].extend(cluster)
            # 去重并重新排序
            unique_quads = list(set([tuple(q) for q in halt_branches[entity]]))
            halt_branches[entity] = [list(q) for q in unique_quads]
            halt_branches[entity].sort(key=lambda x: x[3], reverse=True)
        else:
            # 新实体，直接添加
            halt_branches[entity] = cluster

    print(f"After second-order expansion, halt_branches: {list(halt_branches.keys())}")

    return halt_branches


def perform_second_order_expansion(
    expand_branches: Dict[str, List[List[Any]]],
    entity_search_space: Dict[str, Dict[int, Dict[str, List[str]]]],
    query: List[Any],
    args,
    relation_scores: Dict[str, float] = None
) -> Dict[str, List[List[Any]]]:
    """
    执行二阶历史扩展的主函数 - "关系-实体"双重剪枝

    参数:
        expand_branches: 需要扩展的桥梁实体及其一阶证据簇
        entity_search_space: 实体搜索空间
        query: 查询信息
        args: 参数配置
        relation_scores: 关系评分字典（可选）

    返回:
        融合后的高阶证据簇字典（合并到halt_branches使用）
    """
    if not expand_branches:
        return {}

    # 获取配置参数
    time_window_size = getattr(args, 'time_window_size', 10)
    second_order_len = getattr(args, 'second_order_len', 30)
    top_k_relations = getattr(args, 'top_k_relations', 3)

    # 存储最终的高阶扩展结果
    expanded_clusters = {}

    for bridge_entity, first_order_cluster in expand_branches.items():
        try:
            # ===== 步骤1: 确定动态时序锚点 =====
            anchor_time2 = find_dynamic_time_anchor(first_order_cluster, relation_scores)
            # 这里排序的第一个就是中心事件
            anchor_time = first_order_cluster[0][3]

            # ===== 步骤2: 获取二阶候选集（双向时间窗口）=====
            # TODO:这一块就不太对，是选择前后多少个历史事件
            second_order_quads, second_order_relations = get_entity_edges_in_time_window(
                entity_search_space,
                bridge_entity,
                anchor_time,
                time_window_size,
                second_order_len
            )

            if not second_order_quads or not second_order_relations:
                # 没有二阶数据，保留一阶证据簇
                expanded_clusters[bridge_entity] = first_order_cluster
                continue

            # ===== 步骤3: 上下文感知的二阶关系筛选 =====
            relation_prompt = build_context_aware_relation_prompt(
                first_order_cluster,
                second_order_relations,
                query,
                bridge_entity,
                top_k_relations
            )

            relation_llm_output = get_evaluation_results(relation_prompt)
            pruned_relations_with_scores = parse_relation_pruning_results(relation_llm_output, second_order_relations)

            # 提取保留的关系（即使解析失败也回退到原始关系）
            if pruned_relations_with_scores:
                kept_relations = set([r for r, s in pruned_relations_with_scores[:top_k_relations]])
            else:
                kept_relations = set(second_order_relations)

            # 使用保留的关系对二阶四元组进行硬过滤
            filtered_second_quads = [
                quad for quad in second_order_quads
                if quad[1] in kept_relations
            ]

            if not filtered_second_quads:
                # 过滤后没有剩余，保留一阶证据簇
                expanded_clusters[bridge_entity] = first_order_cluster
                continue

            # ===== 步骤4: 拓扑结构感知的高阶实体复筛 =====
            entity_prompt = build_topology_entity_prompt(first_order_cluster, filtered_second_quads, query, bridge_entity)

            entity_llm_output = get_evaluation_results(entity_prompt)

            # 提取二阶四元组中的尾实体作为候选
            second_order_tails = list(set([quad[2] for quad in filtered_second_quads]))
            kept_tails = parse_entity_pruning_results(entity_llm_output, second_order_tails)

            # 使用保留的尾实体对四元组进行最终过滤
            final_second_quads = [
                quad for quad in filtered_second_quads
                if quad[2] in kept_tails
            ]

            # ===== 步骤5: 证据簇的动态更新与信息融合 =====
            # 将一阶证据簇与最终筛选的二阶交互事件融合
            # 构建"簇-节点"的星状拓扑结构
            merged_cluster = list(first_order_cluster)  # 复制一阶证据簇

            # 添加二阶交互事件
            for quad in final_second_quads:
                merged_cluster.append(quad)

            # 按时间排序
            merged_cluster.sort(key=lambda x: x[3], reverse=True)

            expanded_clusters[bridge_entity] = merged_cluster

        except Exception as e:
            print(f"Error during second-order expansion for {bridge_entity}: {e}")
            # 发生错误时保留一阶证据簇（保守策略）
            expanded_clusters[bridge_entity] = first_order_cluster

    return expanded_clusters

# ============================================================
# 二阶推演 - "关系-实体"双重剪枝高阶历史扩展模块
# ============================================================

def build_context_aware_relation_prompt(
    first_order_cluster: List[List[Any]],
    second_order_relations: List[str],
    query: List[Any],
    bridge_entity: str,
    top_k: int = 5
) -> str:
    """
    构建上下文感知的二阶关系筛选Prompt（关系剪枝）

    将一阶时序证据簇（历史背景）与二阶候选关系集合结合，
    构建联合筛选Prompt，让大模型评估哪些二阶关系最有价值。

    参数:
        first_order_cluster: 一阶证据簇 [[s, r, o, t], ...]
        second_order_relations: 二阶候选关系列表
        query: 查询信息 [subject, relation, target, time]
        bridge_entity: 桥梁实体名称
        top_k: 保留的高价值关系数量

    返回:
        格式化的prompt字符串
    """
    query_subject = query[0]
    query_relation = query[1]
    query_time = query[3]

    parts = []

    # 系统角色与任务
    parts.append("=" * 70 + "\n")
    parts.append("SECOND-ORDER RELATION PRUNING (CONTEXT-AWARE)\n")
    parts.append("=" * 70 + "\n\n")

    parts.append("You are an expert in Temporal Knowledge Graph reasoning. ")
    parts.append("Your task is to select the most valuable second-order relations ")
    parts.append("for high-order path expansion.\n\n")

    # 查询上下文
    parts.append("TARGET QUERY:\n")
    parts.append(f"  ({query_subject}, {query_relation}, ?, on day {query_time})\n")
    parts.append(f"  Bridge Entity: {bridge_entity}\n\n")

    # 一阶历史背景
    parts.append("-" * 70 + "\n")
    parts.append("FIRST-ORDER HISTORICAL CONTEXT (Background Evidence):\n")
    parts.append("-" * 70 + "\n")
    for quad in first_order_cluster:
        s, r, o, t = quad
        parts.append(f"  {s}, {r}, {o}, on the {t}th day\n")
    parts.append("\n")

    # 二阶候选关系
    parts.append("-" * 70 + "\n")
    parts.append("CANDIDATE SECOND-ORDER RELATIONS:\n")
    parts.append("-" * 70 + "\n")
    for idx, rel in enumerate(second_order_relations, 1):
        parts.append(f"  {idx}. {rel}\n")
    parts.append("\n")

    # 筛选标准
    parts.append("=" * 70 + "\n")
    parts.append("SELECTION CRITERIA:\n")
    parts.append("=" * 70 + "\n\n")
    parts.append("Select the Top " + str(top_k) + " most valuable second-order relations based on:\n")
    parts.append("1. Relevance to the target query relation\n")
    parts.append("2. Complementarity to the first-order context\n")
    parts.append("3. Potential to bridge logical gaps\n")
    parts.append("4. Causal reasoning support for the prediction\n\n")

    # 输出格式
    parts.append("=" * 70 + "\n")
    parts.append("OUTPUT FORMAT (STRICT):\n")
    parts.append("=" * 70 + "\n\n")
    parts.append("Output exactly " + str(top_k) + " relations, one per line:\n\n")
    parts.append("[Relation_Name] | [Score]\n\n")
    parts.append("Where [Score] is a confidence value between 0.0 and 1.0\n\n")

    parts.append("Example:\n")
    parts.append("Sign_formal_agreement | 0.92\n")
    parts.append("Engage_in_diplomatic_cooperation | 0.85\n")
    parts.append("Make_a_visit_to | 0.78\n\n")

    parts.append("IMPORTANT:\n")
    parts.append("1. Output ONLY the selected relations with scores\n")
    parts.append("2. Use EXACT relation names from the candidate list\n")
    parts.append("3. NO explanations, NO numbering\n\n")

    return ''.join(parts)


def parse_relation_pruning_results(llm_output: str,
                                   candidate_relations: List[str]) -> List[Tuple[str, float]]:
    """
    解析关系剪枝的大模型输出结果

    参数:
        llm_output: 大模型返回的原始输出
        candidate_relations: 候选关系列表

    返回:
        [(relation, score), ...] 按分数降序排列
    """
    pruned_relations = []

    if not llm_output or not isinstance(llm_output, str):
        return pruned_relations

    lines = llm_output.split('\n')

    for line in lines:
        line = line.strip()

        # 尝试匹配 "Relation | Score" 格式
        if '|' in line:
            parts = line.split('|')
            if len(parts) == 2:
                relation = parts[0].strip()
                try:
                    score = float(parts[1].strip())
                    if relation in candidate_relations and 0.0 <= score <= 1.0:
                        pruned_relations.append((relation, score))
                except ValueError:
                    continue

    # 按分数降序排列
    pruned_relations.sort(key=lambda x: x[1], reverse=True)

    return pruned_relations


def build_topology_entity_prompt(
    first_order_cluster: List[List[Any]],
    second_order_quadruples: List[List[Any]],
    query: List[Any],
    bridge_entity: str
) -> str:
    """
    构建拓扑结构感知的高阶实体复筛Prompt（实体剪枝）

    将一阶证据簇与二阶具体事实（包含尾实体）拼接，
    构成局部子图背景的高阶拓扑推演链路，评估逻辑合理性。

    参数:
        first_order_cluster: 一阶证据簇
        second_order_quadruples: 二阶四元组（已通过关系过滤）
        query: 查询信息
        bridge_entity: 桥梁实体名称

    返回:
        格式化的prompt字符串
    """
    query_subject = query[0]
    query_relation = query[1]
    query_time = query[3]

    parts = []

    # 系统角色与任务
    parts.append("=" * 70 + "\n")
    parts.append("SECOND-ORDER ENTITY PRUNING (TOPOLOGY-AWARE)\n")
    parts.append("=" * 70 + "\n\n")

    parts.append("You are an expert in causal reasoning over temporal knowledge graphs. ")
    parts.append("Your task is to evaluate which second-order paths form coherent, ")
    parts.append("logically valid causal chains.\n\n")

    # 查询上下文
    parts.append("TARGET QUERY:\n")
    parts.append(f"  ({query_subject}, {query_relation}, ?, on day {query_time})\n")
    parts.append(f"  Bridge Entity: {bridge_entity}\n\n")

    # 一阶背景
    parts.append("-" * 70 + "\n")
    parts.append("FIRST-ORDER EVIDENCE (Base Layer):\n")
    parts.append("-" * 70 + "\n")
    for quad in first_order_cluster:
        s, r, o, t = quad
        parts.append(f"  {s}, {r}, {o}, on the {t}th day\n")
    parts.append("\n")

    # 二阶候选链路
    parts.append("-" * 70 + "\n")
    parts.append("CANDIDATE SECOND-ORDER PATHS (Extended Layer):\n")
    parts.append("-" * 70 + "\n\n")

    # 按尾实体分组展示二阶链路
    from collections import defaultdict
    paths_by_tail = defaultdict(list)
    for quad in second_order_quadruples:
        s, r, o, t = quad
        paths_by_tail[o].append(quad)

    for idx, (tail_entity, paths) in enumerate(paths_by_tail.items(), 1):
        parts.append(f"[Path {idx} -> {tail_entity}]\n")
        for quad in paths:
            s, r, o, t = quad
            parts.append(f"  {s}, {r}, {o}, on the {t}th day\n")
        parts.append("\n")

    # 评估标准
    parts.append("=" * 70 + "\n")
    parts.append("EVALUATION CRITERIA:\n")
    parts.append("=" * 70 + "\n\n")
    parts.append("Evaluate each path based on:\n")
    parts.append("1. LOGICAL COHERENCE: Does the path form a valid causal chain?\n")
    parts.append("2. TEMPORAL CONSISTENCY: Do events follow a reasonable timeline?\n")
    parts.append("3. SEMANTIC RELATEDNESS: Is the path relevant to the query context?\n")
    parts.append("4. BRIDGE VALIDITY: Does the bridge entity meaningfully connect layers?\n\n")

    # 输出格式
    parts.append("=" * 70 + "\n")
    parts.append("OUTPUT FORMAT (STRICT):\n")
    parts.append("=" * 70 + "\n\n")
    parts.append("For EACH candidate tail entity, output:\n\n")
    parts.append("[Tail_Entity] | [ACTION]\n\n")
    parts.append("Where [ACTION] is one of: KEEP, DISCARD\n\n")
    parts.append("Example:\n")
    parts.append("China | KEEP\n")
    parts.append("Germany | DISCARD\n")
    parts.append("Japan | KEEP\n\n")

    parts.append("IMPORTANT:\n")
    parts.append("1. Output ONE LINE for EACH of the " + str(len(paths_by_tail)) + " candidates above\n")
    parts.append("2. Use EXACT entity names from the path list\n")
    parts.append("3. Use EXACTLY 'KEEP' or 'DISCARD' (uppercase)\n")
    parts.append("4. NO explanations, NO numbering\n\n")

    return ''.join(parts)


def parse_entity_pruning_results(llm_output: str,
                                 candidate_entities: List[str]) -> List[str]:
    """
    解析实体剪枝的大模型输出结果

    参数:
        llm_output: 大模型返回的原始输出
        candidate_entities: 候选实体列表

    返回:
        保留的实体列表（KEEP的实体）
    """
    kept_entities = []

    if not llm_output or not isinstance(llm_output, str):
        # 回退策略：保守保留所有实体
        return candidate_entities.copy()

    # 构建实体到动作的映射
    entity_actions = {}

    lines = llm_output.split('\n')
    valid_actions = {'KEEP', 'DISCARD'}

    for line in lines:
        line = line.strip()

        # 尝试匹配 "Entity | ACTION" 格式
        if '|' in line:
            parts = line.split('|')
            if len(parts) == 2:
                entity = parts[0].strip()
                action = parts[1].strip().upper()

                if entity in candidate_entities and action in valid_actions:
                    entity_actions[entity] = action
                    continue

        # 容错：尝试其他格式
        for entity in candidate_entities:
            if entity in line:
                for action in valid_actions:
                    if action in line.upper():
                        entity_actions[entity] = action
                        break
                break

    # 提取KEEP的实体
    for entity in candidate_entities:
        action = entity_actions.get(entity, 'KEEP')  # 默认KEEP（保守策略）
        if action == 'KEEP':
            kept_entities.append(entity)

    return kept_entities


def get_entity_edges_in_time_window(entity_search_space, entity, anchor_time,
                                    time_window_size, length):
    """
    获取实体在指定时间锚点周围双向时间窗口内的边（二阶四元组）

    参数:
        entity_search_space: 实体搜索空间字典
        entity: 目标实体
        anchor_time: 动态时序锚点
        time_window_size: 时间窗口大小（向前和向后）
        length: 要获取的边数

    返回:
        四元组列表 [[entity, relation, target, time], ...]
        涉及的关系列表
    """
    if entity not in entity_search_space or length <= 0:
        return [], []

    quadruples = []

    # 双向时间窗口: [anchor_time - time_window_size, anchor_time + time_window_size]
    lower_bound = anchor_time - time_window_size
    upper_bound = anchor_time + time_window_size

    # 遍历实体的所有时间戳
    for t in entity_search_space[entity]:
        if t < lower_bound or t > upper_bound:
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

    return selected_quadruples, list(relations)


def find_dynamic_time_anchor(evidence_cluster: List[List[Any]],
                             relation_scores: Dict[str, float] = None) -> int:
    """
    找出一阶证据簇中综合得分最高的历史事件，将其时间定为动态时序锚点

    参数:
        evidence_cluster: 一阶证据簇 [[s, r, o, t], ...]
        relation_scores: 关系评分字典（可选）

    返回:
        动态时序锚点时间戳
    """
    if not evidence_cluster:
        return 0

    # 如果没有关系评分，直接使用时间最新的四元组
    if not relation_scores:
        evidence_cluster.sort(key=lambda x: x[3], reverse=True)
        return evidence_cluster[0][3]

    # 计算综合分数：关系评分(80%) + 时间新近性(20%)
    # 时间新近性：相对于其他四元组的时间位置
    times = [quad[3] for quad in evidence_cluster]
    max_time = max(times)
    min_time = min(times)
    time_range = max_time - min_time if max_time != min_time else 1

    best_quad = None
    best_score = -1

    for quad in evidence_cluster:
        rel_score = relation_scores.get(quad[1], 0.5)
        # 时间归一化分数：越接近max_time分数越高
        time_score = (quad[3] - min_time) / time_range if time_range > 0 else 1.0
        # 综合分数
        combined_score = 0.8 * rel_score + 0.2 * time_score

        if combined_score > best_score:
            best_score = combined_score
            best_quad = quad

    return best_quad[3] if best_quad else evidence_cluster[0][3]


# ============================================================
# 轻量化二阶扩展 - "锚点提取 + 全局 LLM 筛选" 模块
# ============================================================

def apply_lightweight_second_order_expansion(
    halt_branches: Dict[str, List[List[Any]]],
    expand_branches: Dict[str, List[List[Any]]],
    entity_search_space: Dict[str, Dict[int, Dict[str, List[str]]]],
    query: List[Any],
    top_n: int = 30
) -> Dict[str, List[List[Any]]]:
    """
    轻量化二阶扩展：基于锚点提取 + 全局 LLM 筛选的二阶扩展逻辑

    流程:
        1. 从 expand_branches 的每个桥梁实体提取中心事件（锚点）
        2. 以锚点时间为界，规则拉取该桥梁实体的 10 条历史作为二阶候选
        3. 将二阶候选与锚点拼接为候选链路，汇总到全局列表
        4. 一次全局 LLM 筛选，选出 Top-30 条最有价值的链路
        5. 将选中链路中的二阶尾实体追加到 halt_branches 的对应证据簇中

    参数:
        halt_branches: 当前已有的暂停分支（高置信度候选实体及证据簇）
        expand_branches: 需要二阶扩展的桥梁实体及其一阶证据簇
        entity_search_space: 实体搜索空间
        query: 查询信息 [subject, relation, target, time]
        top_n: LLM 筛选保留的最大链路数（默认30）

    返回:
        (halt_branches, tokens_count, facts_count)
        - halt_branches: 更新后的暂停分支
        - tokens_count: LLM 筛选消耗的 Token 数
        - facts_count: 二阶检索的事实总数
    """
    # 没有 expand_branches，直接返回
    if not expand_branches:
        return halt_branches, 0, 0

    # 统计变量
    tokens_count = 0
    facts_count = 0

    # -------- Step 1 & Step 2: 提取锚点 + 规则拉取二阶候选 --------
    all_candidate_chains = []  # 全局候选链路列表

    for bridge_entity, first_order_cluster in expand_branches.items():
        # Step 1: 中心事件 = 已排序一阶簇的第一个元素
        if not first_order_cluster:
            continue
        anchor_quad = first_order_cluster[0]
        anchor_time = anchor_quad[3]

        # Step 2: 以 anchor_time 为界，拉取桥梁实体在该时间之前的 10 条历史
        second_order_quads, _ = get_entity_edges_before_time(
            entity_search_space, bridge_entity, anchor_time, length=10
        )

        # 统计：累加二阶检索的事实数
        facts_count += len(second_order_quads)

        # 每条二阶历史与中心事件拼接，形成候选链路
        for sq in second_order_quads:
            chain_entry = {
                "chain": [sq, anchor_quad],       # [二阶事实, 一阶中心事实]
                "bridge_entity": bridge_entity,    # 所属桥梁实体
                "second_order_quad": sq,           # 二阶事件本身
                "tail_entity": sq[2],              # 二阶事实的尾实体 C
            }
            all_candidate_chains.append(chain_entry)

    # 没有候选链路，直接返回
    if not all_candidate_chains:
        return halt_branches, 0, facts_count

    # -------- Step 3: 全局 LLM 筛选 --------
    # 如果总数不足 top_n，直接全部保留，无需调用 LLM
    if len(all_candidate_chains) <= top_n:
        selected_indices = list(range(len(all_candidate_chains)))
    else:
        try:
            filter_prompt = build_global_chain_filter_prompt(
                all_candidate_chains, query, top_n
            )
            tokens_count += int(len(filter_prompt.split()) * 1.3)
            llm_output = get_evaluation_results(filter_prompt)
            selected_indices = parse_global_chain_filter_results(
                llm_output, len(all_candidate_chains), top_n
            )
            # 如果解析结果为空，触发 fallback
            if not selected_indices:
                raise ValueError("LLM output parsed to empty ID list")
        except Exception as e:
            # Fallback: LLM 筛选失败时按时间倒序保留前 top_n 条
            print(f"LLM global chain filter failed ({e}), using fallback (top-{top_n} by time).")
            indexed = list(enumerate(all_candidate_chains))
            indexed.sort(key=lambda x: x[1]["second_order_quad"][3], reverse=True)
            selected_indices = [idx for idx, _ in indexed[:top_n]]

    # -------- Step 4: 更新 halt_branches --------
    for idx in selected_indices:
        entry = all_candidate_chains[idx]
        tail_entity_c = entry["tail_entity"]
        second_quad = entry["second_order_quad"]

        # 将二阶事件追加到 halt_branches 中以尾实体 C 为键的证据簇
        if tail_entity_c in halt_branches:
            # 追加并去重
            existing_set = set(tuple(q) for q in halt_branches[tail_entity_c])
            if tuple(second_quad) not in existing_set:
                halt_branches[tail_entity_c].append(second_quad)
                halt_branches[tail_entity_c].sort(key=lambda x: x[3], reverse=True)
        else:
            # 新实体，创建新证据簇
            halt_branches[tail_entity_c] = [second_quad]

    return halt_branches, tokens_count, facts_count


def build_global_chain_filter_prompt(
    candidate_chains: List[Dict[str, Any]],
    query: List[Any],
    top_n: int = 30
) -> str:
    """
    构建全局候选链路筛选 Prompt

    将所有候选链路以带序号的形式展示给 LLM，要求其从中选出
    对预测目标最有效的 top_n 条链路 ID。

    参数:
        candidate_chains: 候选链路列表，每项包含 chain / bridge_entity / second_order_quad 等
        query: 查询信息 [subject, relation, target, time]
        top_n: 需要保留的链路数量

    返回:
        格式化的 prompt 字符串
    """
    query_subject = query[0]
    query_relation = query[1]
    query_time = query[3]

    parts = []

    # 系统角色与任务
    parts.append("=" * 70 + "\n")
    parts.append("GLOBAL CANDIDATE CHAIN RANKING (LIGHTWEIGHT SECOND-ORDER FILTER)\n")
    parts.append("=" * 70 + "\n\n")

    parts.append("You are an expert in Temporal Knowledge Graph (TKG) reasoning. ")
    parts.append("Your task is to rank and select the most valuable second-order chains ")
    parts.append("for predicting the target query.\n\n")

    # 查询上下文
    parts.append("-" * 70 + "\n")
    parts.append("TARGET QUERY:\n")
    parts.append("-" * 70 + "\n")
    parts.append(f"  Predict: ({query_subject}, {query_relation}, ?, on the {query_time}th day)\n\n")

    # 候选链路列表
    parts.append("-" * 70 + "\n")
    parts.append(f"CANDIDATE SECOND-ORDER CHAINS (Total: {len(candidate_chains)}):\n")
    parts.append("-" * 70 + "\n\n")

    for idx, entry in enumerate(candidate_chains, start=1):
        chain = entry["chain"]
        # chain[0] = 二阶事件, chain[1] = 一阶中心事件
        s2, r2, o2, t2 = chain[0]
        s1, r1, o1, t1 = chain[1]
        parts.append(f"ID {idx}: [{s2}, {r2}, {o2}, day {t2}] -> [{s1}, {r1}, {o1}, day {t1}]\n")

    parts.append("\n")

    # 筛选指令
    parts.append("=" * 70 + "\n")
    parts.append("SELECTION INSTRUCTION:\n")
    parts.append("=" * 70 + "\n\n")
    parts.append(f"Select the TOP {top_n} most effective chain IDs for predicting the target query.\n")
    parts.append("Evaluate based on:\n")
    parts.append("  1. Causal relevance to the target relation\n")
    parts.append("  2. Temporal proximity and logical ordering\n")
    parts.append("  3. Entity/bridge entity connection strength\n\n")

    # 输出格式
    parts.append("=" * 70 + "\n")
    parts.append("OUTPUT FORMAT (STRICT):\n")
    parts.append("=" * 70 + "\n\n")
    parts.append(f"Output exactly {min(top_n, len(candidate_chains))} chain IDs, one per line:\n\n")
    parts.append("ID: [number]\n\n")
    parts.append("Example:\n")
    parts.append("ID: 3\n")
    parts.append("ID: 15\n")
    parts.append("ID: 42\n\n")

    parts.append("IMPORTANT:\n")
    parts.append("1. Output ONLY the selected IDs\n")
    parts.append("2. Each ID must be from the candidate list (1 to " + str(len(candidate_chains)) + ")\n")
    parts.append("3. NO explanations, NO extra text\n\n")

    return ''.join(parts)


def parse_global_chain_filter_results(
    llm_output: str,
    total_candidates: int,
    top_n: int
) -> List[int]:
    """
    解析全局链路筛选的 LLM 输出，提取选中的 ID 列表

    参数:
        llm_output: LLM 返回的原始文本
        total_candidates: 候选链路总数
        top_n: 需要保留的最大数量

    返回:
        选中的 0-based 索引列表（对应 candidate_chains 的下标）
    """
    import re

    selected = set()

    if not llm_output or not isinstance(llm_output, str):
        # Fallback: 返回空列表，让上层走 fallback 逻辑
        return []

    lines = llm_output.split('\n')
    for line in lines:
        line = line.strip()
        # 匹配 "ID: 3" / "ID:15" / "3" 等格式
        match = re.search(r'(?:ID\s*[:：]\s*)?(\d+)', line)
        if match:
            try:
                one_based_id = int(match.group(1))
                if 1 <= one_based_id <= total_candidates:
                    selected.add(one_based_id - 1)  # 转为 0-based
            except ValueError:
                continue

    # 限制数量
    result = sorted(selected)[:top_n]

    # 如果解析结果为空，返回空列表触发上层 fallback
    return result
