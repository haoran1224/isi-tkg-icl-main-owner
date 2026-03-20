import os
from collections import defaultdict, deque
from tqdm import tqdm


def load_data(file_path):
    """
    读取数据集文件并返回事实列表。
    忽略可能存在的 label 列，只提取 (s, r, o, t)。
    """
    data = []
    if not os.path.exists(file_path):
        print(f"警告：找不到文件 {file_path}")
        return data

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 提取 s, r, o, t
            data.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    return data


def build_initial_history(train_data, valid_data):
    """
    利用 Train 和 Valid 集构建测试前的全量历史背景图。
    """
    history_triplets = set()
    history_edges = defaultdict(set)

    # 合并训练集和验证集进行预加载
    pre_test_data = train_data + valid_data

    print("阶段 1/2: 正在加载历史背景知识 (Train + Valid)...")
    for s, r, o, t in tqdm(pre_test_data, desc="构建历史图谱", unit="条"):
        history_triplets.add((s, r, o))
        # 构建无向图结构用于跳数查找
        history_edges[s].add(o)
        history_edges[o].add(s)

    return history_triplets, history_edges


def check_multihop_path(start, target, graph, max_depth):
    """使用 BFS 查找图中是否存在两点间的多跳路径"""
    if start not in graph:
        return False

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        current_node, depth = queue.popleft()

        # 达到最大跳数限制则停止当前分支的搜索
        if depth >= max_depth:
            continue

        for neighbor in graph[current_node]:
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return False


def evaluate_test_set(test_data, history_triplets, history_edges, max_hops=3):
    """
    在测试集上按时间步评估并统计四类数据分布。
    """
    results = {"Repeated": 0, "Single-hop": 0, "Multi-hop": 0, "Unseen": 0}

    # 按照时间戳 t 对测试集进行分组，确保严格按时间顺序评估
    time_grouped_data = defaultdict(list)
    for s, r, o, t in test_data:
        time_grouped_data[t].append((s, r, o))

    sorted_times = sorted(time_grouped_data.keys())
    total_processed = len(test_data)

    print("\n阶段 2/2: 开始在 Test 集上进行分类推理统计...")

    # 按时间步进行推演
    for t in tqdm(sorted_times, desc="评估 Test 集进度", unit="时间步"):
        current_time_facts = time_grouped_data[t]

        # 步骤 A：评测当前时间步的所有查询
        for s, r, o in current_time_facts:
            # 1. 重复的历史事实
            if (s, r, o) in history_triplets:
                results["Repeated"] += 1
            # 2. 一跳链路推导
            elif o in history_edges[s]:
                results["Single-hop"] += 1
            # 3. 多跳推理及其他
            else:
                if check_multihop_path(s, o, history_edges, max_hops):
                    results["Multi-hop"] += 1
                else:
                    results["Unseen"] += 1

        # 步骤 B：当前时间步评测完成后，将这些事实并入历史知识中
        # （因为 t 时刻发生的事件，可以作为预测 t+1 时刻事件的依据）
        for s, r, o in current_time_facts:
            history_triplets.add((s, r, o))
            history_edges[s].add(o)
            history_edges[o].add(s)

    # 打印最终统计报表
    print("\n" + "=" * 50)
    print(f"📊 测试集分类统计报告 (总计事实数: {total_processed})")
    print("-" * 50)
    for category, count in results.items():
        percentage = (count / total_processed) * 100 if total_processed > 0 else 0
        print(f"| {category:12} | {count:10d} | {percentage:6.2f}% |")
    print("=" * 50)


def main():
    # 配置文件路径
    train_file = "train.txt"
    valid_file = "valid.txt"
    test_file = "test2.txt"

    # 1. 加载数据
    train_data = load_data(train_file)
    valid_data = load_data(valid_file)
    test_data = load_data(test_file)

    if not test_data:
        print("测试集为空，无法进行评估。")
        return

    # 2. 构建初始历史知识
    history_triplets, history_edges = build_initial_history(train_data, valid_data)

    # 3. 在测试集上评估分类 (默认最大搜索深度为 3 跳)
    evaluate_test_set(test_data, history_triplets, history_edges, max_hops=3)


if __name__ == "__main__":
    main()