# 基于大模型的时序图谱预测方法 V2

## 概述

本项目实现了一种基于大语言模型（LLM）的时序知识图谱（TKG）预测方法，通过融合**局部分支**和**全局分支**的历史信息，利用思维链（Chain of Thought）推理机制进行候选实体的置信度预测。

---

## 设计思路

### 核心思想

针对具体的预测任务（如 `Malaysia, Express_intent_to_cooperate, ?, 8016`），本方法通过以下六个步骤完成预测：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TKG 预测方法 V2 架构                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入: (Malaysia, Express_intent_to_cooperate, ?, 8016)                 │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 步骤1: 历史四元组检索                                             │   │
│  │ 从头实体出发，找到最近的n个时间步的四元组，提取交互的关系类型       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 步骤2: 关系筛选                                                  │   │
│  │ 通过LLM对关系集合进行剪枝，过滤出高价值的历史链路                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 步骤3: 四元组筛选                                                │   │
│  │ 根据筛选得到的关系链路，筛选n个四元组作为高价值历史信息            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│         ┌──────────────────────┴──────────────────────┐                │
│         ↓                                              ↓                │
│  ┌─────────────────┐                        ┌─────────────────┐        │
│  │  局部分支        │                        │  全局分支        │        │
│  │  (Local Branch) │                        │ (Global Branch) │        │
│  ├─────────────────┤                        ├─────────────────┤        │
│  │ 按尾实体分组    │                        │ 使用LLM生成     │        │
│  │ 形成一阶候选    │                        │ 宏观演变状态    │        │
│  │ 线索            │                        │ 的自然语言描述  │        │
│  └─────────────────┘                        └─────────────────┘        │
│         └──────────────────────┬──────────────────────┘                │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 步骤6: 大模型预测 (Chain of Thought)                              │   │
│  │ 融合全局和局部分支，使用思维链进行因果推演                         │   │
│  │ 输出每个候选实体的置信度概率，按置信度降序排列                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  输出: Top-K 预测结果 (带置信度分数)                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 详细步骤说明

#### 步骤1：历史四元组检索
- 从头实体（如 Malaysia）出发
- 找到与该实体交互的最近 n 个时间步的所有四元组
- 从这些四元组中提取所有交互的关系类型

#### 步骤2：关系筛选 ⭐ V2 增强版
- 一个预测实体会连接很多不同的关系
- **使用 `prune_relation_set_v2` 函数**，基于四个维度进行详细评估：
  - **强逻辑路径** (8-10分)：直接前驱、必要条件或关键催化剂
  - **宏观语境设置** (5-7分)：反映整体外交基调或战略倾向
  - **弱或模糊信号** (2-4分)：单方面行动，约束力弱
  - **噪声** (0-1分)：通用高频低信息熵的交互
- 输出包含**信息价值评分**和**极性**（Positive/Negative/Neutral）
- 过滤出高价值的历史链路关系（负极性表示强排除信号）

#### 步骤3：四元组筛选
- 根据筛选得到的关系链路
- 对每个关系下的四元组进行 LLM 评分
- 筛选出高价值的四元组作为历史信息

#### 步骤4：局部分支（Local Branch）
- 将筛选后的四元组按照**尾实体**进行分组
- 将属于同一个尾实体的四元组按时间顺序排列
- 形成多条**一阶候选线索**（Candidate-Specific Patterns）

#### 步骤5：全局分支（Global Branch）⭐ LLM增强
- 忽略具体的尾实体
- 统计头实体（预测的头实体）在近期时间窗口内发起的**各类关系频率**
- **使用大模型生成宏观演变状态的自然语言描述**（Macro State）
- 示例输出：
  > "近期，Malaysia 展现出强烈的积极外交姿态，频繁与多个实体发生 Sign_formal_agreement 和 Engage_in_diplomatic_cooperation，表明其正处于活跃的联盟构建和国际合作阶段。"

#### 步骤6：大模型预测（Chain of Thought）
- 融合全局分支和局部分支的信息
- 构建包含思维链推理要求的 Prompt
- 要求 LLM 进行因果推演，输出候选实体的置信度概率
- 按置信度对所有候选实体进行降序排列
- 输出 Top-K 预测结果

---

## 文件结构

```
isi-tkg-icl-main/
├── run_openai_v2.py              # V2版本启动入口
├── prepare_history_chain_v2.py   # V2版本核心逻辑
├── utils.py                      # 工具函数（兼容原有代码）
├── evaluate.py                   # 评估脚本（兼容原有代码）
├── selfCode/
│   ├── LLMAPI/
│   │   └── qwen_utils.py         # LLM API 调用
│   └── LLM_util/
│       └── score_LLM_chain.py    # LLM 评分工具
└── README_V2.md                  # 本文档
```

---

## 使用方法

### 1. 运行 V2 版本

```bash
python run_openai_v2.py \
    --model chatGLM \
    --dataset ICEWS18 \
    --history_len 50 \
    --history_type entity \
    --history_direction uni \
    --top_k 20 \
    --world_size 1 \
    --rank 0
```

### 2. 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 使用的模型 | `chatGLM` |
| `--dataset` | 数据集名称 | `ICEWS18` |
| `--history_len` | 历史四元组数量 | `50` |
| `--history_type` | 历史类型 | `entity` |
| `--top_k` | 返回Top-K预测结果 | `20` |
| `--world_size` | 并行处理数量 | `1` |
| `--rank` | 当前进程rank | `0` |
| `--use_llm_global` | ⭐ 使用LLM生成全局分支宏观描述 | `True` |

### 3. 评估结果

使用评估脚本计算 MRR、Hits@1、Hits@3、Hits@10 指标：

```bash
python evaluate.py --input outputs/chatGLM_ICEWS18_*.jsonl
```

---

## 核心函数说明

### `prepare_history_chain_v2()`

主要的历史链准备函数，整合了六个步骤的处理逻辑。

**函数签名：**
```python
def prepare_history_chain_v2(
    x,                              # 查询四元组 [subject, relation, target, time]
    entity_search_space,            # 实体搜索空间
    args,                           # 配置参数
    fileChainName=None,             # 链文件名称（兼容）
    global_history_quadruples=None  # 全局历史四元组（兼容）
) -> (model_input, candidates)
```

### `prune_relation_set_v2()` ⭐ 新增

V2 版本的关系剪枝函数，基于详细的四维度评估体系。

**函数签名：**
```python
def prune_relation_set_v2(
    relation_set: List[str],      # 关系集合
    query: List[Any],             # 查询四元组
    chain: List[List[Any]],       # 当前事件链
    top_relation: int = 3        # 返回的 top 关系数量
) -> (pruned_relations, relation_scores_dict)
```

**功能说明：**
- 使用 `get_prune_relation_prompt_v2()` 构建详细评估 prompt
- 评估维度包括：
  1. **强逻辑路径** (8-10分)：直接前驱/必要条件/催化剂
  2. **宏观语境设置** (5-7分)：外交基调/战略倾向
  3. **弱或模糊信号** (2-4分)：单方面行动
  4. **噪声** (0-1分)：通用高频交互
- 输出格式：每个关系包含 **信息价值评分 (0-10)** 和 **极性**
- 负极性评分表示强排除信号（如"军事冲突"对"合作"预测）

### `build_local_branch()`

局部分支：按尾实体分组形成一阶候选线索。

**函数签名：**
```python
def build_local_branch(
    quadruples: List[List[Any]]     # 四元组列表
) -> Dict[str, List[List[Any]]]    # {tail_entity: [(s, r, o, t), ...]}
```

### `build_global_branch()`

全局分支：统计头实体在近期时间窗口内的关系频率（原始统计方法）。

**函数签名：**
```python
def build_global_branch(
    quadruples: List[List[Any]]     # 四元组列表
) -> Dict[str, Any]                # 包含关系频率统计的字典
```

### `build_global_branch_with_llm()` ⭐ 新增

全局分支（LLM增强版）：使用大模型生成宏观演变状态的自然语言描述。

**函数签名：**
```python
def build_global_branch_with_llm(
    quadruples: List[List[Any]],    # 四元组列表
    query: List[Any],               # 查询四元组
    use_llm: bool = True           # 是否使用LLM生成描述
) -> Dict[str, Any]
```

**功能说明：**
- 首先对四元组进行基础统计（关系频率、实体频率、时间跨度）
- 构建包含统计信息的 Prompt，要求 LLM 生成宏观状态描述
- LLM 生成的描述包含：
  - 实体的外交姿态/行为模式
  - 主导的交互类型及其含义
  - 时间趋势或行为转变（如有）
- 如果 LLM 调用失败，自动回退到统计方法

**示例输出：**
```
"近期，Malaysia 展现出强烈的积极外交姿态，频繁与多个实体发生
Sign_formal_agreement 和 Engage_in_diplomatic_cooperation，表明其
正处于活跃的联盟构建和国际合作阶段。"
```

### `get_candidates_from_local_branches()`

从局部分支中获取候选实体列表（尾实体）。

**函数签名：**
```python
def get_candidates_from_local_branches(
    local_branches: Dict[str, List[List[Any]]]  # 按尾实体分组的字典
) -> List[str]                                   # 候选实体列表
```

**功能说明：**
- 从局部分支中提取所有尾实体作为候选实体
- 按每个尾实体的四元组数量降序排列
- 这样确保候选实体都是经过筛选的高价值历史实体

### `get_prune_relation_prompt_v2()` ⭐ 新增

构建 V2 版本的关系剪枝 Prompt（基于 fixPrompt.txt 的设计）。

**函数签名：**
```python
def get_prune_relation_prompt_v2(
    relation_set: List[str],      # 关系集合
    query: List[Any],             # 查询四元组
    chain: List[List[Any]]        # 当前事件链
) -> str
```

**功能说明：**
- 角色设定：高级国际关系分析师和 TKG 特征工程专家
- 评估四个维度：强逻辑路径、宏观语境、弱信号、噪声
- 输出格式要求：关系名、信息价值评分(0-10)、极性、理由

### `parse_relation_scores_v2()` ⭐ 新增

解析 V2 版本的关系评分输出（包含评分和极性）。

**函数签名：**
```python
def parse_relation_scores_v2(
    result_str: str               # LLM 返回的原始字符串
) -> Dict[str, float]             # {relation_name: normalized_score}
```

**功能说明：**
- 解析格式：`[Relation Name]`, `Information Value Score:`, `Polarity:`, `Justification:`
- 归一化评分到 0-1 范围
- 考虑极性：负极性返回负值（强排除信号），中性降低权重

### `build_cot_prompt()`

构建思维链推理 Prompt。

**函数签名：**
```python
def build_cot_prompt(
    local_branches: Dict[str, List[List[Any]]],
    global_branch: Dict[str, Any],
    query: List[Any],
    candidates: List[str]
) -> str
```

---

## 兼容性说明

本 V2 版本完全兼容原有的评估体系和数据结构：

1. **HitsMetric 评估体系**：使用原有的 `utils.py` 中的 `HitsMetric` 类
2. **load_data 数据结构**：使用原有的 `load_data()` 函数加载数据
3. **输出格式**：保持与原版相同的 JSONL 输出格式

---

## 示例输出

### Prompt 示例

```
============================================================
TEMPORAL KNOWLEDGE GRAPH FORECASTING TASK
============================================================

You are an expert reasoning model specialized in Temporal Knowledge Graph (TKG) forecasting.
Your task is to predict the missing object entity for a given query based on historical evidence.

You will be provided with:
1. GLOBAL CONTEXT: The subject entity's macro-level interaction patterns
2. LOCAL EVIDENCE: Candidate-specific historical interaction chains
3. QUERY: The prediction task to solve

Global Historical Context (Subject Entity's Macro State):
近期，Malaysia 展现出强烈的积极外交姿态，频繁与多个实体发生
Sign_formal_agreement 和 Engage_in_diplomatic_cooperation 交互，表明其
正处于活跃的联盟构建和国际合作阶段。从时间趋势来看，该实体的外交活动
呈明显上升趋势，在最近的30天内交互频率达到峰值。

Local Historical Evidence (Candidate-Specific Patterns):
Candidate 1 (China):
  Malaysia, Express_intent_to_cooperate, China, on the 7980th day
  Malaysia, Engage_in_diplomatic_cooperation, China, on the 7950th day

Candidate 2 (Japan):
  Malaysia, Host_a_visit, Japan, on the 7920th day

...
```

### 预测结果示例

```
Reasoning: Based on the global context, Malaysia frequently engages in
Express_intent_to_cooperate. Local evidence shows strong recent interactions
with China, suggesting a continuation of this pattern.

Predictions:
1. China: 0.85
2. Japan: 0.72
3. South_Korea: 0.58
```

---

## 与原版本的主要区别

| 特性 | 原版本 (V1) | 新版本 (V2) |
|------|-------------|-------------|
| 历史处理 | 多轮迭代扩展链路 | 局部分支 + 全局分支 |
| 分支结构 | 单一证据链 | 双分支融合 |
| 推理方式 | 直接预测 | 思维链 (Chain of Thought) |
| 置信度 | 无明确输出 | 明确的置信度分数 |
| 宏观模式 | 未充分利用 | 全局分支统计 |
| 候选筛选 | 隐式处理 | 显式按尾实体分组 |

---

## 依赖环境

- Python 3.8+
- PyTorch
- OpenAI SDK (用于调用 Qwen API)
- 其他依赖见原项目

---

## 注意事项

1. 本版本不修改原有代码，所有新增代码在独立文件中
2. 启动入口为 `run_openai_v2.py`，与原版 `run_openai.py` 并存
3. 如需使用原有功能，请运行 `run_openai.py`
4. LLM API 调用需要配置有效的 API Key

---

## 未来改进方向

1. 添加对尾实体的多跳扩展（二阶、三阶候选线索）
2. 优化思维链 Prompt 的推理深度
3. 实现自适应的历史窗口大小调整
4. 添加候选实体的预筛选机制
5. 支持更多的大语言模型

---

**版本**: V2.0
**更新日期**: 2026-03-21
**作者**: ISI-TKG-ICL Project Team