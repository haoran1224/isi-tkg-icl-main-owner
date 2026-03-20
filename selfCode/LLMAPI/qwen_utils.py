import math
from time import sleep
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # 新加坡和北京地域的API Key不同。获取API Key：https://www.alibabacloud.com/help/zh/model-studio/get-api-key
    api_key="sk-d226dae58dd948c9813e37a74a7e7c43",
    # 以下为新加坡地域base_url，若使用北京地域的模型，需将base_url替换为：https://dashscope.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 新增：专为时序知识图谱(TKG)推理设计的 System Prompt
TKG_SYSTEM_PROMPT = """You are an expert reasoning model specialized in Temporal Knowledge Graph (TKG) forecasting and link prediction.
Your task is to analyze historical event chains (temporal logic paths) to evaluate their relevance and sufficiency for predicting future missing links.
Please note: In TKG forecasting, past events act as behavioral patterns, structural clues, or temporal trends, rather than strict, deterministic causal evidence. 
Please evaluate the information based on whether it provides reasonable contextual support or logical patterns for the target query."""

prompt = """
    你现在是一个时间推理模型，你需要根据给定的时间推理问题，推理出问题的答案。
    target_query = ("Manohar Parrikar", "express intent to meet", "2018/10/16")
    historical_events = [
        ("Manohar Parrikar", "express intent to meet", "India", "2018/06/14"),
        ("Director General (India)", "host a visit for", "Manohar Parrikar", "2018/06/14"), 
        ("Bharatiya Janata", "make a statement about", "Manohar Parrikar", "2018/06/14"),
    ]
    候选项：
    1.Manohar Parrikar
    2.India 
    3.Bharatiya Janata
    task：根据query和候选项，给出最有可能的答案，只需要给出最终输出的数字选项即可，其他的不需要输出
    Do not output options that are not in the list
    If you must infer several {object} that you think may be the answer to the given query based on the given historical 
relations, Your task is to identify which relation is most relevant to answering the query based on the event chain. 
    """

prompt2 = """
You are given a set of relations, a query, and an event history chain. 
If you must infer {object} that you think may be the answer to the given query based on the given historical events, 
what important history chains do you base your predictions on? 
Please rate the importance of each chain between 0~1, Higher scores indicate higher importance

Query:
Malaysia Express_intent_to_cooperate {object} on 8016 time

Available chains:
1. Arrest,_detain,_or_charge_with_legal_action->Query
2. Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)->Query
3. Make_statement->Query
4. Host_a_visit->Query
5. Sign_formal_agreement->Query
6. Engage_in_symbolic_act->Query
7. Express_intent_to_cooperate->Query
8. Engage_in_diplomatic_cooperation->Query
9. Return,_release_person(s)->Query
10. Cooperate_militarily->Query
11. Consult->Query
12. Express_intent_to_provide_humanitarian_aid->Query
13. Make_optimistic_comment->Query
14. Express_intent_to_cooperate_economically->Query
15. Express_intent_to_meet_or_negotiate->Query
16. Express_intent_to_provide_material_aid->Query
17. Make_an_appeal_or_request->Query
18. Express_accord->Query
19. Reject->Query
20. Provide_humanitarian_aid->Query
21. Expel_or_deport_individuals->Query
22. Discuss_by_telephone->Query
23. Praise_or_endorse->Query
24. Criticize_or_denounce->Query
25. Appeal_for_diplomatic_cooperation_(such_as_policy_support)->Query

Output format:
Example: 
1:0.3
2:0.4
3:0.2
Only output the number and score, output in the order of 1, 2, 3, 4, 5, no additional explanation.
"""

prompt3 = """
You are given a set of entities, a query, an event chain, and a relation. 
Your task is to rate the relevance of each entity to answering the query based on the event chain and given relation.


Available entities:
1. South_Korea
2. Japan
3. Canada
4. Vietnam

Current event chain:
7536: Malaysia Express_intent_to_cooperate China

Query:
8016: Malaysia Express_intent_to_cooperate ?

Current relation to consider: Express_intent_to_cooperate

Please select the most relevant entities from the available entities that could help answer the query with the given relation.
Output format: List the numbers of the selected entities separated by commas.
Example: 
1:0.3
2:0.4
3:0.2
Only output the number and score, no additional explanation.
"""

def apply_temperature_and_topk(logprobs, temperature=3.0, top_k=None):
    if not logprobs:
        return []

    # 处理Top-K截断
    if top_k and len(logprobs) > top_k:
        logprobs = logprobs[:top_k]

    # 提取logprob值，并应用温度系数
    tokens = [item.token for item in logprobs]
    log_probs = [item.logprob / temperature for item in logprobs]

    # 计算softmax
    max_log_prob = max(log_probs)  # 数值稳定，防止指数溢出
    exp_probs = [math.exp(log_p - max_log_prob) for log_p in log_probs]
    sum_exp_probs = sum(exp_probs)
    adjusted_probs = [exp_p / sum_exp_probs for exp_p in exp_probs]

    # 组合结果
    return list(zip(tokens, adjusted_probs))


# 评估函数,返回是token-概率对列表, 仅用于评估结果只为第一个内容
def get_evaluation_results_QWEN(prompt, temperature=5.0, top_k=5):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": TKG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        logprobs=True,
        top_logprobs=top_k  # 与TOP_K参数保持一致
    )

    # 解析结果
    choice = completion.choices[0]
    logprobs = choice.logprobs.content[0].top_logprobs
    adjusted_results = apply_temperature_and_topk(logprobs, temperature=temperature, top_k=top_k)
    return adjusted_results


def get_evaluation_results(prompt, max_retries=2):
    got_result = False
    retries = 0
    while not got_result and retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": TKG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0
            )
            got_result = True
        except Exception as e:
            retries += 1
            print(f"LLM API call failed, retrying ({retries}/{max_retries}): {e}")
            sleep(1)
    
    if not got_result:
        print(f"Failed to get LLM response after {max_retries} retries")
        return ""
    
    return completion.choices[0].message.content

def predict_k_chatGLM(prompt, max_retries=2):
    got_result = False
    retries = 0
    while not got_result and retries < max_retries:
        try:
            results = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": TKG_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.0,
            )
            got_result = True
            print(results.choices[0].message.content)
        except Exception:  # pylint: disable=broad-exception-caught
            retries += 1
            return []

    parsed_results = parse_results(results)
    return parsed_results

def parse_results(result):
    return_text = result.choices[0].message.content
    to_return = return_text.split('\n')
    # 遍历所有行，只处理以数字+点+空格开头的行
    results = []
    for line in to_return:
        line = line.strip()
        if line and '. ' in line:
            # 检查行是否以数字+点+空格开头
            parts = line.split('. ', 1)
            if len(parts) > 1 and parts[0].isdigit():
                results.append(parts[1])
    return results


if __name__ == "__main__":
    # 配置参数
    TEMPERATURE = 5.0  # 温度系数，建议2.0-5.0
    TOP_K = 5  # Top-K截断值，与API返回的top_logprobs保持一致

    completion = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content":"介绍一下你自己"},
        ],
        # logprobs=True,
        # top_logprobs=TOP_K  # 与TOP_K参数保持一致
    )
    # --- 如何解析数据 ---
    choice = completion.choices[0]
    print(f"最终选择的答案: {choice.message.content}")

    print(completion.model_dump_json())

    # 获取第一个生成token的logprobs信息
    # 注意：如果输出很长，这里是个列表；针对你的场景通常只有一个token
    first_token_logprobs = choice.logprobs.content[0]

    print("\n--- Original Top Candidates Logprobs ---")
    if first_token_logprobs.top_logprobs:
        for item in first_token_logprobs.top_logprobs:
            prob = math.exp(item.logprob)
            print(f"Token: '{item.token}', Logprob: {prob:.6f}")
    else:
        print("未获取到 top_logprobs 数据，请检查 API 参数支持情况。")

    # 应用温度系数和Top-K调整
    print(f"\n--- Adjusted Logprobs (T={TEMPERATURE}, Top-K={TOP_K}) ---")
    if first_token_logprobs.top_logprobs:
        adjusted_results = apply_temperature_and_topk(first_token_logprobs.top_logprobs, temperature=TEMPERATURE,
                                                      top_k=TOP_K)
        for token, prob in adjusted_results:
            print(f"Token: '{token}', Adjusted Prob: {prob:.6f}")
    else:
        print("未获取到 top_logprobs 数据，请检查 API 参数支持情况。")
