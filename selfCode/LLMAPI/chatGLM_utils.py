import json
from time import sleep

import numpy as np
from zai import ZhipuAiClient
import os

# 获取当前脚本所在目录，构建绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "zhipuai_config.json")

with open(config_path, encoding="utf-8") as f:
    config = json.load(f)

client = ZhipuAiClient(api_key=config["zhipuai_api_key"])


def parse_results(result):
    raw_logprobs = result.choices[0].message.content
    logprobs = [(int(x.strip()), raw_logprobs[x]) for x in raw_logprobs if x.strip().isdecimal()]
    sorted_logprobs = sorted(logprobs, key=lambda tup: tup[1], reverse=True)

    probs = [x[1] for x in sorted_logprobs]
    softmax_probs = np.exp(probs) / np.sum(np.exp(probs), axis=0)

    to_return = [(x[0], p) for x, p in zip(sorted_logprobs, softmax_probs)]
    return to_return


# TODO：目前这一块是只能hit1，就是输出一个结果，但是我们需要输出的是多个结果的概率(原本的函数功能)
def parse_results_chatGLM(result):
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


def parse_textToNumber_chatGLM(result):
    result_text = result.choices[0].message.content

    # 移除所有非数字和非分隔符字符
    cleaned_text = ''
    for char in result_text:
        if char.isdigit() or char in [',', '\n', ' ', ';']:
            cleaned_text += char

    # 将所有分隔符转换为逗号
    for sep in ['\n', ' ', ';']:
        cleaned_text = cleaned_text.replace(sep, ',')

    # 移除连续的逗号
    while ',,' in cleaned_text:
        cleaned_text = cleaned_text.replace(',,', ',')

    # 移除首尾的逗号
    cleaned_text = cleaned_text.strip(',')

    # 分割并转换为整数
    if not cleaned_text:
        return []

    numbers = [int(num.strip()) for num in cleaned_text.split(',') if num.strip().isdigit()]
    return numbers


def predict(prompt, args):
    got_result = False
    while not got_result:
        try:
            results = client.completions.create(
                model="glm-4-air-250414",
                messages=prompt,
                max_tokens=4096,
                temperature=0.0,
                top_p=1,
                n=1,
                stop=["]", "."],
                logprobs=True,
            )
            got_result = True
        except Exception:  # pylint: disable=broad-exception-caught
            sleep(3)

    parsed_results = parse_results(results)  # type: ignore
    return parsed_results


def predict_chatGLM(prompt, args):
    if args.sys_instruction == "":
        prompt = [{"role": "user", "content": prompt}]
    else:
        prompt = [
            {"role": "system", "content": args.sys_instruction},
            {"role": "user", "content": prompt},
        ]

    got_result = False
    while not got_result:
        try:
            results = client.chat.completions.create(
                model="glm-4-air-250414",
                messages=prompt,
                max_tokens=4096,
                temperature=0.0,
            )
            got_result = True
            print(results.choices[0].message.content)
        except Exception:  # pylint: disable=broad-exception-caught
            sleep(3)

    parsed_results = parse_results_chatGLM(results)  # type: ignore
    return parsed_results


"""
    从step1~（k-1）步骤中，将预测的label转换为数字，进行后续处理
"""


def predict_1To_k_minus_1_chatGLM(prompt, args):
    prompt = [{"role": "user", "content": prompt}]

    got_result = False
    while not got_result:
        try:
            results = client.chat.completions.create(
                model="glm-4-air-250414",
                messages=prompt,
                max_tokens=4096,
                temperature=0.3,
            )
            got_result = True
        except Exception:  # pylint: disable=broad-exception-caught
            sleep(3)

    return parse_textToNumber_chatGLM(results)


def predict_k_chatGLM(prompt, args):
    prompt = [{"role": "user", "content": prompt}]

    got_result = False
    while not got_result:
        try:
            results = client.chat.completions.create(
                model="glm-4-air-250414",
                messages=prompt,
                max_tokens=4096,
                temperature=0.0,
            )
            got_result = True
            print(results.choices[0].message.content)
        except Exception:  # pylint: disable=broad-exception-caught
            return []

    parsed_results = parse_results_chatGLM(results)
    return parsed_results


if __name__ == "__main__":
    # 原始输入文本
    prompt3 = """
    You are given a set of entities, a query, an event chain, and a relation. Please score the entities’ contribution to 
    the query on a scale from 0 to 1 (the sum of the scores of all entities is 1).

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

    Output format: 
    Example: 0.3,0.1,0.4,0.2
    Ensure the sum of all scores is 1.0.
    Only output the scores, no additional explanation.
    """

    prompt2 = """
You are given a set of relations, a query, and an event chain. Your task is to rate the relevance of each relation to answering the query based on the event chain.
Please provide a relevance score for each relation between 0~1
Higher scores indicate higher relevance to answering the query.

Available relations:
1. Mobilize_or_increase_armed_forces
2. Occupy_territory
3. Engage_in_diplomatic_cooperation
4. Demand_diplomatic_cooperation_(such_as_policy_support)
5. Make_an_appeal_or_request
6. Express_intent_to_cooperate_militarily
7. Use_conventional_military_force
8. Investigate
9. Praise_or_endorse
10. Express_intent_to_meet_or_negotiate
11. Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)
12. Make_a_visit
13. Cooperate_economically
14. Express_intent_to_cooperate
15. Sign_formal_agreement
16. Demonstrate_or_rally
17. Yield
18. Consult
19. Make_statement
20. Deny_responsibility
21. Meet_at_a_'third'_location
22. Defy_norms,_law
23. Impose_restrictions_on_political_freedoms
24. Engage_in_negotiation
25. Criticize_or_denounce
26. Demand
27. Express_intent_to_ease_administrative_sanctions
28. fight_with_small_arms_and_light_weapons

Current event chain:

Query:
8016: Thailand Express_intent_to_cooperate ?

Output format: A list of scores corresponding to each relation, separated by commas.
Example:
1:0.2
2:0.6
3:0.4
4:0.2
Only output the number and score, no additional explanation.
    """

    result = predict_1To_k_minus_1_chatGLM(prompt2, "")
    print(result)

    # # 遍历所有行，只处理以数字+点+空格开头的行
    # results = []
    # for line in to_return:
    #     line = line.strip()
    #     if line and '. ' in line:
    #         # 检查行是否以数字+点+空格开头
    #         parts = line.split('. ', 1)
    #         if len(parts) > 1 and parts[0].isdigit():
    #             results.append(parts[1])
    #
    # print(results)
