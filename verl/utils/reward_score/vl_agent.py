from openai import OpenAI
import requests
import random
import re
import os
import datetime
import json

from math_verify import parse, verify

openai_api_key = "EMPTY"
openai_api_base_list = [
    # "http://172.30.52.123:8000/v1",
    # "http://10.39.3.123:18901/v1",
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.39.3.123:18901/v1"),
]

client_list = []
model_name_list = []
for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
    try:
        response = requests.get(f"{api_base}/models", timeout=10)
        response.raise_for_status()
        models = response.json()
        model_name_list.append(models['data'][0]['id'])
    except Exception as e:
        print(f"Warning: Failed to fetch models from {api_base}: {e}. Falling back to default model name.")
        # Default fallback for judge model if API is unreachable
        model_name_list.append("judge")



def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}
 
## Student Final Answer
{pred_ans}"""
 
 
def log_judge_interaction(judge_type, question, gold_ans, pred_ans, judge_response, score, extra_info=None):
    log_dir = "/arf/scratch/hvural/DeepEyes/logs/vl_agent"
    try:
        os.makedirs(log_dir, exist_ok=True)
        # Use PID and timestamp for unique files per worker
        log_file = os.path.join(log_dir, f"judge_log_{os.getpid()}.jsonl")
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "judge_type": judge_type,
            "question": question,
            "gold_ans": gold_ans,
            "pred_ans": pred_ans,
            "judge_response": judge_response,
            "score": score
        }
        if extra_info:
            entry.update({"extra_info": extra_info})
        
        def numpy_default(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            return str(obj)

        with open(log_file, "a") as f:
            f.write(json.dumps(entry, default=numpy_default) + "\n")
    except Exception as e:
        print(f" [ERROR] Failed to log judge interaction: {e}")


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_choice_letter(text):
    """
    Extracts the choice letter (A-E) from text.
    Handles formats like 'A)', 'A.', 'Choice: A', etc.
    """
    if not text:
        return ""
    # Look for a standalone letter A-E, or A-E followed by ) or . or : at the beginning or after a word boundary
    # This tries to pick the most likely choice letter.
    match = re.search(r'\b([A-E])\s*[\)\.\:]|(?:\b|^)([A-E])(?:\b|$)', text, re.IGNORECASE)
    if match:
        letter = match.group(1) or match.group(2)
        return letter.upper()
    return text


def call_judge(prompt):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    
    # Check for Equivalence Judgement (Markdown Report style)
    if '## Equivalence Judgement' in response:
        judgement_part = response.split('## Equivalence Judgement')[-1].lower()
        if 'true' in judgement_part and 'false' not in judgement_part:
            return 1.0, response
        elif 'false' in judgement_part and 'true' not in judgement_part:
            return 0.0, response

    if 'Judgement:' in response:
        judgement_part = response.split('Judgement:')[-1].strip()
        if '1' in judgement_part:
            return 1.0, response
        elif '0' in judgement_part:
            return 0.0, response
    
    if response == '1':
        return 1.0, response
    elif response == '0':
        return 0.0, response
    
    print(f' [WARNING] resp format error {response=}')
    return 0.0, response


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    
    # 1. Format checking
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_1 == 0:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0:
        is_format_error = True

    # 2. Component extraction
    reason_text = ""
    if count_think_1 > 0:
        reason_text = predict_str.split("<think>")[-1].split("</think>")[0].strip()
    
    answer_text = ""
    if count_answer_1 > 0:
        raw_answer_text = predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
        answer_text = extract_choice_letter(raw_answer_text)

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        is_format_error = True

    # 3. Judge Auxiliary Reasoning (Extra Info)
    aux_reward = 0.0
    reasoning_judge_resp = "N/A"
    if extra_info and 'question' in extra_info and 'answer' in extra_info:
        aux_question = extra_info['question']
        aux_gold = extra_info['answer']
        # Use more robust prompt for reasoning
        aux_prompt = COMMON_VERIFY_PROMPT.format(
            query=aux_question,
            gold_ans=aux_gold,
            pred_ans=reason_text,
        )
        aux_reward, reasoning_judge_resp = call_judge(aux_prompt)
    
    # 4. Judge Main Answer (ABCDE Choice)
    main_reward = 0.0
    main_judge_resp = "N/A"
    if ground_truth:
        # We use a simple prompt for the main answer consistency check
        # Or we can reuse get_prompt if the judge is good at short answers
        main_prompt = get_prompt(answer_text, ground_truth, "Select the correct option from ABCDE")
        main_reward, main_judge_resp = call_judge(main_prompt)

    # 5. Combined Reward
    format_reward = -1.0 if is_format_error else 0.0
    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    # Tool reward is given if vision tags are used AND at least one part is correct
    tool_reward = 1.0 if count_vision_1 > 0 and (aux_reward > 0.5 or main_reward > 0.5) else 0.0
    
    # Split accuracy reward: 0.4 for reasoning, 0.4 for final answer (total 0.8 as before)
    # Plus format and tool rewards
    score = (0.4 * aux_reward) + (0.4 * main_reward) + (0.2 * format_reward) + (1.2 * tool_reward)
    
    # Log the interaction
    log_judge_interaction(
        judge_type="dual_compute_score",
        question=extra_info.get('question', 'N/A') if extra_info else "N/A",
        gold_ans=f"Main: {ground_truth} | Aux: {extra_info.get('answer', 'N/A') if extra_info else 'N/A'}",
        pred_ans=f"Main: {answer_text} | Reasoning: {reason_text[:100]}...",
        judge_response=f"Reasoning: {reasoning_judge_resp} | Main: {main_judge_resp}",
        score=score,
        extra_info=extra_info
    )
    
    return score

    # reward 2 
    # return 1.0 * acc_reward + 0.2 * format_reward + 1.0 * tool_reward + 0.2 * tool_reward_base
    # reward 3
    # tool_reward_alpha = 1.2 if count_vision_1 > 0 else 0.0
    # return 1.0 * acc_reward * tool_reward_alpha + 0.2 * format_reward
    # reward 4
    # extra_reward = tool_reward_base * (count_vision_1 - 1) * (1 - acc_reward)
    # return  0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base  + 0.2 * extra_reward



def compute_common_reasoning(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = extract_answer(predict_no_think) # predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    if not answer_text:
        acc_reward = 0.0
        is_format_error = True
    elif len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True
    else:
        question_text = extra_info['question']
        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]
        full_prompt = COMMON_VERIFY_PROMPT.format(
            query=question_text,
            gold_ans=ground_truth,
            pred_ans=answer_text,
        )

        acc_reward = 0.0
        for ix in range(8):
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.5,
            )
            response = chat_response.choices[0].message.content.strip()
            judgement = response.split('## Equivalence Judgement')[-1].lower()
            if 'true' in judgement and 'false' not in judgement:
                acc_reward = 1.0
                break
            elif 'false' in judgement and 'true' not in judgement:
                acc_reward = 0.0
                break
            else:
                print(f' [ERROR] judgement format invalid: {judgement}')
                continue

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {answer_text=}, {acc_reward=}, {format_reward=}')
    
    score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    
    # Log the interaction
    log_judge_interaction(
        judge_type="compute_common_reasoning",
        question=extra_info['question'],
        gold_ans=ground_truth,
        pred_ans=answer_text,
        judge_response=response if 'response' in locals() else "N/A (No answer extracted)",
        score=score,
        extra_info=extra_info
    )
    
    return score


def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        print(f' [ERROR math] verify bug output: ')


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False
    # predict_str = "<think>" + predict_str
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split('</think>')[-1].strip()
    answer_pattern = r'\\boxed{([^}]+)}'
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(extra_info['question'], ground_truth, model_answer) else 0.0
    
    format_reward = -1.0 if is_format_error else 0.0
    print(f' [DEBUG] query={extra_info["question"]}, {ground_truth=}, {model_answer=}, {acc_reward=}, {format_reward=}')
    
    score = 1.2 * acc_reward + 0.4 * format_reward
    
    # Log the interaction
    log_judge_interaction(
        judge_type="compute_score_math",
        question=extra_info['question'],
        gold_ans=ground_truth,
        pred_ans=model_answer,
        judge_response= "N/A (Math Rule/Generative)",
        score=score,
        extra_info=extra_info
    )
    
    return score


if __name__ == '__main__':
    predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {'answer': 'The woman is to the left of the man who is holding the camera.', 'id': 0, 'image': '/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg', 'pred_ans': 'The woman is to the right of the man who is holding the camera.', 'question': 'Is the woman to the left or to the right of the man who is holding the camera?'}

    score = compute_score(predict_str, ground_truth, extra_info)
    print(f"Score: {score}")