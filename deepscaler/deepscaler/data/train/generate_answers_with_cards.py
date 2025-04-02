import requests
import json,re
from concurrent.futures import ThreadPoolExecutor
import tqdm
import os

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution



def request_api(prompt,model_name,temperature=0.6,max_tokens=None,top_p=0.7,top_k=50,frequency_penalty=0,n=1):

    url = "https://api.siliconflow.cn/v1/chat/completions"

    headers = {
        "Authorization": "Bearer sk-mphnsxemiwmceqritbpybmdacqazoaztpfrvtqtqdavbnbdw",
        "Content-Type": "application/json"
    }

    messages = []

    payload = {
        "model": model_name, # Qwen/Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "n": n,
        "response_format": {"type": "text"},
    }


    response = requests.request("POST", url, json=payload, headers=headers)
    output = response.text
    output = json.loads(output)["choices"]
    return output


def single_thread_eval_generation(data, model_name):
    prompt = data.pop('gen_prompt')
    response_list = request_api(prompt, model_name, temperature=0.7, max_tokens=4096, n=10)

    data['Generation'] = []

    for response_dict in response_list:
        response = response_dict["message"]["content"]
        result = False

        if "\\boxed" in response:
            extract_answer = extract_boxed_answer(response).strip()
            if(extract_answer==data["answer"]):
                result = True
        else:
            extract_answer = None

        generation = {
            "response": response,
            "extracted_answer": extract_answer,
            "result": result
        }

        data['Generation'].append(generation)


def multi_thread_response_generation(data_list, model_name, max_workers=5, type='eval'):
    if type == 'eval':
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(
                tqdm.tqdm(
                    executor.map(lambda x: single_thread_eval_generation(x, model_name,), data_list),
                    total=len(data_list)
                )
            )

def get_eval_outputs(all_problems, output_file_path, model_name, max_workers=5):
    data_list = []

    with open("prompt_ans.txt", "r") as f:
        prompt_template = f.read()
    
    for data in all_problems:
        problem = data["problem"]
        prompt = prompt_template.replace("<problem>", problem)
        gen_prompt = prompt
        data['gen_prompt'] = gen_prompt
        data_list.append(data)
    multi_thread_response_generation(data_list, model_name, max_workers, type='eval')

    isFile = os.path.isfile(output_file_path)

    if(isFile):
        with open(output_file_path, 'r',encoding='utf-8') as file:
            existing_problems = json.load(file)
        existing_problems.extend(all_problems)
        with open(output_file_path, 'w',encoding='utf-8') as file:
            json.dump(existing_problems, file, indent=2, ensure_ascii=False)
    else:
        with open(output_file_path, 'w',encoding='utf-8') as file:
            json.dump(all_problems, file, indent=2, ensure_ascii=False)

def main():


    # 读取MR-GSM8K.json文件
    with open(f'aime_6.json',encoding='utf-8') as json_file:
        all_problems = json.load(json_file)[140:150]

    eval_output_path = "7b_qwen_generated_answers_structured_full.json"

    eval_model = "Qwen/Qwen2.5-7B-Instruct" # Qwen/Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    eval_responses = get_eval_outputs(all_problems, eval_output_path, eval_model, max_workers=5)

    return


if __name__ == "__main__":
    main()