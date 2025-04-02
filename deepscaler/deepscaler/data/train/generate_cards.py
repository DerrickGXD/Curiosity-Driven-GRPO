import requests
import json,re
from concurrent.futures import ThreadPoolExecutor
import tqdm


def request_api(prompt,model_name,temperature=0.0,max_tokens=None,top_p=1,top_k=1,frequency_penalty=0,n=1):
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
        "stop": ["null"],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "frequency_penalty": frequency_penalty,
        "n": 1,
        "response_format": {"type": "text"},
    }


    response = requests.request("POST", url, json=payload, headers=headers)
    output = response.text
    output = json.loads(output)["choices"][0]["message"]["content"]
    return output


def single_thread_eval_generation(data, model_name):
    prompt = data.pop('gen_prompt')
    response = request_api(prompt, model_name)

    data[f'Generation'] = {
        "response": response
    }


def multi_thread_response_generation(data_list, model_name, max_workers=5, type='eval'):
    if type == 'eval':
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(
                tqdm.tqdm(
                    executor.map(lambda x: single_thread_eval_generation(x, model_name,), data_list),
                    total=len(data_list)
                )
            )

def get_eval_outputs(all_problems, output_file_path,model_name, prompt_path, max_workers=5):
    data_list = []

    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    
    for data in all_problems:
        gen_prompt = prompt_template.replace("<question>", data['problem'])
        # gen_prompt = gen_prompt.replace("<solution>", str(data['solution']))
        data['gen_prompt'] = gen_prompt
        data_list.append(data)
    multi_thread_response_generation(data_list, model_name, max_workers, type='eval')
    with open(output_file_path, 'w',encoding='utf-8') as file:
        json.dump(all_problems, file, indent=2, ensure_ascii=False)

def main():


    # 读取MR-GSM8K.json文件
    with open(f'aime.json',encoding='utf-8') as json_file:
        all_problems = json.load(json_file)

    # 读取前100个含有ground truth的问题
    all_problems = all_problems[:5]

    eval_output_path = "7b_generated_cards_wo_answers_aime.json"

    prompt_path = "prompt_gen_card_without_ans.txt"

    eval_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # Qwen/Qwen2.5-7B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    eval_responses = get_eval_outputs(all_problems, eval_output_path, eval_model, prompt_path, max_workers=5)

    return


if __name__ == "__main__":
    main()