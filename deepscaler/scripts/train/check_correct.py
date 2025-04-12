import json
from collections import defaultdict
import re

train_path = "/data/projects/13003098/derrick/Curiosity-Driven-GRPO/deepscaler/scripts/train/train_output/qwen7b_bs_64_rollout_4_reasoningpattern_0.5_reasoningpatterndescription_cossimemb_1.0_calculation_cossimemb_1.0_answer_5.0.json"

dataset_path = "/data/projects/13003098/derrick/Curiosity-Driven-GRPO/deepscaler/deepscaler/data/train/aime_6_correct_0.json"

with open(dataset_path, "r") as f:
    dataset = json.load(f)

with open(train_path, "r") as f:
    train_data = json.load(f)

dataset_str_to_id = {}

data_id_to_correct = {}
data_id_to_bleu = {}
data_id_to_response = {}
data_id_to_answer = {}

specific_id = 1

until_epoch = 11

for i, data in enumerate(dataset):
    dataset_str_to_id[data["problem"]] = i
    data_id_to_correct[i] = []
    data_id_to_bleu[i] = []
    data_id_to_response[i] = []
    data_id_to_answer[i] = []

for key in train_data:
    if(int(key)>until_epoch):
        break

    epoch_id_to_correct = defaultdict(int)
    epoch_id_to_bleu = defaultdict(list)
    epoch_id_to_response = defaultdict(list)
    epoch_id_to_answer = defaultdict(list)


    for data in train_data[key]:
        # clean_prompt = data["prompt"].replace("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n", "")
        # clean_prompt = clean_prompt.split('Let\'s think step by step')[0].strip()

        match = re.search(r'Problem\s*:\s*(.*)', data["prompt"], re.DOTALL)
        clean_prompt = match.group(1).strip()
        clean_prompt = clean_prompt.split('<|im_end|>')[0].strip()

        data_id = dataset_str_to_id[clean_prompt]

        response = data["response"]

        if(data["score"]["rm_score"]):
            epoch_id_to_correct[data_id] += 1
        else:
            epoch_id_to_correct[data_id] += 0

        # epoch_id_to_bleu[data_id].append(data["score"]["bleu_score"])
        # epoch_id_to_response[data_id].append(data["response"])
        # epoch_id_to_answer[data_id].append(data["extract_answer"])

    for idx in epoch_id_to_correct:
        data_id_to_correct[idx].append(epoch_id_to_correct[idx])

    for idx in epoch_id_to_bleu:
        data_id_to_bleu[idx].append(epoch_id_to_bleu[idx])

    for idx in epoch_id_to_response:
        data_id_to_response[idx].append(epoch_id_to_response[idx])

    for idx in epoch_id_to_response:
        data_id_to_answer[idx].append(epoch_id_to_answer[idx])

# print(data_id_to_bleu)
# for idx in data_id_to_bleu:
#     data_id_to_bleu[idx] = [sum(x) for x in data_id_to_bleu[idx]]
# print(data_id_to_bleu)

# print(len(data_id_to_correct))

questions_with_one_correct_answer = 0

for x in data_id_to_correct:
    if(sum(data_id_to_correct[x])>0):
        questions_with_one_correct_answer += 1

print(f"{questions_with_one_correct_answer} QUESTIONS WITH ONE EXPLORED CORRECT ANSWER")


# epoch_taken_to_get_correct_answer = []

# def first_non_zero_index(lst):
#     for i, val in enumerate(lst):
#         if val != 0:
#             return i + 1
#     return -1

# first_non_zero_dict = {}

# for x in data_id_to_correct:
#     success_epoch = first_non_zero_index(data_id_to_correct[x])
#     if(success_epoch>0):
#         first_non_zero_dict[x] = success_epoch

# print(first_non_zero_dict)
    