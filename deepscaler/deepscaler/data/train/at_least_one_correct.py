import json


file_name = "7b_qwen_generated_answers_structured_full.json"
with open(file_name, "r") as f:
    data = json.load(f)


for d in data:

    correct_sample = False
    for i in d["Generation"]:
        if(i["result"]):
            correct_sample = True
            break
            
    d["at_least_one_correct"] = correct_sample

with open(file_name, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
