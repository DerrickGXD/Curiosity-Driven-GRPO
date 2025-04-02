import json

with open("aime_6_correct_num.json", "r") as f:
    data = json.load(f)

print(len(data))

correct_dict = {}
for i in range(9):
    correct_dict[i] = []

with open("aime_6.json", "r") as f:
    actual_data = json.load(f)


for d in actual_data:
    if(d["problem"] in data):
        num_correct = data[d["problem"]]["correct"]
        correct_dict[num_correct].append(d)

for i in correct_dict:
    print(i, len(correct_dict[i]))


for i in range(9):
    with open(f"aime_6_correct_{i}.json", "w") as f:
        json.dump(correct_dict[i], f, ensure_ascii=False, indent=4)
