import json
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

class RewardCalculation():

    def extract_data(self, response):
        pattern = {
            "Reasoning Pattern": r"Reasoning Pattern\s*:\s*(.*?)Reasoning Pattern Description",
            "Reasoning Pattern Description": r"Reasoning Pattern Description\s*:\s*(.*?)Calculation",
            "Answer": r"\\boxed\{(.*?)\}"
        }

        extracted_data = {}
        for key, regex in pattern.items():
            matches = re.findall(regex, response, re.DOTALL)
            extracted_data[key] = matches[-1].strip()

        # Extract Calculation based on presence of Answer
        calculation_match = re.search(r"Calculation\s*:\s*(.*)", response, re.DOTALL)
        if calculation_match:
            calculation_text = calculation_match.group(1).strip()
            # If "Answer:" exists, extract up to it
            if "Answer:" in calculation_text:
                calculation_text = calculation_text.split("Answer:")[0].strip()
            extracted_data["Calculation"] = calculation_text
        else:
            raise RuntimeError("Calculation not found")

        return extracted_data


    def __call__(self, response, epoch=None):

        format_correct = True
        try:
            extracted_data = self.extract_data(response)
        except:
            format_correct = False

        if(format_correct):
            section_rewards = {
                "Reasoning Pattern": -0.5,
                "Reasoning Pattern Description": -0.6,
                "Calculation": -0.7,
                "Answer": -1.0
            }

            tokens = word_tokenize(response)
            reward_tokens = [0] * len(tokens)

            section_type = ["Reasoning Pattern", "Reasoning Pattern Description", "Calculation"]
            last_position = 0
            
            for key in section_type:

                section_text = extracted_data[key]
                section_tokens = word_tokenize(section_text)
                section_len = len(section_tokens)
                
                # Search for the section tokens in the whole text, starting from the last position
                for i in range(last_position, len(tokens) - section_len + 1):
                    if tokens[i:i + section_len] == section_tokens:
                        # Found the section, now insert the reward after the last token of the section
                        reward = section_rewards.get(key, 0)
                        reward_tokens[i + section_len - 1] = reward
                        last_position = i + section_len  # Update position for next section
                        break

            reward_tokens[-1] = section_rewards["Answer"]
        else:
            reward_tokens[-1] = -1.0

        print(reward_tokens)

                


reward_calculation = RewardCalculation()


with open("7b_qwen_generated_answers_structured_full.json", "r") as f:
    data = json.load(f)

outputs = data

has_answer = 0

for out in outputs:
    generation = out["Generation"]
    for g in generation:
        response = g["response"]
        has_answer += reward_calculation(response)
