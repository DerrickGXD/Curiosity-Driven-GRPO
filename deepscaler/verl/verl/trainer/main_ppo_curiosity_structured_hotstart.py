# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

# curiosity class
from curiosity_redteam.self_bleu import SelfBleuReward
from curiosity_redteam.sentence_embed import CosineSentenceEmbeddingReward
from deepscaler.rewards.math_reward import deepscaler_reward_fn

import os
import re
import json

def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        return deepscaler_reward_fn


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config=None, output_filename=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config = config
        self.first_evaluation = {}
        self.existing_reasoning_pattern = {}

        self.epoch = 0

        # if(num_examine==0):
        #     self.output_folder = "/data/projects/13003098/derrick/Curiosity-Driven-GRPO/deepscaler/scripts/train/train_output"
        #     self.output_filename = f"{self.output_folder}/{output_filename}.json"
        # else:
        #     self.output_folder = "/data/projects/13003098/derrick/Curiosity-Driven-GRPO/deepscaler/scripts/train/validation_output"
        #     self.output_filename = f"{self.output_folder}/{output_filename}.json"

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        print("Output filename", self.output_filename)

        with open(self.output_filename, "w") as f:
            json.dump({}, f)

        self.bleu_tokenizer = lambda x: self.tokenizer.batch_decode(self.tokenizer(x, return_tensors="pt")["input_ids"][0].unsqueeze(1))
        if config.curiosity.reasoning_pattern_description_penalty_type == "bleu":
            self.reasoning_pattern_description_reward_module = SelfBleuReward(
                grams=config.curiosity.bleu_reward_grams,
                sample_size=config.curiosity.bleu_n_samples,
                tokenizer=self.bleu_tokenizer,
                is_reasoning_pattern=True
            )
        elif config.curiosity.reasoning_pattern_description_penalty_type == "cossimemb":
            self.reasoning_pattern_description_reward_module = CosineSentenceEmbeddingReward(
                n_samples=config.curiosity.cossimemb_n_samples,
                impl=config.curiosity.cossimemb_impl,
                device="cpu",
                is_reasoning_pattern=True
            )
        else:
            raise NotImplementedError(f"Reasoning Pattern Description Penalty not found : {config.curiosity.reasoning_pattern_description_penalty_type}")

        if config.curiosity.calculation_penalty_type == "bleu":
            self.calculation_reward_module = SelfBleuReward(
                grams=config.curiosity.bleu_reward_grams,
                sample_size=config.curiosity.bleu_n_samples,
                tokenizer=self.bleu_tokenizer
            )
        elif config.curiosity.reasoning_pattern_description_penalty_type == "cossimemb":
            self.calculation_reward_module = CosineSentenceEmbeddingReward(
                n_samples=config.curiosity.cossimemb_n_samples,
                impl=config.curiosity.cossimemb_impl,
                device="cpu"
            )
        else:
            raise NotImplementedError(f"Calculation Penalty not found : {config.curiosity.calculation_penalty_type}")

    def extract_data(self, response):
        pattern = {
            "Reasoning Pattern": r"Reasoning Pattern\s*:\s*(.*?)\n\n",
            "Reasoning Pattern Description": r"Reasoning Pattern Description\s*:\s*(.*?)\n\n",
            "Answer": r"\\boxed\{(.*?)\}"
        }

        extracted_data = {}
        for key, regex in pattern.items():
            matches = re.findall(regex, response, re.DOTALL)
            if(matches):
                extracted_data[key] = matches[-1].strip()
            else:
                extracted_data[key] = None

        # Extract Calculation based on presence of Answer
        calculation_match = re.search(r"Calculation\s*:\s*(.*)", response, re.DOTALL)
        if calculation_match:
            calculation_text = calculation_match.group(1).strip()
            # If "Answer:" exists, extract up to it
            if "Answer:" in calculation_text:
                calculation_text = calculation_text.split("Answer:")[0].strip()
            extracted_data["Calculation"] = calculation_text
        else:
            extracted_data["Calculation"] = None

        return extracted_data

    def validate_and_preprocess_chain(self, s):
        pattern = r'^\s*(SA|OST|CoT|DC|SRR)(\s*->\s*(SA|OST|CoT|DC|SRR))*\s*$'
        
        if not re.fullmatch(pattern, s):
            return None  # Return None if the string is invalid
        
        # Preprocess: Remove unnecessary spaces around '->' and trim the string
        cleaned = re.sub(r'\s*->\s*', '->', s.strip())
        return cleaned


    def __call__(self, data: DataProto, epoch=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        rm_tensor = torch.zeros(len(data), dtype=torch.float32)
        format_correct_tensor = torch.zeros(len(data), dtype=torch.float32)
        reasoning_pattern_reward_tensor = torch.zeros(len(data), dtype=torch.float32)
        reasoning_pattern_description_reward_tensor = torch.zeros(len(data), dtype=torch.float32)
        calculation_reward_tensor = torch.zeros(len(data), dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()

        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # record prompt and response

            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
                
            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            # compute_score_fn = _select_rm_score_fn(data_source)
            # rm_score, extract_answer = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)


            """
            Compute Self-BLEU rewards as diveristy penalty
                1. Compute Self-BLEU score for each generated response
                2. Update the references in Self-BLEU score 
            """

            extracted_data = self.extract_data(response_str)

            section_penalty = {}

            if extracted_data["Reasoning Pattern"]!=None:
                reasoning_pattern = self.validate_and_preprocess_chain(extracted_data["Reasoning Pattern"])
                # no valid reasoning pattern, reasoning pattern description will be invalid, so no reward for both reasoning pattern and reasoning pattern
                if(reasoning_pattern==None):
                    extracted_data["Reasoning Pattern Description"] = None
                    section_penalty["Reasoning Pattern"] = 1.0
                else:
                    # if there is a valid reasoning pattern, reward it with 1.0. encourage model to produce valid reasoning pattern.
                    section_penalty["Reasoning Pattern"] = 0.0
            else:
                extracted_data["Reasoning Pattern Description"] = None
                section_penalty["Reasoning Pattern"] = 1.0
                reasoning_pattern = None


            reward_tokens = torch.zeros(valid_response_length)
            reasoning_pattern_exists = False

            if(reasoning_pattern!=None):
                if(prompt_str in self.existing_reasoning_pattern):
                    if reasoning_pattern in self.existing_reasoning_pattern[prompt_str]:
                        reasoning_pattern_exists = True
                else:
                    self.existing_reasoning_pattern[prompt_str] = set()

            if(extracted_data["Reasoning Pattern Description"]!=None):
                if(reasoning_pattern_exists):
                    #if reasoning pattern exist, then we have to impose a penalty. If reasoning pattern does not exist, maximum reward for reasoning pattern description  
                    section_penalty["Reasoning Pattern Description"] = self.reasoning_pattern_description_reward_module(prompt_str, [extracted_data["Reasoning Pattern Description"]], reasoning_pattern)
                else:
                    section_penalty["Reasoning Pattern Description"] = 0.0
            else:
                section_penalty["Reasoning Pattern Description"] = 1.0

            if(extracted_data["Calculation"]!=None):
                section_penalty["Calculation"] = self.calculation_reward_module(prompt_str, [extracted_data["Calculation"]])
            else:
                section_penalty["Calculation"] = 1.0

            section_type = ["Reasoning Pattern", "Reasoning Pattern Description", "Calculation"]
            last_position = 0

            if(extracted_data["Answer"]!=None):
                if(extracted_data["Answer"]==ground_truth):
                    rm_score = 1.0
                else:
                    rm_score = 0.0
            else:
                rm_score = 0.0

            for key in section_type:

                if(extracted_data[key]!=None):
                    # only if the extracted data is not None, we can reward the model
                    section_text = extracted_data[key]
                    section_tokens = self.tokenizer.encode(section_text)
                    section_len = len(section_tokens)
                
                    # Search for the section tokens in the whole text, starting from the last position
                    for j in range(last_position, len(valid_response_ids) - section_len + 1):
                        if valid_response_ids[j:j + section_len] == section_tokens:
                            # Found the section, now insert the reward after the last token of the section

                            penalty = section_penalty.get(key, 0)

                            if(key=="Reasoning Pattern"):
                                coef = self.config.curiosity.reasoning_pattern_reward_coef
                            elif(key=="Reasoning Pattern Description"):
                                coef = self.config.curiosity.reasoning_pattern_description_reward_coef
                            else:
                                coef = self.config.curiosity.calculation_reward_coef

                            reward_tokens[j + section_len - 1] = (1 - penalty) * coef
                            last_position = j + section_len  # Update position for next section
                            break


            reward_tokens[-1] += rm_score * self.config.curiosity.answer_reward_coef

            reasoning_pattern_reward = 1 - section_penalty["Reasoning Pattern"]
            reasoning_pattern_description_reward = 1 - section_penalty["Reasoning Pattern Description"]
            calculation_reward = 1 - section_penalty["Calculation"]
            extract_answer = extracted_data["Answer"]

            if(extracted_data["Reasoning Pattern"]!=None and extracted_data["Reasoning Pattern Description"]!=None and extracted_data["Calculation"]!=None and extracted_data["Answer"]!=None):
                format_correct = True
            else:
                format_correct = False
            
            score_dict = {
                            "rm_score": rm_score, 
                            "format_correct": format_correct, 
                            "reasoning_pattern_reward": reasoning_pattern_reward,
                            "reasoning_pattern_description_reward" : reasoning_pattern_description_reward, 
                            "calculation_reward": calculation_reward          
                        }            

            return i, reward_tokens, valid_response_length, score_dict, prompt_str, response_str, extract_answer, extracted_data, reasoning_pattern

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        data_list = []
        # Fill reward tensor with results
        for i, reward_tokens, valid_response_length, score_dict, prompt_str, response_str, extract_answer, extracted_data, reasoning_pattern in results:
            reward_tensor[i, :valid_response_length] = reward_tokens
            rm_tensor[i] = score_dict["rm_score"]
            format_correct_tensor[i] = score_dict["format_correct"]
            reasoning_pattern_reward_tensor[i] = score_dict["reasoning_pattern_reward"]
            reasoning_pattern_description_reward_tensor[i] = score_dict["reasoning_pattern_description_reward"]
            calculation_reward_tensor[i] = score_dict["calculation_reward"]


            if self.num_examine == 0 and extracted_data!=None:
                if reasoning_pattern != None and extracted_data["Reasoning Pattern Description"] != None:
                    self.existing_reasoning_pattern[prompt_str].add(reasoning_pattern)
                    self.reasoning_pattern_description_reward_module.append_reference(prompt_str, extracted_data["Reasoning Pattern Description"], reasoning_pattern=reasoning_pattern)
                
                if extracted_data["Calculation"] != None:
                    self.calculation_reward_module.append_reference(prompt_str, extracted_data["Calculation"])

            extracted_data["Reasoning Pattern"] = reasoning_pattern

            
            new_data = {"prompt": prompt_str, "response": response_str, "extract_answer": extract_answer, "score": score_dict, "extracted_data": extracted_data}
            data_list.append(new_data)

        
        # with open(self.output_filename, "r") as f:
        #     data_json = json.load(f)
        
        # data_json[self.epoch] = data_list

        # with open(self.output_filename, "w") as f:
        #     json.dump(data_json, f, indent=4, ensure_ascii=False)

        self.epoch += 1

        return reward_tensor, rm_tensor, format_correct_tensor, reasoning_pattern_reward_tensor, reasoning_pattern_description_reward_tensor, calculation_reward_tensor


import ray
import hydra
import json

@hydra.main(config_path='config', config_name='ppo_trainer_curiosity_structured', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    print(ray.cluster_resources())
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id


    # Note that we always use function-based RM for validation

    filename_parameters = f"qwen7b_bs_{config.data.train_batch_size}_rollout_{config.actor_rollout_ref.rollout.n}_reasoningpattern_{config.reward_model.curiosity.reasoning_pattern_reward_coef}_reasoningpatterndescription_{config.reward_model.curiosity.reasoning_pattern_description_penalty_type}_{config.reward_model.curiosity.reasoning_pattern_description_reward_coef}_calculation_{config.reward_model.curiosity.calculation_penalty_type}_{config.reward_model.curiosity.calculation_reward_coef}_answer_{config.reward_model.curiosity.answer_reward_coef}"

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, config=config.reward_model, output_filename=filename_parameters)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, config=config.reward_model, output_filename=filename_parameters)


    # hot start
    with open("/home/happywwy/Curiosity-Driven-GRPO/deepscaler/scripts/data/hot_start", "r") as f:
        hot_start_data = json.load(f)

    hot_start_data = hot_start_data["0"]

    for hot_start in hot_start_data:
        prompt_str = hot_start["prompt"]
        extracted_data = hot_start["extracted_data"]

        if(prompt_str not in reward_fn.existing_reasoning_pattern):
            reward_fn.existing_reasoning_pattern[prompt_str] = set()
        
        if(extracted_data["Reasoning Pattern"]!=None and extracted_data["Reasoning Pattern Description"]!=None)
            reward_fn.reasoning_pattern_description_reward_module.append_reference(prompt_str, extracted_data["Reasoning Pattern Description"], reasoning_pattern=extracted_data["Reasoning Pattern"])
            reward_fn.existing_reasoning_pattern[prompt_str].add(extracted_data["Reasoning Pattern"])

    # end of hot start


    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
                            
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
