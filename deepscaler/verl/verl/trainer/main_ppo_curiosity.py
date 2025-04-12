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
from curiosity_redteam.clean_reward import GiberishPenalty

from deepscaler.rewards.math_reward import deepscaler_reward_fn

import os



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

        self.epoch = 0

        if(num_examine==0):
            self.output_folder = "train_output"
            self.output_filename = f"{self.output_folder}/{output_filename}_compareanswer_{self.config.curiosity.compare_answer}_ignorerm_{self.config.curiosity.ignore_rm_score}_hybridreward_{self.config.curiosity.hybrid}" + ".json"
        else:
            self.output_folder = "validation_output"
            self.output_filename = f"{self.output_folder}/deepseek_7b_performance.json"

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)


        with open(self.output_filename, "w") as f:
            json.dump({}, f)


        if(self.num_examine==0):
            if config.curiosity.bleu_reward_coef != 0:
                self.bleu_tokenizer = bleu_tokenizer = lambda x: self.tokenizer.batch_decode(self.tokenizer(x, return_tensors="pt")["input_ids"][0].unsqueeze(1))
                self.bleu_reward_module = SelfBleuReward(
                    grams=config.curiosity.bleu_reward_grams,
                    sample_size=config.curiosity.bleu_n_samples,
                    tokenizer=bleu_tokenizer,
                )

            if config.curiosity.cossimemb_reward_coef != 0:
                self.cossimemb_reward_module = CosineSentenceEmbeddingReward(
                    n_samples=config.curiosity.cossimemb_n_samples,
                    impl=config.curiosity.cossimemb_impl,
                    device="cpu"
                )

            if config.curiosity.giberish_penalty_coef != 0:
                self.giberish_penalty_module = GiberishPenalty("cpu")

    def _aggregate_traj_reward(self, rm_score, bleu_score, cossimemb_score, giberish_score):
        bleu = self.config.curiosity.bleu_reward_coef * bleu_score
        cossimemb = self.config.curiosity.cossimemb_reward_coef * cossimemb_score
        giberish = self.config.curiosity.giberish_penalty_coef * giberish_score
        # return rm_score + bleu + cossimemb + giberish
        return rm_score



    def __call__(self, data: DataProto, epoch=None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        rm_tensor = torch.zeros(len(data), dtype=torch.float32)
        bleu_tensor = torch.zeros(len(data), dtype=torch.float32)
        cossimemb_tensor = torch.zeros(len(data), dtype=torch.float32)
        giberish_tensor = torch.zeros(len(data), dtype=torch.float32)

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
            compute_score_fn = _select_rm_score_fn(data_source)
            rm_score, extract_answer = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

            """
            Compute Self-BLEU rewards as diveristy penalty
                1. Compute Self-BLEU score for each generated response
                2. Update the references in Self-BLEU score 
            """


            if self.config.curiosity.bleu_reward_coef == 0 or self.num_examine==1:
                bleu_score = 0.
            else:
                if(self.config.curiosity.hybrid and rm_score):
                    # if hybrid activates, and response is correct, then can ignore noise so reward is maximum
                    bleu_score = 0.
                else:
                    if self.config.curiosity.bleu_reward_include_prompts:
                        bleu_score = self.bleu_reward_module(prompt_str, [sequences_str])
                    else:
                        if self.config.curiosity.compare_answer:
                            if(extract_answer==None):
                                str_to_compare = "None"
                            else:
                                str_to_compare = extract_answer
                        else:     
                            str_to_compare = response_str

                        bleu_score = self.bleu_reward_module(prompt_str, [str_to_compare])

            """
            Compute SimEmd rewards as diversity penalty
            """
            if self.config.curiosity.cossimemb_reward_coef == 0 or self.num_examine==1:
                cossimemb_score = 0.
            else:
                if(self.config.curiosity.hybrid and rm_score):
                    # if hybrid activates, and response is correct, then can ignore noise so reward is maximum
                    cossimemb_score = 0.
                else:
                    if self.config.curiosity.cossimemb_reward_include_prompts:
                        cossimemb_score = self.cossimemb_reward_module(prompt_str, [sequences_str])
                    else:
                        cossimemb_score = self.cossimemb_reward_module(prompt_str, [response_str])
            
            # Implementing textual similarity and target embedding similarity might not make sense here... skip

            """
            Compute gibberish penalty
            """                
            if self.config.curiosity.giberish_penalty_coef == 0 or self.num_examine==1:
                giberish_score = 0.
            else:
                giberish_score = self.giberish_penalty_module([response_str])
                    

            if self.config.curiosity.ignore_rm_score:
                score = self._aggregate_traj_reward(0, bleu_score, cossimemb_score, giberish_score)
            else:
                score = self._aggregate_traj_reward(rm_score * 2, bleu_score, cossimemb_score, giberish_score)


            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)

            score_dict = {"rm_score": rm_score, "bleu_score": bleu_score, "cossimemb_score": cossimemb_score, "giberish_score": giberish_score}            

            return i, score, valid_response_length, score_dict, prompt_str, sequences_str, response_str, extract_answer

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        data_list = []
        # Fill reward tensor with results
        for i, score, valid_response_length, score_dict, prompt_str, sequences_str, response_str, extract_answer in results:
            reward_tensor[i, valid_response_length - 1] = score
            rm_tensor[i] = score_dict["rm_score"]
            bleu_tensor[i] = score_dict["bleu_score"]
            cossimemb_tensor[i] = score_dict["cossimemb_score"]
            giberish_tensor[i] = score_dict["giberish_score"]

            if(prompt_str not in self.first_evaluation):
                self.first_evaluation[prompt_str] = {"correct": 0, "incorrect": 0}
            
            if(score_dict["rm_score"]):
                self.first_evaluation[prompt_str]["correct"] += 1
            else:
                self.first_evaluation[prompt_str]["incorrect"] += 1

            if self.num_examine == 0 and self.config.curiosity.bleu_reward_coef != 0:
                if self.config.curiosity.bleu_reward_include_prompts:
                    self.bleu_reward_module.append_reference(prompt_str, sequences_str)
                else:
                    if self.config.curiosity.compare_answer:
                        if(extract_answer==None):
                            str_to_compare = "None"
                        else:
                            str_to_compare = extract_answer
                    else:     
                        str_to_compare = response_str

                    self.bleu_reward_module.append_reference(prompt_str, str_to_compare)

            if self.num_examine == 0 and self.config.curiosity.cossimemb_reward_coef != 0:
                if self.config.curiosity.cossimemb_reward_include_prompts:
                    self.cossimemb_reward_module.append_reference(prompt_str, sequences_str)
                else:
                    self.cossimemb_reward_module.append_reference(prompt_str, response_str)
            
            new_data = {"prompt": prompt_str, "response": response_str, "extract_answer": extract_answer, "score": score_dict}
            data_list.append(new_data)

        
        with open(self.output_filename, "r") as f:
            data_json = json.load(f)
        
        data_json[self.epoch] = data_list

        with open(self.output_filename, "w") as f:
            json.dump(data_json, f, indent=4, ensure_ascii=False)

        self.epoch += 1


        return reward_tensor, rm_tensor, bleu_tensor, cossimemb_tensor, giberish_tensor


import ray
import hydra
import json

@hydra.main(config_path='config', config_name='ppo_trainer_curiosity', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}}, _temp_dir='/dev/shm')

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


    filename_parameters = f"bs_{config.data.train_batch_size}_minibs_{config.actor_rollout_ref.actor.ppo_mini_batch_size}_rollout_{config.actor_rollout_ref.rollout.n}_bleu_{config.reward_model.curiosity.bleu_reward_coef}_cossimemb_{config.reward_model.curiosity.cossimemb_reward_coef}_giberish_{config.reward_model.curiosity.giberish_penalty_coef}"

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, config=config.reward_model, output_filename=filename_parameters)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, config=config.reward_model, output_filename=filename_parameters)

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
