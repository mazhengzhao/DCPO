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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict

class NaiveRewardManager_ori:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor



def _compute_confidence_ci(log_prob: torch.Tensor,
                           mask: torch.Tensor,
                           epsilon: float = 1e-8) -> torch.Tensor:

    masked_log_prob = log_prob * mask

    # Calculate the actual length of each response
    sequence_lengths = mask.sum(dim=-1)

    # Sum the log probabilities for each sequence
    sum_log_prob = masked_log_prob.sum(dim=-1)

    # Compute the mean log probability, handle division by zero for empty sequences
    mean_log_prob = sum_log_prob / (sequence_lengths + epsilon)

    # Ci is the exponent of the mean log probability
    ci = torch.exp(mean_log_prob)

    return ci
# 假设这个辅助函数已定义在某处
# def _compute_confidence_ci(log_prob, mask, epsilon=1e-8):
#     """Computes confidence as the average log probability of the generated tokens."""
#     sum_log_probs = (log_prob * mask).sum(dim=-1)
#     num_tokens = mask.sum(dim=-1)
#     return sum_log_probs / (num_tokens + epsilon)

import json
from typing import Optional
class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False, save_analysis_path: Optional[str] = None):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        #TODO: New added
        responses = data.batch['responses']
        is_save_mode = save_analysis_path is not None
        # 只有在需要保存时，才强制计算置信度
        confidences_list = [0.0] * len(data)
        if is_save_mode:
            response_length = responses.size(1)
            all_attention_mask = data.batch['attention_mask']
            response_only_mask = all_attention_mask[:, -response_length:]
            log_probs = data.batch['response_log_probs']
            confidences_tensor = _compute_confidence_ci(log_probs, response_only_mask)
            confidences_list = confidences_tensor.cpu().tolist()
        all_samples_data_for_json = []
        # ########

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward


            #TODO: NEW ADD
            # d. 准备单个样本的数据字典
            single_sample_info = {}
            if isinstance(score, dict):
                accuracy = score.get("score", 0.0)
                single_sample_info.update(score)  # 将评分函数返回的字典内容全部加入
            else:
                accuracy = score
                single_sample_info["accuracy"] = accuracy

            # 如果 'score' 存在，也用 'accuracy' 作为标准键名
            if 'score' in single_sample_info:
                single_sample_info['accuracy'] = single_sample_info['score']

            confidence = confidences_list[i]
            single_sample_info["confidence"] = confidence

            # e. 收集信息到 reward_extra_info (用于返回)
            for key, value in single_sample_info.items():
                reward_extra_info[key].append(value)

            # f. 如果是保存模式，构建并收集完整的样本字典
            if is_save_mode:
                # 把所有想要保存的信息都放进这个字典里
                full_sample_for_json = {
                    "input": prompt_str,
                    "output": response_str,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    **single_sample_info  # 使用字典解包合并所有评分和置信度信息
                }
                all_samples_data_for_json.append(full_sample_for_json)

            # #######


            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
                # --- 3. 循环结束后的保存逻辑 (修改为保存 JSON) ---
        #TODO:NEW ADDED
        if is_save_mode:
            print(f"Aggregating data for saving to {save_analysis_path}...")
            try:
                # 使用 json.dump 将列表写入文件
                with open(save_analysis_path, 'w', encoding='utf-8') as f:
                    json.dump(all_samples_data_for_json, f, ensure_ascii=False, indent=4)
                print(f"Analysis data successfully saved as JSON to {save_analysis_path}")
            except Exception as e:
                print(f"Error! Failed to save analysis JSON data: {e}")
        # ########
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
