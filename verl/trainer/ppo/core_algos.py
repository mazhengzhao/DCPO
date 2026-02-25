# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F
import torch.nn.functional as F

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == 'fixed':
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == 'adaptive':
        assert kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {kl_ctrl.horizon}'
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

def _grpo_normalize(scores: torch.Tensor, index: np.ndarray, eps: float):
        """
        GRPO outcome normalization (grouped by prompt index).

        Args:
            scores: (bs,)
            index: numpy array of prompt ids

        Returns:
            normalized advantages: (bs,)
        """
        id2scores = defaultdict(list)

        for i in range(len(scores)):
            id2scores[index[i]].append(scores[i])

        id2mean = {}
        id2std = {}

        for k, v in id2scores.items():
            if len(v) == 1:
                id2mean[k] = torch.tensor(0.0, device=scores.device)
                id2std[k] = torch.tensor(1.0, device=scores.device)
            else:
                stacked = torch.stack(v)
                id2mean[k] = stacked.mean()
                id2std[k] = stacked.std()

        advantages = scores.clone()
        for i in range(len(scores)):
            advantages[i] = (scores[i] - id2mean[index[i]]) / (
                id2std[index[i]] + eps
            )

        return advantages

def compute_confidence_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   confidence_pos,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    if isinstance(confidence_pos, torch.Tensor):
        confidence_pos = confidence_pos.long()
    else:
        confidence_pos = torch.tensor(confidence_pos, device=token_level_rewards.device).long()

    bs, response_len = token_level_rewards.shape

    # --------------------------------------------------
    # 1. Extract outcome scores
    # --------------------------------------------------
    answer_scores = torch.zeros(bs, device=token_level_rewards.device)
    confidence_scores = torch.zeros(bs, device=token_level_rewards.device)

    for i in range(bs):
        cpos = confidence_pos[i].item()
        confidence_scores[i] = token_level_rewards[i, cpos]
        answer_scores[i] = token_level_rewards[i].sum() - confidence_scores[i]

    # --------------------------------------------------
    # 2. GRPO normalize separately
    # --------------------------------------------------
    answer_adv = _grpo_normalize(
        answer_scores, index, epsilon
    )
    confidence_adv = _grpo_normalize(
        confidence_scores, index, epsilon
    )

    # --------------------------------------------------
    # 3. Build token-level advantage
    # --------------------------------------------------
    advantages = torch.zeros_like(token_level_rewards)

    for i in range(bs):
        cpos = confidence_pos[i].item()
        for t in range(response_len):
            if not response_mask[i, t]:
                continue
            if t >= cpos:
                advantages[i, t] = confidence_adv[i]
            else:
                advantages[i, t] = answer_adv[i]

    # --------------------------------------------------
    # 4. Returns = advantages (GRPO is outcome-based)
    # --------------------------------------------------
    returns = advantages.clone()

    return advantages, returns
    

#TODO：new added
def _compute_confidence_ci(old_log_prob: torch.Tensor,
                           response_mask: torch.Tensor,
                           epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes the sequence-level confidence Ci from token-level log probabilities.
    Ci = exp(mean(log_probs over the sequence length)).

    Args:
        old_log_prob: (torch.Tensor)
            Log probabilities of tokens from the old policy. Shape: (bs, response_length).
        response_mask: (torch.Tensor)
            Mask for the response tokens. Shape: (bs, response_length).

    Returns:
        ci: (torch.Tensor)
            Sequence-level confidence. Shape: (bs,).
    """

    # Mask out padding tokens to avoid them affecting the mean
    masked_log_prob = old_log_prob * response_mask

    # Calculate the actual length of each response
    sequence_lengths = response_mask.sum(dim=-1)

    # Sum the log probabilities for each sequence
    sum_log_prob = masked_log_prob.sum(dim=-1)

    # Compute the mean log probability, handle division by zero for empty sequences
    mean_log_prob = sum_log_prob / (sequence_lengths + epsilon)

    # Ci is the exponent of the mean log probability
    ci = torch.exp(mean_log_prob)

    return ci




def compute_advantage_CCPO_BCE(token_level_rewards: torch.Tensor,
                                 old_log_prob: torch.Tensor,
                                 response_mask: torch.Tensor,
                                 index: np.ndarray,
                                 epsilon: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the confidence-based advantage tilde_A_i = (R_i - m) / (1 - C_i).
    'm' is the mean reward calculated per prompt group.
    'C_i' is the sequence-level confidence.

    Args:
        token_level_rewards: `(torch.Tensor)`
            Shape: (bs, response_length)
        old_log_prob: `(torch.Tensor)`
            Log probabilities for computing C_i. Shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            Shape: (bs, response_length)
        index: `(np.ndarray)`
            Array of prompt indices to group rewards. Shape: (bs,)
        epsilon: `(float)`
            Small value to prevent division by zero.

    Returns:
        advantages: `(torch.Tensor)`
            Shape: (bs, response_length)
        returns: `(torch.Tensor)`
            Shape: (bs, response_length)
    """
    # 1. Calculate sequence-level reward R_i
    scores = token_level_rewards.sum(dim=-1)
    device = scores.device

    # 2. Calculate sequence-level confidence C_i
    # are computed for any part of it, including the intermediate confidence 'ci'.
    with torch.no_grad():
        masked_log_prob = old_log_prob * response_mask
        # Calculate actual lengths of each response
        sequence_lengths = response_mask.sum(dim=-1)
        # Sum log probabilities for each sequence
        sum_log_prob = masked_log_prob.sum(dim=-1)
        # Compute the mean log probability, handle division by zero
        mean_log_prob = sum_log_prob / (sequence_lengths + epsilon)
        # Ci is the exponent of the mean log probability
        ci = torch.exp(mean_log_prob)

    # 3. Group scores by prompt index and compute mean (m) for each group
    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i].item())

        for idx, score_list in id2score.items():
            id2mean[idx] = torch.tensor(np.mean(score_list), device=device)

        # 4. Calculate advantage tilde_A_i = (R_i - m) / (1 - C_i)
        advantages_scalar = torch.zeros_like(scores)
        for i in range(bsz):
            numerator = scores[i] - id2mean[index[i]]
            # Denominator includes C_i for this specific sequence
            denominator = 1.0 - ci[i]
            advantages_scalar[i] = numerator / (denominator + epsilon)

    # 5. Broadcast advantages to token level
    advantages = advantages_scalar.unsqueeze(-1) * response_mask

    # In this formulation, returns are the same as advantages
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor,
                                                           response_mask: torch.Tensor,
                                                           index: torch.Tensor,
                                                           epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss




def compute_policy_loss(old_log_prob,
                        log_prob,
                        advantages,
                        response_mask,
                        cliprange=None,
                        cliprange_low=None,
                        cliprange_high=None,
                        clip_ratio_c=3.0,
                        loss_agg_mode="token-mean",
                        ):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low,
                                           1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1,
                                    pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)


    total_policy_loss = pg_loss

    return total_policy_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower





def compute_policy_loss_new(
        algorithm_name: str,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        # 可选参数，根据算法需要传入
        token_level_rewards: torch.Tensor = None,
        beta: float = 0.01,
        loss_agg_mode: str = "token-mean",
        epsilon: float = 1e-8,
        cliprange=None,
        cliprange_low=None,
        cliprange_high=None,
        clip_ratio_c=3.0,
):
    """
    Computes the policy loss for various RL algorithms.

    Args:
        algorithm_name (str): The name of the algorithm.
            Choices: "grpo", "cspo", "gpg", "ours_mse", "ours_bce".
        old_log_prob (torch.Tensor): Log probabilities from the old policy. Shape: (bs, seq_len).
        log_prob (torch.Tensor): Log probabilities from the current policy. Shape: (bs, seq_len).
        advantages (torch.Tensor): Computed advantages, specific to each algorithm. Shape: (bs, seq_len).
        response_mask (torch.Tensor): Mask for response tokens. Shape: (bs, seq_len).
        token_level_rewards (torch.Tensor, optional): Rewards for "ours" methods. Shape: (bs, seq_len).
        beta (float, optional): Regularization coefficient. Defaults to 0.01.
        cliprange (float, optional): PPO clip range for GRPO. Defaults to 0.2.
        clip_ratio_c (float, optional): Dual-clip PPO parameter for GRPO. Defaults to 3.0.
        loss_agg_mode (str, optional): Loss aggregation mode for GRPO/GPG. Defaults to "token-mean".
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-8.

    Returns:
        A tuple containing:
        - pg_loss (torch.Tensor): The final computed policy loss, ready for optimization.
        - pg_clipfrac (float): The fraction of clipped samples (for GRPO).
        - approx_kl (float): The approximate KL divergence.
        - pg_clipfrac_lower (float): The fraction of lower-clipped samples (for GRPO).
    """

    # --- 1. Pre-compute common quantities ---
    approx_kl = verl_F.masked_mean(old_log_prob - log_prob, response_mask)
    # --- FIX 1: Initialize as Tensors on the correct device ---
    device = log_prob.device
    pg_clipfrac = torch.tensor(0.0, device=device)
    pg_clipfrac_lower = torch.tensor(0.0, device=device)
    regularization_loss = torch.tensor(0.0, device=device)
    # --- 新增：为BCE的两个部分初始化返回值 ---
    positive_bce_part = torch.tensor(0.0, device=device)
    negative_bce_part = torch.tensor(0.0, device=device)
    # --- 2. Select loss calculation based on algorithm name ---
    if algorithm_name == "grpo":
        # GRPO uses a PPO-style clipped objective with a token-level ratio.
        assert clip_ratio_c > 1.0, f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

        negative_approx_kl = log_prob - old_log_prob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        pg_losses1 = -advantages * ratio
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low,
                                               1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_pg_losses1 = torch.maximum(pg_losses1,
                                        pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    elif algorithm_name == "ccpo_bce":
        # Objective: log(C_i) * Ã_i + β * (R_i*log(C_i) + (1-R_i)*log(1-C_i))
        if token_level_rewards is None:
            raise ValueError("`token_level_rewards` must be provided for 'ours_bce'.")

        ci_new = _compute_confidence_ci(log_prob, response_mask, epsilon)
        rewards_scalar = token_level_rewards.sum(dim=-1)
        advantages_scalar = advantages[:, 0]  # This should be Ã_i

        # Clamp ci_new for numerical stability in log
        ci_clamped = torch.clamp(ci_new, min=epsilon, max=1.0 - epsilon)

        policy_term = torch.log(ci_clamped) * advantages_scalar

        positive_bce_term_unscaled = rewards_scalar * torch.log(ci_clamped)
        negative_bce_term_unscaled = (1 - rewards_scalar) * torch.log(1 - ci_clamped)
        bce_term = beta * ( positive_bce_term_unscaled + negative_bce_term_unscaled)

        positive_bce_part = positive_bce_term_unscaled.mean()
        negative_bce_part = negative_bce_term_unscaled.mean()
        regularization_loss = - bce_term.mean()

        objective = policy_term + bce_term
        pg_loss = -objective.mean()

    else:
        raise ValueError(f"Unknown algorithm_name: '{algorithm_name}'. "
                         "Choices are: 'grpo', 'ccpo_bce'.")

    return pg_loss, pg_clipfrac, approx_kl, pg_clipfrac_lower, regularization_loss, positive_bce_part, negative_bce_part





def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
