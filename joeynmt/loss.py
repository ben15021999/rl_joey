# coding: utf-8
"""
Module to implement training loss
"""

import torch
from torch import nn, Tensor
from torch.autograd import Variable
from joeynmt.metrics import bleu
import numpy as np


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='sum')
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index,
            as_tuple=False)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1),
                vocab_size=log_probs.size(-1))
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape \
                == targets.shape
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss

class ReinforceLoss(nn.Module):
    """
    Reinforce Loss
    """
    def __init__(self, baseline, use_cuda, reward: str="bleu"):
        super(ReinforceLoss, self).__init__()
        self.baseline = baseline
        self.reward = reward
        self.bleu = []
        
    def forward(self, predicted, gold, log_probs):
        """
        Compute the reinforce loss using logprobs and bleu scores
        :param predicted: predicted sentences
        :param gold: gold sentences
        :return: loss, rewards for logging, unscaled rewards for logging
        """
        bleu_scores = [bleu([prediction], [gold_ref]) \
                for prediction, gold_ref in zip(predicted, gold)]
        # save unscaled rewards for logging
        unscaled_rewards = bleu_scores
        if self.reward == "constant":
            bleu_scores = [1 for log_prob in log_probs]
        elif self.reward == "scaled_bleu":
            def scale(reward, a, b, minim, maxim):
                if maxim-minim == 0:
                    return 0
                return (((b-a)*(reward - minim))/(maxim-minim)) + a
            # local scale
            maxim = max(bleu_scores)
            minim = min(bleu_scores)
            bleu_scores = [scale(score, -0.5, 0.5, minim, maxim) \
                for score in bleu_scores]
        elif self.reward == "bleu":
            if self.baseline == "average_reward_baseline":
                # global average
                self.bleu.extend(bleu_scores)
                average_bleu = np.mean(self.bleu)
                bleu_scores = [score - average_bleu for score in bleu_scores]
        # calculate PG loss with rewards and log probs
        loss = sum([-log_prob*bleu_score \
                for log_prob, bleu_score in zip(log_probs, bleu_scores)])
        return loss, bleu_scores, unscaled_rewards
