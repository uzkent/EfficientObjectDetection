"""
This function pretrains the policy network using the high resolution classifier
output-explained as pretraining the policy network in the paper.
How to Run on the CIFAR10 and CIFAR100 Datasets:
    python pretrain.py --model R32_C10, R32_C100
       --lr 1e-3
       --cv_dir checkpoint directory
       --batch_size 512
       --ckpt_hr_cl Load the checkpoint from the directory (hr_classifier)
How to Run on the fMoW Dataset:
    python pretrain.py --model R34_fMoW
       --lr 1e-3
       --cv_dir checkpoint directory
       --batch_size 1024
       --ckpt_hr_cl Load the checkpoint from the directory (hr_classifier)
"""
import os
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import random
import tqdm
import utils
import utils_detector
import torch.optim as optim

from tensorboard_logger import configure, log_value
from torch.distributions import Multinomial, Bernoulli
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='PatchDrop Pre-Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--penalty', type=float, default=-0.5, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--sigma', type=float, default=0.1, help='multiplier for the entropy loss')
args = parser.parse_args()

def test(epoch):
    # Test the policy network
    matches, rewards, metrics, policies, set_labels = [], [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        # Initiate the policy
        policy = torch.zeros((inputs.size(1), num_actions)).cuda()

        # ---------------------------------------
        policy[:, :] = 0
        for ind in range(policy.size(0)):
            indexes = random.sample(range(0, 16), 6)
            policy[ind, indexes] = 1
        # ---------------------------------------

        outputs, targets, batch_labels = utils.get_detected_boxes(policy, targets)
        metrics += utils_detector.get_batch_statistics(outputs, targets, 0.5)

        # Find the reward for baseline and sampled policy
        reward, _ = utils.compute_reward(offset_fd, offset_cd, policy.data)

        set_labels += batch_labels.tolist()
        rewards.append(reward)
        policies.append(policy.data)

    # Compute the Precision and Recall Performance of the Agent and Detectors
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(true_positives, pred_scores, pred_labels, set_labels)

    print 'Test - AP: %.2f | AR : %.2f'%(AP[0], recall[0])
    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)

    print 'Test - Rw: %.2E | S: %.3f | V: %.3f | #: %d'%(reward, sparsity, variance, len(policy_set))

#--------------------------------------------------------------------------------------------------------#
num_actions = 16
num_window_side = 4
_, testset = utils.get_dataset(args.model, args.data_dir)
testloader = torchdata.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_actions)

test(0)
