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
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load_fd', default=None, help='checkpoint to load agent from')
parser.add_argument('--load_cd', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
args = parser.parse_args()

def test(epoch):
    # Test the policy network
    agent_fd.eval()
    agent_cd.eval()

    metrics, set_labels, num_total, num_sampled = [], [], 0., 0.
    for batch_idx, (inputs_cd, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs_cd = Variable(inputs_cd, volatile=True)
        inputs_cd = inputs_cd.cuda()

        # Get the low resolution agent images
        probs = F.sigmoid(agent_cd(inputs_cd))

        # Sample the policy from the agents output
        policy_cd = probs.data.clone()
        policy_cd[policy_cd<0.5] = 0.0
        policy_cd[policy_cd>=0.5] = 1.0
        policy_cd = Variable(policy_cd)

        # Select the images to run Fine Detector on
        for xind in range(num_windows_cd):
            for yind in range(num_windows_cd):
                # Get the low resolution agent images
                probs = F.sigmoid(agent_fd(inputs_cd[:, :, xind*img_size_fd:xind*img_size_fd+img_size_fd,
                        yind*img_size_fd:yind*img_size_fd+img_size_fd]))

                # Sample the policy from the agents output
                policy_fd = probs.data.clone()
                policy_fd[policy_fd<0.5] = 0.0
                policy_fd[policy_fd>=0.5] = 1.0
                policy_fd = Variable(policy_fd)

                # -----------------------------------------------
                index_ft = xind*num_windows_cd + yind
                if policy_cd[:, index_ft] == 0:
                    policy_fd[:, :] = 0
                # -----------------------------------------------
                num_sampled += (policy_fd == 1).sum().cpu().numpy().tolist()
                num_total += policy_fd.size(1)

                # Compute the Batch-wise metrics
                targets_ind = ['{}_{}_{}'.format(str(targets[0].numpy().tolist()), str(xind), str(yind))]

                outputs, targets_np, batch_labels = utils.get_detected_boxes(policy_fd, targets_ind, num_windows_fd, base_dir_fd,
                                                    base_dir_cd, base_dir_gt)
                metrics += utils_detector.get_batch_statistics(outputs, targets_np, 0.5)

                set_labels += batch_labels.tolist()

    # Compute the Precision and Recall Performance of the Agent and Detectors
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(true_positives, pred_scores, pred_labels, set_labels)

    print 'Test - AP: %.2f | AR : %.2f | RS : %.2f'%(AP[0], recall[0], num_sampled/num_total)

#--------------------------------------------------------------------------------------------------------#
num_windows_fd = 2
num_windows_cd = 4
num_actions_fd = 4
num_actions_cd = 16
img_size_fd = 112
_, testset = utils.get_dataset(args.model, args.data_dir)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=16)
agent_cd = utils.get_model(args.model, num_actions_cd)
agent_fd = utils.get_model(args.model, num_actions_fd)

base_dir_fd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/fd_output_txt_small/'
base_dir_cd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/cd_output_txt_small/'
base_dir_gt = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/labels/'

# ---- Load the pre-trained model ----------------------
checkpoint_fd = torch.load(args.load_fd)
checkpoint_cd = torch.load(args.load_cd)
agent_fd.load_state_dict(checkpoint_fd['agent'])
agent_cd.load_state_dict(checkpoint_cd['agent'])
agent_fd.cuda()
agent_cd.cuda()

# Save the args to the checkpoint directory
test(0)
