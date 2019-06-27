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
import torch.optim as optim

from utils import utils, utils_detector
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

def read_confidences(image_ids, file_dir, num_windows):
    offset_cd = np.zeros((len(image_ids), num_windows, num_windows))
    for index, img_id in enumerate(image_ids):
        for rw in range(num_windows):
            for cl in range(num_windows):
                path = '{}{}_{}_{}_ds'.format(file_dir, img_id, str(rw), str(cl))
                if os.path.exists(path):
                    offset_cd[index, rw, cl] = np.loadtxt(path).reshape([-1,7])[:, 4].mean()
                else:
                    offset_cd[index, rw, cl] = 1.0

    return torch.from_numpy(offset_cd)

def test(epoch):
    # Test the policy network
    metrics, set_labels, num_sampled, num_total = [], [], 0., 0.
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        # Initiate the policy
        policy_cd = torch.zeros((inputs.size(0), num_actions_cd))
        conf_cd = read_confidences(targets, base_dir, num_windows_cd)

        indices_sample = conf_cd[:,:].cpu().numpy().flatten().argsort()[:7]
        policy_cd[:, indices_sample] = 1

        counter = 0
        for xind in range(num_windows_cd):
            for yind in range(num_windows_cd):

                policy_fd = torch.zeros((inputs.size(0), num_actions_fd))

                targets_ind = ['{}_{}_{}'.format(str(targets[0].numpy().tolist()), str(xind), str(yind))]
                conf_fd = read_confidences(targets_ind, base_dir_cd, num_windows_fd)

                if policy_cd[:, counter] == 0:
                    policy_fd[:, :] = 0
                else:
                    indices_sample = conf_fd[:, :].cpu().numpy().flatten().argsort()[:4]
                    policy_fd[:, indices_sample] = 1

                num_sampled += (policy_fd == 1).sum().numpy().tolist()
                num_total += policy_fd.size(1)

                outputs, targets_np, batch_labels = utils.get_detected_boxes(policy_fd, targets_ind, num_windows_fd,
                                                    base_dir_fd, base_dir_cd, base_dir_gt)
                metrics += utils_detector.get_batch_statistics(outputs, targets_np, 0.5)

                set_labels += batch_labels.tolist()
                counter += 1

    # Compute the Precision and Recall Performance of the Agent and Detectors
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(true_positives, pred_scores, pred_labels, set_labels)

    print 'Test - AP: %.2f | AR : %.2f | RS : %.2f'%(AP[0], recall[0], num_sampled/num_total)

#--------------------------------------------------------------------------------------------------------#
num_windows_cd = 4
num_windows_fd = 2
num_actions_cd = 16
num_actions_fd = 4
_, testset = utils.get_dataset(args.model, args.data_dir)
testloader = torchdata.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)

base_dir = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/cd_output_txt/'
base_dir_fd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/fd_output_txt_small/'
base_dir_cd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/cd_output_txt_small/'
base_dir_gt = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/labels/'

test(0)
