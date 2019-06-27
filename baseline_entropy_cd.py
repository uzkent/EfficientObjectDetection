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

def read_confidences(image_ids):
    base_dir_cd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/cd_output_txt/'
    offset_cd = np.zeros((len(image_ids), num_windows, num_windows))
    for index, img_id in enumerate(image_ids):
        for rw in range(num_windows):
            for cl in range(num_windows):
                path = '{}{}_{}_{}_ds'.format(base_dir_cd, img_id.cpu().numpy().tolist(), str(rw), str(cl))
                if os.path.exists(path):
                    try:
                        offset_cd[index, rw, cl] = np.loadtxt(path)[:, 4].mean()
                    except Exception as error:
                        offset_cd[index, rw, cl] = np.loadtxt(path)[4]
                else:
                    offset_cd[index, rw, cl] = 1.0

    return torch.from_numpy(offset_cd)

def test(epoch):
    # Test the policy network
    matches, rewards, metrics, policies, set_labels = [], [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        # Initiate the policy
        policy = torch.zeros((inputs.size(1), num_actions)).cuda()

        # Compute the Batch-wise metrics
        conf_cd = read_confidences(targets)

        indices_sample = conf_cd[:,:].cpu().numpy().flatten().argsort()[:7]
        policy[:, indices_sample] = 1

        outputs, targets, batch_labels = utils.get_detected_boxes(policy, targets, num_windows, base_dir_fd, base_dir_cd, base_dir_gt)
        metrics += utils_detector.get_batch_statistics(outputs, targets, 0.5)

        set_labels += batch_labels.tolist()
        policies.append(policy.data)

    # Compute the Precision and Recall Performance of the Agent and Detectors
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = utils_detector.ap_per_class(true_positives, pred_scores, pred_labels, set_labels)

    print 'Test - AP: %.2f | AR : %.2f'%(AP[0], recall[0])

#--------------------------------------------------------------------------------------------------------#
num_actions = 16
num_windows = 4
_, testset = utils.get_dataset(args.model, args.data_dir)
testloader = torchdata.DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_actions)

base_dir_reward_fd = '/atlas/u/buzkent/EfficientObjectDetection/data/xView/reward_fd/'
base_dir_reward_cd = '/atlas/u/buzkent/EfficientObjectDetection/data/xView/reward_cd/'
base_dir_fd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/fd_output_txt/'
base_dir_cd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/building/cd_output_txt/'
base_dir_gt = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/labels/'

test(0)
