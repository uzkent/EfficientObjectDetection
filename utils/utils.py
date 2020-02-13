import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
import numpy as np
import pandas as pd
import pdb
import shutil
import json
import copy
import utils_detector
import torch.nn.functional as F
import time
from random import randint, sample

from dataset.xView_dataloader import CustomDatasetFromImages
from constants import base_dir_gt, base_dir_cd, base_dir_fd, base_dir_reward_cd, base_dir_reward_fd
from constants import num_actions_coarse, num_windows_cd, num_windows_fd, img_size_fd, img_size_cd

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)
    return data

def xywh2xyxy(x):
    y = np.zeros((x.shape))
    y[:,0] = x[:, 0] - x[:, 2] / 2.
    y[:,1] = x[:, 1] - x[:, 3] / 2.
    y[:,2] = x[:, 0] + x[:, 2] / 2.
    y[:,3] = x[:, 1] + x[:, 3] / 2.
    return y

def get_detected_boxes(policy, file_dirs, metrics, set_labels):
    for index, file_dir_st in enumerate(file_dirs):
         counter, outputs_img = 0, np.zeros((1,7))
         for xind in range(num_windows):
             for yind in range(num_windows):
                 # ---------------- Read Ground Truth ----------------------------------
                 outputs_all = []
                 gt_path = '{}{}_{}_{}_{}_{}.txt'.format(base_dir_gt, file_dir_st, str(xind), str(yind))
                 if os.path.exists(gt_path):
                     gt = np.loadtxt(gt_path).reshape([-1, 5])
                     targets = np.hstack((np.zeros((gt.shape[0], 1)), gt))
                     targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                     # ----------------- Read Detections -------------------------------
                     if policy[index, counter] == 1:
                         preds_dir = '{}{}_{}_{}_{}_{}'.format(base_dir_fd, file_dir_st, str(xind), str(yind))
                         targets[:, 2:] *= img_size_fd
                         if os.path.exists(preds_dir):
                             preds = np.loadtxt(preds_dir).reshape([-1,7])
                             outputs_all.append(torch.from_numpy(preds))
                     else:
                         preds_dir = '{}{}_{}_{}_{}_{}'.format(base_dir_cd, file_dir_st, str(xind), str(yind))
                         targets[:, 2:] *= img_size_cd
                         if os.path.exists(preds_dir):
                             preds = np.loadtxt(preds_dir).reshape([-1,7])
                             outputs_all.append(torch.from_numpy(preds))
                     set_labels += targets[:, 1].tolist()
                     metrics += utils_detector.get_batch_statistics(outputs_all, torch.from_numpy(targets), 0.5)
                 else:
                     continue
                 counter += 1

    return metrics, set_labels

def read_offsets(image_ids, base_dir_fd, base_dir_cd, num_actions):
    offset_fd = torch.zeros((len(image_ids), num_actions)).cuda()
    offset_cd = torch.zeros((len(image_ids), num_actions)).cuda()
    for index, img_id in enumerate(image_ids):
        offset_fd[index, :] = torch.from_numpy(np.loadtxt('{}{}'.format(base_dir_fd, img_id)).flatten())
        offset_cd[index, :] = torch.from_numpy(np.loadtxt('{}{}'.format(base_dir_cd, img_id)).flatten())

    return offset_fd, offset_cd

def performance_stats(policies, rewards):
    # Print the performace metrics including the average reward, average number
    # and variance of sampled num_patches, and number of unique policies
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set([''.join(p) for p in policy_set])

    return reward, num_unique_policy, variance, policy_set

def compute_reward(offset_fd, offset_cd, reward_patch_acc, policy, beta, sigma):
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    offset_cd += beta
    reward_patch_diff = (offset_fd - offset_cd)*policy + -1*((offset_fd - offset_cd)*(1-policy))
    reward_patch_acqcost = (policy.size(1) - policy.sum(dim=1)) / policy.size(1)
    reward_img = reward_patch_diff.sum(dim=1) + sigma * reward_patch_acqcost
    reward = reward_img.unsqueeze(1)
    return reward.float()

def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
       transforms.Scale(img_size),
       transforms.RandomCrop(img_size),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
       transforms.Scale(img_size),
       transforms.CenterCrop(img_size),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
    ])

    return transform_train, transform_test

def rearrange_batch(inputs, targets, selected_index, train_time=True, num_windows=3, img_size=55):
    num_actions = num_windows * num_windows
    if train_time:
        inputs_partitioned = torch.zeros((inputs.size(0), 3, img_size, img_size))
    else:
        inputs_partitioned = torch.zeros((num_actions*inputs.size(0), 3, img_size, img_size))
    counter, targets_partitioned = 0, []
    for ind in range(inputs.size(0)):
        for xind in range(num_windows):
            for yind in range(num_windows):
                x1 = xind*img_size
                x2 = x1 + img_size
                y1 = yind*img_size
                y2 = y1 + img_size
                if train_time:
                    if (xind*num_windows+yind) == selected_index:
                        inputs_partitioned[ind, :, :, :] = inputs[ind, :, x1:x2, y1:y2]
                        targets_partitioned.append('{}_{}_{}'.format(str(targets[ind].cpu().numpy().tolist()), str(xind), str(yind)))
                else:
                    inputs_partitioned[counter, :, :, :] = inputs[ind, :, x1:x2, y1:y2]
                    targets_partitioned.append('{}_{}_{}'.format(str(targets[ind]), str(xind), str(yind)))
                    counter += 1

    return inputs_partitioned, tuple(targets_partitioned)

def agent_chosen_input(input_org, policy, mappings, interval):
    """ Make the inputs variable size images using the determined policy
    """
    input_full = input_org.clone()
    sampled_img = torch.zeros([input_org.shape[0], input_org.shape[1], input_org.shape[2], input_org.shape[3]])
    for pl_ind in range(policy.shape[1]):
        mask = (policy[:, pl_ind] == 1).cpu()
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval] = input_full[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval]
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval] *= mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
    input_org = sampled_img

    return input_org.cuda()

def action_space_model(fine=True):
    # Model the action space by dividing the image space into equal size patches
    img_size = 175
    interval = 55
    mappings = []
    for cl in range(0, img_size, interval):
        for rw in range(0, img_size, interval):
            mappings.append([cl, rw])

    return mappings, img_size, interval

def get_dataset(img_size, root='data/'):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root+'train.csv', transform_train)
    testset = CustomDatasetFromImages(root+'valid.csv', transform_test)

    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(num_output=16):
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)

    return agent
