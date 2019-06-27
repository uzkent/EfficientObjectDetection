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
import utils_detector

from random import randint, sample
from xView_dataloader import CustomDatasetFromImages
img_size = 416

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

def get_detected_boxes(policy, file_dirs, num_windows, base_dir_fd, base_dir_cd, base_dir_gt):
    outputs_all, targets, counter_temp = [], np.zeros((1, 6)), 0
    for index, file_dir in enumerate(file_dirs):
         counter, outputs_img = 0, np.zeros((1,7))
         for xind in range(num_windows):
             for yind in range(num_windows):
                 file_dir_st = file_dir
                 # ---------------- Read Ground Truth -----------------------------
                 gt_path = '{}{}_{}_{}.txt'.format(base_dir_gt, file_dir_st, str(xind), str(yind))
                 if os.path.exists(gt_path):
                     gt = np.loadtxt(gt_path).reshape([-1, 5])
                     gt = np.hstack((counter_temp*np.ones((gt.shape[0], 1)), gt))
                     targets = np.vstack((targets, gt))
                     # ---------------- Read Detections -------------------------------
                     if policy[index, counter] == 1:
                         preds_dir = '{}{}_{}_{}'.format(base_dir_fd, file_dir_st, str(xind), str(yind))
                         if os.path.exists(preds_dir):
                             outputs_all.append(torch.from_numpy(np.loadtxt(preds_dir).reshape([-1,7])))
                         else:
                             outputs_all.append(torch.from_numpy(np.zeros((1, 7))))
                     else:
                         preds_dir = '{}{}_{}_{}_ds'.format(base_dir_cd, file_dir_st, str(xind), str(yind))
                         if os.path.exists(preds_dir):
                             outputs_all.append(torch.from_numpy(np.loadtxt(preds_dir).reshape([-1,7])))
                         else:
                             outputs_all.append(torch.from_numpy(np.zeros((1, 7))))
                     counter_temp += 1
                     counter += 1
                 else:
                     counter += 1
                     continue
                 # ----------------------------------------------------------------

    # Rescale target
    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    targets[:, 2:] *= img_size
    targets = np.delete(targets, 0, 0)
    return outputs_all, torch.from_numpy(targets), targets[:,1]

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

def compute_reward(offset_fd, offset_cd, policy, alpha=0.1, beta=0.1):
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    reward_patch_acc = (offset_cd-offset_fd)*policy + -1*((offset_cd-offset_fd)*(1-policy))
    reward_patch_acqcost = policy.sum(dim=1)/policy.size(1)
    reward_patch_rtcost = policy.sum(dim=1)/policy.size(1)
    reward_img = reward_patch_acc.sum(dim=1) + 7.50 * reward_patch_acqcost + 7.50 * reward_patch_rtcost
    reward = reward_img.unsqueeze(1)
    return reward.float(), reward

def get_transforms(rnet, dset):
    if dset == 'CD':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
           transforms.Scale(448),
           transforms.RandomCrop(448),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
           transforms.Scale(448),
           transforms.CenterCrop(448),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])

    if dset == "FD":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
           transforms.Scale(112),
           transforms.RandomCrop(112),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
           transforms.Scale(112),
           transforms.CenterCrop(112),
           transforms.ToTensor(),
           transforms.Normalize(mean, std)
        ])

    return transform_train, transform_test

def agent_chosen_input(input_org, policy, mappings, interval):
    """ Make the inputs variable size images using the determined policy
        The high resolution images (32x32 C10) will be changed to low resolution images (24x24 C10)
        if the action is set to 0.
    """
    input_full = input_org.clone()
    sampled_img = torch.zeros([input_org.shape[0], input_org.shape[1], input_org.shape[2], input_org.shape[3]])
    for pl_ind in range(policy.shape[1]):
        mask = (policy[:, pl_ind] == 1).cpu()
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval] = input_full[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval]
        sampled_img[:, :, mappings[pl_ind][0]:mappings[pl_ind][0]+interval, mappings[pl_ind][1]:mappings[pl_ind][1]+interval] *= mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
    input_org = sampled_img

    return input_org.cuda()

def action_space_model(dset):
    img_size = 448
    interval = 112
    # Model the action space by dividing the image space into equal size patches
    mappings = []
    for cl in range(0, img_size, interval):
        for rw in range(0, img_size, interval):
            mappings.append([cl, rw])

    return mappings, img_size, interval

# Pick from the datasets available and the hundreds of models we have lying around depending on the requirements.
def get_dataset(model, root='data/'):
    rnet, dset = model.split('_')
    transform_train, transform_test = get_transforms(rnet, dset)
    trainset = CustomDatasetFromImages(root+'/xView/train.csv', transform_train)
    testset = CustomDatasetFromImages(root+'/xView/valid.csv', transform_test)

    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(model, num_output=16):
    os.environ['TORCH_MODEL_ZOO'] = '/atlas/u/buzkent/EfficientObjectDetection/cv/pretrained/'
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)

    return agent
