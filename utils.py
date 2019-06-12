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
from random import randint, sample

from xView_dataloader import CustomDatasetFromImages
mappings_labels = pd.read_csv('/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/mapping_xview_labels.csv')

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)

    return data

def get_detected_boxes(policy, file_dirs):
    base_dir_fd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/fine_dt_output/'
    base_dir_cd = '/atlas/u/buzkent/PyTorch-YOLOv3/data/custom/fine_dt_output_ds/'
    base_dir_gt = '/atlas/u/buzkent/Single_Shot_Object_Detector/prepare_dataset/train_chips_xview/'
    outputs_all, targets = [], np.zeros((1, 6))
    policy[:, :] = 0
    for index, file_dir in enumerate(file_dirs):
         counter, outputs_img = 0, np.zeros((1,7))
         for xind in range(6):
             for yind in range(6):
                 file_dir_st = str(file_dir).split(')')[0].split('(')[1]
                 # ---------------- Read Detections -------------------------------
                 if policy[index, counter] == 1:
                     preds_dir = '{}{}_{}_{}'.format(base_dir_fd, file_dir_st, str(xind), str(yind))
                     if os.path.exists(preds_dir):
                         outputs_img = np.vstack((outputs_img, np.loadtxt(preds_dir)))
                 else:
                     preds_dir = '{}{}_{}_{}_ds'.format(base_dir_cd, file_dir_st, str(xind), str(yind))
                     if os.path.exists(preds_dir):
                         outputs_img = np.vstack((outputs_img, np.loadtxt(preds_dir)))
                 # ----------------------------------------------------------------
                 # ---------------- Read Ground Truth -----------------------------
                 gt = read_json('{}{}_{}_{}.json'.format(base_dir_gt, file_dir_st, str(xind), str(yind)))
                 if gt:
                     for gt_ind in gt:
                         try:
                             targets = np.vstack((targets, [index, mappings_labels[mappings_labels['labels']==gt_ind[1]].index.item(),
                                        gt_ind[0][0], gt_ind[0][1], gt_ind[0][2], gt_ind[0][3]]))
                         except Exception as error:
                             print(error)
                 # ----------------------------------------------------------------
                 counter += 1
         outputs_all.append(torch.from_numpy(outputs_img))

    # Project the ground truth back to the image domain
    targets[:,2:] = 416*targets[:,2:]
    return outputs_all, torch.from_numpy(np.delete(targets, 0, 0)), targets[:,1]

def read_groundtruth(image_ids):
    base_dir = '/atlas/u/buzkent/Single_Shot_Object_Detector/prepare_dataset/train_chips_xview/'
    object_interest = [73, 74, 76, 79, 77, 71, 72] #[11, 12, 13, 15]
    patch_groundtruth = torch.zeros((len(image_ids), 16)).cuda()
    for index, img_id in enumerate(image_ids):
        counter = 0
        for xind in range(4):
            for yind in range(4):
                temp = read_json('{}{}_{}_{}{}'.format(base_dir, img_id, str(xind), str(yind), '.json'))
                intersection = [i for i in object_interest if i in temp]
                if intersection:
                    patch_groundtruth[index, counter] = 1
                counter += 1
    return patch_groundtruth

def read_offsets(image_ids):
    base_dir_fd = '/atlas/u/buzkent/EfficientObjectDetection/data/xView/reward_fd/'
    base_dir_cd = '/atlas/u/buzkent/EfficientObjectDetection/data/xView/reward_cd/'
    offset_fd = torch.zeros((len(image_ids), 36)).cuda()
    offset_cd = torch.zeros((len(image_ids), 36)).cuda()
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
    reward_patch = (offset_cd - offset_fd) * policy + -(offset_cd - offset_fd) * (1-policy)
    reward_img = reward_patch.sum(dim=1)
    reward = reward_img.unsqueeze(1)
    return reward.float(), reward

def get_transforms(rnet, dset):

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
    testset = CustomDatasetFromImages(root+'/xView/test.csv', transform_test)

    return trainset, testset

def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(model):
    os.environ['TORCH_MODEL_ZOO'] = '/atlas/u/buzkent/EfficientObjectDetection/cv/pretrained/'
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, 36)

    return agent
