import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import torchvision.models as torchmodels
import numpy as np
import pdb
import shutil
import json
from random import randint, sample

from xView_dataloader import CustomDatasetFromImages

def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

def read_json(filename):
    with open(filename) as dt:
        data = json.load(dt)

    labels = []
    for dt in data:
        labels.append(dt[-1])

    return labels

def read_groundtruth(image_ids):
    base_dir = '/atlas/u/buzkent/Single_Shot_Object_Detector/prepare_dataset/train_chips_xview/'
    object_interest = 73
    patch_groundtruth = torch.zeros((len(image_ids), 16)).cuda()
    for index, img_id in enumerate(image_ids):
        counter = 0
        for xind in range(4):
            for yind in range(4):
                temp = read_json('{}{}_{}_{}{}'.format(base_dir, img_id, str(xind), str(yind), '.json'))
                if object_interest in temp:
                    patch_groundtruth[index, counter] = 1
                counter += 1
    return patch_groundtruth

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

def compute_reward(patch_groundtruth, policy):
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    reward = (patch_groundtruth == policy).sum(dim=1).float() / policy.size(1)
    reward = reward.unsqueeze(1)
    return reward.float()

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
    agent.fc = torch.nn.Linear(num_ftrs, 16)

    return agent
