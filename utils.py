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

def compute_reward(patch_groundtruth, policy, alpha=0.1, beta=0.1):
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    reward_acc = ((patch_groundtruth == policy).float() * patch_groundtruth).sum(dim=1) / patch_groundtruth.sum(dim=1)
    reward_acc[reward_acc!=reward_acc] = 0
    #reward_fcost = alpha * (policy.size(1) - policy.sum(dim=1)) / policy.size(1)
    #reward_rtcost = beta * (policy.size(1) - policy.sum(dim=1)) / policy.size(1)
    #reward = reward_acc + reward_fcost + reward_rtcost

    reward = (patch_groundtruth == policy).sum(dim=1).float() / policy.size(1)
    reward = reward.unsqueeze(1)
    return reward.float(), reward_acc

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
    agent.fc = torch.nn.Linear(num_ftrs, 16)

    return agent
