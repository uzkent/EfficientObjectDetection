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
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import tqdm
import utils
import torch.optim as optim
from torch.distributions import Multinomial, Bernoulli
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='PatchDrop Pre-Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--ckpt_hr_cl', help='checkpoint directory for the high resolution classifier')
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

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):
    # This steps trains the policy network and high resolution classifier jointly
    # rnet represents the high resolution classifier
    # rnet_lr represents the low resolution classifier

    agent.train()
    matches, rewards, rewards_baseline, policies = [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs = Variable(inputs)
        if not args.parallel:
    	    inputs = inputs.cuda()

        # Get the low resolution agent images
        probs = F.sigmoid(agent(inputs))
        probs = probs*args.alpha + (1-args.alpha) * (1-probs)

        # Sample the policies from the Bernoulli distribution characterized by agent's output
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        # Agent sampled high resolution images
        patch_groundtruth = utils.read_groundtruth(targets)

        # Find the reward for baseline and sampled policy
        reward_map, _ = utils.compute_reward(patch_groundtruth, policy_map.data)
        reward_sample, _ = utils.compute_reward(patch_groundtruth, policy_sample.data)
        advantage = reward_sample.cuda().float() - reward_map.cuda().float()

        # Find the loss for only the policy network
        loss = -distr.log_prob(policy_sample)
        loss = loss * Variable(advantage).expand_as(policy_sample)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())
        policies.append(policy_sample.data.cpu())

    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)

    print 'Train: %d | Rw: %.2E | S: %.3f | V: %.3f | #: %d'%(epoch, reward, sparsity, variance, len(policy_set))

    log_value('train_reward', reward, epoch)
    log_value('train_sparsity', sparsity, epoch)
    log_value('train_variance', variance, epoch)
    log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    log_value('train_unique_policies', len(policy_set), epoch)

def test(epoch):
    # Test the policy network and the high resolution classifier
    agent.eval()

    matches, rewards, reward_accuracies, policies = [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs = Variable(inputs, volatile=True)
        inputs_dem = inputs.clone()
        if not args.parallel:
            inputs = inputs.cuda()

        # Get the low resolution agent images
        probs = F.sigmoid(agent(inputs))

        # Sample the policy from the agents output
        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0
        policy = Variable(policy)

        # Agent sampled high resolution images
        patch_groundtruth = utils.read_groundtruth(targets)

        # Find the reward for baseline and sampled policy
        reward, reward_acc = utils.compute_reward(patch_groundtruth, policy.data)

        rewards.append(reward)
        reward_accuracies.append(reward_acc)
        policies.append(policy.data)

    accuracy = torch.stack(reward_accuracies).mean()
    reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards)

    print 'Test - Acc: %.3f | Rw: %.2E | S: %.3f | V: %.3f | #: %d'%(accuracy, reward, sparsity, variance, len(policy_set))

    log_value('test_reward', reward, epoch)
    log_value('test_accuracy', accuracy, epoch)
    log_value('test_sparsity', sparsity, epoch)
    log_value('test_variance', variance, epoch)
    log_value('test_unique_policies', len(policy_set), epoch)

    # save the model --- agent
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()

    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': reward,
    }

    torch.save(state, args.cv_dir+'/ckpt_E_%d_R_%.2E'%(epoch, reward))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
testloader = torchdata.DataLoader(testset, batch_size=10, shuffle=True, num_workers=0)
agent = utils.get_model(args.model)

# ------- PatchDrop Action Space for fMoW -----------------------
start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print 'loaded agent from', args.load

# Parallelize the models if multiple GPUs available - Important for Large Batch Size
if args.parallel:
    agent = nn.DataParallel(agent)

agent.cuda()

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Save the args to the checkpoint directory
configure(args.cv_dir+'/log', flush_secs=5)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch % 10 == 0:
        test(epoch)
