import os
import pickle
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch
# import d4rl
import matplotlib.pyplot as plt

import absl.app
import absl.flags
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorboardX import SummaryWriter

from conservative_sac import ConservativeSAC
from replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch
from model import TanhGaussianPolicy, FullyConnectedQFunction, VAE
from sampler import StepSampler, TrajSampler
from utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from utils import WandBLogger
from viskit.logging import logger, setup_logger

FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-expert-v2',
    max_traj_length=1000,
    seed=1,
    device='cuda:1',
    save_model=True,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=2000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=8,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),

    beta=1e-3,
    model_path='weights/model_halfcheetah-medium-expert-v2_seed0.pkl',
    knock_mode='n',
    
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    knock_dict = ['n', 's', 'm', 'l']
    assert FLAGS.knock_mode in knock_dict

    if FLAGS.knock_mode == 'n':
        knock_level = 0
    elif FLAGS.knock_mode == 's':
        knock_level = 1
    elif FLAGS.knock_mode == 'm':
        knock_level = 2
    else:
        knock_level = 3

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length, test_mode=True, seed=FLAGS.seed)

    path = FLAGS.model_path

    file = open(path, 'rb')
    sac = pickle.load(file)

    sac.torch_to_device(FLAGS.device)

    from KLversion.model import SamplerPolicy
    sampler_policy = SamplerPolicy(sac.policy, FLAGS.device)

    trajs = eval_sampler.sample(
        sampler_policy, FLAGS.eval_n_trajs, deterministic=True, knock_level=knock_level
    )

    average_return = np.mean([np.sum(t['rewards']) for t in trajs])
    average_traj_length = np.mean([len(t['rewards']) for t in trajs])
    average_normalizd_return = np.mean(
        [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
    )

    print("number of eval trajs: {}\n"
          "model_seed: {}\n"
          "average_return: {}\n"
          " average_traj_length: {}\n"
          " average_normalizd_return: {}".format(FLAGS.eval_n_trajs, FLAGS.seed,
        average_return, average_traj_length, average_normalizd_return))


if __name__ == '__main__':
    absl.app.run(main)
