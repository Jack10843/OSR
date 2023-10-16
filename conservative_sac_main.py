import os
import pickle
import random
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch
import d4rl

import absl.app
import absl.flags
from tensorboardX import SummaryWriter

from KLversion.conservative_sac import ConservativeSAC
from KLversion.replay_buffer import batch_to_torch, get_d4rl_dataset, subsample_batch
from KLversion.model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy, VAE
from KLversion.sampler import StepSampler, TrajSampler
from KLversion.utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from KLversion.utils import WandBLogger
from viskit.logging import logger, setup_logger

FLAGS_DEF = define_flags_with_default(
    env='hopper-medium-expert-v2',
    max_traj_length=1000,
    seed=1,
    device='cuda:0',
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
    eval_n_trajs=5,

    cql=ConservativeSAC.get_default_config(),
    logging=WandBLogger.get_default_config(),

    beta=1e-3,
    limited_valuable=False,
    limited_rate=0.6
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

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    if FLAGS.limited_valuable:
        env_name = FLAGS.env.split("-")
        eval_sampler_r = TrajSampler(gym.make(env_name[0] + "-random-v2").unwrapped, FLAGS.max_traj_length)
        random_dataset = get_d4rl_dataset(eval_sampler_r.env)
        random_dataset['rewards'] = random_dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
        random_dataset['actions'] = np.clip(random_dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    policy = TanhGaussianPolicy(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
    )

    qf1 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction(
        eval_sampler.env.observation_space.shape[0],
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
    )
    target_qf2 = deepcopy(qf2)

    inverse_dynamics = VAE(
        eval_sampler.env.observation_space.shape[0] * 2,
        eval_sampler.env.action_space.shape[0],
        FLAGS.device
    )
    if FLAGS.limited_valuable:
        writer = SummaryWriter('limited_valuable{}/cql_beta{}_alpha10_lambda1/{}_runs/mfSDC_{}_seed{}'.format(FLAGS.limited_rate, FLAGS.beta, FLAGS.env, FLAGS.env, FLAGS.seed))
    else:
        writer = SummaryWriter(
            'beta{}_alpha12_lambda2/{}_runs/cql_{}_seed{}'.format(FLAGS.beta, FLAGS.env, FLAGS.env, FLAGS.seed))

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf1, qf2, target_qf1, target_qf2, inverse_dynamics, FLAGS.device, beta=FLAGS.beta)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    print('pretraining the inverse dynamics....')
    for i in range(int(1000000 / FLAGS.batch_size) * 2):
        batch = subsample_batch(dataset, FLAGS.batch_size)
        batch = batch_to_torch(batch, FLAGS.device)
        loss = sac.train_inv(batch)
        print("inv_loss: {}".format(loss))

    # sac.inverse_dynamics.eval()
    best = 0

    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch * 2):
                if FLAGS.limited_valuable:
                    rand_batch = subsample_batch(random_dataset, FLAGS.batch_size)
                    rand_batch = batch_to_torch(rand_batch, FLAGS.device)
                    batch = subsample_batch(dataset, FLAGS.batch_size)
                    batch = batch_to_torch(batch, FLAGS.device)

                    rdm = torch.rand(FLAGS.batch_size, 1).to(FLAGS.device)
                    mask = torch.where(rdm < FLAGS.limited_rate, 1, 0).to(FLAGS.device)

                    mask_o = mask.expand(FLAGS.batch_size, eval_sampler.env.observation_space.shape[0])
                    mask_a = mask.expand(FLAGS.batch_size, eval_sampler.env.action_space.shape[0])

                    batch['observations'] = mask_o * rand_batch['observations'] + (1 - mask_o) * batch['observations']
                    batch['actions'] = mask_a * rand_batch['actions'] + (1 - mask_a) * batch['actions']
                    batch['rewards'] = mask.squeeze() * rand_batch['rewards'] + (1 - mask.squeeze()) * batch['rewards']
                    batch['next_observations'] = mask_o * rand_batch['next_observations'] + (1 - mask_o) * batch['next_observations']
                    batch['dones'] = mask.squeeze() * rand_batch['dones'] + (1 - mask.squeeze()) * batch['dones']
                    metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))
                else:
                    batch = subsample_batch(dataset, FLAGS.batch_size)
                    batch = batch_to_torch(batch, FLAGS.device)
                    metrics.update(prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac'))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                writer.add_scalar('average_return', np.mean([np.sum(t['rewards']) for t in trajs]), epoch)
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                )
                writer.add_scalar('average_normalizd_return', np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                ), epoch)
                score = np.mean([np.sum(t['rewards']) for t in trajs])
                if FLAGS.save_model and score > 0.99 * best and epoch >= 150:
                    # save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    # wandb_logger.save_pickle(save_data, 'model.pkl')
                    best = score
                    if FLAGS.limited_valuable:
                        file = open('limited_valuable{}/beta{}_alpha10_lambda1/weights/cql_model_{}_seed{}.pkl'.format(
                            FLAGS.limited_rate, FLAGS.beta, FLAGS.env, FLAGS.seed), 'wb')
                    else:
                        file = open('beta{}_alpha12_lambda2/weights/cql_model_{}_seed{}.pkl'.format(FLAGS.beta, FLAGS.env, FLAGS.seed), 'wb')
                    pickle.dump(sac, file)
                    file.close()

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    # if FLAGS.save_model:
    #     save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
    #     wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
