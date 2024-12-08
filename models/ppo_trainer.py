import time
import random
import math
import os
import wandb
import pickle
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from log.wandb_logger import WandbLogger
from models.policy.base_policy import BasePolicy
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.evaulators.sf_evaluator import Evaluator


# model-free policy trainer
class PPOTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 1000,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_interval: int = 2,
        grid_type: int = 0,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        # training parameters
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._init_epoch = init_epoch
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.std_limit = 0.5
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.grid_type = grid_type

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # train loop
        for e in trange(self._init_epoch, self._epoch, desc=f"PPO Epoch"):
            ### training loop
            self.policy.train()
            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                batch, sample_time = self.sampler.collect_samples(
                    self.policy,
                    grid_type=self.grid_type,
                )

                loss_dict, update_time = self.policy.learn(batch)

                # Logging further info
                self.num_env_steps += len(batch["rewards"])
                loss_dict["PPO/sample_time"] = sample_time
                loss_dict["PPO/update_time"] = update_time

                self.write_log(loss_dict, iter_idx=int(e * self._step_per_epoch + it))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            ### Eval Loop
            self.policy.eval()
            rew_mean, rew_std, _, _ = self.evaluator(
                self.policy,
                env_step=self.num_env_steps,
                epoch=e + 1,
                iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                dir_name="PPO",
                grid_type=self.grid_type,
            )

            self.last_reward_mean.append(rew_mean)
            self.last_reward_std.append(rew_std)

            self.save_model(e + 1)

        self.logger.print(
            "total PPO training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

        return self._epoch

    def save_model(self, e):
        # save checkpoint
        if e % self.log_interval == 0:
            self.policy.save_model(self.logger.checkpoint_dirs[1], e)

        # save the best model
        if (
            np.mean(self.last_reward_mean) > self.last_max_reward
            and np.mean(self.last_reward_std) <= self.std_limit
        ):
            self.policy.save_model(self.logger.log_dirs[1], e, is_best=True)
            self.last_max_reward = np.mean(self.last_reward_mean)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values for each key
        sum_dict = {key: 0 for key in dict_list[0].keys()}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                sum_dict[key] += value

        # Calculate the average for each key
        avg_dict = {key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()}

        return avg_dict

    def write_log(self, logging_dict: dict, iter_idx: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(int(iter_idx), display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, int(iter_idx))