import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms

from models.evaulators import PPO_Evaluator
from models import OCTrainer
from utils import *


class OptionCritic:
    def __init__(self, env: gym.Env, logger, writer, args):
        """
        This is a naive PPO wrapper that includes all necessary training pipelines for HRL.
        This trains SF network and train PPO according to the extracted features by SF network
        """
        self.env = env

        # define buffers and sampler for Monte-Carlo sampling
        self.sampler = OnlineSampler(
            training_envs=self.env,
            state_dim=args.s_dim,
            feature_dim=args.sf_dim,
            action_dim=args.a_dim,
            hc_action_dim=args.num_vector + 1,
            agent_num=args.agent_num,
            min_option_length=args.min_option_length,
            min_cover_option_length=args.min_cover_option_length,
            episode_len=args.episode_len,
            episode_num=args.episode_num,
            num_cores=args.num_cores,
            gamma=args.gamma,
        )

        # object initialization
        self.logger = logger
        self.writer = writer
        self.args = args

        # param initialization
        self.curr_epoch = 0

        # SF checkpoint b/c plotter will only be used
        self.sf_path, self.op_path, self.hc_path, self.oc_path, self.ppo_path = (
            self.logger.checkpoint_dirs
        )

        self.plotter = Plotter(
            grid_size=args.grid_size,
            img_tile_size=args.img_tile_size,
            sf_path=self.sf_path,
            op_path=self.op_path,
            hc_path=self.hc_path,
            oc_path=self.oc_path,
            ppo_path=self.ppo_path,
            log_dir=logger.log_dir,
            device=args.device,
        )

        ### Define evaulators tailored for each process
        # each evaluator has slight deviations
        self.oc_evaluator = PPO_Evaluator(
            logger=logger,
            writer=writer,
            training_env=self.env,
            plotter=self.plotter,
            renderPlot=args.rendering,
            render_fps=args.render_fps,
            dir=self.ppo_path,
            log_interval=args.ppo_log_interval,
            eval_ep_num=10,
        )

    def run(self):
        self.train_ppo()
        torch.cuda.empty_cache()

    def train_ppo(self):
        self.sampler.initialize(episode_num=self.args.oc_episode_num)

        ### Call network param and run
        self.oc_network = call_ocNetwork(self.args)
        print_model_summary(self.oc_network, model_name="Option-Critic model")
        if not self.args.import_oc_model:
            oc_trainer = OCTrainer(
                policy=self.oc_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.oc_evaluator,
                epoch=self.curr_epoch + self.args.oc_epoch,
                init_epoch=self.curr_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.oc_log_interval,
                grid_type=self.args.grid_type,
            )
            final_epoch = oc_trainer.train()
        else:
            final_epoch = self.curr_epoch + self.args.PPO_epoch

        self.curr_epoch += final_epoch