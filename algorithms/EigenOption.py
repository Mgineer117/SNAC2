import torch
import torch.nn as nn
import random
import uuid
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms

from models.evaulators import (
    SF_Evaluator,
    OP_Evaluator,
    UG_Evaluator,
    HC_Evaluator,
)
from models import SFTrainer, OPTrainer, HCTrainer
from utils import *
from utils.call_env import call_env


class EigenOption:
    """
    The difference from SNAC is two-fold:
        - it does not have reward-predictive feature
        - it does only pick top n number of eigenvectors
            - this heuristics ignores other info of eigs
    """

    def __init__(
        self, env: gym.Env, sf_network: nn.Module, prev_epoch: int, logger, writer, args
    ):
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
        self.sf_network = sf_network
        self.logger = logger
        self.writer = writer
        self.args = args

        # param initialization
        self.curr_epoch = prev_epoch

        # SF checkpoint b/c plotter will only be used
        self.sf_path, self.ppo_path, self.op_path, self.ug_path, self.hc_path = (
            self.logger.checkpoint_dirs
        )

        self.plotter = Plotter(
            grid_size=args.grid_size,
            img_tile_size=args.img_tile_size,
            sf_path=self.sf_path,
            ppo_path=self.ppo_path,
            op_path=self.op_path,
            hc_path=self.hc_path,
            log_dir=logger.log_dir,
            device=args.device,
        )

        ### Define evaulators tailored for each process
        # each evaluator has slight deviations
        evaluator_params = {
            "logger": logger,
            "writer": writer,
            "training_env": self.env,
            "plotter": self.plotter,
            "gridPlot": True,
            "renderPlot": args.rendering,
            "render_fps": args.render_fps,
            "eval_ep_num": args.eval_episodes,
        }
        if args.env_name in ("PointNavigation"):
            evaluator_params.update({"gridPlot": False})

        self.op_evaluator = OP_Evaluator(
            dir=self.op_path, log_interval=args.op_log_interval, **evaluator_params
        )
        self.hc_evaluator = HC_Evaluator(
            dir=self.hc_path,
            log_interval=args.hc_log_interval,
            min_option_length=args.min_option_length,
            **evaluator_params,
        )

    def run(self):
        self.train_op()
        torch.cuda.empty_cache()
        self.train_hc()
        torch.cuda.empty_cache()

    def train_op(self):
        """
        This discovers the eigenvectors via clustering for each of reward and state decompositions.
        --------------------------------------------------------------------------------------------
        """
        if not self.args.import_op_model:
            self.option_vals, self.options, _ = get_eigenvectors(
                self.env,
                self.sf_network,
                self.sampler,
                self.plotter,
                self.args,
                draw_map=self.args.draw_map,
            )
            self.op_network = call_opNetwork(
                self.sf_network, self.args, self.option_vals, self.options
            )
            print_model_summary(self.op_network, model_name="OP model")
            op_trainer = OPTrainer(
                policy=self.op_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.op_evaluator,
                val_options=self.op_network._option_vals,
                epoch=self.curr_epoch + self.args.OP_epoch,
                init_epoch=self.curr_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.op_log_interval,
                grid_type=self.args.grid_type,
            )

            if self.sf_network.psiNet is None:
                final_epoch = op_trainer.train()
            else:
                final_epoch = op_trainer.evaluate(epoch=3)
                print(f"\n+++Psi-Network exists, so no OP training takes place+++")

        else:
            self.op_network = call_opNetwork(self.sf_network, self.args)
            final_epoch = self.curr_epoch + self.args.OP_epoch + self.args.Psi_epoch
        self.curr_epoch += final_epoch

    def train_hc(self):
        self.hc_network = call_hcNetwork(
            self.sf_network.feaNet, self.op_network, self.args
        )
        print_model_summary(self.hc_network, model_name="HC model")
        if not self.args.import_hc_model:
            hc_trainer = HCTrainer(
                policy=self.hc_network,
                sampler=self.sampler,
                logger=self.logger,
                writer=self.writer,
                evaluator=self.hc_evaluator,
                prefix="HC",
                epoch=self.curr_epoch + self.args.HC_epoch,
                init_epoch=self.curr_epoch,
                step_per_epoch=self.args.step_per_epoch,
                eval_episodes=self.args.eval_episodes,
                log_interval=self.args.hc_log_interval,
                grid_type=self.args.grid_type,
            )
            hc_trainer.train()
        self.curr_epoch += self.args.HC_epoch
