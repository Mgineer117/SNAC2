from itertools import chain
from typing import Final, Literal, TypeAlias, TypedDict
from typing import Any, Iterable, SupportsFloat, TypeVar

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
import numpy as np
import random
from numpy.typing import NDArray

from gym_multigrid.core.constants import *
from gym_multigrid.utils.window import Window
from gym_multigrid.core.agent import Agent, PolicyAgent, AgentT, MazeActions
from gym_multigrid.core.grid import Grid
from gym_multigrid.core.object import Goal, Wall
from gym_multigrid.core.world import RoomWorld
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.typing import Position


class Maze(MultiGridEnv):
    """
    Environment for capture the flag with multiple agents with N blue agents and M red agents.
    """

    def __init__(
        self,
        grid_type: int = 0,
        width=20,
        height=20,
        max_steps=1000,
        see_through_walls=False,
        agent_view_size=7,
        partial_observability=False,
        render_mode=None,
        highlight_visible_cells=True,
        tile_size=32,
    ):
        self.grid_type = grid_type

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.world = RoomWorld
        self.actions_set = MazeActions

        see_through_walls: bool = False

        self.agents = [
            Agent(
                self.world,
                color="blue",
                bg_color="light_blue",
                view_size=agent_view_size,
                actions=self.actions_set,
                type="agent",
            )
        ]

        # Define positions for goals and agents
        self.goal_positions = [(17, 17)]
        self.agent_positions = [(12, 7)]

        self.grids = {}
        self.grid_imgs = {}

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            agents=self.agents,
            agent_view_size=agent_view_size,
            actions_set=self.actions_set,
            partial_obs=partial_observability,
            world=self.world,
            render_mode=render_mode,
            highlight_visible_cells=highlight_visible_cells,
            tile_size=tile_size,
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height, self.world)

        # Explicit maze structure based on the image
        maze_structure = [
            "####################",
            "#                  #",
            "#            #     #",
            "#       ###  ####  #",
            "#              #   #",
            "#####          #   #",
            "#   #   #      #   #",
            "#   #   #      #   #",
            "#   #   #      #   #",
            "#   #   ########   #",
            "#   #      #   #   #",
            "#   #      #       #",
            "#          #       #",
            "#   #      #   #   #",
            "#   #      #   #####",
            "#   #      #       #",
            "#  #########       #",
            "#                  #",
            "#                  #",
            "####################",
        ]

        # Translate the maze structure into the grid
        for y, row in enumerate(maze_structure):
            for x, cell in enumerate(row):
                if cell == "#":
                    self.grid.set(x, y, Wall(self.world))
                elif cell == " ":
                    self.grid.set(x, y, None)

        # Place the goal
        goal = Goal(self.world, 0)
        self.put_obj(goal, *self.goal_positions[self.grid_type])
        goal.init_pos, goal.cur_pos = self.goal_positions[self.grid_type]

        # Place the agent
        for agent in self.agents:
            self.place_agent(agent, pos=self.agent_positions[self.grid_type])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        obs, info = super().reset(seed=seed, options=options)

        ### NOTE: not multiagent setting
        self.agent_pos = self.agents[0].pos

        ### NOTE: NOT MULTIAGENT SETTING
        observations = {"image": obs[0][:, :, 0:1]}
        return observations, info

    def step(self, actions):
        self.step_count += 1

        ### NOTE: MULTIAGENT SETTING NOT IMPLEMENTED
        actions = [actions]
        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))

        for i in order:
            if (
                self.agents[i].terminated
                or self.agents[i].paused
                or not self.agents[i].started
            ):
                continue

            # Get the current agent position
            curr_pos = self.agents[i].pos
            done = False

            # Rotate left
            if actions[i] == self.actions.left:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, -1)))
                fwd_cell = self.grid.get(*fwd_pos)

                dist_reward = np.linalg.norm(
                    np.array(self.goal_positions[self.grid_type]) - np.array(fwd_pos),
                    ord=2,
                )

                dist_norm_reward = dist_reward / np.linalg.norm(
                    [self.width, self.height], ord=2
                )

                rewards += 0.5 * dist_norm_reward  # 0 ~ 0.5 of weak reward signals

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards += self._reward(i, rewards, 1)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards += self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Rotate right
            elif actions[i] == self.actions.right:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (0, +1)))
                fwd_cell = self.grid.get(*fwd_pos)

                dist_reward = np.linalg.norm(
                    np.array(self.goal_positions[self.grid_type]) - np.array(fwd_pos),
                    ord=2,
                )

                dist_norm_reward = dist_reward / np.linalg.norm(
                    [self.width, self.height], ord=2
                )

                rewards += 0.5 * dist_norm_reward  # 0 ~ 0.5 of weak reward signals

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards += self._reward(i, rewards, 1)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards += self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            # Move forward
            elif actions[i] == self.actions.up:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (-1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)

                dist_reward = np.linalg.norm(
                    np.array(self.goal_positions[self.grid_type]) - np.array(fwd_pos),
                    ord=2,
                )

                dist_norm_reward = dist_reward / np.linalg.norm(
                    [self.width, self.height], ord=2
                )

                rewards += 0.5 * dist_norm_reward  # 0 ~ 0.5 of weak reward signals

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards += self._reward(i, rewards, 1)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards += self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif actions[i] == self.actions.down:
                # Get the contents of the cell in front of the agent
                fwd_pos = tuple(a + b for a, b in zip(curr_pos, (+1, 0)))
                fwd_cell = self.grid.get(*fwd_pos)

                dist_reward = np.linalg.norm(
                    np.array(self.goal_positions[self.grid_type]) - np.array(fwd_pos),
                    ord=2,
                )

                dist_norm_reward = dist_reward / np.linalg.norm(
                    [self.width, self.height], ord=2
                )

                rewards += 0.1 * dist_norm_reward  # 0 ~ 0.5 of weak reward signals

                if fwd_cell is not None:
                    if fwd_cell.type == "goal":
                        done = True
                        rewards += self._reward(i, rewards, 1)
                    elif fwd_cell.type == "switch":
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                    elif fwd_cell.type == "ball":
                        rewards += self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            elif actions[i] == self.actions.stay:
                # Get the contents of the cell in front of the agent
                fwd_pos = curr_pos
                fwd_cell = self.grid.get(*fwd_pos)
                self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                assert False, "unknown action"

        ### NOTE: not multiagent setting
        self.agent_pos = self.agents[0].pos

        terminated = done
        truncated = True if self.step_count >= self.max_steps else False

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [
                self.grid.encode_for_agents(agent_pos=self.agents[i].pos)
                for i in range(len(actions))
            ]

        obs = [self.world.normalize_obs * ob for ob in obs]

        ### NOTE: not multiagent
        observations = {"image": obs[0][:, :, 0:1]}

        return observations, rewards, terminated, truncated, {}
