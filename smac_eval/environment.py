from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sample_factory.envs.env_utils import register_env
from sample_factory.utils.attr_dict import AttrDict
from smacv2.env import StarCraftCapabilityEnvWrapper

from smac_eval.training_config import SMACv2Config


def get_smacv2_obs_shape(cfg: SMACv2Config) -> int:
    smac_kwargs = cfg.__dict__
    env = StarCraftCapabilityEnvWrapper(**smac_kwargs)
    try:
        env.reset()
        sample_obs = env.get_obs_agent(0)
        obs_shape = sample_obs.shape[0]
    finally:
        env.close()
    return int(obs_shape)


class SMACv2Env(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None):
        super().__init__()
        self.episode_reward_sum = 0
        self.name = full_env_name
        if isinstance(cfg, AttrDict):
            self.cfg = cfg
        else:
            self.cfg = cfg.__dict__ or {}
        self.env_config = env_config or {}

        # Unpack SMACv2 config
        smac_kwargs = self.cfg['environment']['env_extra_config']
        self.env = StarCraftCapabilityEnvWrapper(**smac_kwargs)

        env_info = self.env.get_env_info()
        self.num_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']

        # Define spaces
        self.obs_shape = self.cfg['environment']['obs_shape']
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Dict({
            'obs':         spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(self.n_actions,), dtype=np.int8),
        })
        self.is_multiagent = True
        self.render_mode = render_mode

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Reset the underlying smac-v2 environment
        agent_obs, _ = self.env.reset()

        # Get the initial available actions for all agents
        avail_actions = self.env.get_avail_actions()

        observations = []
        for obs, action in zip(agent_obs, avail_actions):
            obs_dict = {
                'obs':         np.array(obs, dtype=np.float32),
                'action_mask': np.array(action, dtype=np.int8),
            }
            observations.append(obs_dict)

        return observations, {}

    def step(self, actions):
        """
        :param actions: list/np.array of length num_agents
        :return: observation, reward, terminated, truncated, infos
        """

        reward, terminated, info = self.env.step(actions)

        agent_obs = self.env.get_obs()
        avail_actions = self.env.get_avail_actions()

        num_dead = 0
        for acts in avail_actions:
            num_dead += int(acts[0] == 1)

        observations = []
        for obs, action in zip(agent_obs, avail_actions):
            obs_dict = {
                'obs':         np.array(obs, dtype=np.float32),
                'action_mask': np.array(action, dtype=np.int8),
            }
            observations.append(obs_dict)
        rewards = [reward] * self.num_agents

        terminated_flags = [terminated] * self.num_agents

        truncated = info.get('episode_limit', False)

        truncated_flags = [truncated] * self.num_agents

        info['episode_extra_stats'] = dict()
        self.episode_reward_sum += reward

        if any(terminated_flags) or any(truncated_flags):
            info['episode_extra_stats'] |= {
                'episode_reward': self.episode_reward_sum
            }
            self.episode_reward_sum = 0

            # Get stats from env and add them to info
            if self.env.env.battles_game != 0:
                info['episode_extra_stats'] |= self.env.get_stats()

        infos = [info] * self.num_agents

        # Environment auto-reset
        if terminated or truncated:
            observations, _ = self.reset()

        return observations, rewards, terminated_flags, truncated_flags, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_smacv2_env(full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None):
    return SMACv2Env(full_env_name, cfg, env_config, render_mode)


def register_smacv2_env(env_name):
    register_env(env_name, make_smacv2_env)
