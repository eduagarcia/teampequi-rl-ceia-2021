import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = "./ray_results/PPO_alphastar_v3/PPO_Soccer_50cc5_00000_0_2021-12-06_17-51-42/selfplay_checkpoints/main_agent_0_72863_striker-433.pkl"
POLICY_NAME = "main_agent_0"  # this may be useful when training with selfplay


class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env, checkpoint='default'):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        self.name = "SELfPLAYV3"
        ray.init(ignore_reinit_error=True)

        checkpoints, extra_checkpoints = self._get_checkpoints()

        if checkpoint == 'default':
            checkpoint_path = CHECKPOINT_PATH
        elif checkpoint == 'latest':
            checkpoint_path = checkpoints[max(list(checkpoints.keys()))]
        else:
            checkpoint_path = checkpoints[int(checkpoint)]

        config = ppo.DEFAULT_CONFIG.copy()
        config['num_gpus'] = 0
        config['num_workers'] = 0
        config['model']["fcnet_hiddens"] = [512, 512]

        # create the Trainer from config
        # load state from checkpoint
        #agent.restore(checkpoint_path)
        # get policy for evaluation
        self.wrappers = []
        wrapped_env = env
        for wrapper in self.wrappers:
            wrapped_env = wrapper(wrapped_env)

        self.policies = []
        for t in ['striker', 'goalie']:
            with open(checkpoint_path.replace('striker', t), 'rb') as f:
                weights = pickle.load(f)
                policy = PPOTorchPolicy(wrapped_env.observation_space, wrapped_env.action_space, config)
                policy.set_weights(weights)
                self.policies.append(policy)

        

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """The act method is called when the agent is asked to act.
        Args:
            observation: a dictionary where keys are team member ids and
                values are their corresponding observations of the environment,
                as numpy arrays.
        Returns:
            action: a dictionary where keys are team member ids and values
                are their corresponding actions, as np.arrays.
        """
        obs = observation
        for wrapper in self.wrappers:
            obs = wrapper.preprocess_obs(obs)

        actions = {}
        for player_id in obs:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # as we only need the action, we discard the other elements
            actions[player_id], *_ = self.policies[player_id].compute_single_action(
                obs[player_id]
            )
        return actions

    def _get_checkpoints(self):
        checkpoints = {}
        extra_checkpoints = {
            'league_exploiter': {
                0: {},
                1: {}
            },
            'main_exploiter': {
                0: {}
            }
        }
        chepoints_dir = os.path.dirname(CHECKPOINT_PATH)
        for filename in os.listdir(chepoints_dir):
            if filename.endswith('.pkl') and 'striker' in filename:
                checkpoint = int(filename.split('-')[-1][:-len('.pkl')])
                checkpoint_path = os.path.join(chepoints_dir, filename)
                if filename.startswith('main_agent_0'):
                    checkpoints[checkpoint] = checkpoint_path
                else:
                    if filename.startswith('league_exploiter_agent_0'):
                        data = extra_checkpoints['league_exploiter'][0]
                    elif filename.startswith('league_exploiter_agent_1'):
                        data = extra_checkpoints['league_exploiter'][1]
                    elif filename.startswith('main_exploiter_agent_0'):
                        data = extra_checkpoints['main_exploiter'][0]
                    else:
                        continue
                    
                    data[checkpoint] = checkpoint_path
        return checkpoints, extra_checkpoints