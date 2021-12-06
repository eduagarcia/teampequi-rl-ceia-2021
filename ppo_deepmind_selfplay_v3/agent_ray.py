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
from wrappers import SingleObsWrapper, MultiagentTeamObsWrapper


ALGORITHM = "PPO"
CHECKPOINT_PATH = "./ray_results/PPO_deepmind_selfplay_v3/PPO_Soccer_690bc_00000_0_2021-12-05_18-07-52/selfplay_checkpoints/main_agent_650"
POLICY_NAME = "main_agent"  # this may be useful when training with selfplay


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

        checkpoints = self._get_checkpoints()

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
        self.wrappers = [SingleObsWrapper, MultiagentTeamObsWrapper]
        wrapped_env = env
        for wrapper in self.wrappers:
            wrapped_env = wrapper(wrapped_env)

        self.policies = []
        for t in ['_striker', '_goalie']:
            with open(checkpoint_path+t+'.pkl', 'rb') as f:
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
        chepoints_dir = os.path.dirname(CHECKPOINT_PATH)
        for filename in os.listdir(chepoints_dir):
            if filename.endswith('.pkl'):
                checkpoint = int(filename.split('_')[-2])
                checkpoint_path = os.path.join(chepoints_dir, "_".join(filename.split('_')[:-1]))
                checkpoints[checkpoint] = checkpoint_path
        return checkpoints