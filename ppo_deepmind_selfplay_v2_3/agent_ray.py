import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "PPO"
CHECKPOINT_PATH = "./ray_results/PPO_deepmind_selfplay_v2_3/PPO_Soccer_7eed3_00000_0_2021-12-01_03-18-27/checkpoint_000300/checkpoint-300"
POLICY_NAME = "current_team"  # this may be useful when training with selfplay


class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env, checkpoint='latest'):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        ray.init(ignore_reinit_error=True)

        checkpoints = self._get_checkpoints()
        if checkpoint == 'latest':
            checkpoint_path = checkpoints[max(list(checkpoints.keys()))]
        else:
            checkpoint_path = checkpoints[int(checkpoint)]

        # Load configuration from checkpoint file.
        config_path = ""
        if checkpoint_path:
            config_dir = os.path.dirname(checkpoint_path)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0

        # create a dummy env since it's required but we only care about the policy
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(checkpoint_path)
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)

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
        actions = {}
        for player_id in observation:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # as we only need the action, we discard the other elements
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions

    def _get_checkpoints(self):
        checkpoints = {}
        chepoints_dir = os.path.join(os.path.dirname(CHECKPOINT_PATH), '..')
        for folder in os.listdir(chepoints_dir):
            if folder.startswith('checkpoint_'):
                checkpoint = int(folder.split('_')[-1])
                checkpoint_path = os.path.join(chepoints_dir, folder, f'checkpoint-{checkpoint}')
                checkpoints[checkpoint] = checkpoint_path
        return checkpoints