import pickle
import os
from typing import Dict

from gym_unity.envs import ActionFlattener
import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv, EnvType
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface


ALGORITHM = "SAC"
CHECKPOINT_PATH = "ray_results/checkpoint_003000/checkpoint-3000"
dir_path = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_PATH = os.path.join(dir_path, CHECKPOINT_PATH)
POLICY_NAME = "SAC_1"  # this may be useful when training with selfplay

class RayAgent(AgentInterface):
    """
    RayAgent is an agent that uses ray to train a model.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RayAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        self.flattener = ActionFlattener(env.action_space.nvec)
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
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

        config["env_config"] = {
            "flatten_branched": True,
            "multiagent": False,
            "num_envs_per_worker": 2,
            "single_player": True,
        }

        # create a dummy env since it's required but we only care about the policy
        from utils import create_rllib_env

        tune.registry.register_env("Soccer", create_rllib_env)

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(CHECKPOINT_PATH)
        # get policy for evaluation
        self.policy = agent.get_policy()

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
            action, *_ = self.policy.compute_single_action(
                observation[player_id]
            )
            actions[player_id] = self.flattener.lookup_action(action)
        return actions
