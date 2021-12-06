import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

import soccer_twos
from soccer_twos import AgentInterface, EnvType


ALGORITHM = "PPO"
CHECKPOINT_PATH = "ray_results/checkpoint_005601/checkpoint-5601"
dir_path = os.path.dirname(os.path.realpath(__file__))
CHECKPOINT_PATH = os.path.join(dir_path, CHECKPOINT_PATH)
POLICY_NAME = "policy_01"  # this may be useful when training with selfplay

from ray.rllib.agents.ppo import ppo

def policy_mapping_fn(agent_id):
    if agent_id == 0:
        return "policy_01" # Choose 01 policy for agent_01
    else:
        return np.random.choice(["policy_01", "policy_02", "policy_03", "policy_04"],1,
                                p=[.8, .2/3, .2/3, .2/3])[0]

temp_env = soccer_twos.make(variation=EnvType.multiagent_player)
obs_space = temp_env.observation_space
act_space = temp_env.action_space
temp_env.close()

default_config = ppo.DEFAULT_CONFIG.copy()


custom_config = {
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "log_level": "INFO",
        "framework": "torch",
        "ignore_worker_failures": True,
        "train_batch_size": 1024,
        #"sgd_minibatch_size": 10000,
        "lr": 3e-4,
        "lambda": .95,
        "gamma": .998,
        "entropy_coeff": 0.01,
        "kl_coeff": 1.0,
        "clip_param": 0.2,
        "num_sgd_iter": 10,
        "observation_filter": "NoFilter",  # breaks the action mask
        #"vf_share_layers": True,
        "vf_loss_coeff": 1e-4,    #VF loss is error^2, so it can be really out of scale compared to the policy loss. 
                                #Ref: https://github.com/ray-project/ray/issues/5278
        "vf_clip_param": 100.0,
        "multiagent": {
            "policies": {
                "policy_01": (None, obs_space, act_space, {}),
                "policy_02": (None, obs_space, act_space, {}),
                "policy_03": (None, obs_space, act_space, {}),
                "policy_04": (None, obs_space, act_space, {})
            },
            "policy_mapping_fn": tune.function(policy_mapping_fn),
            #"policies_to_train": ["policy_01"]
        },
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": 1,
            "variation": EnvType.multiagent_player,
        },
    }

default_config.update(custom_config)

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
        self.name = "GoiabaFinalForm"
        ray.init(ignore_reinit_error=True)

        config = default_config.copy()
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
        agent.restore(CHECKPOINT_PATH)
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
