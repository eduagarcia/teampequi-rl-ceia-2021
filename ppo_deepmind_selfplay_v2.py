from pickle5 import pickle
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks
from soccer_twos import EnvType

from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
import os

from utils import create_rllib_env

class PrioritizedSelfPlay(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.current_team = "blue_team"
        self.opponent_team = "yellow_team"
        self.opponent_iter = 0
        self.policy_history = []

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        ## Save current checkpoint
        checkpoint_dir = os.path.join(trainer.logdir, 'selfplay_checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        weights_filepath = os.path.join(checkpoint_dir, str(trainer.iteration)+'.pkl')
        
        with open(weights_filepath, 'wb') as f:
            weights = trainer.get_weights([self.current_team])[self.current_team]
            pickle.dump(weights, f)

        #Save the current result
        hist = {
            'iteraction': trainer.iteration,
            'weights_filepath': weights_filepath,
            'current_team': self.current_team,
            'opponent_team': self.opponent_team,
            'opponent_iter': self.opponent_iter,
            'result': result['policy_reward_mean'][self.current_team]
        }
        print('Policy Saved:', hist)
        self.policy_history.append(hist)

        #sample opponent policy
        opponent_policy = np.random.choice(self.policy_history)
        print('sampled opponent policy:', opponent_policy)
        with open(opponent_policy['weights_filepath'], 'rb') as f:
            opponent_weights = pickle.load(f)
        self.opponent_iter = opponent_policy['iteraction']

        #swap teams for next iter:
        temp = self.current_team
        self.current_team = self.opponent_team
        self.opponent_team = temp

        trainer.set_weights({self.current_team: trainer.get_weights([self.opponent_team])[self.opponent_team],
                            self.opponent_team: opponent_weights,
                            })

        self.counter += 1

NUM_ENVS_PER_WORKER = 3


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    

    analysis = tune.run(
        "PPO",
        name="PPO_deepmind_selfplay_v2",
        config={
            "num_gpus": 1,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "ignore_worker_failures": True,
            "train_batch_size": 4000,
            "sgd_minibatch_size": 256,
            "lr": 3e-4,
            "lambda": .95,
            "gamma": .99,
            "clip_param": 0.2,
            "num_sgd_iter": 20,
            "rollout_fragment_length": 200,
            "model": {
                "fcnet_hiddens": [512, 512],
                "vf_share_layers": False
            },
            "multiagent": {
                "policies": {
                    "blue_team": (None, obs_space, act_space, {}),
                    "yellow_team": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": lambda id: "blue_team" if id < 2 else "yellow_team"
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
            },
            'callbacks': MultiCallbacks([PrioritizedSelfPlay])
        },
        stop={
            "timesteps_total": 15000000,  # 15M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
