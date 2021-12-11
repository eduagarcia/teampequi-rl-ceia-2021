import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 2


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "SAC",
        name="SAC_2",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 3,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            
            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "flatten_branched": True,
                "single_player": True,
            },
            "buffer_size":int(1e5),
            
        },
        stop={
            #"timesteps_total": 18000,  # 20M
             "time_total_s": 86400, # 24h
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
