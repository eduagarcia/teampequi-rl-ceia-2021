import ray
from ray import tune
from soccer_twos import EnvType
import numpy as np
import os

from utils import create_rllib_env
from ray.rllib.agents.ppo import ppo
from ray.tune.logger import pretty_print

#Fix ray problem with detection GPU on local worker
ray.get_gpu_ids = lambda *_: [0]

NUM_ENVS_PER_WORKER = 3
CHECKPOINT_PATH = 'ray_results/ppo_selfpay_deepmind'

def policy_mapping_fn(agent_id):
    if agent_id == 0:
        return "policy_01" # Choose 01 policy for agent_01
    else:
        return np.random.choice(["policy_01", "policy_02", "policy_03", "policy_04"],1,
                                p=[.8, .2/3, .2/3, .2/3])[0]
    
ray.init(num_gpus=1)

tune.registry.register_env("Soccer", create_rllib_env)
temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
obs_space = temp_env.observation_space
act_space = temp_env.action_space
temp_env.close()

config = ppo.DEFAULT_CONFIG.copy()

custom_config = {
        "num_gpus": 1,
        "num_workers": 4,
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
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
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "variation": EnvType.multiagent_player,
        },
    }

config.update(custom_config)

if __name__ == "__main__":
    trainer = ppo.PPOTrainer(
        env="Soccer",
        config=config
    )

    for i in range(int(1e7)): # train iter
        print(pretty_print(trainer.train()))
        
        if i % 5 == 0:
            trainer.set_weights({"policy_04": trainer.get_weights(["policy_03"])["policy_03"],
                                "policy_03": trainer.get_weights(["policy_02"])["policy_02"],
                                "policy_02": trainer.get_weights(["policy_01"])["policy_01"],
                                })
        
        if i % 100 == 0:
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            trainer.save_checkpoint(CHECKPOINT_PATH)