from pickle5 import pickle
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks
from soccer_twos import EnvType

from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
import os
import uuid

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 3
NUM_MAIN_AGENTS = 1
NUM_LEAGUE_AGENTS = 5 # num of slots to rotate at each iteration
NUM_MAIN_EXPLOITERS = 2
NUM_LEAGUE_EXPLOITERS = 2 

POLICIES_IDS = {
    'main_agents': [f"main_agent_{i}" for i in range(NUM_MAIN_AGENTS)],
    'league_agents': [f"league_agent_{i}" for i in range(NUM_LEAGUE_AGENTS)],
    'main_exploiter_agents': [f"main_exploiter_agent_{i}" for i in range(NUM_MAIN_EXPLOITERS)],
    'main_exploiter_oponnent_agents': [f"main_exploiter_oponnent_agent_{i}" for i in range(NUM_MAIN_EXPLOITERS)],
    'league_exploiter_agents': [f"league_exploiter_agent_{i}" for i in range(NUM_LEAGUE_EXPLOITERS)]
}

class Elo:
	def __init__(self,k,g=1,homefield = 100):
		self.ratingDict  	= {}	
		self.k 				= k
		self.g 				= g
		self.homefield		= homefield

	def addPlayer(self,name,rating = 1500):
		self.ratingDict[name] = rating
		
	def gameOver(self, winner, loser, winnerHome):
		if winnerHome:
			result = self.expectResult(self.ratingDict[winner] + self.homefield, self.ratingDict[loser])
		else:
			result = self.expectResult(self.ratingDict[winner], self.ratingDict[loser]+self.homefield)

		self.ratingDict[winner] = self.ratingDict[winner] + (self.k*self.g)*(1 - result)  
		self.ratingDict[loser] 	= self.ratingDict[loser] + (self.k*self.g)*(0 - (1 -result))
		
	def expectResult(self, p1, p2):
		exp = (p2-p1)/400.0
		return 1/((10.0**(exp))+1)

class PrioritizedFictitiousSelfPlay(DefaultCallbacks):

    def __init__(self, freq_save = 5, freq_run = 1):
        super().__init__()
        self.freq_save  = freq_save
        self.freq_run = freq_run
        self.counter = 0
        self.opponent_iter = 0
        self.policy_history = {}

    def on_train_result(self, whatisithis, trainer, result: dict, **kwargs) -> None:
        print("whatisithis", whatisithis)
        print("kwargs", kwargs)
        """
        self.counter += 1

        #Current result
        current_hist = {
            'iteraction': trainer.iteration,
            'weights_filepath': None,
            'opponent_iter': self.opponent_iter,
            'result': result['policy_reward_mean']['main_agent']
        }

        ## Save current checkpoint
        if self.counter % self.freq_save == 0:
            checkpoint_dir = os.path.join(trainer.logdir, 'selfplay_checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            weights_filepath = os.path.join(checkpoint_dir, str(trainer.iteration)+'.pkl')
            
            with open(weights_filepath, 'wb') as f:
                weights = trainer.get_weights(['main_agent'])['main_agent']
                pickle.dump(weights, f)
            
            current_hist['weights_filepath'] = weights_filepath
            print('Policy Saved:', current_hist)
            self.policy_history.append(current_hist)

        #sample opponent policy
        if self.counter % self.freq_run == 0:
            #50% of the time current policy, 50% random policy
            if np.random.choice([0, 1]) == 0 or len(self.policy_history) == 0:
                opponent_policy = current_hist
                opponent_weights = trainer.get_weights(['main_agent'])['main_agent']
            else:
                opponent_policy = np.random.choice(self.policy_history)
                with open(opponent_policy['weights_filepath'], 'rb') as f:
                    opponent_weights = pickle.load(f)
            
        print('sampled opponent policy:', opponent_policy)
        self.opponent_iter = opponent_policy['iteraction']


        trainer.set_weights({'main_agent': trainer.get_weights(['main_agent'])['main_agent'],
                            'opponent_team': opponent_weights,
                            })
        """

# based on https://github.com/ray-project/ray/issues/7023
#Select a random team (blue or yellow) to give the current_policy or opponent_policy
class MatchMaker:
    def __init__(self, policy_ids, n_agents=4):
        self.policy_ids = policy_ids
        self.n_agents = n_agents

        self.team_vs_policy_select = []

    def policy_mapping_fn(self, agent_id):
        if agent_id == 0:
            agent_to_train = np.random.choice("main", "exploiters")
            if agent_to_train == "main":
                seed = np.random.random()

                main_policy_id = np.random.choice(self.policy_ids['main_agents'])
                if seed < 0.35:
                    #main agent selfplay
                    self.team_vs_policy_select = [main_policy_id, main_policy_id]
                elif seed > 0.35 and seed < 0.85:
                    #main agent vs sampled oponent from league
                    league_policy_id = np.random.choice(self.policy_ids['league_agents'])
                    self.team_vs_policy_select = [main_policy_id, league_policy_id]
                    np.random.shuffle(self.team_vs_policy_select)
                else:
                    #main agent vs exploiters
                    exploiter_policy_id = np.random.choice(self.policy_ids['league_exploiter_agents'] + self.policy_ids['main_exploiter_agents'])
                    self.team_vs_policy_select = [main_policy_id, exploiter_policy_id]
                    id = int(exploiter_policy_id.split('_')[-1])
                    if id % 2 == 1:
                        self.team_vs_policy_select = self.team_vs_policy_select[::-1]

                if self.team_select is None:
                    self.team_select = np.random.choice([0, 1])
            else:
                exploiter_to_train = np.random.choice("main", "league")
                if exploiter_to_train == "main":
                    exploiter_policy_id = np.random.choice(self.policy_ids['main_exploiter_agents'])
                    id = int(exploiter_policy_id.split('_')[-1])
                    self.team_vs_policy_select = [exploiter_policy_id, f"main_exploiter_oponnent_agent_{id}"]
                    if id % 2 == 1:
                        self.team_vs_policy_select = self.team_vs_policy_select[::-1]
                if exploiter_to_train == "league":
                    exploiter_policy_id = np.random.choice(self.policy_ids['league_exploiter_agents'])
                    id = int(exploiter_policy_id.split('_')[-1])
                    league_policy_id = np.random.choice(self.policy_ids['league_agents'])
                    self.team_vs_policy_select = [exploiter_policy_id, league_policy_id]
                    if id % 2 == 1:
                        self.team_vs_policy_select = self.team_vs_policy_select[::-1]
            
        if agent_id < 2:
            policy_id = self.team_vs_policy_select[0]
        else:
            policy_id = self.team_vs_policy_select[1]

        return policy_id

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    policies = {}

    for policy_type in range(POLICIES_IDS):
        for policy_id in POLICIES_IDS[policy_type]:
            policies[policy_id] = (None, obs_space, act_space, {})

    policies_to_train = POLICIES_IDS['main_agents'] + POLICIES_IDS['main_exploiter_agents'] + POLICIES_IDS['league_exploiter_agents']
    
    matchmaker = MatchMaker(POLICIES_IDS)

    analysis = tune.run(
        "PPO",
        name="PPO_deepmind_selfplay_v2_1",
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
                "policies": policies,
                "policy_mapping_fn": matchmaker.policy_mapping_fn,
                "policies_to_train": policies_to_train
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
            },
            'callbacks': MultiCallbacks([PrioritizedFictitiousSelfPlay])
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
