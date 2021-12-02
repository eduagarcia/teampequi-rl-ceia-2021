from pickle5 import pickle
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks
from soccer_twos import EnvType

from ray.rllib.policy import Policy, PolicySpec, RandomPolicy
from ray.rllib.policy.sample_batch import SampleBatch

import numpy as np
import os
import uuid

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 3
NUM_MAIN_AGENTS = 1
NUM_LEAGUE_AGENTS = 5 # num of slots to rotate at each iteration

POLICIES_IDS = {
    'main_agents': [f"main_agent_{i}" for i in range(NUM_MAIN_AGENTS)],
    'league_agents': [f"league_agent_{i}" for i in range(NUM_LEAGUE_AGENTS)],

}

#TODO: https://github.com/ray-project/ray/blob/fd13bac9b3fc2e7142065c759f2c9fc1c753e912/rllib/examples/self_play_league_based_with_open_spiel.py

# The code for the algorithm of pfsp, priority fictitious self-play
# https://github.com/liuruoze/mini-AlphaStar/blob/8c18233cf6e68abb581292c36f4059d7d950fc69/alphastarmini/core/ma/pfsp.py
def pfsp(win_rates, weighting="linear"):
    weightings = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x)**2,
    }
    fn = weightings[weighting]
    probs = fn(np.asarray(win_rates))
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm

class Agent(object):
    def __init__(self, 
                 name, 
                 initial_weights=None,
                 steps = 0,
                 iters = 0,
                 episodes = 0,
                 elo=1500,
                 wins = 0,
                 loses = 0):

        self.name = name
        self.episodes = iters
        self.steps = steps
        self.episodes = episodes
        self.elo = elo
        self.wins = wins
        self.loses = loses

        self.weights = initial_weights

    def set_elo(self, new_elo):
        self.elo = new_elo

    def get_elo(self):
        return self.elo

class League:
    def __init__(self,k=20,g=1):
        self.players = {}	
        self.k 	= k
        self.g 	= g

    def add_player(self, agent: Agent):
        self.players[agent.name] = agent

    def update_result(self, winner, loser):
        result = self.expectResult(winner, loser)
        self.set_player_elo(winner, self.get_player_elo(winner) + (self.k*self.g)*(1 - result))
        self.set_player_elo(loser, self.get_player_elo(loser) + (self.k*self.g)*(0 - (1 -result)))

    def get_player(self, agent):
        if isinstance(agent, Agent):
            name = agent.name
        else:
            name = agent
        return self.players[name]
    
    def get_player_elo(self, agent):
        return self.get_player(agent).get_elo()

    def set_player_elo(self, agent, elo):
        return self.get_player(agent).set_elo(elo)

    def expectResult(self, p1, p2):
        r_p1, r_p2 = self.get_player_elo(p1), self.get_player_elo(p2)
        exp = (r_p2-r_p1)/400.0
        return 1/((10.0**(exp))+1)

    def get_probs_against_league(self, agent):
        opponents = []
        probs = []
        player = self.get_player(agent)
        for name, opponent in self.players.items():
            if name == player.name:
                continue
            opponents.append(opponent)
            probs.append(self.expectResult(player, opponent))
        return opponents, probs

    def sample_opponents(self, agent, n_agents=5, weighting="squared"):
        sampled_opponents = []
        opponents, probs = self.get_probs_against_league(agent)
        for i in range(n_agents):
            sampled_opponents.append(np.random.choice(opponents, p=pfsp(probs, weighting=weighting)))
        return sampled_opponents

class PrioritizedFictitiousSelfPlay(DefaultCallbacks):

    def __init__(self, freq_save = 5, freq_run = 1):
        super().__init__()
        self.freq_save  = freq_save
        self.freq_run = freq_run
        self.policies_ids = POLICIES_IDS

        policies_to_train = POLICIES_IDS['main_agents'] + POLICIES_IDS['main_exploiter_agents'] + POLICIES_IDS['league_exploiter_agents']


    def on_train_result(self, whatisithis, trainer, result: dict, **kwargs) -> None:
        
        for policy_id, rewards in result["hist_stats"].items():
            mo = re.match("^policy_(.+)_reward$", policy_id)
            if mo is None:
                continue
            policy_id = mo.group(1)

            # Calculate this policy's win rate.
            won = 0
            for r in rewards:
                if r > 0.0:  # win = 1.0; loss = -1.0
                    won += 1
            win_rate = won / len(rewards)
            self.win_rates[policy_id] = win_rate

            # Policy is frozen; ignore.
            if policy_id in self.non_trainable_policies:
                continue
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
            seed = np.random.random()
            main_policy_id = np.random.choice(self.policy_ids['main_agents'])
            if seed < 0.6:
                #main agent selfplay
                self.team_vs_policy_select = [main_policy_id, main_policy_id]
            elif seed > 0.6:
                #main agent vs sampled oponent from league
                league_policy_id = np.random.choice(self.policy_ids['league_agents'])
                self.team_vs_policy_select = [main_policy_id, league_policy_id]
                np.random.shuffle(self.team_vs_policy_select)
            
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
        name="ppo_mini_alphastar",
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
