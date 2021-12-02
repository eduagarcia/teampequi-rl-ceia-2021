import pickle
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks
from soccer_twos import EnvType

from typing import Dict, Optional
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.agents.trainer import Trainer

import numpy as np
import os
import uuid
import re

from utils import create_rllib_env
from alphastar import AlphaStarAgent, League

INITIAL_WEIGHTS = './ray_results/PPO_deepmind_selfplay_v2_1/PPO_Soccer_8853e_00000_0_2021-11-29_12-10-48/selfplay_checkpoints/1805.pkl'
dir_path = os.path.dirname(os.path.realpath(__file__))
INITIAL_WEIGHTS = os.path.join(dir_path, INITIAL_WEIGHTS)


LEAGUE_WEIGHTS = []
NUM_PRETRAINED_LEAGUE_AGENTS_PER_DIR = 5
league_paths = [
    './ray_results/PPO_deepmind_selfplay_v2_1/PPO_Soccer_8853e_00000_0_2021-11-29_12-10-48/selfplay_checkpoints',
    './ray_results/PPO_deepmind_selfplay_v2_2/PPO_Soccer_06a9e_00000_0_2021-12-01_02-10-40/selfplay_checkpoints'
]
for i, path in enumerate(league_paths):
    abs_path = os.path.join(dir_path, path)
    checkpoint_files = [filename for filename in os.listdir(abs_path) if filename.endswith('.pkl')]
    chekpoints = np.asarray([int(filename[:-len('.pkl')]) for filename in checkpoint_files])
    chekpoints = chekpoints[chekpoints > 800]
    if i == 0:
        name = 'goiabav2'
        chekpoints = chekpoints[chekpoints < 1400]
    else:
        name = 'goiabav3'
    selection = np.round(np.linspace(0, len(chekpoints)-1, num=NUM_PRETRAINED_LEAGUE_AGENTS_PER_DIR)).astype(int)
    for checkpoint in chekpoints[selection]:
        LEAGUE_WEIGHTS.append((name, checkpoint, os.path.join(abs_path, str(checkpoint)+'.pkl')))
    

NUM_MAIN_AGENTS = 1
NUM_MAIN_EXPLOITERS = 2
NUM_LEAGUE_EXPLOITERS = 2 

NUM_ENVS_PER_WORKER = NUM_MAIN_AGENTS + NUM_MAIN_EXPLOITERS + NUM_LEAGUE_EXPLOITERS

POLICIES_IDS = {
    'main_agents': [f"main_agent_{i}" for i in range(NUM_MAIN_AGENTS)],
    'main_agents_freezed': [f"main_agent_{i}_0" for i in range(NUM_MAIN_AGENTS)],
    'main_opponent_agents': [f"main_agent_{i}_sync" for i in range(NUM_MAIN_AGENTS)],
    'main_exploiter_agents': [f"main_exploiter_agent_{i}" for i in range(NUM_MAIN_EXPLOITERS)],
    'main_exploiter_opponent_agents': [f"main_exploiter_agent_{i}_sync" for i in range(NUM_MAIN_EXPLOITERS)],
    'league_exploiter_agents': [f"league_exploiter_agent_{i}" for i in range(NUM_LEAGUE_EXPLOITERS)],
    'league_exploiter_opponent_agents': [f"league_exploiter_agent_{i}_sync" for i in range(NUM_LEAGUE_EXPLOITERS)]
}


class PrioritizedFictitiousSelfPlay(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.trainable_policies = POLICIES_IDS['main_agents']  + POLICIES_IDS['main_exploiter_agents'] + POLICIES_IDS['league_exploiter_agents']
        self.oponnent_policies = POLICIES_IDS['main_agents_freezed'] + POLICIES_IDS['main_opponent_agents']  + POLICIES_IDS['main_exploiter_opponent_agents'] + POLICIES_IDS['league_exploiter_opponent_agents']
        self.freezed_main_policies = POLICIES_IDS['main_agents_freezed']
        self.initial_weight_policy = self.freezed_main_policies[0]

        self.initial_agents = [AlphaStarAgent('main_agent_0')]
        for name, checkpoint, checkpoint_path in LEAGUE_WEIGHTS:
            agent = AlphaStarAgent(name, initial_weights=checkpoint_path)
            agent.set_steps(checkpoint)
            self.initial_agents.append(agent)

        self.league = League(
            self.initial_agents,
            main_players=NUM_MAIN_AGENTS,
            main_exploiters=NUM_MAIN_EXPLOITERS,
            league_exploiters=NUM_LEAGUE_EXPLOITERS
        )
        self.matches = {
            POLICIES_IDS['main_agents'][0]: POLICIES_IDS['main_agents_freezed'][0]
        }
        self.field_side = [0]
        self.start_config = True

    # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
    #                       result: dict, **kwargs) -> None:
    #     print('on_learn_on_batch', policy)
    #     print('on_learn_on_batch result', result)
    #     print('on_learn_on_batch len trainbatch', len(train_batch))

    # def on_episode_end(self,
    #                    *,
    #                    worker,
    #                    base_env: BaseEnv,
    #                    policies: Dict[PolicyID, Policy],
    #                    episode: MultiAgentEpisode,
    #                    env_index: Optional[int] = None,
    #                    **kwargs) -> None:
    #     """Runs when an episode is done.

    #     Args:
    #         worker (RolloutWorker): Reference to the current rollout worker.
    #         base_env (BaseEnv): BaseEnv running the episode. The underlying
    #             env object can be gotten by calling base_env.get_unwrapped().
    #         policies (Dict[PolicyID, Policy]): Mapping of policy id to policy
    #             objects. In single agent mode there will only be a single
    #             "default_policy".
    #         episode (MultiAgentEpisode): Episode object which contains episode
    #             state. You can use the `episode.user_data` dict to store
    #             temporary data, and `episode.custom_metrics` to store custom
    #             metrics for the episode.
    #         env_index (EnvID): Obsoleted: The ID of the environment, which the
    #             episode belongs to.
    #         kwargs: Forward compatibility placeholder.
    #     """
    #     #has all policies with their object
    #     #print('policies', list(policies.keys()))
    #     #{(0, 'main_agent_0'): 0.0, (1, 'main_agent_0'): 0.0, (2, 'main_agent_0'): 0.0, (3, 'main_agent_0'): 0.0}
    #     print('rewards', episode.agent_rewards)
        
        
    #     print('episode length', episode.length)
    #     print('episode hist_data', episode.hist_data)

    #     #print('timesteps', self.global_timesteps)
    #     #self.global_timesteps += episode.length * 4
    #     #print('timesteps', self.global_timesteps)
    #     #self.league.update(home_player, away_player, outcome)
    #     #if home_player.ready_to_checkpoint():
    #     #    self.league.add_player(home_player.checkpoint())

    def on_train_result(self, *, trainer: Trainer, result: dict, **kwargs) -> None:
        def _sync_weights(from_policy_id, to_policy_id):
            print("syncing weights", from_policy_id, to_policy_id)
            main_state = trainer.get_policy(from_policy_id).get_state()
            pol_map = trainer.workers.local_worker().policy_map
            pol_map[to_policy_id].set_state(main_state)
            # We need to sync the just copied local weights to all the
            # remote workers as well.
            trainer.workers.sync_weights(policies=[to_policy_id])

        def _create_checkpoint(policy_id, new_policy_id):
            print("creating checkpoint", policy_id, new_policy_id)
            new_policy = trainer.add_policy(
                policy_id=new_policy_id,
                policy_cls=type(trainer.get_policy(policy_id)),
                policy_mapping_fn=lambda *_: new_policy_id,
                policies_to_train=self.trainable_policies,
            )
            main_state = trainer.get_policy(policy_id).get_state()
            new_policy.set_state(main_state)
            # We need to sync the just copied local weights to all the
            # remote workers as well.
            trainer.workers.sync_weights(policies=[new_policy_id])

        if self.start_config:
            if os.path.exists(INITIAL_WEIGHTS):
                with open(INITIAL_WEIGHTS, 'rb') as f:
                    initial_weights = pickle.load(f)
                    pol_map = trainer.workers.local_worker().policy_map
                    pol_map['main_agent_0'].set_weights(initial_weights)
                    trainer.workers.sync_weights(policies=['main_agent_0'])
            else:
                raise(f"Inital Weights not found on {INITIAL_WEIGHTS}")

            for policy_id in self.trainable_policies:
                _sync_weights('main_agent_0', policy_id)
            for policy_id in self.freezed_main_policies:
                _sync_weights('main_agent_0', policy_id)

            for agent in self.initial_agents[1:]:
                new_policy_id = agent.name + '_' + str(agent.get_steps())
                with open(agent.get_weights(), 'rb') as f:
                    weights = pickle.load(f)
                    new_policy = trainer.add_policy(
                        policy_id=new_policy_id,
                        policy_cls=type(trainer.get_policy('main_agent_0')),
                        policy_mapping_fn=lambda *_: new_policy_id,
                        policies_to_train=self.trainable_policies,
                    )
                    new_policy.set_weights(weights)
                    trainer.workers.sync_weights(policies=[new_policy_id])     

            self.start_config = False

        for policy_id, rewards in result["hist_stats"].items():
            mo = re.match("^policy_(.+)_reward$", policy_id)
            if mo is None:
                continue
            policy_id = mo.group(1)

            if policy_id in self.trainable_policies:
                home_player = self.league.get_player_by_id(policy_id)
                
                away_player_policy_id = self.matches[policy_id]
                if away_player_policy_id.endswith('_sync'):
                    away_player_policy_id = away_player_policy_id[:-len('_sync')]
                away_player = self.league.get_player_by_id(away_player_policy_id)

                for reward in rewards:
                    if reward > 0:
                        outcome = 'win'
                    elif reward < 0:
                        outcome = 'loss'
                    else:
                        outcome = 'draw'
                    self.league.update(home_player, away_player, outcome)

                home_player.agent.set_steps(home_player.agent.get_steps()+1)
                home_player.agent.set_weights(trainer.iteration)

                if home_player.ready_to_checkpoint():
                    checkpoint = home_player.checkpoint()
                    self.league.add_player(checkpoint)
                    _create_checkpoint(home_player.name, checkpoint.name)
                    if home_player.agent.get_weights() == 'inital':
                        _sync_weights(self.initial_weight_policy, home_player.name) 

        self.field_side = []
        self.matches = {}
        for player in self.league.get_learning_players():
            opponent = player.get_match()[0] #TODO: Que merda é esse True e False?
            opponent_name = opponent.name
            if opponent.name in self.trainable_policies:
                opponent_name = opponent_name + '_sync'
                _sync_weights(opponent.name, opponent_name)
            self.matches[player.name] = opponent_name
            self.field_side.append(np.random.choice([0, 1]))
        
        print('matches', self.matches)
        print('field_side', self.field_side)

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            i = episode.episode_id % len(self.matches)
            player, opponent = list(self.matches.items())[i]
                        
            #print('policy_mapping', agent_id, episode.episode_id, player, opponent)

            if self.field_side[i] == 0:
                team_order = [player, opponent]
            else:
                team_order = [opponent, player]

            if agent_id == 0 or agent_id == 1:
                return team_order[0]
            else:
                return team_order[1]

        def _set(worker):
            worker.set_policy_mapping_fn(policy_mapping_fn)
            worker.set_policies_to_train(self.trainable_policies)

        trainer.workers.foreach_worker(_set)

        self.league.print_league_stats()

if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = temp_env.env.observation_space
    act_space = temp_env.env.action_space
    temp_env.env.close()

    policies = {}

    for policy_type in POLICIES_IDS.keys():
        for policy_id in POLICIES_IDS[policy_type]:
            policies[policy_id] = (None, obs_space, act_space, {})

    policies_to_train = POLICIES_IDS['main_agents'] + POLICIES_IDS['main_exploiter_agents'] + POLICIES_IDS['league_exploiter_agents']
    
    def default_policy_mapping_fn(agent_id):
        player = POLICIES_IDS['main_agents'][0]
        opponent = POLICIES_IDS['main_agents_freezed'][0]

        team_order = [player, opponent]

        if agent_id == 0 or agent_id == 1:
            return team_order[0]
        else:
            return team_order[1]

    analysis = tune.run(
        "PPO",
        name="PPO_alphastar_v1",
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
                "policy_mapping_fn": default_policy_mapping_fn,#matchmaker.policy_mapping_fn,
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
            "timesteps_total": 150000000,  # 150M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        #resume=True
        #restore="./ray_results/PPO_selfplay_1/PPO_alphastar_v1/PPO_Soccer_8aaa7_00000_0_2021-12-02_01-44-32/checkpoint_000200/checkpoint-200",
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
