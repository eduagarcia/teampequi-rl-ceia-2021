import pickle
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks
from soccer_twos import EnvType

from typing import Dict, Optional
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.agents.trainer import Trainer

import numpy as np
import os
import uuid
import re

from utils import create_wrapped_rllib_env
from wrappers import SingleObsWrapper, PreviousActionWrapper, MultiagentTeamObsWrapper, AgentIdInfoWrapper, RandomEnvWrapper, CurriculumWrapper
from custom_reward import WhoWonWrapper, BallInTeamFieldSidePenaltyWrapper, BallBehindPenaltyWrapper, ColisionRewardWrapper, FinalRewardMultiplier

from alphastar import AlphaStarAgent, League

INITIAL_WEIGHTS = './ray_results/PPO_deepmind_selfplay_v2_1/PPO_Soccer_8853e_00000_0_2021-11-29_12-10-48/selfplay_checkpoints/1805.pkl'
dir_path = os.path.dirname(os.path.realpath(__file__))
INITIAL_WEIGHTS = os.path.join(dir_path, INITIAL_WEIGHTS)


LEAGUE_WEIGHTS = []
league_paths = [
    './ray_results/PPO_deepmind_selfplay_v2_1/PPO_Soccer_8853e_00000_0_2021-11-29_12-10-48/selfplay_checkpoints',
    './ray_results/PPO_deepmind_selfplay_v2_2/PPO_Soccer_06a9e_00000_0_2021-12-01_02-10-40/selfplay_checkpoints',
    './ray_results/PPO_deepmind_selfplay_v4_continue/PPO_Soccer_92714_00000_0_2021-12-04_03-08-17/selfplay_checkpoints'
]
for i, path in enumerate(league_paths):
    abs_path = os.path.join(dir_path, path)
    checkpoint_files = [filename for filename in os.listdir(abs_path) if filename.endswith('.pkl')]
    chekpoints = np.asarray([int(filename[:-len('.pkl')]) for filename in checkpoint_files])
    chekpoints.sort()
    chekpoints = chekpoints[chekpoints > 800]
    model_definitions =  {
            "fcnet_hiddens": [512, 512],
            "vf_share_layers": False
        }
    if i == 0:
        num = 3
        name = 'goiabav2'
        chekpoints = chekpoints[chekpoints < 1500]
    elif i == 1:
        num = 3
        name = 'goiabav3'
    else:
        num = 6
        model_definitions['fcnet_hiddens'] = [1024,512,256]
        model_definitions['fcnet_activation'] = "relu"
        name= 'goiabav4'
        chekpoints = chekpoints[chekpoints > 2000]
    selection = np.round(np.linspace(0, len(chekpoints)-1, num=num)).astype(int)
    for checkpoint in chekpoints[selection]:
        LEAGUE_WEIGHTS.append((name, checkpoint, os.path.join(abs_path, str(checkpoint)+'.pkl'), model_definitions))

from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
class CustomPPOPolicty(Policy):
    """Hand-coded policy that returns checkpoint actions."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        checkpoint_path = self.config['checkpoint_path']
        model_definitions = self.config['model_definitions']
        config = ppo.DEFAULT_CONFIG.copy()
        for config_point, value in model_definitions.items():
            config['model'][config_point] = value
        with open(checkpoint_path, 'rb') as f:
            print('Initiating freezed policy from:', checkpoint_path)
            weights = pickle.load(f)
            self.policy = PPOTorchPolicy(self.observation_space, self.action_space, config)
            self.policy.set_weights(weights)

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return self.policy.compute_actions(obs_batch)

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        return self.policy.get_weights()

    def set_weights(self, weights):
        self.policy.set_weights(weights)
    

NUM_MAIN_AGENTS = 1
NUM_MAIN_EXPLOITERS = 1
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

trainable_policies = POLICIES_IDS['main_agents']  + POLICIES_IDS['main_exploiter_agents'] + POLICIES_IDS['league_exploiter_agents']
oponnent_policies = POLICIES_IDS['main_agents_freezed'] + POLICIES_IDS['main_opponent_agents']  + POLICIES_IDS['main_exploiter_opponent_agents'] + POLICIES_IDS['league_exploiter_opponent_agents']
freezed_main_policies = POLICIES_IDS['main_agents_freezed']
initial_weight_policy = freezed_main_policies[0]

initial_agents = [AlphaStarAgent('main_agent_0')]
for name, checkpoint, checkpoint_path, model_definitions in LEAGUE_WEIGHTS:
    agent = AlphaStarAgent(name, initial_weights=checkpoint_path)
    agent.set_steps(checkpoint)
    agent.model_definitions = model_definitions
    initial_agents.append(agent)

@ray.remote
class Coordinator:
    def __init__(self):
        self.trainable_policies = trainable_policies
        self.oponnent_policies = oponnent_policies
        self.freezed_main_policies = freezed_main_policies
        self.initial_weight_policy = initial_weight_policy

        self.initial_agents = initial_agents

        self.league = League(
            self.initial_agents,
            main_players=NUM_MAIN_AGENTS,
            main_exploiters=NUM_MAIN_EXPLOITERS,
            league_exploiters=NUM_LEAGUE_EXPLOITERS
        )

        self.match_idx = 0

    def get_league(self):
        return self.league

    def get_match(self):
        self.match_idx += 1
        
        player = self.league.get_learning_player(self.match_idx % len(self.trainable_policies))
        opponent = player.get_match()[0] #TODO: Que merda é esse True e False?
        opponent_name = opponent.name
        if opponent.name in self.trainable_policies:
            opponent_name = opponent_name + '_sync'

        field_side = 1 - player.metadata['last_field_side']
        player.metadata['last_field_side'] = field_side

        if field_side == 0:
            return player.name, opponent_name
        else:
            return opponent_name, player.name   

    def update_result(self, policies, rewards, steps):
        #example result
        ##{(0, 'main_agent_0'): 0.0, (1, 'main_agent_0'): 0.0, (2, 'main_agent_0'): 0.0, (3, 'main_agent_0'): 0.0}

        if policies[0] in self.trainable_policies:
            home_policy = policies[0]
            away_policy = policies[1]
            home_reward = rewards[0]
        else:
            home_policy = policies[1]
            away_policy = policies[0]
            home_reward = rewards[1]

        if away_policy.endswith('_sync'):
            away_policy = away_policy[:-len('_sync')]

        home_player = self.league.get_player_by_id(home_policy)       
        away_player = self.league.get_player_by_id(away_policy)

        if home_reward > 0:
            outcome = 'win'
        elif home_reward < 0:
            outcome = 'loss'
        else:
            outcome = 'draw'

        self.league.update(home_player, away_player, outcome)

        home_player.agent.set_steps(home_player.agent.get_steps()+1)

    def check_checkpoints(self, iteration):
        ready_for_checkpoint = []
        for player in self.league.get_learning_players():
            player.agent.set_weights(iteration)

            if player.ready_to_checkpoint():
                checkpoint = player.checkpoint()
                self.league.add_player(checkpoint)
                ready_for_checkpoint.append((player.name, checkpoint.name, player.agent.get_weights() == 'inital'))
        return ready_for_checkpoint

class PrioritizedFictitiousSelfPlay(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.trainable_policies = trainable_policies
        self.oponnent_policies = oponnent_policies
        self.freezed_main_policies = freezed_main_policies
        self.initial_weight_policy = initial_weight_policy

        self.initial_agents = initial_agents

        self.start_config = True

    def _save_weights(self, trainer, policy_id):
        checkpoint_dir = os.path.join(trainer.logdir, 'selfplay_checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        weights_filepath = os.path.join(checkpoint_dir, policy_id+'-'+str(trainer.iteration)+'.pkl')
        
        with open(weights_filepath, 'wb') as f:
            weights = trainer.get_policy(policy_id).get_weights()
            pickle.dump(weights, f)

    def on_episode_end(self,
                       *,
                       worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Runs when an episode is done.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (Dict[PolicyID, Policy]): Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (EnvID): Obsoleted: The ID of the environment, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """
        id = uuid.uuid4()
        coordinator = ray.get_actor("coordinator_alphastar")

        #example result
        ##{(0, 'main_agent_0'): 0.0, (1, 'main_agent_0'): 0.0, (2, 'main_agent_0'): 0.0, (3, 'main_agent_0'): 0.0}
        ep_rewards = episode.agent_rewards

        agents = list(ep_rewards.keys())
        policies = [agents[0][1][:-len('_striker')], agents[3][1][:-len('_striker')]]
        
        #rewards = list(ep_rewards.values())
        #rewards = [rewards[0]+rewards[1], rewards[2]+rewards[3]]

        result = episode.last_info_for(0)['result']
        
        if result == 0:
            rewards = [2, -2]
        elif result == 1:
            rewards = [-2, 2]
        else:
            rewards = [0, 0]
        
        coordinator.update_result.remote(policies, rewards, episode.length)

        match = ray.get(coordinator.get_match.remote())

        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            #print(id, 'running policy mapping', agent_id, match)
            if agent_id == 0 or agent_id == 1:
                if agent_id == 0:
                    return match[0]+'_striker'
                else:
                    return match[0]+'_goalie'
            else:
                if agent_id == 3:
                    return match[1]+'_striker'
                else:
                    return match[1]+'_goalie'

        worker.set_policy_mapping_fn(policy_mapping_fn)

    def on_train_result(self, *, trainer: Trainer, result: dict, **kwargs) -> None:
        coordinator = ray.get_actor("coordinator_alphastar")

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
                    pol_map['main_agent_0_striker'].set_weights(initial_weights)
                    trainer.workers.sync_weights(policies=['main_agent_0_striker'])
            else:
                raise(f"Inital Weights not found on {INITIAL_WEIGHTS}")

            for t in ['_striker', '_goalie']:
                for policy_id in self.trainable_policies:
                    _sync_weights('main_agent_0_striker', policy_id+t)
                for policy_id in self.freezed_main_policies:
                    _sync_weights('main_agent_0_striker', policy_id+t)

                #for agent in self.initial_agents[1:]:
                #    policy_id = agent.name + '_' + str(agent.get_steps())
                #    with open(agent.get_weights(), 'rb') as f:
                #        weights = pickle.load(f)
                #        pol_map = trainer.workers.local_worker().policy_map
                #        pol_map[policy_id+t].set_weights(weights)
                #        trainer.workers.sync_weights(policies=[policy_id+t])

            self.start_config = False

        for policy_id in self.trainable_policies:
            for t in ['_striker', '_goalie']:
                _sync_weights(policy_id+t, policy_id+'_sync'+t)

        checkpoint_data = ray.get(coordinator.check_checkpoints.remote(trainer.iteration))

        for policy_id, checkpoint_id, initial_weights in checkpoint_data:
            for t in ['_striker', '_goalie']:
                _create_checkpoint(policy_id+t, checkpoint_id+t)
                self._save_weights(trainer, checkpoint_id+t)
                if initial_weights:
                    _sync_weights(self.initial_weight_policy+t, policy_id+t)

        league = ray.get(coordinator.get_league.remote())
        league.print_league_stats()

if __name__ == "__main__":
    ray.init()

    coordinador = Coordinator.options(name="coordinator_alphastar").remote()

    create_rllib_env = create_wrapped_rllib_env(
        wrappers = [
            WhoWonWrapper,
            #FinalRewardMultiplier,
            #BallInTeamFieldSidePenaltyWrapper,
            #BallBehindPenaltyWrapper,
            #ColisionRewardWrapper,
            #SingleObsWrapper,
            #MultiagentTeamObsWrapper,
            RandomEnvWrapper
            ]
    )

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env({"variation": EnvType.multiagent_player})
    obs_space = temp_env.env.observation_space
    act_space = temp_env.env.action_space
    temp_env.env.close()

    policies = {}
    _trainable_policies = []

    for t in ['_striker', '_goalie']:
        for policy_type in POLICIES_IDS.keys():
            for policy_id in POLICIES_IDS[policy_type]:
                policies[policy_id+t] = (None, obs_space, act_space, {})

        for policy_id in trainable_policies:
            _trainable_policies.append(policy_id+t)

        for agent in initial_agents[1:]:
            policy_id = agent.name + '_' + str(agent.get_steps())
            policies[policy_id+t] = PolicySpec(
                policy_class=CustomPPOPolicty,
                config={'checkpoint_path': agent.get_weights(), 'model_definitions': agent.model_definitions})
        
    def default_policy_mapping_fn(agent_id):
        print('Running default policy mapping')
        player = POLICIES_IDS['main_agents_freezed'][0]
        opponent = POLICIES_IDS['main_agents_freezed'][0]

        team_order = [player, opponent]

        if agent_id == 0 or agent_id == 1:
            if agent_id == 0:
                return team_order[0]+'_striker'
            else:
                return team_order[0]+'_goalie'
        else:
            if agent_id == 3:
                return team_order[1]+'_striker'
            else:
                return team_order[1]+'_goalie'

    config = {
            "num_gpus": 1,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "ignore_worker_failures": True,
            "train_batch_size": int(4000 * len(_trainable_policies)),
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
                "policy_mapping_fn": default_policy_mapping_fn,
                "policies_to_train": _trainable_policies
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.multiagent_player,
            },
            'callbacks': MultiCallbacks([PrioritizedFictitiousSelfPlay])
        }

    analysis = tune.run(
        "PPO",
        name="PPO_alphastar_v3",
        config=config,
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
