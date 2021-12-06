import pickle
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks
from soccer_twos import EnvType, wrappers

from typing import Dict, Optional, List
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
import collections

from utils import create_wrapped_rllib_env
from wrappers import SingleObsWrapper, PreviousActionWrapper, MultiagentTeamObsWrapper, AgentIdInfoWrapper, RandomEnvWrapper, CurriculumWrapper
from custom_reward import WhoWonWrapper, BallInTeamFieldSidePenaltyWrapper, BallBehindPenaltyWrapper, ColisionRewardWrapper, FinalRewardMultiplier
from alphastar import pfsp
from copy import deepcopy

from ray.tune.checkpoint_manager import Checkpoint
from ray.tune.callback import Callback
from ray.tune.trial import Trial

NUM_ENVS_PER_WORKER = 5

trainable_policies = ['main_agent']
self_play_policies = [policy_id+'_sync' for policy_id in trainable_policies]
initial_policies = ['random_policy', 'do_nothing_policy']
league_policies = ['random_policy', 'do_nothing_policy']

checkpoint_freq = 100
restore = False
RESTORE_PATH = None
#restore = True
#RESTORE_PATH = 'ray_results/PPO_deepmind_selfplay_v3/PPO_Soccer_b5b30_00000_0_2021-12-05_10-53-22/checkpoint_000600/checkpoint-600'

class RandomPolicy(Policy):
    """Hand-coded policy that returns random actions."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space_for_sampling = self.action_space


    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [self.action_space_for_sampling.sample() for _ in obs_batch], \
               [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

class DoNothingPolicy(Policy):
    """Hand-coded policy that keeps the agent still."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_nothing_action = np.asarray([0,0,0])


    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [self.do_nothing_action for _ in obs_batch], \
               [], {}
    
    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


@ray.remote
class Coordinator:
    def __init__(self, restore_path=None):
        self.trainable_policies = deepcopy(trainable_policies)
        self.initial_policies = deepcopy(initial_policies)
        self.league = deepcopy(league_policies)
        self.state = None

        
        if restore_path is not None:
            restore_league_path = os.path.join(os.path.dirname(restore_path), '..', 'selfplay_checkpoints')
            restore_checkpoint_id = int(os.path.dirname(restore_path).split('_')[-1])
            checkpoints = []
            for filename in os.listdir(restore_league_path):
                if filename.endswith('.pkl'):
                    checkpoint_name = "_".join(filename.split('_')[:-1])
                    checkpoint_id = int(checkpoint_name.split('_')[-1])
                    if checkpoint_id < restore_checkpoint_id:
                        checkpoints.append(checkpoint_name)
            for agent_id in set(checkpoints):
                self.league.append(agent_id)

            state_file = os.path.join(restore_league_path, 'state')
            if os.path.exists(state_file):
                with open(state_file, 'rb') as f:
                    self.state = pickle.load(f)
        
        if self.state is None:
            self.state = collections.defaultdict(
                lambda *_: {
                    'stage': 'historical',
                    'last_field_side': 1,
                    'wins': 0,
                    'losses': 0,
                    'draws': 0,
                    'games': 0,
                    'steps': 0
                }
            )

            for policy_id in self.trainable_policies:
                self.state[policy_id]['stage'] = 'initial'
                self.state[policy_id+'_sync']['stage'] = 'sync'
                #self.league.append(policy_id+'_sync')
                #self.state[policy_id]['stage'] = 'league'

        print('league', self.league)
        self.decay = 0.99
        self.save_period = 50
        self.limit_period = 600
        self.match_idx = 0

    def get_state(self):
        return self.state

    def get_win_rate(self, policy_id):
        policy_state = self.state[policy_id]
        if policy_state['games'] <= 20:
            return 0.5
        else:
            return (policy_state['wins'] + 0.5*policy_state['draws'])/policy_state['games'] 

    def _pfsp_branch(self):
        league_data = [
            (player, self.get_win_rate(player)) for player in self.league
        ]
        league = [player_data[0] for player_data in league_data]
        win_rates = [1 - player_data[1] for player_data in league_data]
        return np.random.choice(
            league, p=pfsp(win_rates, weighting="squared"))

    def get_match(self):
        self.match_idx += 1
        
        player = self.trainable_policies[self.match_idx % len(self.trainable_policies)]
        #If player is in stage 'initial' play only agains random policies, else plays with league or selfplay
        if  self.state[player]['stage'] == 'initial':
            opponent = np.random.choice(self.initial_policies)
        else:
            seed = np.random.random()
            # 30 % Selfplay
            if seed < 0.3:
                opponent = player+'_sync'
            # 40% Prioritized SelfPlay
            elif seed > 0.3 and seed < 0.7:
                opponent = self._pfsp_branch()
            # 15% Random of league
            elif seed < 0.85:
                np.random.choice(self.league)
            # 15% Harder player in the League
            else:
                win_rates = np.asarray([1 - self.get_win_rate(player) for player in self.league])
                opponent = self.league[win_rates.argmin()]
            # seed = np.random.choice([0, 1])
            # if seed == 0:
            #     #self_play
            #     opponent = player+'_sync'
            # else:
            #     #league_agent
            #     opponent = np.random.choice(self.league)

        field_side = 1 - self.state[player]['last_field_side']
        self.state[player]['last_field_side'] = field_side

        if field_side == 0:
            return player, opponent
        else:
            return opponent, player 

    def update_result(self, policies, rewards, steps):

        if policies[0] in self.trainable_policies:
            home_policy = policies[0]
            away_policy = policies[1]
            home_reward = rewards[0]
        else:
            home_policy = policies[1]
            away_policy = policies[0]
            home_reward = rewards[1]

        #if away_policy.endswith('_sync'):
        #    away_policy = away_policy[:-len('_sync')]

        for stats in ('games', 'wins', 'losses', 'draws'):
            self.state[home_policy][stats] *= self.decay
            self.state[away_policy][stats] *= self.decay

        self.state[home_policy]['games'] += 1
        self.state[away_policy]['games'] += 1
        if home_reward > 0:
             self.state[home_policy]['wins'] += 1
             self.state[away_policy]['losses'] += 1
        elif home_reward < 0:
            self.state[home_policy]['losses'] += 1
            self.state[away_policy]['wins'] += 1
        else:
            self.state[home_policy]['draws'] += 1
            
            self.state[away_policy]['draws'] += 1

        self.state[home_policy]['steps'] += steps

    def save_state(self, logdir):
        with open(os.path.join(logdir, 'selfplay_checkpoints', 'state'), 'wb') as f:
            pickle.dump(self.state, f)

    def reset_win_rate(self):
        for policy_id in self.trainable_policies:
            for data in ['wins', 'losses', 'draws', 'games']:
                self.state[policy_id][data] = 0

    def activate_league_stage(self):
        for policy_id in self.trainable_policies:
            self.state[policy_id]['stage'] = 'league'

    def check_checkpoints(self, iteration):
        ready_for_checkpoint = []
        for policy_id in self.trainable_policies:
            policy_state = self.state[policy_id]
            checkpoint_name = policy_id+'_'+str(iteration)
            if policy_state['stage'] == 'initial':
                #if self.get_win_rate(policy_id) > 0.85 or iteration >= self.limit_period:
                #    self.state[policy_id]['stage'] = 'league'
                #    ready_for_checkpoint.append((policy_id, checkpoint_name))
                #    self.league.append(checkpoint_name)
                #    self.state[checkpoint_name]['steps'] = self.state[policy_id]['steps']
                if iteration % int(self.save_period*2) == 0:
                    ready_for_checkpoint.append((policy_id, checkpoint_name))
                    self.league.append(checkpoint_name)
                    self.state[checkpoint_name]['steps'] = self.state[policy_id]['steps']
            else:
                if iteration % self.save_period == 0:
                    ready_for_checkpoint.append((policy_id, checkpoint_name))
                    self.league.append(checkpoint_name)
                    self.state[checkpoint_name]['steps'] = self.state[policy_id]['steps']
        return ready_for_checkpoint

    def print_league(self):
        print("############################### League Historic ##################################")
        
        header = ["name", "win_rate", "games","wins","losses", "draws", "stage", "steps"]
        data = [header]
        for player in self.state:
            data.append([
                player,
                round(self.get_win_rate(player), 4),
                round(self.state[player]['games'], 4),
                round(self.state[player]['wins'], 4),
                round(self.state[player]['losses'], 4),
                round(self.state[player]['draws'], 4),
                self.state[player]['stage'],
                self.state[player]["steps"]
            ])
        s = [[str(e) for e in row] for row in data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print("##############################################################################")

class PrioritizedSelfPlay(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.trainable_policies = deepcopy(trainable_policies)
        self.save_period = 50
        self.task_id = 0
        self.stage = 0

    def _save_weights(self, trainer, policy_id):
        checkpoint_dir = os.path.join(trainer.logdir, 'selfplay_checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        weights_filepath = os.path.join(checkpoint_dir, policy_id+'.pkl')
        
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
        coordinator = ray.get_actor("coordinator_league")

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
        coordinator = ray.get_actor("coordinator_league")

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

        for policy_id in self.trainable_policies:
            for t in ['_striker', '_goalie']:
                _sync_weights(policy_id+t, policy_id+'_sync'+t)

        checkpoint_data = ray.get(coordinator.check_checkpoints.remote(trainer.iteration))

        for policy_id, checkpoint_id in checkpoint_data:
            for t in ['_striker', '_goalie']:
                _create_checkpoint(policy_id+t, checkpoint_id+t)
                self._save_weights(trainer, checkpoint_id+t)

        if trainer.iteration % checkpoint_freq == 0:
            coordinator.save_state.remote(trainer.logdir)

        
        win_rates = []
        result['win_rates'] = {}
        for policy_id in self.trainable_policies:
            win_rate = ray.get(coordinator.get_win_rate.remote(policy_id))
            result['win_rates'][policy_id] = win_rate
        
        print('Current Task:', str(self.task_id), 'Stage:', 'initial' if self.stage == 0 else 'league')
        result['task_id'] = self.task_id
        result['stage'] = self.stage

        #Curriculum Learning
        if np.mean(np.asarray(win_rates)) > 0.85 and self.stage == 0:
            self.task_id += 1

            trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_task(self.task_id)))

            coordinator.reset_win_rate.remote()
            if self.task_id == 5:
                self.stage = 1
                coordinator.activate_league_stage.remote()


        coordinator.print_league.remote()
        #state = ray.get(coordinator.get_state.remote())

if __name__ == "__main__":
    ray.init()

    if restore:
        coordinador = Coordinator.options(name="coordinator_league").remote(restore_path=RESTORE_PATH)
    else:
        coordinador = Coordinator.options(name="coordinator_league").remote()

    create_rllib_env = create_wrapped_rllib_env(
        wrappers = [
            WhoWonWrapper,
            #FinalRewardMultiplier,
            #BallInTeamFieldSidePenaltyWrapper,
            #BallBehindPenaltyWrapper,
            #ColisionRewardWrapper,
            SingleObsWrapper,
            MultiagentTeamObsWrapper,
            CurriculumWrapper
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
        for policy_id in (trainable_policies + self_play_policies):
            policies[policy_id+t] = PolicySpec()

        for policy_id in trainable_policies:
            _trainable_policies.append(policy_id+t)
        
        policies['random_policy'+t] = PolicySpec(policy_class=RandomPolicy)
        policies['do_nothing_policy'+t] = PolicySpec(policy_class=DoNothingPolicy)
    

    def default_policy_mapping_fn(agent_id):
        print('Running default policy mapping')
        player = trainable_policies[0]
        opponent = 'random_policy'

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
            "ignore_worker_failures": False,
            "train_batch_size": int(8 * 1000 * len(trainable_policies)),
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
            'callbacks': MultiCallbacks([PrioritizedSelfPlay])
        }

    analysis = tune.run(
        "PPO",
        name="PPO_deepmind_selfplay_v3",
        config=config,
        stop={
            "timesteps_total": 150000000,  # 150M
            # "time_total_s": 14400, # 4h
        },
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        #resume=restore,
        restore=RESTORE_PATH if restore else None
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
