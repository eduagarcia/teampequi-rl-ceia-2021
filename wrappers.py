from mlagents_envs import base_env
import numpy as np
import copy
import gym
from gym_unity.envs import ActionFlattener
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.models.preprocessors import OneHotPreprocessor
import uuid
from soccer_twos.wrappers import MultiAgentUnityWrapper
import random
from ray.rllib.utils.annotations import override


class SingleObsWrapper(gym.core.Wrapper):
    """
    A wrapper to unstack the agent observation, returning only the last information
    """

    def __init__(self, env):
        super(SingleObsWrapper, self).__init__(env)
        self.env = env

        #original obs format:
        # size = 336
        #[fowardt_-2(88), fowardt_-1(88), fowardt_0(88), backwardt_-2(24), backwardt_-1(24), backwardt_0(24)]

        #single obs format:
        # size = 112
        # [fowardt_0(88), backwardt_0(24)]
        
        self.observation_space = gym.spaces.Box(
            0, 1, dtype=np.float32, shape=(112,)
        )

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        return (
            self._preprocess_obs(obs),
            rewards,
            done,
            info,
        )
    
    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def _preprocess_obs(self, obs):
        return SingleObsWrapper.preprocess_obs(obs)

    @staticmethod
    def preprocess_obs(obs, **kwargs):
        new_obs = {}
        for agent_id, agent_obs in obs.items():
            new_agent_obs = np.zeros((112,))
            new_agent_obs[:88] = agent_obs[176:264]
            new_agent_obs[88:] = agent_obs[312:]
            new_obs[agent_id] = new_agent_obs
        return new_obs

class MultiagentTeamObsWrapper(gym.core.Wrapper):
    """
    A wrapper for multiagent a environment.
    Join the observation of each team for each agent.
    """

    def __init__(self, env):
        super(MultiagentTeamObsWrapper, self).__init__(env)
        self.env = env

        # duplicate obs space (concatenate team players)
        self.observation_space = gym.spaces.Box(
            0, 1, dtype=np.float32, shape=(env.observation_space.shape[0] * 2,)
        )

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        return (
            self._preprocess_obs(obs),
            rewards,
            done,
            info,
        )
    
    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def _preprocess_obs(self, obs):
        return {
            0: np.concatenate((obs[1], obs[0])),
            1: np.concatenate((obs[0], obs[1])),
            2: np.concatenate((obs[3], obs[2])),
            3: np.concatenate((obs[2], obs[3])),
        }

    @staticmethod
    def preprocess_obs(obs, **kwargs):
        return {
            0: np.concatenate((obs[1], obs[0])),
            1: np.concatenate((obs[0], obs[1])),
        }

class AgentIdInfoWrapper(gym.core.Wrapper):
    """
    A wrapper for multiagent a environment.
    Adds agent id information to the observation.
    """

    def __init__(self, env):
        super(AgentIdInfoWrapper, self).__init__(env)
        self.env = env

        self.observation_space = gym.spaces.Box(
            0, 1, dtype=np.float32, shape=(env.observation_space.shape[0]+2,)
        )

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        return (
            self._preprocess_obs(obs),
            rewards,
            done,
            info,
        )
    
    def reset(self):
        return self._preprocess_obs(self.env.reset())

    def _preprocess_obs(self, obs):
        return {
            0: np.concatenate(([0, 1], obs[0])),
            1: np.concatenate(([1, 0], obs[1])),
            2: np.concatenate(([0, 1], obs[2])),
            3: np.concatenate(([1, 0], obs[3])),
        }

    @staticmethod
    def preprocess_obs(obs, **kwargs):
        return {
            0: np.concatenate(([0, 1], obs[0])),
            1: np.concatenate(([1, 0], obs[1])),
        }

class PreviousActionWrapper(gym.core.Wrapper):
    """
    A wrapper that adds agent previous action information to the observation.
    """

    def __init__(self, env):
        super(PreviousActionWrapper, self).__init__(env)
        self.env = env

        self.preprocessor = OneHotPreprocessor(env.action_space)

        # duplicate obs space (concatenate team players)
        self.observation_space = gym.spaces.Box(
            0, 1, dtype=np.float32, shape=(env.observation_space.shape[0]+self.preprocessor.shape[0],)
        )

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.zero_action = self.preprocessor.transform(0)
        else:
            self.zero_action = self.preprocessor.transform(np.zeros(env.action_space.shape).astype(int))

        self.last_action = {
            0: self.zero_action,
            1: self.zero_action,
            2: self.zero_action,
            3: self.zero_action,
        }

    def step(self, action):
        for agent_id, a in action.items():
            self.last_action[agent_id] = self.preprocessor.transform(a)

        obs, rewards, done, info = self.env.step(action)

        return (
            self._preprocess_obs(obs),
            rewards,
            done,
            info,
        )
    
    def reset(self):
        self.last_action = {
            0: self.zero_action,
            1: self.zero_action,
            2: self.zero_action,
            3: self.zero_action,
        }
        return self._preprocess_obs(self.env.reset())

    def _preprocess_obs(self, obs):
        return {
            0: np.concatenate((self.last_action[0], obs[0])),
            1: np.concatenate((self.last_action[1], obs[1])),
            2: np.concatenate((self.last_action[2], obs[2])),
            3: np.concatenate((self.last_action[3], obs[3])),
        }

    @staticmethod
    def preprocess_obs(obs, **kwargs):
         return {
            0: np.concatenate((kwargs['last_action'][0], obs[0])),
            1: np.concatenate((kwargs['last_action'][1], obs[1])),
        }

class RandomEnvWrapper(gym.core.Wrapper):
    """
    A wrapper randomizes the envoriment. 
    (By default Does only randomize Ball position and Player rotation)
    """

    def __init__(self, env, randomize = {'ball': ['position'], 'player': ['rotation_y']}, min_divergence = 0.3, k=1):
        super(RandomEnvWrapper, self).__init__(env)
        self.env = env
        self.randomize = randomize
        self.base_env = self.env.unwrapped
        while not isinstance(self.base_env, MultiAgentUnityWrapper):
            self.base_env = self.base_env.unwrapped
        self.env_channel = self.base_env._env._side_channel_manager._side_channels_dict[uuid.UUID('3f07928c-2b0e-494a-810b-5f0bbb7aaeca')]
        
        self.k = 1
        self.min_divergence = min_divergence

        self.default_watch_position = {
            'ball_info': {
                'position': [0., 0.],
                'velocity': [0., 0.]
            },
            'player_info': {
                0: {
                    'position': [-8.190001, -1.2],
                    'rotation_y': 90.0,
                    'velocity': [0., 0.],
                },
                1: {
                    'position': [-8.190001,  1.2],
                    'rotation_y': 90.0,
                    'velocity': [0., 0.],
                },
                2: {
                    'position': [8.190001, 1.2],
                    'rotation_y': 270.0,
                    'velocity': [0., 0.],
                },
                3: {
                    'position': [8.190001, -1.2],
                    'rotation_y': 270.0,
                    'velocity': [0., 0.],
                }
            }
        }

        self.default_train_position = {
            'ball_info': {
                'position': [1.0909986, 1.8254881],
                'velocity': [0., 0.]
            },
            'player_info': {
                0: {
                    'position': [-9.031397, -1.2],
                    'rotation_y': 87.729774,
                    'velocity': [0., 0.],
                },
                1: {
                    'position': [-6.2403193,  1.2],
                    'rotation_y': 85.95333,
                    'velocity': [0., 0.],
                },
                2: {
                    'position': [6.4539313, 1.2],
                    'rotation_y': 277.19568,
                    'velocity': [0., 0.],
                },
                3: {
                    'position': [6.6643953, -1.2],
                    'rotation_y': 270.04953,
                    'velocity': [0., 0.],
                }
            }
        }


        VELOCITY_RANGE = 5
        self.limits = {
            'ball_info': {
                'position': ([-10, 10], [-5, 5]),
                'velocity': ([-VELOCITY_RANGE, VELOCITY_RANGE], [-VELOCITY_RANGE, VELOCITY_RANGE])
            },
            'player_info': {
                0: {
                    'position': ([-5, 18], [-5, 5]),
                    'rotation_y': ([-80,270],),
                    'velocity': ([-VELOCITY_RANGE, VELOCITY_RANGE], [-VELOCITY_RANGE, VELOCITY_RANGE]),
                },
                1: {
                    'position': ([-5, 18], [-5, 5]),
                    'rotation_y': ([-80,270],),
                    'velocity': ([-VELOCITY_RANGE, VELOCITY_RANGE], [-VELOCITY_RANGE, VELOCITY_RANGE]),
                },
                2: {
                    'position': ([-18, 5], [-5, 5]),
                    'rotation_y':([-270,80],),
                    'velocity': ([-VELOCITY_RANGE, VELOCITY_RANGE], [-VELOCITY_RANGE, VELOCITY_RANGE]),
                },
                3: {
                    'position': ([-18, 5], [-5, 5]),
                    'rotation_y': ([-270,80],),
                    'velocity': ([-VELOCITY_RANGE, VELOCITY_RANGE], [-VELOCITY_RANGE, VELOCITY_RANGE]),
                }
            }
        }

    def reset(self):
        obs = self.env.reset()
        base = self.default_watch_position#np.random.choice([self.default_train_position, self.default_watch_position])
        max_divergence = np.random.random()
        if max_divergence < self.min_divergence:
            state = base
            max_divergence = 0
        else:
            state = copy.deepcopy(base)
            for data_point, limits in self.limits['ball_info'].items():
                if data_point not in self.randomize['ball']:
                    continue
                p1 = np.random.uniform(limits[0][0],limits[0][1])
                if len(limits) == 2:
                    p2 = np.random.uniform(limits[1][0],limits[1][1])
                    state['ball_info'][data_point][0] += p1*max_divergence*self.k
                    state['ball_info'][data_point][1] += p2*max_divergence*self.k
                else:
                    state['ball_info'][data_point] += p1
            for agent_id in self.limits['player_info']:
                for data_point, limits in self.limits['player_info'][agent_id].items():
                    if data_point not in self.randomize['player']:
                        continue
                    p1 = np.random.uniform(limits[0][0],limits[0][1])
                    if len(limits) == 2:
                        p2 = np.random.uniform(limits[1][0],limits[1][1])
                        state['player_info'][agent_id][data_point][0] += p1*max_divergence*self.k
                        state['player_info'][agent_id][data_point][1] += p2*max_divergence*self.k
                    else:
                        state['player_info'][agent_id][data_point] += p1*max_divergence*self.k
        #print("Randomizing ambient", max_divergence, self.k)
        #print("setted state", state)
        self.env_channel.set_parameters(
            ball_state = state['ball_info'],
            players_states = state['player_info']
        )
        return obs

class CurriculumWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super(CurriculumWrapper, self).__init__(env)
        self.env = env
        self.current_env = env
        self.cur_level = 0
        self.number_of_tasks = 6

    def reset(self):
        return self.current_env.reset()
    
    def step(self, action):
        return self.current_env.step(action)

    #@override(TaskSettableEnv)
    def sample_tasks(self, n_tasks):
        """Implement this to sample n random tasks."""
        return [random.randint(0, self.number_of_tasks) for _ in range(n_tasks)]
    
    #@override(TaskSettableEnv)
    def set_task(self, task) -> None:
        """Sets the specified task to the current environment
        Args:
            task: task of the meta-learning environment
        """
        if task == 0:
            self.current_env = self.env
        elif task == 1:
            randomize = {'ball': [], 'player': ['rotation_y']}
            self.current_env = RandomEnvWrapper(self.env, randomize=randomize)
        elif task == 2:
            randomize = {'ball': ['position'], 'player': ['rotation_y']}
            self.current_env = RandomEnvWrapper(self.env, randomize=randomize)
        elif task == 3:
            randomize = {'ball': ['position'], 'player': ['rotation_y', 'velocity']}
            self.current_env = RandomEnvWrapper(self.env, randomize=randomize)
        elif task == 4:
            randomize = {'ball': ['position', 'velocity'], 'player': ['rotation_y', 'velocity']}
            self.current_env = RandomEnvWrapper(self.env, randomize=randomize)
        elif task == 5:
            randomize = {'ball': ['position', 'velocity'], 'player': ['rotation_y', 'velocity']}
            self.current_env = RandomEnvWrapper(self.env, randomize=randomize, min_divergence=0.5, k=0.7)
        else:
            print(f"Invalid Task number {task}, setting enviroment task 0")
            self.current_env = self.env
            task = 0

        self.cur_level = task

    #@override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_level

    def next_task(self):
        self.set_task((self.cur_level + 1) %self.number_of_tasks)