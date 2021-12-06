from typing import Dict

import gym
import numpy as np
from soccer_twos import AgentInterface


class RandomAgent(AgentInterface):
    """
    Random Agent is an agent that always returns a random action.
    """

    def __init__(self, env: gym.Env):
        """Initialize the RandomAgent.
        Args:
            env: the competition environment.
        """
        super().__init__()
        self.name = "RANDOM"


        self.action_space = env.action_space
        self.zero_action = np.asarray([0,0,0])

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
            actions[player_id] = self.zero_action
        return actions
