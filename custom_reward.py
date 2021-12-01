import math
import numpy as np
import gym

def angle_vector(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

class CustomRewardWrapper(gym.core.Wrapper):
    """
    A wrapper for multiagent team-controlled environment.
    Uses a 2x2 (4 players) environment to expose a 1x1 (2 teams) environment.
    """

    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.env = env
        self.goals = [[16.5, 0], [-16.5, 0]]
        self.step_counter = 0 

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        self.step_counter += 1

        return (
            obs,
            self._preprocess_reward(rewards, info),
            done,
            info,
        )
    
    def reset(self):
        return self.env.reset()

    def _preprocess_reward(self, rewards, info):
        agent_ids = list(rewards.keys())
        new_rewards = {agent_id: reward for agent_id, reward in rewards.items()}
        
        if abs(list(rewards.values())[0]) > 0:
            for agent_id, reward in rewards.items():    
                new_rewards[agent_id] = reward*20*(1-self.step_counter/5000)
            return new_rewards

        ball = info[agent_ids[0]]['ball_info']
        for team_id in [0, 1]:
            reward = 0

            opponent_goal = self.goals[team_id]
            vector_ball_to_goal = [opponent_goal[0] - ball['position'][0], opponent_goal[1] - ball['position'][1]]
            ball_velocity_norm = np.linalg.norm(ball['velocity'])
            if ball_velocity_norm > 1e-3 and not np.isnan(ball_velocity_norm):
                angle = angle_vector(vector_ball_to_goal, ball['velocity'])
                if not np.isnan(angle):
                    normalized_angle = (math.pi/2 - angle)/(math.pi/2)
                    reward += normalized_angle*ball_velocity_norm/40
            
            for agent_id in [team_id*2, (team_id*2)+1]:
                new_rewards[agent_id] += reward
        #print(new_rewards)
        return new_rewards

