import math
import numpy as np
import gym

def angle_vector(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


class WhoWonWrapper(gym.core.Wrapper):
    """
    A wrapper to add the info of which team won

    Team Blue: 0
    Team Orange: 0
    """

    def __init__(self, env):
        super(WhoWonWrapper, self).__init__(env)
        self.env = env
        
    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        return (
            obs,
            rewards,
            done,
            self._preprocess_info(info, rewards, done),
        )
    
    def _preprocess_info(self, info, rewards, done):
        for agent_id in info:
            info[agent_id]['result'] = 'undefined'
            if max(done.values()):
                if rewards[0] > 0:
                    info[agent_id]['result'] = 0
                elif rewards[0] < 0:
                    info[agent_id]['result'] = 1
                else:
                    info[agent_id]['result'] = 'draw'
        return info

class CustomRewardWrapper(gym.core.Wrapper):
    """
    A wrapper for custom rewards
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
            self._preprocess_reward(rewards, info, done),
            done,
            info,
        )

    def _preprocess_reward(self, rewards, info, done):
        agent_ids = list(rewards.keys())
        new_rewards = {agent_id: reward for agent_id, reward in rewards.items()}
        
        if max(done.values()):
            for agent_id, reward in rewards.items():    
                new_rewards[agent_id] = reward*100#*(1-self.step_counter/5000)
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
        print(new_rewards)
        return new_rewards

class ColisionRewardWrapper(gym.core.Wrapper):
    """
    A wrapper for custom rewards
    """

    def __init__(self, env):
        super(ColisionRewardWrapper, self).__init__(env)
        self.env = env
        self.goals = [[16.5, 0], [-16.5, 0]]
        
        self.step_counter = 0
        self.last_info = None

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        ep_data = (
            obs,
            self._preprocess_reward(rewards, info, done),
            done,
            info
        )

        self._gatter_info(info)

        return ep_data

    def reset(self):
        self.step_counter = 0
        self.last_info = None
        return self.env.reset()

    def _gatter_info(self, info):
        self.step_counter += 1
        self.last_info = info

    def _dist_to_the_ball(self, info, ball_position):
        pos = np.zeros((4,))
        for agent_id in info:
            agent_position = info[agent_id]['player_info']['position']
            dist = np.linalg.norm(agent_position-ball_position)
            pos[agent_id] = dist
        return pos

    def _preprocess_reward(self, rewards, info, done):
        if max(done.values()):
            self.step_counter = 0
            self.last_info = None
            return rewards

        if self.last_info is None or not self.last_info:
            return rewards

        agent_ids = list(rewards.keys())
        new_rewards = {agent_id: reward for agent_id, reward in rewards.items()}

        last_ball_info = self.last_info[agent_ids[0]]['ball_info']
        ball_info = info[agent_ids[0]]['ball_info']

        last_ball_speed = np.linalg.norm(last_ball_info['velocity'])
        ball_speed = np.linalg.norm(ball_info['velocity'])

        last_ball_position = last_ball_info['position']
        ball_position = ball_info['position']

        dist_traveld_ball_position = np.linalg.norm(ball_position-last_ball_position)

        # If ball speed increased
        if last_ball_speed < ball_speed and dist_traveld_ball_position > 0.5:
            # wich angent now is closest to the ball where it as before?
            dist_by_angent = self._dist_to_the_ball(info, last_ball_position)
            agent_id = dist_by_angent.argmin()

            if agent_id == 0 or agent_id == 1:
                opponent_goal = self.goals[0]
            else:
                opponent_goal = self.goals[1]

            vector_ball_to_goal = [opponent_goal[0] - ball_info['position'][0], opponent_goal[1] - ball_info['position'][1]]
            if ball_speed > 1e-3 and not np.isnan(ball_speed):
                angle = angle_vector(vector_ball_to_goal, ball_info['velocity'])
                if not np.isnan(angle):
                    normalized_angle = (math.pi/2 - angle)/(math.pi/2)
                    new_rewards[agent_id] += 0.05*normalized_angle
                    #print(f"Agent {agent_id} kicked the ball for {0.05*normalized_angle}")

        return new_rewards

class BallBehindPenaltyWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super(BallBehindPenaltyWrapper, self).__init__(env)
        self.env = env

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        return (
            obs,
            self._preprocess_reward(rewards, info, done),
            done,
            info,
        )

    def _preprocess_reward(self, rewards, info, done):
        #If game over, do nothing
        if max(done.values()):
            return rewards

        agent_ids = list(rewards.keys())
        new_rewards = {agent_id: reward for agent_id, reward in rewards.items()}
        ball_position = info[0]['ball_info']['position']
        
        for agent_id in new_rewards:
            agent_position = info[agent_id]['player_info']['position']
            if agent_id == 0 or agent_id == 1:
                if agent_position[0] > ball_position[0]:
                    new_rewards[agent_id] += -0.005
                    #print(f"Agent {agent_id} is behind the ball {-0.005}")
            else:
                if agent_position[0] < ball_position[0]:
                    new_rewards[agent_id] += -0.005
                    #print(f"Agent {agent_id} is behind the ball {-0.005}")

        return new_rewards

class BallInTeamFieldSidePenaltyWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super(BallInTeamFieldSidePenaltyWrapper, self).__init__(env)
        self.env = env

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        return (
            obs,
            self._preprocess_reward(rewards, info, done),
            done,
            info,
        )

    def _preprocess_reward(self, rewards, info, done):
        #If game over, do nothing
        if max(done.values()):
            return rewards

        agent_ids = list(rewards.keys())
        new_rewards = {agent_id: reward for agent_id, reward in rewards.items()}
        ball_position = info[0]['ball_info']['position']
        
        for agent_id in new_rewards:
            ball_position = info[agent_id]['ball_info']['position']
            if agent_id == 0 or agent_id == 1:
                if ball_position[0] < 0:
                    new_rewards[agent_id] += -0.001

                if ball_position[0] < 12:
                    new_rewards[agent_id] += -0.004
            else:
                if ball_position[0] > 0:
                    new_rewards[agent_id] += -0.001

                if ball_position[0] > 12:
                    new_rewards[agent_id] += -0.004

        return new_rewards

class FinalRewardMultiplier(gym.core.Wrapper):
    def __init__(self, env):
        super(FinalRewardMultiplier, self).__init__(env)
        self.env = env
        self.multiplier = 2
        #self.historic = 

    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        return (
            obs,
            self._preprocess_reward(rewards, done),
            done,
            info,
        )

    def _preprocess_reward(self, rewards, done):
        #If game over, do nothing
        if max(done.values()):
            for agent_id, reward in rewards.items():
                rewards[agent_id] = reward*self.multiplier
        
        return rewards

class StrikerGoalieWrapper(gym.core.Wrapper):
    """
    A wrapper for rewards for the striker and goalie game
    
    agents
    0: Blue Striker
    1: Blue Goalie  
    2: Orange Goalie
    3: Orange Striker
    
    """

    def __init__(self, env):
        super(StrikerGoalieWrapper, self).__init__(env)
        self.env = env
        
    def step(self, action):
        obs, rewards, done, info = self.env.step(action)

        return (
            obs,
            self._preprocess_reward(info, done),
            done,
            info,
        )

    def _preprocess_reward(self, info, done):
        #End of Game
        if max(done.values()): 
            result = info[0]['result']           
            
            #Win (Team 0 Blue)
            if result == 0:
                winner = 0
            #Lost (Team 0 Blue)
            elif result == 1:
                winner = 1
            #Draw
            else:
                return {
                    0: -2,
                    1: 0,
                    2: 0,
                    3: -2,
                }
            
            return {
                0: 2 if winner == 0 else 0,
                1: 0 if winner == 0 else -2,
                2: 0 if winner == 1 else -2,
                3: 2 if winner == 1 else 0,
            }

        #Existial Penaly for Strikers
        #Existial Bonus for Goalies
        default_reward = {
            0: -0.001,
            1: 0.001,
            2: 0.001,
            3: -0.001,
        }

        #Penality if goalies pass the middle camp
        default_reward[1] += -0.005 if info[1]['player_info']['position'][0] > 0 else 0
        default_reward[2] += -0.005 if info[2]['player_info']['position'][0] < 0 else 0

        #Bonus if goalies is close to the goal
        default_reward[1] += 0.001 if info[1]['player_info']['position'][0] < -10 else 0
        default_reward[2] += 0.001 if info[2]['player_info']['position'][0] > 10 else 0

        #Penality if ball is in the goalies camp
        #Bonus if ball is in the opposite goalie camp
        default_reward[1] += -0.003 if info[1]['ball_info']['position'][0] < 0 else 0.003
        default_reward[2] += -0.003 if info[2]['ball_info']['position'][0] > 0 else 0.003

        return default_reward