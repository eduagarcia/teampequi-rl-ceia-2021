# AlphaStar archteture from https://github.com/liuruoze/mini-AlphaStar

import numpy as np
import collections

STEPS_UNTIL_CHECKPOINT = 5000
MULTIPLIER_EXPLOITER = 3


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



class AlphaStarAgent(object):
    """A alphastar agent for starcraft.
    Demonstrates agent interface.
    In practice, this needs to be instantiated with the right neural network
    architecture.
    """

    def __init__(self, name, initial_weights='initial'):
        # AlphaStarAgent use raw actions
        self.name = name
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None

        self.weights = initial_weights

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        #assert self.weights == self.agent_nn.get_weights()
        return self.weights
    
    def get_steps(self):
        return self.steps

    def set_steps(self, steps):
        self.steps = steps

#Payoff
class Payoff:

    def __init__(self):
        self._players = []
        self._players_by_id = {}
        self._wins = collections.defaultdict(lambda: 0)
        self._draws = collections.defaultdict(lambda: 0)
        self._losses = collections.defaultdict(lambda: 0)
        self._games = collections.defaultdict(lambda: 0)
        self._decay = 0.99

    def _win_rate(self, _home, _away):
        #if self._games[_home, _away] == 0:
        if self._games[_home, _away] <= 20:
            return 0.5

        return (self._wins[_home, _away]
                + 0.5 * self._draws[_home, _away]) / self._games[_home, _away]

    def __getitem__(self, match):
        home, away = match

        if isinstance(home, Player):
            home = [home]
        if isinstance(away, Player):
            away = [away]

        win_rates = np.array([[self._win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
            win_rates = win_rates.reshape(-1)

        return win_rates

    def update(self, home, away, result):
        for stats in (self._games, self._wins, self._draws, self._losses):
            stats[home, away] *= self._decay
            stats[away, home] *= self._decay

        self._games[home, away] += 1
        self._games[away, home] += 1
        if result == "win":
            self._wins[home, away] += 1
            self._losses[away, home] += 1
        elif result == "draw":
            self._draws[home, away] += 1
            self._draws[away, home] += 1
        else:
            self._wins[away, home] += 1
            self._losses[home, away] += 1

    def add_player(self, player):
        self._players.append(player)
        self._players_by_id[player.name] = player

    def get_players_num(self):
        return len(self._players)

    def get_players_by_id(self, name):
        return self._players_by_id[name]

    def print_stats(self, name):
        player = name
        if isinstance(name, str):
            player = self.get_players_by_id(name)
        print("############################### League Historic ##################################")
        print(f"################################ {player.name} ###################################")
        
        header = ["vs","win_rate", "games","wins","losses", "draws"]
        data = [header]
        for opponent in self._players:
            data.append([
                opponent.name,
                self._win_rate(player,opponent),
                self._games[player,opponent],
                self._wins[player,opponent],
                self._losses[player,opponent],
                self._draws[player,opponent]
            ])
        s = [[str(e) for e in row] for row in data]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print("##############################################################################")


    @property
    def players(self):
        return self._players

#https://github.com/liuruoze/mini-AlphaStar/blob/8c18233cf6e68abb581292c36f4059d7d950fc69/alphastarmini/core/ma/player.py

class Player(object):
    def __init__(self):
        self.metadata = {
            'last_field_side': 0
        }

    @property
    def learner(self):
        return self._learner

    def set_learner(self, learner):
        self._learner = learner

    @property
    def actors(self):
        return self._actors

    def add_actor(self, actor):
        self._actors.append(actor)

    def get_match(self):
        pass

    def ready_to_checkpoint(self):
        return False

    def _create_checkpoint(self):
        # AlphaStar： return Historical(self, self.payoff)
        return Historical(self, self._payoff)

    @property
    def payoff(self):
        return self._payoff

    def checkpoint(self):
        raise NotImplementedError

    def setup(self, obs_spec, action_spec):
        self.agent.setup(obs_spec, action_spec)

    def reset(self):
        self.agent.reset()

class Historical(Player):

    def __init__(self, agent, payoff):
        super().__init__()
        # AlphaStar： self._agent = Agent(agent.race, agent.get_weights())
        self.agent = AlphaStarAgent(name=f"{agent.agent.name}_{agent.agent.get_steps()}", initial_weights=agent.agent.get_weights())
        self._payoff = payoff
        self._parent = agent
        self.name = f"{agent.agent.name}_{agent.agent.get_steps()}"
        self.type = "Historical"
        self._actors = []

    @property
    def parent(self):
        return self._parent

    def get_match(self):
        raise ValueError("Historical players should not request matches")

    def ready_to_checkpoint(self):
        return False


class MainPlayer(Player):

    def __init__(self, name, agent, payoff):
        super().__init__()
        self.agent = AlphaStarAgent(name=name, initial_weights=agent.get_weights())
        # actually the _payoff maintains all the players and their fight results
        # maybe this should be the league, making it more reasonable
        self._payoff = payoff
        self._checkpoint_step = 0
        self.name = name
        self.type = "MainPlayer"
        self._actors = []

    def _pfsp_branch(self):
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="squared")), True

    def _selfplay_branch(self, opponent):
        # Play self-play match
        if self._payoff[self, opponent] > 0.3:
            return opponent, False

        # If opponent is too strong, look for a checkpoint
        # as curriculum
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="variance")), True

    def _verification_branch(self, opponent):
        # Check exploitation
        exploiters = set([
            player for player in self._payoff.players
            if isinstance(player, MainExploiter)
        ])
        # Q: What is the player.parent?
        # A: This is only the property of Historical
        exp_historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent in exploiters
        ]
        win_rates = self._payoff[self, exp_historical]
        if len(win_rates) and win_rates.min() < 0.3:
            return np.random.choice(
                exp_historical, p=pfsp(win_rates, weighting="squared")), True

        # Check forgetting
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]

        def remove_monotonic_suffix(win_rates, players):
            #if not win_rates:
            if win_rates.size == 0:
                return win_rates, players

            for i in range(len(win_rates) - 1, 0, -1):
                if win_rates[i - 1] < win_rates[i]:
                    return win_rates[:i + 1], players[:i + 1]

            return np.array([]), []

        win_rates, historical = remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < 0.7:
            return np.random.choice(
                historical, p=pfsp(win_rates, weighting="squared")), True

        return None

    def get_match(self):
        coin_toss = np.random.random()

        # Make sure you can beat the League
        if coin_toss < 0.5:
            return self._pfsp_branch()

        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        # Verify if there are some rare players we omitted
        if coin_toss < 0.5 + 0.15:
            request = self._verification_branch(opponent)
            if request is not None:
                return request

        return self._selfplay_branch(opponent)

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < STEPS_UNTIL_CHECKPOINT:
            return False

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > STEPS_UNTIL_CHECKPOINT*2

    def checkpoint(self):
        self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()


class MainExploiter(Player):

    def __init__(self, name, agent, payoff):
        super().__init__()
        self.agent = AlphaStarAgent(name=name, initial_weights=agent.get_weights())
        self._initial_weights = agent.get_weights()
        self._payoff = payoff
        self._checkpoint_step = 0
        self.name = name
        self.type = "MainExploiter"
        self._actors = []

    def get_match(self):
        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        if self._payoff[self, opponent] > 0.1:
            return opponent, True

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]

        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="variance")), True

    def checkpoint(self):
        self.agent.set_weights('inital')
        self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < STEPS_UNTIL_CHECKPOINT*MULTIPLIER_EXPLOITER:
            return False

        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        win_rates = self._payoff[self, main_agents]
        return win_rates.min() > 0.7 or steps_passed > STEPS_UNTIL_CHECKPOINT*2*MULTIPLIER_EXPLOITER


class LeagueExploiter(Player):

    def __init__(self, name, agent, payoff):
        super().__init__()
        self.agent = AlphaStarAgent(name=name, initial_weights=agent.get_weights())
        self._initial_weights = agent.get_weights()
        self._payoff = payoff
        self._checkpoint_step = 0
        self.name = name
        self.type = "LeagueExploiter"
        self._actors = []

    def get_match(self):
        historical = [
            player for player in self._payoff.players 
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(

            historical, p=pfsp(win_rates, weighting="linear_capped")), True

    def checkpoint(self):
        if np.random.random() < 0.25:
            self.agent.set_weights('initial')
        self._checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()

    def ready_to_checkpoint(self):
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < STEPS_UNTIL_CHECKPOINT*MULTIPLIER_EXPLOITER:
            return False
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > STEPS_UNTIL_CHECKPOINT*2*MULTIPLIER_EXPLOITER


#https://github.com/liuruoze/mini-AlphaStar/blob/8c18233cf6e68abb581292c36f4059d7d950fc69/alphastarmini/core/ma/league.py
class League(object):

    def __init__(self,
                 initial_agents,
                 main_players=1,
                 main_exploiters=1,
                 league_exploiters=2):
        self._payoff = Payoff()
        self._learning_players = []

        for i in range(main_players):
            main_player = MainPlayer(f"main_agent_{i}", initial_agents[0], self._payoff)
            self._learning_players.append(main_player)
            # add Historcal (snapshot) player
            self._payoff.add_player(main_player.checkpoint())

        for i in range(main_exploiters):
            self._learning_players.append(
                MainExploiter(f"main_exploiter_agent_{i}", initial_agents[0], self._payoff))

        for i in range(league_exploiters):
            self._learning_players.append(
                LeagueExploiter(f"league_exploiter_agent_{i}", initial_agents[0], self._payoff))

        for historical_agent in initial_agents[main_players:]:
            player = MainPlayer(historical_agent.name, historical_agent, self._payoff)
            player.agent.set_steps(historical_agent.get_steps())
            player._checkpoint_step = historical_agent.get_steps()
            self._payoff.add_player(Historical(player, self._payoff))

        # add MP, ME, LE player
        for player in self._learning_players:
            self._payoff.add_player(player)

        self._learning_players_num = len(self._learning_players)

    def update(self, home, away, result):
        return self._payoff.update(home, away, result)

    def get_learning_player(self, idx):
        return self._learning_players[idx]

    def get_learning_players(self):
        return self._learning_players

    def get_learning_players_by_type(self, type):
        players = []
        cls_type = "MainPlayer"
        if type == 'exploiter':
            cls_type = "MainExploiter"
        elif type == 'league_exploiter':
            cls_type = "LeagueExploiter"
        
        for player in self._learning_players:
            if player.type == cls_type:
                players.append(player)
        
        return players

    def add_player(self, player):
        self._payoff.add_player(player)

    def get_learning_players_num(self):
        return self._learning_players_num

    def get_players_num(self):
        return self._payoff.get_players_num()

    def get_player_by_id(self, name) -> Player:
        return self._payoff.get_players_by_id(name)

    def print_league_stats(self):
        self._payoff.print_stats(self._learning_players[0])