import argparse
import importlib
import inspect
import logging
import os
import sys
import collections
import json
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from ray.tune.logger import pretty_print
from ray.tune.utils.util import SafeFallbackEncoder

import soccer_twos
from soccer_twos.agent_interface import AgentInterface

def get_agent_class(module):
    for class_name, class_type in inspect.getmembers(module, inspect.isclass):
        if class_name != "AgentInterface" and issubclass(class_type, AgentInterface):
            logging.info(f"Found agent {class_name} in module {module.__name__}")
            return class_type

    raise ValueError(
        "No AgentInterface subclass found in module {}".format(module.__name__)
    )

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = str(iteration)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent} {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def collect_episodes(
        env,
        agent1,
        agent2,
        n_episodes: int = 200,
) -> List[Dict]:
    """Gathers new episodes metrics from the given evaluators."""
    episodes = []

    printProgressBar(0, n_episodes, prefix = 'Progress:', suffix = f'/ {n_episodes} Complete', length = 50)
    for i in range(n_episodes):
        obs = env.reset()
        if i % 2 == 0:
            team_order = [agent1, agent2]
            team_agent_1 = "blue_team"
            team_agent_2 = "orange_team"
        else:
            team_order = [agent2, agent1]
            team_agent_1 = "orange_team"
            team_agent_2 = "blue_team"

        steps = 0
        episode_done = False
        blue_team_reward = 0
        orange_team_reward = 0
        
        while not episode_done:
            blue_team_actions = team_order[0].act({0: obs[0], 1: obs[1]})
            orange_team_actions = team_order[1].act({0: obs[3], 1: obs[2]})
            actions = {
                0: blue_team_actions[0],
                1: blue_team_actions[1],
                2: orange_team_actions[1],
                3: orange_team_actions[0],
            }

            # step
            obs, reward, done, info = env.step(actions)
            steps += 1

            # logging
            blue_team_reward = reward[0] + reward[1]
            orange_team_reward = reward[2] + reward[3]
            if max(done.values()):  # if any agent is done
                episode_done = True
        
        episodes.append({
            "episode_length": steps,
            "agent_1_reward": blue_team_reward if team_agent_1 == "blue_team" else orange_team_reward,
            "agent_2_reward": blue_team_reward if team_agent_2 == "blue_team" else orange_team_reward,
            "team_agent_1": team_agent_1,
            "team_agent_2": team_agent_2
        })

        printProgressBar(i+1, n_episodes, prefix = 'Progress:', suffix = f'/ {n_episodes} Complete', length = 50)
        
    return episodes


def summarize_episodes(
    episodes: List[Dict],
    agent_1_name: str,
    agent_2_name: str
) -> Dict:
    """Summarizes a set of episode metrics.

    Args:
        episodes: smoothed set of episodes including historical ones
        agent_1_name: First agent name
        agent_2_name: Second agent name
    """


    episode_lengths = []
    episode_rewards = []
    hist_stats = {
            agent_name: {
                    "rewards": [],
                    "blue_team": collections.defaultdict(list),
                    "orange_team": collections.defaultdict(list),
            }
            for agent_name in (agent_1_name, agent_2_name)
        }



    for episode in episodes:
        episode_lengths.append(episode["episode_length"])
        episode_rewards.append(episode["agent_1_reward"] + episode["agent_2_reward"])
        
        for agent_id, agent_name in [(1, agent_1_name), (2, agent_2_name)]:
            team = episode[f"team_agent_{agent_id}"]
            reward = episode[f"agent_{agent_id}_reward"]
            hist_stats[agent_name]["rewards"].append(reward)
            hist_stats[agent_name][team]["rewards"].append(reward)

    if episode_rewards:
        min_reward = min(episode_rewards)
        max_reward = max(episode_rewards)
        avg_reward = np.mean(episode_rewards)
    else:
        min_reward = float("nan")
        max_reward = float("nan")
        avg_reward = float("nan")
    if episode_lengths:
        avg_length = np.mean(episode_lengths)
    else:
        avg_length = float("nan")

    # Show as histogram distributions.
    hist_stats["episode_reward"] = episode_rewards
    hist_stats["episode_lengths"] = episode_lengths

    policies = {}
    for agent_name in (agent_1_name, agent_2_name):
        total_rewards = hist_stats[agent_name]["rewards"]
        total_results = []
        for reward in total_rewards:
            if reward > 0:
                total_results.append(1)
            elif reward < 0:
                total_results.append(-1)
            else:
                total_results.append(0)
        policies[agent_name] = {
            f"policy_reward_min": min(total_rewards),
            f"policy_reward_max": max(total_rewards),
            f"policy_reward_mean": np.mean(total_rewards),
            f"policy_total_games": len(total_rewards),
            f"policy_wins": total_results.count(1),
            f"policy_losses": total_results.count(-1),
            f"policy_draws": total_results.count(0),
            f"policy_win_rate": total_results.count(1)/len(total_rewards)
        }
        for team in ("blue_team", "orange_team"):
            team_rewards = hist_stats[agent_name][team]["rewards"]
            team_results = []
            for reward in team_rewards:
                if reward > 0:
                    team_results.append(1)
                elif reward < 0:
                    team_results.append(-1)
                else:
                    team_results.append(0)
            policies[agent_name][team] = {
                f"policy_{team}_reward_min": min(team_rewards),
                f"policy_{team}_reward_max": max(team_rewards),
                f"policy_{team}_reward_mean": np.mean(team_rewards),
                f"policy_{team}_total_games": len(team_rewards),
                f"policy_{team}_wins": team_results.count(1),
                f"policy_{team}_losses": team_results.count(-1),
                f"policy_{team}_draws": team_results.count(0),
                f"policy_{team}_win_rate": team_results.count(1)/len(team_rewards)
            }

    return dict(
        episode_reward_max=max_reward,
        episode_reward_min=min_reward,
        episode_reward_mean=avg_reward,
        episode_len_mean=avg_length,
        episodes_this_eval=len(episodes),
        policies=policies,
        hist_stats=dict(hist_stats))

def load_agent(agent_module_name: str, base_port = None) -> AgentInterface:
    """Loads a AgentInterface based on his module name"""

    agent_module = importlib.import_module(agent_module_name)

    env = soccer_twos.make(base_port=base_port)
    agent = get_agent_class(agent_module)(env)
    env.close()

    return agent


def evaluate(
    agent1_module_name: str,
    agent2_module_name: str = None,
    n_episodes: int = 200,
    base_port = None
) -> Dict:
    """Evaluates two agents against each other"""

    if agent2_module_name is None:
        agent2_module_name = agent1_module_name
    else:
        agent2_module_name = agent2_module_name

    agent1 = load_agent(agent1_module_name, base_port=base_port)
    agent2 = load_agent(agent2_module_name, base_port=base_port)

    env = soccer_twos.make(
        base_port=base_port,
    )
    
    episodes_data = collect_episodes(env, agent1, agent2, n_episodes)

    env.close()

    result = summarize_episodes(episodes_data, agent1_module_name, agent2_module_name)

    #with open('evaluation.json', 'w') as f:
    #    json.dump(result, f, cls=SafeFallbackEncoder)
    
    return result


if __name__ == "__main__":
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    parser = argparse.ArgumentParser(description="Evaluation script soccer-twos.")
    parser.add_argument("-m", "--agent-module", help="Selfplay Agent Module")
    parser.add_argument("-m1", "--agent1-module", help="Team 1 Agent Module")
    parser.add_argument("-m2", "--agent2-module", help="Team 2 Agent Module")
    parser.add_argument("-e", "--episodes", type=int, default=200, help="Number of Episodes to Evaluate")
    parser.add_argument("-p", "--base-port", type=int, help="Base Communication Port")
    args = parser.parse_args()

    if args.agent_module:
        agent1_module_name = args.agent_module
        agent2_module_name = args.agent_module
    elif args.agent1_module and args.agent2_module:
        agent1_module_name = args.agent1_module
        agent2_module_name = args.agent2_module
    else:
        parser.print_help(sys.stderr)
        raise ValueError("Must specify selfplay (-m) or team (-m1, -m2) agent modules")

    # import agent modules
    logging.info(f"Loading {agent1_module_name}")
    logging.info(f"Loading {agent2_module_name}")
    logging.info(f"Number of Episodes to Evaluate {args.episodes}")
    logging.info(f"Base Communication Port {args.base_port}")

    result = evaluate(agent1_module_name, agent2_module_name, args.episodes, args.base_port)

    print(pretty_print(result))