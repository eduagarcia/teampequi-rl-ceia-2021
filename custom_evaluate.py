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
from evaluate import collect_episodes, summarize_episodes, get_agent_class

def load_agent(agent_module_name: str, checkpoint: str = 'default', base_port = None) -> AgentInterface:
    """Loads a AgentInterface based on his module name"""

    agent_module = importlib.import_module(agent_module_name)

    env = soccer_twos.make(base_port=base_port)

    if checkpoint == 'default':
        agent = get_agent_class(agent_module)(env)
    else:
        agent = get_agent_class(agent_module)(env, checkpoint)

    env.close()

    return agent


def evaluate(
    agent1_module_name: str,
    agent2_module_name: str = None,
    agent1_checkpoint: str = 'default',
    agent2_checkpoint: str = 'default',
    n_episodes: int = 200,
    base_port = 50039,
    worker_id = 0
) -> Dict:
    """Evaluates two agents against each other"""

    if agent2_module_name is None:
        agent2_module_name = agent1_module_name
    else:
        agent2_module_name = agent2_module_name

    agent1 = load_agent(agent1_module_name, agent1_checkpoint, base_port=base_port)
    agent2 = load_agent(agent2_module_name, agent2_checkpoint, base_port=base_port)

    env = soccer_twos.make(
        base_port=base_port, worker_id=worker_id
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
    parser.add_argument("-c", "--checkpoint", help="Checkpoint Number")
    parser.add_argument("-c1", "--checkpoint-1", help="Checkpoint Number for Agent 1")
    parser.add_argument("-c2", "--checkpoint-2", help="Checkpoint Number for Agent 2")
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

    if args.checkpoint:
        checkpoint_1 = args.checkpoint
        checkpoint_2 = args.checkpoint
    elif args.checkpoint_1 and args.checkpoint_2:
        checkpoint_1 = args.checkpoint_1
        checkpoint_2 = args.checkpoint_2
    else:
        checkpoint_1 = "default"
        checkpoint_2 = "default"

    # import agent modules
    logging.info(f"Loading {agent1_module_name}-{checkpoint_1}")
    logging.info(f"Loading {agent2_module_name}-{checkpoint_2}")
    logging.info(f"Number of Episodes to Evaluate {args.episodes}")
    logging.info(f"Base Communication Port {args.base_port}")

    result = evaluate(
        agent1_module_name,
        agent2_module_name,
        checkpoint_1,
        checkpoint_2,
        args.episodes,
        args.base_port)

    print(pretty_print(result))