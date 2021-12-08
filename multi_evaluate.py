from typing import Match
from custom_evaluate import evaluate
from copy import deepcopy
import numpy as np
from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool as Pool
import itertools
import pandas as pd
from ray.tune.logger import pretty_print

goiabav2 = {
    'name': 'goiabav2',
    'cls': 'ppo_deepmind_selfplay_v2_1',
    'checkpoints': [
        800, 1200, 1500, 1877
    ]
}

goiabav3 = {
    'name': 'goiabav3',
    'cls': 'ppo_deepmind_selfplay_v2_2',
    'checkpoints': [
        800, 1200, 1500, 1877
    ]
}

goiabav4 = {
    'name': 'goiabav4',
    'cls': 'ppo_deepmind_selfplay_v4',
    'checkpoints': [
        2000, 3000, 4000, 4200, 4500, 4800, 5000, 5500, 6047
    ]
}

alphastar = {
    'name': 'pequistar',
    'cls': 'ppo_alphastar_v3',
    'checkpoints': [
        109, 217, 325, 397, 433
    ]
}

baseline = {
    'name': 'baseline',
    'cls': 'ceia_baseline_agent',
    'checkpoint': 'default'
}

runners = [goiabav2, goiabav3, goiabav4, alphastar]

players = []
for r in runners:
    #players.append([])
    for c in r['checkpoints']:
        p = deepcopy(r)
        p['checkpoint'] = c
        players.append(p)

def compete(players):
    p1, p2 = players
    print(f"Running {p1['name']}-{p1['checkpoint']} vs {p2['name']}-{p2['checkpoint']}")
    result = evaluate(p1['cls'], p2['cls'], str(p1['checkpoint']), str(p2['checkpoint']), n_episodes=300, worker_id=np.random.randint(0, 1000))
    print(pretty_print(result))
    return (p1, p2, result)


data = []
for player in players:
    result = compete((player, baseline))
    for p1, p2, r in [result]:
        data.append({
            'p1': p1['name'] + '-' + str(p1['checkpoint']),
            'p2': p2['name'] + '-' + str(p2['checkpoint']),
            'episode_len': r['episode_len_mean'],
            'p1_winrate': r['policies'][p1['cls']]['policy_win_rate'],
            'p2_winrate': r['policies'][p2['cls']]['policy_win_rate'],
            'result': r
        })
    pd.DataFrame(data).to_csv('result.csv', index=False)
