from pettingzoo.utils import parallel_to_aec
from pettingzoo.utils.env import ParallelEnv

import random
import time
import numpy as np

def performance_benchmark(env):
    """
    Some simple modifications over PettingZoo's `performance_benchmark` (e.g. we return some of the data).
    """
    cycles = 0
    turn = 0
    env.reset()
    start = time.time()
    end = 0

    while True:
        cycles += 1
        for agent in env.agent_iter(
            env.num_agents
        ):  # step through every agent once with observe=True
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]).tolist())
            else:
                action = env.action_space(agent).sample()
            env.step(action)
            turn += 1

            if all(env.terminations.values()) or all(env.truncations.values()):
                env.reset()

        if time.time() - start > 5:
            end = time.time()
            break

    length = end - start

    turns_per_time = turn / length
    cycles_per_time = cycles / length
    return turns_per_time, cycles_per_time

def benchmark_env(env, iterations=3):
    """
    Provides benchmarking for PettingZoo but with a customisable number of iterations
    """
    # Ensure we have an AEC env
    e = env
    if isinstance(e, ParallelEnv):
        e = parallel_to_aec(env)

    turns_results = []
    cycles_results = []

    # Allow iterations to be either an int or an iterable
    if isinstance(iterations, int):
        run_iter = range(iterations)
    else:
        run_iter = iterations

    n_runs = 0
    for i in run_iter:
        print(f"Starting performance benchmark {i}")
        turns_per_time, cycles_per_time = performance_benchmark(e)
        turns_results.append(turns_per_time)
        cycles_results.append(cycles_per_time)
        n_runs += 1

    if n_runs == 0:
        print("No benchmark runs executed.")
        return 0.0, 0.0

    avg_turns = float(np.mean(turns_results))
    avg_cycles = float(np.mean(cycles_results))

    print(f"Average over {n_runs} runs:")
    print(f"  turns per time:  {avg_turns}")
    print(f"  cycles per time: {avg_cycles}")

    return avg_turns, avg_cycles
