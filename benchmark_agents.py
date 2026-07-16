import argparse

from elevator_animation import ElevatorAnimation
from agents.standard_elevator_v7_controller import StandardElevatorV7Controller
from envs.elevator_v7 import ElevatorV7Env
from stable_baselines3 import PPO
from tqdm import tqdm
from multiprocessing import Process

import torch
import numpy as np
import random


def benchmark_agent(model_filepath, num_episodes=100, num_elevators_start=1, num_elevators_end=1, num_floors_start=3, num_floors_end=3, animation_delay=-1):
    RANDOM_SEED = 456
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    env = ElevatorV7Env(curriculum=True,
                        num_elevators_start=num_elevators_start,
                        num_elevators_end=num_elevators_end,
                        num_floors_start=num_floors_start,
                        num_floors_end=num_floors_end,
                        episode_len=100,
                        random_seed=RANDOM_SEED)

    if model_filepath == "Standard Controller":
        model = StandardElevatorV7Controller(env)
    else:
        model = PPO.load(model_filepath, env=env)

    print(f"Benchmarking {model_filepath}")

    if animation_delay >= 0:
        animation = ElevatorAnimation(env, title=model_filepath, delay=animation_delay, padding=150, width=1000)

    all_rewards = []
    for _ in tqdm(range(num_episodes)):
        obs = env.reset(override_curriculum=True)
        done = False
        reward_sum = 0
        while not done:
            # elevator animation
            if animation_delay >= 0:
                animation.set_environment(env)
                animation.draw_environment()

            # get action and update environment
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
        all_rewards.append(reward_sum)

    all_rewards = np.array(all_rewards)
    mean = np.mean(all_rewards)
    std = np.std(all_rewards)

    print(f"{model_filepath}: {mean=}, {std=}")


def main(args):
    num_floors = int(args.num_floors)
    models = args.models
    models.append("Standard Controller")
    params = {
        'num_episodes': 100,
        'num_elevators_start': 1,
        'num_elevators_end': 1,
        'num_floors_start': num_floors,
        'num_floors_end': num_floors,
        'animation_delay': float(args.animation_delay)
    }

    procs = []

    for model_name in models:
        proc = Process(target=benchmark_agent, args=(model_name,), kwargs=params)
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_floors', '-n', default=3, help='number of floors for environment')
    parser.add_argument('--models', '-m', nargs='+', help='Paths to models to be evaluated', required=True)
    parser.add_argument('--animation_delay', '-a', default=-1, help='Animation delay (in seconds) between timesteps, animation off by default')
    args = parser.parse_args()
    main(args)
