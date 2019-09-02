import ray
from ray import tune
import gym
from ray.rllib.models import ModelCatalog
import ray.rllib.agents.dqn as DQNAgent
import time
from itertools import zip_longest
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.policy_client import PolicyClient
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--num-episodes", type=int, required=True)

class CPEnviroment(object):
    def __init__(self):
        self.env = gym.make("CartPole-v0")
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()

def get_action(episode, obs):
    return client.get_action(episode, obs)

def step(env,action):
    return env.step(action)

def log_return_and_end(episode, rew, obs, done):
    client.log_returns(episode, rew)
    if done:
        client.end_episode(episode, obs)

def create_envs(cls, num_envs=4):
    ray.init()
    return [cls() for _ in range(num_envs)]

def reset(env):
    return env.reset()

def start_ep(client, num_envs=4):
    return client.start_episode(), False

@ray.remote
def run_episode(env, client):
    start=time.time()
    obs = reset(env)
    ep, done = start_ep(client)
    tot_rew = 0
    time.sleep(5)
    while not done:
        action = get_action(ep, obs)
        transition = step(env, action)
        rew = transition[1]
        tot_rew += rew
        done = transition[2]
        obs = transition[0]
        log_return_and_end(ep, rew, obs, done)
    print("Episode reward", tot_rew)

if __name__ == '__main__':
    args = parser.parse_args()
    client = PolicyClient("http://localhost:9900")
    envs = create_envs(CPEnviroment, args.num_envs)
    start = time.time()
    for _ in range(args.num_episodes):
        ray.get([run_episode.remote(env, client) for env in envs])
    print('All', time.time()-start)