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
from parallel import CPEnviroment, get_action, step, log_return_and_end, create_envs, reset, start_ep

parser = argparse.ArgumentParser()
parser.add_argument("--num-episodes", type=int, required=True)

def run_episode(env, client):
	start=time.time()
	obs = reset(env)
	ep, done = start_ep(client)
	tot_rew = 0
	time.sleep(5)
	while not done:
		env.render()
		action = get_action(client, ep, obs)
		transition = step(env, action)
		rew = transition[1]
		tot_rew += rew
		done = transition[2]
		obs = transition[0]
		log_return_and_end(client, ep, rew, obs, done)
	print("Episode reward", tot_rew)

if __name__ == '__main__':
	is_test = True
	args = parser.parse_args()
	client = PolicyClient("http://localhost:9900")
	envs = create_envs(CPEnviroment, 1)
	start = time.time()
	for _ in range(args.num_episodes):
		run_episode(envs[0], client)
	print('All', time.time()-start)