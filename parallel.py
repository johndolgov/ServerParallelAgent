import ray
from ray import tune
import gym
from ray.rllib.models import ModelCatalog
import ray.rllib.agents.dqn as DQNAgent
import time
from itertools import zip_longest
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.policy_client import PolicyClient

@ray.remote
class CPEnviroment(object):
    def __init__(self):
        self.env = gym.make("CartPole-v0")
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        time.sleep(1)
        return self.env.step(action)
    
    def render(self):
        return self.env.render()

@ray.remote
def action(episode, obs):
    return client.get_action(episode, obs)

def step(env,action):
    return env.step.remote(action)

@ray.remote
def log_return_and_end(episode, rew, obs, done):
    client.log_returns(episode, rew)
    if done:
        client.end_episode(episode, obs)

def create_envs(cls, num_envs=4):
    ray.init()
    return [cls.remote() for _ in range(num_envs)]

def reset(envs):
    return [env.reset.remote() for env in envs]

def start_ep(client, num_envs=4):
    return [client.start_episode() for _ in range(num_envs)], [False for _ in range(num_envs)]

def run_episode(envs, client, num_envs=4):
    obsvs = reset(envs)
    eps, done = start_ep(client, num_envs)
    tot_rew = [0 for _ in range(num_envs)]
    while done or tot_rew[0]==30:
        print('Start')
        start = time.time()
        actions = ray.get([action.remote(ep, obs) for ep, obs in zip(eps, obsvs)])
        print('Action', time.time() - start)
        start = time.time()
        start = time.time()
        transitions = ray.get([step(env, action) for env, action in zip(envs, actions)])
        print('Step', time.time() - start)
        rews = [tr[1] for tr in  transitions]
        tot_rew = [sum(x) for x in zip_longest(rews, tot_rew, fillvalue=0)]
        done = [tr[2] for tr in  transitions]
        obsvs = [tr[0] for tr in transitions]
        start = time.time()
        ray.get([log_return_and_end.remote(ep , rew, obs, d) for ep, rew, obs, d in zip(eps, rews, obsvs, done)])
        print('LogEnd', time.time() - start)
        eps = [ep for ep, d in zip(eps, done) if not d]
        obsvs = [ray.put(tr[0]) for tr, d in zip(transitions, done) if not d]
        envs = [env for env, d in zip(envs, done) if not d]
        done = [d for d in done if not d]
    print("Episode reward", tot_rew)

if __name__ == '__main__':
	client = PolicyClient("http://localhost:9900")
	envs = create_envs(CPEnviroment, 10)

	for _ in range(20):
		start = time.time()
		run_episode(envs, client, 10)
		print('All', time.time()-start)