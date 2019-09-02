import ray
from ray.rllib.agents.dqn import DQNAgent
from model import CustomModel
from ray.rllib.models import ModelCatalog
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.serving_env import ServingEnv
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.remote_vector_env import RemoteVectorEnv
from ray.rllib.utils.policy_server import PolicyServer
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
import os
from gym import spaces
import numpy as np
import time
import argparse

SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900

parser = argparse.ArgumentParser()
parser.add_argument("--action-size", type=int, required=True)
parser.add_argument("--observation-size", type=int, required=True)
parser.add_argument("--checkpoint-file", type=str, required=True)

class CartPoleServer(ExternalEnv):
    def __init__(self, config):
        self.config = config
        ExternalEnv.__init__(self, action_space=spaces.Discrete(config['action_size']), 
            observation_space=spaces.Box(low=-10, high=10, shape=(config['observation_size'], ), dtype=np.float32))

    def run(self):
        print(f'Started server host:{SERVER_ADDRESS} port:{SERVER_PORT} with checkpoint: {self.config["checkpoint_file"]}')
        server = PolicyServer(self, SERVER_ADDRESS, SERVER_PORT)
        server.serve_forever()

    def get_action(self, episode_id, observation):
        """Record an observation and get the on-policy action.
        Arguments:
            episode_id (str): Episode id returned from start_episode().
            observation (obj): Current environment observation.
        Returns:
            action (obj): Action from the env action space.
        """
        #time.sleep(1)
        print('Get action to episode', episode_id)
        episode = self._get(episode_id)
        return episode.wait_for_action(observation)

    def log_returns(self, episode_id, reward, info=None):
        """Record returns from the environment.
        The reward will be attributed to the previous action taken by the
        episode. Rewards accumulate until the next action. If no reward is
        logged before the next action, a reward of 0.0 is assumed.
        Arguments:
            episode_id (str): Episode id returned from start_episode().
            reward (float): Reward from the environment.
            info (dict): Optional info dict.
        """

        #time.sleep(1)
        episode = self._get(episode_id)
        episode.cur_reward += reward
        if info:
            episode.cur_info = info or {}

    def end_episode(self, episode_id, observation):
        """Record the end of an episode.
        Arguments:
            episode_id (str): Episode id returned from start_episode().
            observation (obj): Current environment observation.
        """

        #time.sleep(1)
        episode = self._get(episode_id)
        self._finished.add(episode.episode_id)
        episode.done(observation)


if __name__ == '__main__':
	
    args = parser.parse_args()
    ray.init()
    register_env('srv', lambda config: CartPoleServer(config))

    ModelCatalog.register_custom_model("CM", CustomModel)
    dqn = DQNAgent(env='srv', config={'num_workers': 0, 'env_config': {'observation_size': args.observation_size, 'action_size': args.action_size, 'checkpoint_file': args.checkpoint_file},'model': {'custom_model': 'CM', 'custom_options': {}, },'learning_starts': 150})
    if os.path.exists(args.checkpoint_file):
        checkpoint_path = open(args.checkpoint_file).read()
        print("Restoring from checkpoint path", checkpoint_path)
        dqn.restore(checkpoint_path)

    while True:
        print(pretty_print(dqn.train()))
        checkpoint_path = dqn.save()
        print("Last checkpoint", checkpoint_path)
        with open(args.checkpoint_file, "w") as f:
            f.write(checkpoint_path)