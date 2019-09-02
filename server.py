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

SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900
CHECKPOINT_FILE = "test"

class CartPoleServer(ExternalEnv):
    def __init__(self, config):
    	super(CartPoleServer, self).__init__(action_space=spaces.Discrete(2), 
    		observation_space=spaces.Box(low=-10, high=10, shape=(4, ), dtype=np.float32))
    def run(self):
        print(f'Started server host:{SERVER_ADDRESS} port:{SERVER_PORT} with checkpoint: {CHECKPOINT_FILE}')
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
        time.sleep(1)
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

        time.sleep(1)
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

        time.sleep(1)
        episode = self._get(episode_id)
        self._finished.add(episode.episode_id)
        episode.done(observation)


if __name__ == '__main__':
	
	def env_creator():
		return CartPoleServer()

	ray.init()
	register_env('srv', lambda config: CartPoleServer(config))

	ModelCatalog.register_custom_model("CM", CustomModel)
	dqn = DQNAgent(env='srv', config={'num_workers': 0, 'model': {'custom_model': 'CM', 'custom_options': {}, },'learning_starts': 150})
	if os.path.exists(CHECKPOINT_FILE):
		checkpoint_path = open(CHECKPOINT_FILE).read()
		print("Restoring from checkpoint path", checkpoint_path)
		dqn.restore(checkpoint_path)

	while True:
		print(pretty_print(dqn.train()))
		checkpoint_path = dqn.save()
		print("Last checkpoint", checkpoint_path)
		with open(CHECKPOINT_FILE, "w") as f:
			f.write(checkpoint_path)