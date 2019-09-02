import tensorflow as tf
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel


class CustomModel(DistributionalQModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)

        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            32,
            name="my_layer1",
            activation=tf.nn.relu)(self.inputs)
        layer_2 = tf.keras.layers.Dense(
            64,
            name="my_layer2",
            activation=tf.nn.relu)(layer_1)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=tf.nn.relu)(layer_2)
        self.base_model = tf.keras.Model(self.inputs, layer_out)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict['obs'])
        return model_out, state




