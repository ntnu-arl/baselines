import tensorflow as tf
from baselines.common.models import get_network_builder
from baselines.a2c.utils import ortho_init, conv
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, name, network='mlp', **network_kwargs):
        super(Model, self).__init__(name=name)
        self.network = network
        self.network_kwargs = network_kwargs

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_variables if 'layer_normalization' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, ob_robot_state_shape, ob_pcl_shape, name='actor', network='mlp_rmf_actor', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions
        self.robot_state_shape = ob_robot_state_shape
        self.pcl_shape = ob_pcl_shape
        # input layer
        robot_state_input = tf.keras.Input(shape=ob_robot_state_shape)
        # CNN layers for pcl
        self.pcl_encoder = get_network_builder('pcl_encoder')()(input_shape=ob_pcl_shape)
        x = tf.keras.layers.concatenate([robot_state_input, self.pcl_encoder.outputs[0]])
        # FC layers
        h1 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='fc1', activation='relu')(x)
        h2 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='fc2', activation='relu')(h1) 
        #print('pcl enoder input:', self.pcl_encoder.inputs[0])
        #print('pcl enoder output:', self.pcl_encoder.outputs[0])
        self.actor_sub_net = tf.keras.Model(inputs=[robot_state_input, self.pcl_encoder.inputs[0]], outputs=[h2], name='actor_sub_net')
        print('actor sub net summary')
        self.actor_sub_net.summary()
        #tf.keras.utils.plot_model(self.actor_sub_net, "actor_sub_net.png", show_shapes=True)
        # output layer
        self.output_layer = tf.keras.layers.Dense(units=self.nb_actions,
                                                  activation=tf.keras.activations.tanh,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        _ = self.output_layer(self.actor_sub_net.outputs[0])

    @tf.function
    def call(self, obs):
        #print('obs shape:', obs.shape)
        #print('input layers:', self.actor_sub_net.layers)
        return self.output_layer(self.actor_sub_net([obs[:,0:self.robot_state_shape], obs[:,self.robot_state_shape:]]))


class Critic(Model):
    def __init__(self, nb_actions, ob_robot_state_shape, ob_pcl_shape, name='critic', network='mlp_rmf_critic', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True
        self.robot_state_shape = ob_robot_state_shape
        # self.network_builder = get_network_builder(network)(**network_kwargs)((ob_shape[0] + nb_actions,))
        # input layer
        robot_state_input = tf.keras.Input(shape=ob_robot_state_shape)
        action_input = tf.keras.Input(shape=nb_actions)
        # CNN layers for pcl
        self.pcl_encoder = get_network_builder('pcl_encoder')()(input_shape=ob_pcl_shape) # (4,15,4)
        x = tf.keras.layers.concatenate([robot_state_input, self.pcl_encoder.outputs[0], action_input])
        # FC layers
        h1 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='fc1', activation='relu')(x)
        h2 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='fc2', activation='relu')(h1) 

        self.critic_sub_net = tf.keras.Model(inputs=[robot_state_input, self.pcl_encoder.inputs[0], action_input], outputs=[h2], name='critic_sub_net')
        print('critic sub net summary')
        self.critic_sub_net.summary()
        #tf.keras.utils.plot_model(self.critic_sub_net, "critic_sub_net.png", show_shapes=True)
        # output layer        
        self.output_layer = tf.keras.layers.Dense(units=1,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                                  name='output')
        _ = self.output_layer(self.critic_sub_net.outputs[0])

    @tf.function
    def call(self, obs, actions):
        #x = tf.concat([obs, actions], axis=-1) # this assumes observation and action can be concatenated
        x = self.critic_sub_net([obs[:,0:self.robot_state_shape], obs[:,self.robot_state_shape:], actions])
        return self.output_layer(x)

    @property
    def output_vars(self):
        return self.output_layer.trainable_variables
