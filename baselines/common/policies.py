import tensorflow as tf
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.dagger.dagger import build_actor_model
import gym

DAGGER_ACTOR_WEIGHT = '/home/arl/models/dagger/dagger_pcl_20_0_1s.h5'

class PolicyWithValue(tf.Module):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        Parameters:
        ----------
        ac_space        action space

        policy_network  keras network for policy

        value_network   keras network for value

        estimate_q      q value or v value

        """

        self.policy_network = policy_network # not including final layer

        # MODIFY: load weights from dagger actor
        print('PolicyWithValue: trpo actor model')
        self.policy_network.summary()
        print('PolicyWithValue: dagger actor model')
        self.dagger_model = build_actor_model(8, (8, 45, 4), 3)
        self.dagger_model.summary()
        self.dagger_model.load_weights(DAGGER_ACTOR_WEIGHT)
        self.policy_network.get_layer('conv1').set_weights(self.dagger_model.get_layer('conv1').get_weights())
        self.policy_network.get_layer('conv2').set_weights(self.dagger_model.get_layer('conv2').get_weights())
        self.policy_network.get_layer('fc1').set_weights(self.dagger_model.get_layer('fc1').get_weights())
        self.policy_network.get_layer('fc2').set_weights(self.dagger_model.get_layer('fc2').get_weights())

        self.value_network = value_network or policy_network # not including final layer
        self.estimate_q = estimate_q
        self.initial_state = None

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(policy_network.output_shape, ac_space, init_scale=0.01, activation=tf.keras.activations.tanh)
        self.pdtype.matching_fc.set_weights(self.dagger_model.get_layer('output').get_weights())

        if estimate_q:
            assert isinstance(ac_space, gym.spaces.Discrete)
            self.value_fc = fc(self.value_network.output_shape, 'q', ac_space.n)
        else:
            self.value_fc = fc(self.value_network.output_shape, 'vf', 1)

    @tf.function
    def step(self, observation):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     batched observation data

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """
        #print('observation.shape:', observation.shape)
        latent = self.policy_network([observation[:,0:8], observation[:,8:]])
        pd, pi = self.pdtype.pdfromlatent(latent)
        action = pd.sample()
        #action = pi # when play
        neglogp = pd.neglogp(action)
        value_latent = self.value_network([observation[:,0:8], observation[:,8:]])
        vf = tf.squeeze(self.value_fc(value_latent), axis=1)
        return action, vf, None, neglogp

    @tf.function
    def value(self, observation):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        Returns:
        -------
        value estimate
        """
        value_latent = self.value_network([observation[:,0:8], observation[:,8:]])
        result = tf.squeeze(self.value_fc(value_latent), axis=1)
        return result

