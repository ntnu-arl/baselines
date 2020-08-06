import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from baselines.a2c.utils import ortho_init
from baselines.common.rotors_wrappers import RotorsWrappers
from baselines.dagger.pid import PID
#from baselines.dagger.buffer import Buffer
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorboardX import SummaryWriter
import timeit

batch_size = 32
steps = 100
nb_training_epoch = 50
dagger_itr = 100
critic_lr = 1e-3
total_episodes = 100
gamma = 0.99 # Discount factor for future rewards
tau = 0.005 # Used to update target networks
buffer_capacity=50000
batch_size=64

import tensorflow as tf

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, nb_obs))
        self.action_buffer = np.zeros((self.buffer_capacity, nb_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, nb_obs))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = actor(next_state_batch) # from dagger, no target actor
            y = reward_batch + gamma * target_critic(tf.concat([next_state_batch, tf.cast(target_actions, dtype='float64')], axis=-1)) #([next_state_batch, target_actions])
            critic_value = critic(tf.concat([state_batch, tf.cast(action_batch, dtype='float64')], axis=-1)) #([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic.trainable_variables)
        )

        # with tf.GradientTape() as tape:
        #     actions = actor_model(state_batch)
        #     critic_value = critic_model([state_batch, actions])
        #     # Used `-value` as we want to maximize the value given
        #     # by the critic for our actions
        #     actor_loss = -tf.math.reduce_mean(critic_value)

        # actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        # actor_optimizer.apply_gradients(
        #     zip(actor_grad, actor_model.trainable_variables)
        # )
        return critic_loss

def get_teacher_action(expert, obs, action_space):
    action = expert.calculate(obs)
    #print('action before clip:', action)
    action = tf.clip_by_value(action, action_space.low, action_space.high)
    #print('action after clip:', action)
    action = np.array([action])
    return action

# build network
def build_actor_model(input_shape, output_shape): # BatchNormalization??????????
    x_input = tf.keras.Input(shape=input_shape)
    h = x_input
    h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)), # too fancy initialization =))
                               name='mlp_fc1', activation='relu')(h)
    h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc2', activation='relu')(h)
    h = tf.keras.layers.Dense(units=output_shape, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                               name='output', activation=tf.keras.activations.tanh)(h)

    model = tf.keras.Model(inputs=[x_input], outputs=[h])

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model

def build_critic_model(input_shape):
    x_input = tf.keras.Input(shape=input_shape)
    h = x_input
    h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc1', activation='relu')(h)
    h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc2', activation='relu')(h)
    h = tf.keras.layers.Dense(units=1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                               name='output')(h)

    model = tf.keras.Model(inputs=[x_input], outputs=[h])

    #optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    # model.compile(loss='mse',
    #               optimizer=optimizer,
    #               metrics=['mse'])
    return model    

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

# Limiting GPU memory growth: https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#tf.keras.backend.set_floatx('float32')

# Initialize
env = RotorsWrappers()
expert = PID([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

nb_actions = env.action_space.shape[-1]
nb_obs = env.observation_space.shape[-1]

# actor
actor = build_actor_model(nb_obs, nb_actions)
actor.summary()

# critic
critic = build_critic_model(nb_obs + nb_actions)
critic.summary()

target_critic = build_critic_model(nb_obs + nb_actions)
target_critic.set_weights(critic.get_weights())

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

# buffer
buffer = Buffer(buffer_capacity, batch_size)

obs_all = np.zeros((0, nb_obs))
actions_all = np.zeros((0, nb_actions))
rewards_all = np.zeros((0, ))

obs_list = []
action_list = []
reward_list = []

# Collect data with expert in first iteration
obs = env.reset()
print('Collecting data...')
for i in range(steps):
    # if i == 0:
    #     act = np.array([0.0])
    # else:
    #     act = get_teacher_action(ob)

    action = get_teacher_action(expert, obs, env.action_space)
    obs, reward, done, _ = env.step(action)
    obs_list.append(obs)
    action_list.append(action)
    reward_list.append(np.array([reward]))

    if done:
        obs = env.reset()

env.pause()

print('Packing data into arrays...')
for obs, act, rew in zip(obs_list, action_list, reward_list):
    obs_all = np.concatenate([obs_all, np.reshape(obs, [1,nb_obs])], axis=0)
    actions_all = np.concatenate([actions_all, act], axis=0)
    rewards_all = np.concatenate([rewards_all, rew], axis=0)

# First train for actor network
actor.fit(obs_all, actions_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=0)
output_file = open('results.txt', 'w')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
writer = SummaryWriter(comment="-rmf_dagger")

# Aggregate and retrain actor network
for itr in range(dagger_itr):
    obs_list = []

    obs = env.reset()
    reward_sum = 0.0

    epoch_actor_losses = []
    for i in range(steps): #??????????????????
        #print('obs:', obs)
        #start = timeit.default_timer()
        action = actor(np.reshape(obs, [1,nb_obs]), training=False) * env.action_space.high # assume symmetric action space (low = -high)
        #stop = timeit.default_timer()
        #print('Time for actor prediction: ', stop - start)        
        #print('action:', action) # action = [[.]]
        new_obs, reward, done, _ = env.step(action)
        buffer.record((obs, action, reward, new_obs))
        obs = new_obs

        reward_sum += reward

        # train critic
        if (buffer.buffer_counter >= buffer.batch_size):
            actor_loss = buffer.learn()
            update_target(tau)
            epoch_actor_losses.append(actor_loss)

        if done is True:
            obs = env.reset()
            continue
        else:
            obs_list.append(obs)

    env.pause()    
    print('Episode done ', 'itr ', itr, ',i ', i, 'reward sum ', reward_sum)
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n'%(i, reward_sum))
    writer.add_scalar("rewward_sum", reward_sum, itr)
    writer.add_scalar("actor_losses", np.mean(epoch_actor_losses), itr)

    #if i==(steps-1):
    #    break

    for obs in obs_list:
        obs_all = np.concatenate([obs_all, np.reshape(obs, [1,nb_obs])], axis=0)
        actions_all = np.concatenate([actions_all, get_teacher_action(expert, obs, env.action_space)], axis=0)

    # train actor
    actor.fit(obs_all, actions_all,
                  batch_size=batch_size,
                  epochs=nb_training_epoch,
                  shuffle=True,
                  validation_split=0.2, verbose=0,
                  callbacks=[early_stop, tfdocs.modeling.EpochDots()])              

actor.save('dagger_actor_6state', include_optimizer=False) # should we include optimizer?
actor.save_weights('weight_actor_6state.h5')

critic.save('dagger_critic_6state', include_optimizer=False) # should we include optimizer?
critic.save_weights('weight_critic_6state.h5')