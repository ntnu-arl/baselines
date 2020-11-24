import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from baselines.a2c.utils import ortho_init
#from baselines.dagger.buffer import Buffer
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorboardX import SummaryWriter
import timeit
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from collections import deque
import gym
import gym_arc

batch_size = 32
steps = 2048
nb_training_epoch = 50
dagger_itr = 20 #200
gamma = 0.99 # Discount factor for future rewards
stddev = 0.1

import tensorflow as tf

class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

def get_teacher_action(expert, obs, action_space):
    action = expert.calculate(obs)
    #print('action before clip:', action)
    action = tf.clip_by_value(action, action_space.low, action_space.high)
    #print('action after clip:', action)
    action = np.array([action])
    return action

# build network
def build_actor_model(input_shape, output_shape):
    with tf.compat.v1.variable_scope('dagger_actor_model', reuse=tf.compat.v1.AUTO_REUSE):
        x_input = tf.keras.Input(shape=input_shape)
        h = x_input
        h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)), # too fancy initialization =))
                                name='mlp_fc1', activation=tf.tanh)(h)
        h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                name='mlp_fc2', activation=tf.tanh)(h)
        h = tf.keras.layers.Dense(units=output_shape, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                name='output', activation=tf.keras.activations.tanh)(h)
        #h = tf.keras.layers.Dense(units=output_shape, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
        #                           name='output')(h)

        model = tf.keras.Model(inputs=[x_input], outputs=[h])

        optimizer = tf.keras.optimizers.Adam(lr=1e-4)

        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mse'])
    return model

if __name__ == '__main__':
    #print('sys.argv:', sys.argv)
    tf.compat.v1.enable_eager_execution()
    play = False
    load_path = None
    save_path = None
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--play':
            play = True
            print('PLAY')
            continue
        index = sys.argv[i].find('--load_path')
        if (index != -1):
            load_path = sys.argv[i][12:]
            continue
        index = sys.argv[i].find('--save_path')
        if (index != -1):
            save_path = sys.argv[i][12:]
            continue 

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

    # Initialize env and expert
    env = gym.make('ARC-v0')

    nb_actions = env.action_space.shape[-1]
    nb_obs = env.observation_space.shape[-1]
    #print('Env nb_actions:', nb_actions, ", nb_obs:", nb_obs)

    # actor
    actor = build_actor_model(nb_obs, nb_actions)
    actor.summary()
    if (load_path != None):
        print(bcolors.OKBLUE + 'load model from path:', load_path, bcolors.ENDC)
        actor.load_weights(load_path)    

    writer = SummaryWriter(comment="-ARC_RL_dagger")

    episode_rew_queue = deque(maxlen=10)
    max_mean_return = -100000    

    if play:
        obs = env.reset()
        scaled_obs = obs / env.observation_space.high
        reward_sum = 0.0
        itr = 0
        while True:
            for i in range(steps):
                #print('obs:', obs)
                robot_state = np.reshape(obs, [1, nb_obs])
                #start = timeit.default_timer()
                action = actor(robot_state, training=False)  # assume symmetric action space (low = -high)
                #stop = timeit.default_timer()
                #print('Time for actor prediction: ', stop - start)

                action = tf.clip_by_value(action, -1.0, 1.0) # output from actor NN will actually be in the range (-1, 1)
                action = action.numpy()
                
                obs, reward, done, info = env.step(action[0] * env.action_space.high) # scale actor NN output to actual scale
                scaled_obs = obs / env.observation_space.high
                reward_sum += reward

                env.render(False)

                if done is True:
                    episode_rew_queue.appendleft(reward_sum)
                    reward_sum = 0.0
                    obs = env.reset()
                    scaled_obs = obs / env.observation_space.high

            mean_return = np.mean(episode_rew_queue)
            print('Itr (', steps, ' steps):', itr, ', mean return:', mean_return)
            writer.add_scalar("mean return", mean_return, itr)
            itr += 1 

    else: # training mode
        obs_all = np.zeros((0, nb_obs))
        actions_all = np.zeros((0, nb_actions))
        rewards_all = np.zeros((0, ))

        obs_list = []
        action_list = []
        reward_list = []

        # Collect data with expert in first iteration
        obs = env.reset()
        scaled_obs = obs / env.observation_space.high
        print('Collecting data...')
        for i in range(steps):
            expert_action = env.get_controller_result()
            obs_list.append(obs)
            action_list.append(env.encode_control(expert_action) / env.action_space.high)
            
            obs, reward, done, info = env.step(expert_action)
            
            scaled_obs = obs / env.observation_space.high
            reward_list.append(np.array([reward]))
            
            env.render(False)
            
            if done:
                obs = env.reset()

        print('Packing data into arrays...')
        for obs, act, rew in zip(obs_list, action_list, reward_list):
            obs_all = np.concatenate([obs_all, np.reshape(obs, [1,nb_obs])], axis=0)
            #print('actions_all:', actions_all, ', actions_all.shape:', actions_all.shape, ', act:', act, 'act.shape:', act.shape)
            actions_all = np.concatenate([actions_all, np.reshape(act, [1,nb_actions])], axis=0)
            rewards_all = np.concatenate([rewards_all, rew], axis=0)

        # First train for actor network
        actor.fit(obs_all, actions_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=0)

        # action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))

        # Aggregate and retrain actor network
        for itr in range(dagger_itr):
            obs_list = []
            action_list = []

            obs = env.reset()
            scaled_obs = obs / env.observation_space.high
            reward_sum = 0.0

            for i in range(steps):
                #print('obs:', obs)
                expert_action = env.get_controller_result()
                obs_list.append(obs)
                action_list.append(env.encode_control(expert_action) / env.action_space.high )
                
                #start = timeit.default_timer()
                action = actor(np.reshape(obs, [1,nb_obs]), training=False)  # assume symmetric action space (low = -high)
                #stop = timeit.default_timer()
                #print('Time for actor prediction: ', stop - start)        
                #print('action:', action) # action = [[.]]
                
                action = tf.clip_by_value(action, -1.0, 1.0)
                #action = action.eval(session=tf.compat.v1.Session())
                action = action.numpy()
                new_obs, reward, done, _ = env.step(action[0] * env.action_space.high)
                new_obs_scaled = new_obs / env.observation_space.high
                scaled_obs = new_obs_scaled
                obs = new_obs
                env.render(False)

                reward_sum += reward

                if done is True:
                    episode_rew_queue.appendleft(reward_sum)
                    reward_sum = 0
                    obs = env.reset()
                    scaled_obs = obs / env.observation_space.high
                    continue

            mean_return = np.mean(episode_rew_queue)
            print('Itr (', steps, ' steps):', itr, ', mean return:', mean_return)
            writer.add_scalar("mean return", mean_return, itr)

            #if i==(steps-1):
            #    break

            for obs, act in zip(obs_list, action_list):
                obs_all = np.concatenate([obs_all, np.reshape(obs, [1,nb_obs])], axis=0)
                actions_all = np.concatenate([actions_all, np.reshape(act, [1,nb_actions])], axis=0)

            # train actor
            actor.fit(obs_all, actions_all,
                        batch_size=batch_size,
                        epochs=nb_training_epoch,
                        shuffle=True,
                        verbose=0)

        actor.save_weights('dagger_actor_weight.h5')
        if (save_path != None):
            #actor.save('dagger_actor_pcl', include_optimizer=False) # should we include optimizer?
            print('save weights to file:', save_path)
            actor.save_weights(save_path + '/dagger_pcl.h5')