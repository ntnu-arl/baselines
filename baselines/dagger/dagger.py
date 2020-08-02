import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from baselines.a2c.utils import ortho_init
from baselines.common.rotors_wrappers import RotorsWrappers
from baselines.dagger.pid import PID
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorboardX import SummaryWriter
import timeit

batch_size = 32
steps = 1000
nb_training_epoch = 50
dagger_itr = 5

def get_teacher_action(expert, obs, action_space):
    action = expert.calculate(obs)
    #print('action before clip:', action)
    action = tf.clip_by_value(action, action_space.low, action_space.high)
    #print('action after clip:', action)
    action = np.array([action])
    return action

# build network
def build_model(input_shape, output_shape):
    x_input = tf.keras.Input(shape=input_shape)
    h = x_input
    h1 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc1', activation='relu')(h)
    h2 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc2', activation='relu')(h1)
    h3 = tf.keras.layers.Dense(units=output_shape, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                               name='output', activation=tf.keras.activations.tanh)(h2)

    model = tf.keras.Model(inputs=[x_input], outputs=[h3])

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model

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

# Initialize
env = RotorsWrappers()
expert = PID([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

nb_actions = env.action_space.shape[-1]
nb_obs = env.observation_space.shape[-1]

model = build_model(nb_obs, nb_actions)
model.summary()

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

# First train
model.fit(obs_all, actions_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=0)
output_file = open('results.txt', 'w')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
writer = SummaryWriter(comment="-rmf_dagger")

# Aggregate and retrain
for itr in range(dagger_itr):
    obs_list = []

    obs = env.reset()
    reward_sum = 0.0

    for i in range(steps): #??????????????????
        #print('obs:', obs)
        #start = timeit.default_timer()
        action = model(np.reshape(obs, [1,nb_obs]), training=False) * env.action_space.high # assume symmetric action space (low = -high)
        #stop = timeit.default_timer()
        #print('Time for actor prediction: ', stop - start)        
        #print('action:', action) # action = [[.]]
        obs, reward, done, _ = env.step(action)
        if done is True:
            obs = env.reset()
            continue
            #break
        else:
            obs_list.append(obs)
        reward_sum += reward
    env.pause()    
    print('Episode done ', 'itr ', itr, ',i ', i, 'reward sum ', reward_sum)
    output_file.write('Number of Steps: %02d\t Reward: %0.04f\n'%(i, reward_sum))
    writer.add_scalar("rewward_sum", reward_sum, itr)

    #if i==(steps-1):
    #    break

    for obs in obs_list:
        obs_all = np.concatenate([obs_all, np.reshape(obs, [1,nb_obs])], axis=0)
        actions_all = np.concatenate([actions_all, get_teacher_action(expert, obs, env.action_space)], axis=0)

    model.fit(obs_all, actions_all,
                  batch_size=batch_size,
                  epochs=nb_training_epoch,
                  shuffle=True,
                  validation_split=0.2, verbose=0,
                  callbacks=[early_stop, tfdocs.modeling.EpochDots()])

model.save('dagger_6state')   