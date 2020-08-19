import sys
import rospy
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
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from collections import deque
from planner_msgs.srv import planner_search, planner_searchRequest
from geometry_msgs.msg import Pose

batch_size = 32
steps = 2000
nb_training_epoch = 50
dagger_itr = 100
dagger_buffer_size = 40000
gamma = 0.99 # Discount factor for future rewards
tau = 0.001 # Used to update target networks
buffer_capacity=50000 # unused now!
stddev = 0.1

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

        self.cnt = 0

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
            y = reward_batch + gamma * target_critic(tf.concat([next_state_batch, tf.cast(target_actions, dtype='float64')], axis=-1))
            critic_value = critic(tf.concat([state_batch, tf.cast(action_batch, dtype='float64')], axis=-1))
            #y = reward_batch + gamma * target_critic([next_state_batch, tf.cast(target_actions, dtype='float64')])
            #critic_value = critic([state_batch, tf.cast(action_batch, dtype='float64')])             
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            # self.cnt = self.cnt + 1
            # if (self.cnt == 100):
            #     self.cnt = 0
            #     print('y:', y)
            #     print('critic_value:', critic_value)

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

def get_teacher_action(planner, controller, augmented_obs, action_space):
    request = planner_searchRequest()
    request.use_current_state = False
    source = Pose()
    source.position.x = augmented_obs[0]
    source.position.y = augmented_obs[1]
    source.position.z = augmented_obs[2]
    source.orientation.x = 0.0
    source.orientation.y = 0.0
    source.orientation.z = 0.0
    source.orientation.w = 1.0
    target = Pose()
    target.position.x = augmented_obs[6]
    target.position.y = augmented_obs[7]
    target.position.z = augmented_obs[8]
    target.orientation.x = 0.0
    target.orientation.y = 0.0
    target.orientation.z = 0.0
    target.orientation.w = 1.0
    request.source = source
    request.target = target
    response = planner(request)
    path = response.path

    if (len(path) > 1):
        waypoint = path[1].position
        vel_setpoint = np.array([waypoint.x - augmented_obs[0], waypoint.y - augmented_obs[1], waypoint.z - augmented_obs[2]])
        if (len(path) > 5):
            vel_magnitude = 1.0
        else:
            vel_magnitude = 0.15 * (len(path) - 1)
        vel_setpoint_norm = np.linalg.norm(vel_setpoint)
        if (vel_setpoint_norm != 0):
            vel_setpoint = vel_setpoint * vel_magnitude / vel_setpoint_norm

        obs = np.array([waypoint.x - augmented_obs[0], waypoint.y - augmented_obs[1], waypoint.z - augmented_obs[2],
                    vel_setpoint[0] - augmented_obs[3], vel_setpoint[1] - augmented_obs[4], vel_setpoint[2] - augmented_obs[5]])
        action = controller.calculate(obs)
        action = tf.clip_by_value(action, -1.0, 1.0)
        action = np.array([action])
    elif (len(path) == 1):
        action = np.array([[0.0, 0.0, 0.0]])
    else: # cannot find a path
        action = np.array([])        
    return action

def pcl_encoder(input_shape):
    print('pcl_encoder input shape is {}'.format(input_shape))
    inputs = tf.keras.Input(shape=input_shape[0] * input_shape[1] * input_shape[2])
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], input_shape[2]))(inputs)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name='conv1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 3), padding='same', name='pool1')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 3), padding='same', name='pool2')(x)
    # Generate the latent vector
    latent = tf.keras.layers.Flatten()(x)
    encoder = tf.keras.Model(inputs, latent, name='encoder')
    #encoder.summary()
    return encoder

# build network
def build_backbone(ob_robot_state_shape, ob_pcl_shape):
    # input layer
    robot_state_input = tf.keras.Input(shape=ob_robot_state_shape)
    # CNN layers for pcl
    pcl_encoder_submodel = pcl_encoder(input_shape=ob_pcl_shape)
    x = tf.keras.layers.concatenate([robot_state_input, pcl_encoder_submodel.outputs[0]])
    # FC layers
    h1 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                name='fc1', activation='relu')(x)
    h2 = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                                name='fc2', activation='relu')(h1)
    backbone_model = tf.keras.Model(inputs=[robot_state_input, pcl_encoder_submodel.inputs[0]], outputs=[h2], name='backbone_net')
    return backbone_model

def build_actor_model(ob_robot_state_shape, ob_pcl_shape, nb_actions):
    backbone = build_backbone(ob_robot_state_shape, ob_pcl_shape)
    # output layer
    output_layer = tf.keras.layers.Dense(units=nb_actions,
                                        name='output',
                                        activation=tf.keras.activations.tanh,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))(backbone.outputs[0])
    model = tf.keras.Model(inputs=[backbone.inputs], outputs=[output_layer], name='actor_net')
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model

def build_critic_model(input_shape):
    x_input = tf.keras.Input(shape=input_shape)
    h = x_input
    h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc1', activation=tf.keras.activations.tanh)(h)
    h = tf.keras.layers.Dense(units=64, kernel_initializer=ortho_init(np.sqrt(2)),
                               name='mlp_fc2', activation=tf.keras.activations.tanh)(h)
    h = tf.keras.layers.Dense(units=1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                               name='output')(h)

    model = tf.keras.Model(inputs=[x_input], outputs=[h])

    #optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    # model.compile(loss='mse',
    #               optimizer=optimizer,
    #               metrics=['mse'])
    return model

# def build_critic_model(input_shape):
#     # State as input
#     state_input = layers.Input(shape=(6))
#     state_out = layers.Dense(16, activation="relu")(state_input)
#     state_out = layers.BatchNormalization()(state_out)
#     state_out = layers.Dense(32, activation="relu")(state_out)
#     state_out = layers.BatchNormalization()(state_out)

#     # Action as input
#     action_input = layers.Input(shape=(3))
#     action_out = layers.Dense(32, activation="relu")(action_input)
#     action_out = layers.BatchNormalization()(action_out)

#     # Both are passed through seperate layer before concatenating
#     concat = layers.Concatenate()([state_out, action_out])

#     out = layers.Dense(64, activation="relu")(concat)
#     out = layers.BatchNormalization()(out)
#     out = layers.Dense(64, ac0.6tivation="relu")(out)
#     out = layers.BatchNormalization()(out)
#     outputs = layers.Dense(1)(out)

#     # Outputs single value for give state-action
#     model = tf.keras.Model([state_input, action_input], outputs)

#     return model

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))

    target_critic.set_weights(new_weights)

if __name__ == '__main__':
    #print('sys.argv:', sys.argv)
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

    # Initialize
    env = RotorsWrappers()
    planner_service = rospy.ServiceProxy('/gbplanner/search', planner_search)

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

    pid = PID([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

    nb_actions = env.action_space.shape[-1]
    nb_obs = env.observation_space.shape[-1]

    # actor
    actor = build_actor_model(ob_robot_state_shape=env.ob_robot_state_shape, ob_pcl_shape=env.ob_pcl_shape, nb_actions=nb_actions)
    actor.summary()
    if (load_path != None):
        print('load model from path:', load_path)
        actor.load_weights(load_path)

    writer = SummaryWriter(comment="-rmf_dagger_pcl")

    episode_rew_queue = deque(maxlen=10)

    max_mean_return = -100000

    if play:
        obs = env.reset()
        reward_sum = 0.0
        itr = 0
        while True:
            for i in range(steps):
                #print('obs:', obs)
                robot_state = np.reshape(obs[0:env.ob_robot_state_shape], [1, env.ob_robot_state_shape])
                pcl_feature = np.reshape(obs[env.ob_robot_state_shape:], [1, env.pcl_feature_size])
                #start = timeit.default_timer()
                action = actor([robot_state, pcl_feature], training=False)  # assume symmetric action space (low = -high)
                #stop = timeit.default_timer()
                #print('Time for actor prediction: ', stop - start)

                action = tf.clip_by_value(action, -1.0, 1.0)

                obs, reward, done, _ = env.step(action * env.action_space.high)
                reward_sum += reward

                if done is True:
                    episode_rew_queue.appendleft(reward_sum)
                    reward_sum = 0.0
                    obs = env.reset()

            mean_return = np.mean(episode_rew_queue)
            print('Episode done ', 'itr ', itr, 'mean return', mean_return)
            writer.add_scalar("mean return", mean_return, itr)
            itr += 1 

    else: # training mode
        robot_state_all = np.zeros((0, env.ob_robot_state_shape))
        pcl_feature_all = np.zeros((0, env.pcl_feature_size))
        print('robot_state_all:', robot_state_all)
        print('robot_state_all.shape:', robot_state_all.shape)
        actions_all = np.zeros((0, nb_actions))
        rewards_all = np.zeros((0, ))

        obs_list = []
        augmented_obs_list = []
        action_list = []
        reward_list = []

        # Collect data with expert in first iteration
        obs = env.reset()
        augmented_obs = env.get_augmented_obs()
        print('Collecting data...')
        for i in range(steps):
            # if i == 0:
            #     act = np.array([0.0])
            # else:
            #     act = get_teacher_action(ob)

            action = get_teacher_action(planner_service, pid, augmented_obs, env.action_space)
            while len(action) == 0:
                print('no expert path')
                obs = env.reset()
                augmented_obs = env.get_augmented_obs()
                action = get_teacher_action(planner_service, pid, augmented_obs, env.action_space)                
            obs, reward, done, _ = env.step(action * env.action_space.high)
            augmented_obs = env.get_augmented_obs()
            obs_list.append(obs)
            augmented_obs_list.append(augmented_obs)
            action_list.append(action)
            reward_list.append(np.array([reward]))

            if done:
                obs = env.reset()
                augmented_obs = env.get_augmented_obs()

        env.pause()

        print('Packing data into arrays...')
        for obs, act, rew in zip(obs_list, action_list, reward_list):
            robot_state = np.reshape(obs[0:env.ob_robot_state_shape], [1, env.ob_robot_state_shape])
            pcl_feature = np.reshape(obs[env.ob_robot_state_shape:], [1, env.pcl_feature_size])
            robot_state_all = np.concatenate([robot_state_all, robot_state], axis=0)
            pcl_feature_all = np.concatenate([pcl_feature_all, pcl_feature], axis=0)
            actions_all = np.concatenate([actions_all, act], axis=0)
            rewards_all = np.concatenate([rewards_all, rew], axis=0)

        # First train for actor network
        actor.fit([robot_state_all, pcl_feature_all], actions_all, batch_size=batch_size, epochs=nb_training_epoch, shuffle=True, verbose=0)
        output_file = open('results.txt', 'w')

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        # Aggregate and retrain actor network
        dagger_buffer_cnt = 0
        for itr in range(dagger_itr):
            obs_list = []
            augmented_obs_list = []

            obs = env.reset()
            reward_sum = 0.0

            for i in range(steps):
                #print('obs:', obs)
                robot_state = np.reshape(obs[0:env.ob_robot_state_shape], [1, env.ob_robot_state_shape])
                pcl_feature = np.reshape(obs[env.ob_robot_state_shape:], [1, env.pcl_feature_size])
                #start = timeit.default_timer()
                action = actor([robot_state, pcl_feature], training=False)  # assume symmetric action space (low = -high)
                #stop = timeit.default_timer()
                #print('Time for actor prediction: ', stop - start)
                #print('action:', action) # action = [[.]]

                action = tf.clip_by_value(action, -1, 1)

                new_obs, reward, done, _ = env.step(action * env.action_space.high)
                augmented_obs = env.get_augmented_obs()
                obs = new_obs

                reward_sum += reward

                if done is True:
                    episode_rew_queue.appendleft(reward_sum)
                    reward_sum = 0
                    obs = env.reset()
                    augmented_obs = env.get_augmented_obs()
                    continue
                else:
                    obs_list.append(obs)
                    augmented_obs_list.append(augmented_obs)

            env.pause()
            mean_return = np.mean(episode_rew_queue)
            print('Episode done ', 'itr ', itr, ',i ', i, 'mean return', mean_return)
            writer.add_scalar("mean return", mean_return, itr)
            if (mean_return > max_mean_return):
                max_mean_return = mean_return
                actor.save_weights('dagger_pcl_itr' + str(itr) + '.h5')

            #if i==(steps-1):
            #    break

            for obs, augmented_obs in zip(obs_list, augmented_obs_list):
                teacher_action = get_teacher_action(planner_service, pid, augmented_obs, env.action_space)
                if len(teacher_action) == 0:
                    print('found no expert path')
                    continue
                #print('obs', obs)
                robot_state = np.reshape(obs[0:env.ob_robot_state_shape], [1, env.ob_robot_state_shape])
                pcl_feature = np.reshape(obs[env.ob_robot_state_shape:], [1, env.pcl_feature_size])
                #print('robot_state:', robot_state)
                if (len(actions_all) < dagger_buffer_size):
                    robot_state_all = np.concatenate([robot_state_all, robot_state], axis=0)
                    pcl_feature_all = np.concatenate([pcl_feature_all, pcl_feature], axis=0)
                    actions_all = np.concatenate([actions_all, teacher_action], axis=0)
                else: # buffer is full
                    robot_state_all[dagger_buffer_cnt] = robot_state
                    pcl_feature_all[dagger_buffer_cnt] = pcl_feature
                    actions_all[dagger_buffer_cnt] = teacher_action
                    dagger_buffer_cnt += 1
                    if (dagger_buffer_cnt == dagger_buffer_size):
                        print('reset dagger_buffer_cnt')
                        dagger_buffer_cnt = 0

            # train actor
            actor.fit([robot_state_all, pcl_feature_all], actions_all,
                            batch_size=batch_size,
                            epochs=nb_training_epoch,
                            shuffle=True)
                            #validation_split=0.2, verbose=0,
                            #callbacks=[early_stop, tfdocs.modeling.EpochDots()])
        
        actor.save_weights('dagger_pcl.h5')
        if (save_path != None):
            #actor.save('dagger_actor_pcl', include_optimizer=False) # should we include optimizer?
            print('save weights to file:', save_path)
            actor.save_weights(save_path + '/dagger_pcl.h5')

