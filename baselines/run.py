#!/home/huan/anaconda3/envs/tf-gpu/bin/python
import rospy
from baselines.common.rotors_wrappers import RotorsWrappers

import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
from collections import deque
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines import logger
from importlib import import_module
#from baselines.common.lidar_feature_ext import LidarFeatureExtract

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    if args.save_path:
        alg_kwargs['save_path'] = args.save_path

    if args.load_actor_dagger_path:
        alg_kwargs['load_actor_dagger_path'] = args.load_actor_dagger_path

    if args.load_critic_dagger_path:
        alg_kwargs['load_critic_dagger_path'] = args.load_critic_dagger_path

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)
    elif env_type == 'rotors':
        env = make_env(env_id, env_type, seed=seed)
    else:
        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if (env_type == 'mujoco'):
            env = VecNormalize(env)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    if env_id == 'rotors-rmf':
        return 'rotors','rotors-rmf'

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # ROS stuff
    #rospy.init_node('RL_node', anonymous=True, log_level=rospy.FATAL)

    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    analyze_plots = False

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, save_path, max_to_keep=None)
        manager.save()

    if args.play:
        logger.log("Running trained model")
        obs = env.reset()
        if not isinstance(env, VecEnv):
            obs = np.expand_dims(np.array(obs), axis=0)

        state = model.initial_state if hasattr(model, 'initial_state') else None

        new_goal = True
        reach_goal_trajectory_list = []

        episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        episode_rew_queue = deque(maxlen=10)
        queue_cnt = 0

        avrg_RMS = np.array([])
        all_goals = np.array([])
        #num_sims = 10
        #sim_done = False
        #sim_ctr = 0

        while True:
            if new_goal:
                reach_goal_trajectory = np.array([])
                new_goal = False

            if state is not None:
                actions, _, state, _ = model.step(obs)
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, _ = env.step(actions.numpy())

            reach_goal_trajectory = np.concatenate((reach_goal_trajectory, obs[0:3]))
            reach_goal_trajectory_list.append(obs[0:3])

            if not isinstance(env, VecEnv):
                obs = np.expand_dims(np.array(obs), axis=0)
            episode_rew += rew
            #env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done

            if done_any:
                for i in np.nonzero(done)[0]:
                    episode_rew_queue.appendleft(episode_rew[i])
                    episode_rew[i] = 0
                    print('episode_rew mean={}'.format(np.mean(episode_rew_queue)))

                #Plots
                if analyze_plots == True:
                    env.pause()

                    env.position_xyz_response()
                    env.velocity_xyz_response()
                    RMS, goal = env.compare_trajectory_with_optimal(analyze_plots)
                    avrg_RMS = np.append(avrg_RMS, RMS)
                    all_goals = np.append(all_goals, goal)

                    env.unpause()


                new_goal = True
                obs = env.reset()
                obs = np.array([obs])


        #print(f'Avrg RMS is: {np.mean(avrg_RMS)}')
        #print("The goals are:")
        #for i in range(num_sims):
        #    print(all_goals[i])

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
