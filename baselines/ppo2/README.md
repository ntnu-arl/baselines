# PPO2

- Original paper: https://arxiv.org/abs/1707.06347
- Baselines blog post: https://blog.openai.com/openai-baselines-ppo/

- `python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4` runs the algorithm for 40M frames = 10M timesteps on an Atari Pong. See help (`-h`) for more options.
- `python -m baselines.run --alg=ppo2 --env=Ant-v2 --num_timesteps=1e6` runs the algorithm for 1M frames on a Mujoco Ant environment.
- also refer to the repo-wide [README.md](../../README.md#training-models)

To run with ARC model:

- `python3 -m baselines.run --alg=ppo2 --env=ARC-v0 --num_timesteps=1e7 --num_env=5 --save_path=path_to_save`
- `python3 -m baselines.run --alg=ppo2 --env=ARC-v0 --num_timesteps=0 --num_env=1 --play --load_path=path_to_load`

To edit the path to weight file of dagger actor, change `DAGGER_ACTOR_WEIGHT` [variable](https://github.com/unr-arl/baselines/blob/0dc5386b284031ff788db7c6b06b4903f1a74492/baselines/a2c/utils.py#L9) in `baselines/a2c/utils.py` file 

# TODO:

When `--play`, change this line of code [code](https://github.com/unr-arl/baselines/blob/2f0405364d7d3e6476e1548eaaa8424dc1c24541/baselines/common/policies.py#L52) to

```
self.action = self.pi
```

to remove variance from running policy.