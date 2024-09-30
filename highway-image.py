import warnings
from functools import partial as bind
import time  # Import the time module

import dreamerv3
import embodied

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

def main():

  start_time = time.time()  # Record the start time

  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size12m'],  # Changed from 100m to 12m
      'logdir': f'~/logdir/{embodied.timestamp()}-example',
      'run.train_ratio': 32,
      'run.steps': 10000
  })
  config = embodied.Flags(config).parse()

  print('Logdir:', config.logdir)
  logdir = embodied.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def make_agent(config):
    env = make_env(config)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent

  def make_logger(config):
    logdir = embodied.Path(config.logdir)
    return embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandbOutput(logdir.name, config=config),
    ])

  def make_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=embodied.Path(config.logdir) / 'replay',
        online=config.replay.online)

  def make_env(config, env_id=0):
    import gymnasium as gym
    import highway_env
    from embodied.envs import from_gym

    env = gym.make('highway-v0')
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)
    return env

  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context,
  )

  embodied.run.train(
      bind(make_agent, config),
      bind(make_replay, config),
      bind(make_env, config),
      bind(make_logger, config), args)

  end_time = time.time()  # Record the end time
  total_time = end_time - start_time
  print(f"Total training time: {total_time:.2f} seconds")  # Print the time taken

if __name__ == '__main__':
  main()
