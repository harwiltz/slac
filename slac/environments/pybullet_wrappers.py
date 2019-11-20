from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gym
import gym.spaces
import numpy as np
from tf_agents.environments import wrappers

class PixelObservationsPyBulletWrapper(wrappers.PyEnvironmentBaseWrapper):

  def __init__(self, gym_env, observations_whitelist=None, render_kwargs=None):
    super(PixelObservationsPyBulletWrapper, self).__init__(gym_env)
    if observations_whitelist is None:
      self._observations_whitelist = ['state', 'pixels']
    else:
      self._observations_whitelist = observations_whitelist

    observation_spaces = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observation_spaces['state'] = self._env.observation_space
      elif observation_name == 'pixels':
        image_shape = (
          240, 320, 3)
        image_space = gym.spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        observation_spaces['pixels'] = image_space
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels", got %s.' % observation_name)
    print("THIS IS THE OBSERVATION SPACE: {}".format(gym.spaces.Dict(observation_spaces)))
    self.observation_space = gym.spaces.Dict(observation_spaces)

  def _modify_observation(self, observation):
    observations = collections.OrderedDict()
    for observation_name in self._observations_whitelist:
      if observation_name == 'state':
        observations['state'] = observation
      elif observation_name == 'pixels':
        image = self.render()[::-1, :, :]
        observations['pixels'] = image
      else:
        raise ValueError('observations_whitelist can only have "state" '
                         'or "pixels", got %s.' % observation_name)
    return observations

  def _step(self, action):
    observation, reward, done, info = self._env.step(action)
    observation = self._modify_observation(observation)
    return observation, reward, done, info

  def _reset(self):
    observation = self._env.reset()
    return self._modify_observation(observation)

  def render(self, mode='rgb_array'):
    return self._env.render(mode=mode)
