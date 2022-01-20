# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for DQN agent."""

from absl.testing import absltest
import tensorflow as tf
import acme
from acme import specs
from acme.agents.tf import dqn
from acme.testing import fakes
from acme import wrappers
from absl import flags
import bsuite
import dm_env
import gym
from absl import app

import numpy as np
import sonnet as snt
from gym_minigrid.wrappers import *


def make_environment(
    environment) -> dm_env.Environment:
  """Creates an OpenAI Gym environment."""


  # Make sure the environment obeys the dm_env.Environment interface.
  environment = wrappers.GymWrapper(environment)
  # Clip the action returned by the agent to the environment spec.
  environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment

def _make_Qnetwork(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([256, 256, action_spec.num_values]),
  ])

def _make_qnetwork(action_spec: specs.DiscreteArray) -> snt.Module: #takes in s + s' + action, spits out probability
  return dqn.ConditionalProductNetwork(output_dims=action_spec.num_values,categorical=True)

def _make_feat_network(action_spec: specs.DiscreteArray) -> snt.Module: #lol this just makes features, so we'll just flatten for now
  return snt.Sequential([snt.Conv2D(32,4,2),tf.nn.leaky_relu,snt.BatchNorm(True,True),snt.Conv2D(64,4,2),tf.nn.leaky_relu,
                         snt.BatchNorm(True,True),snt.Flatten(),snt.Linear(64)
  ])

def _make_rnetwork(action_spec: specs.DiscreteArray) -> snt.Module: #takes in just s and action, spits out probability
  return dqn.RNetwork(output_dims=action_spec.num_values,categorical=True)




def run(_):
  flags.DEFINE_integer('num_episodes',100,'number of episodes to run')
  FLAGS = flags.FLAGS


  env = gym.make('MiniGrid-Empty-8x8-v0')
  env = RGBImgPartialObsWrapper(env)  # Get pixel observations
  env = ImgObsWrapper(env)  # Get rid of the 'mission' field

  environment = make_environment(env)

  spec = specs.make_environment_spec(environment)
    # Create a fake environment to test with.

    # Construct the agent.
  agent = dqn.DQNEmpowerment(
        environment_spec=spec,
        Qnetwork=_make_Qnetwork(spec.actions),
        qnetwork = _make_qnetwork(spec.actions),
        feat_network = _make_feat_network(spec.actions),
        feat_dims=64,
        rnetwork = _make_rnetwork(spec.actions),
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(run)
