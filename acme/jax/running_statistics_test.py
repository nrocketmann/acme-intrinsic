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

"""Tests for running statistics utilities."""

import functools
import math
from typing import NamedTuple

from absl.testing import absltest
from acme import specs
from acme.jax import running_statistics
import jax
from jax import test_util as jtu
from jax.config import config as jax_config
import jax.numpy as jnp
import tree

update_and_validate = functools.partial(
    running_statistics.update, validate_shapes=True)


class TestNestedSpec(NamedTuple):
  # Note: the fields are intentionally in reverse order to test ordering.
  a: specs.Array
  b: specs.Array


class RunningStatisticsTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    jax_config.update('jax_enable_x64', False)

  def test_normalize(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.float32))

    x = jnp.arange(200, dtype=jnp.float32).reshape(20, 2, 5)
    x1, x2, x3, x4 = jnp.split(x, 4, axis=0)

    state = update_and_validate(state, x1)
    state = update_and_validate(state, x2)
    state = update_and_validate(state, x3)
    state = update_and_validate(state, x4)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized)
    std = jnp.std(normalized)
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std))

  def test_init_normalize(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.float32))

    x = jnp.arange(200, dtype=jnp.float32).reshape(20, 2, 5)
    normalized = running_statistics.normalize(x, state)

    self.assertAllClose(normalized, x)

  def test_one_batch_dim(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.float32))

    x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized, axis=0)
    std = jnp.std(normalized, axis=0)
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std))

  def test_clip(self):
    state = running_statistics.init_state(specs.Array((), jnp.float32))

    x = jnp.arange(5, dtype=jnp.float32)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state, max_abs_value=1.0)

    mean = jnp.mean(normalized)
    std = jnp.std(normalized)
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std) * math.sqrt(0.6))

  def test_nested_normalize(self):
    state = running_statistics.init_state({
        'a': specs.Array((5,), jnp.float32),
        'b': specs.Array((2,), jnp.float32)
    })

    x1 = {
        'a': jnp.arange(20, dtype=jnp.float32).reshape(2, 2, 5),
        'b': jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)
    }
    x2 = {
        'a': jnp.arange(20, dtype=jnp.float32).reshape(2, 2, 5) + 20,
        'b': jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2) + 8
    }
    x3 = {
        'a': jnp.arange(40, dtype=jnp.float32).reshape(4, 2, 5),
        'b': jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2)
    }

    state = update_and_validate(state, x1)
    state = update_and_validate(state, x2)
    state = update_and_validate(state, x3)
    normalized = running_statistics.normalize(x3, state)

    mean = tree.map_structure(lambda x: jnp.mean(x, axis=(0, 1)), normalized)
    std = tree.map_structure(lambda x: jnp.std(x, axis=(0, 1)), normalized)
    tree.map_structure(lambda x: self.assertAllClose(x, jnp.zeros_like(x)),
                       mean)
    tree.map_structure(lambda x: self.assertAllClose(x, jnp.ones_like(x)), std)

  def test_validation(self):
    state = running_statistics.init_state(specs.Array((1, 2, 3), jnp.float32))

    x = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
    with self.assertRaises(AssertionError):
      update_and_validate(state, x)

    x = jnp.arange(3, dtype=jnp.float32).reshape(1, 1, 3)
    with self.assertRaises(AssertionError):
      update_and_validate(state, x)

  def test_int_not_normalized(self):
    state = running_statistics.init_state(specs.Array((), jnp.int32))

    x = jnp.arange(5, dtype=jnp.int32)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state)

    self.assertArraysEqual(normalized, x)

  def test_pmap_update_nested(self):
    local_device_count = jax.local_device_count()
    assert local_device_count > 1
    state = running_statistics.init_state({
        'a': specs.Array((5,), jnp.float32),
        'b': specs.Array((2,), jnp.float32)
    })

    x = {
        'a': (jnp.arange(15 * local_device_count,
                         dtype=jnp.float32)).reshape(local_device_count, 3, 5),
        'b': (jnp.arange(6 * local_device_count,
                         dtype=jnp.float32)).reshape(local_device_count, 3, 2),
    }

    devices = jax.local_devices()
    state = jax.device_put_replicated(state, devices)
    pmap_axis_name = 'i'
    state = jax.pmap(
        functools.partial(update_and_validate, pmap_axis_name=pmap_axis_name),
        pmap_axis_name)(state, x)
    state = jax.pmap(
        functools.partial(update_and_validate, pmap_axis_name=pmap_axis_name),
        pmap_axis_name)(state, x)
    normalized = jax.pmap(running_statistics.normalize)(x, state)

    mean = tree.map_structure(lambda x: jnp.mean(x, axis=(0, 1)), normalized)
    std = tree.map_structure(lambda x: jnp.std(x, axis=(0, 1)), normalized)
    tree.map_structure(lambda x: self.assertAllClose(x, jnp.zeros_like(x)),
                       mean)
    tree.map_structure(lambda x: self.assertAllClose(x, jnp.ones_like(x)), std)

  def test_different_structure_normalize(self):
    spec = TestNestedSpec(
        a=specs.Array((5,), jnp.float32), b=specs.Array((2,), jnp.float32))
    state = running_statistics.init_state(spec)

    x = {
        'a': jnp.arange(20, dtype=jnp.float32).reshape(2, 2, 5),
        'b': jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)
    }

    with self.assertRaises(TypeError):
      state = update_and_validate(state, x)

  def test_weights(self):
    state = running_statistics.init_state(specs.Array((), jnp.float32))

    x = jnp.arange(5, dtype=jnp.float32)
    x_weights = jnp.ones_like(x)
    y = 2 * x + 5
    y_weights = 2 * x_weights
    z = jnp.concatenate([x, y])
    weights = jnp.concatenate([x_weights, y_weights])

    state = update_and_validate(state, z, weights)

    self.assertEqual(state.mean, (jnp.mean(x) + 2 * jnp.mean(y)) / 3)
    big_z = jnp.concatenate([x, y, y])
    normalized = running_statistics.normalize(big_z, state)
    self.assertAlmostEqual(jnp.mean(normalized), 0., places=6)
    self.assertAlmostEqual(jnp.std(normalized), 1., places=6)


if __name__ == '__main__':
  absltest.main()
