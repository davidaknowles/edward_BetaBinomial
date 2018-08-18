# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Modified by David A Knowles knowles84@gmail.com
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
# ==============================================================================
"""The Kumaraswamy distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.python.ops.distributions import bijector
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import transformed_distribution
from tensorflow.python.ops.distributions import uniform
from tensorflow.python.util import deprecation

from edward.models.random_variable import RandomVariable

__all__ = [
    "Kumaraswamy",
]

_kumaraswamy_sample_note = """Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`."""


class bijectors_Kumaraswamy(bijector.Bijector):
  """Compute `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a), X in [0, 1]`.

  This bijector maps inputs from `[0, 1]` to [0, 1]`. The inverse of the
  bijector applied to a uniform random variable `X ~ U(0, 1) gives back a
  random variable with the [Kumaraswamy distribution](
  https://en.wikipedia.org/wiki/Kumaraswamy_distribution):

  ```none
  Y ~ Kumaraswamy(a, b)
  pdf(y; a, b, 0 <= y <= 1) = a * b * y ** (a - 1) * (1 - y**a) ** (b - 1)
  ```
  """

  def __init__(self,
               concentration1=None,
               concentration0=None,
               validate_args=False,
               name="kumaraswamy"):
    """Instantiates the `Kumaraswamy` bijector.

    Args:
      concentration1: Python `float` scalar indicating the transform power,
        i.e., `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)` where `a` is
        `concentration1`.
      concentration0: Python `float` scalar indicating the transform power,
        i.e., `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)` where `b` is
        `concentration0`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    self._graph_parents = []
    self._name = name
    self._validate_args = validate_args

    with self._name_scope("init", values=[concentration1, concentration0]):
      concentration1 = self._maybe_assert_valid_concentration(
          ops.convert_to_tensor(concentration1, name="concentration1"),
          validate_args=validate_args)
      concentration0 = self._maybe_assert_valid_concentration(
          ops.convert_to_tensor(concentration0, name="concentration0"),
          validate_args=validate_args)

    self._concentration1 = concentration1
    self._concentration0 = concentration0
    super(bijectors_Kumaraswamy, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  @property
  def concentration1(self):
    """The `a` in: `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)`."""
    return self._concentration1

  @property
  def concentration0(self):
    """The `b` in: `Y = g(X) = (1 - (1 - X)**(1 / b))**(1 / a)`."""
    return self._concentration0

  def _forward(self, x):
    x = self._maybe_assert_valid(x)
    return math_ops.exp(
        math_ops.log1p(-math_ops.exp(math_ops.log1p(-x) / self.concentration0))
        / self.concentration1)

  def _inverse(self, y):
    y = self._maybe_assert_valid(y)
    return math_ops.exp(math_ops.log1p(
        -(1 - y**self.concentration1)**self.concentration0))

  def _inverse_log_det_jacobian(self, y):
    y = self._maybe_assert_valid(y)
    return (
        math_ops.log(self.concentration1) + math_ops.log(self.concentration0) +
        (self.concentration1 - 1) * math_ops.log(y) +
        (self.concentration0 - 1) * math_ops.log1p(-y**self.concentration1))

  def _maybe_assert_valid_concentration(self, concentration, validate_args):
    """Checks the validity of a concentration parameter."""
    if not validate_args:
      return concentration
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(
            concentration,
            message="Concentration parameter must be positive."),
    ], concentration)

  def _maybe_assert_valid(self, x):
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_non_negative(
            x,
            message="sample must be non-negative"),
        check_ops.assert_less_equal(
            x, array_ops.ones([], self.concentration0.dtype),
            message="sample must be no larger than `1`."),
    ], x)
    
def _harmonic_number(x):
  """Compute the harmonic number from its analytic continuation.

  Derivation from [here](
  https://en.wikipedia.org/wiki/Digamma_function#Relation_to_harmonic_numbers)
  and [Euler's constant](
  https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant).

  Args:
    x: input float.

  Returns:
    z: The analytic continuation of the harmonic number for the input.
  """
  one = array_ops.ones([], dtype=x.dtype)
  return math_ops.digamma(x + one) - math_ops.digamma(one)


class distributions_Kumaraswamy(transformed_distribution.TransformedDistribution):
  """Kumaraswamy distribution.

  The Kumaraswamy distribution is defined over the `(0, 1)` interval using
  parameters
  `concentration1` (aka "alpha") and `concentration0` (aka "beta").  It has a
  shape similar to the Beta distribution, but is reparameterizeable.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, beta) = alpha * beta * x**(alpha - 1) * (1 - x**alpha)**(beta -
  1)
  ```

  where:

  * `concentration1 = alpha`,
  * `concentration0 = beta`,

  Distribution parameters are automatically broadcast in all functions; see
  examples for details.

  #### Examples

  ```python
  # Create a batch of three Kumaraswamy distributions.
  alpha = [1, 2, 3]
  beta = [1, 2, 3]
  dist = Kumaraswamy(alpha, beta)

  dist.sample([4, 5])  # Shape [4, 5, 3]

  # `x` has three batch entries, each with two samples.
  x = [[.1, .4, .5],
       [.2, .3, .5]]
  # Calculate the probability of each pair of samples under the corresponding
  # distribution in `dist`.
  dist.prob(x)         # Shape [2, 3]
  ```

  ```python
  # Create batch_shape=[2, 3] via parameter broadcast:
  alpha = [[1.], [2]]      # Shape [2, 1]
  beta = [3., 4, 5]        # Shape [3]
  dist = Kumaraswamy(alpha, beta)

  # alpha broadcast as: [[1., 1, 1,],
  #                      [2, 2, 2]]
  # beta broadcast as:  [[3., 4, 5],
  #                      [3, 4, 5]]
  # batch_Shape [2, 3]
  dist.sample([4, 5])  # Shape [4, 5, 2, 3]

  x = [.2, .3, .5]
  # x will be broadcast as [[.2, .3, .5],
  #                         [.2, .3, .5]],
  # thus matching batch_shape [2, 3].
  dist.prob(x)         # Shape [2, 3]
  ```

  """

  def __init__(self,
               concentration1=None,
               concentration0=None,
               validate_args=False,
               allow_nan_stats=True,
               name="Kumaraswamy"):
    """Initialize a batch of Kumaraswamy distributions.

    Args:
      concentration1: Positive floating-point `Tensor` indicating mean
        number of successes; aka "alpha". Implies `self.dtype` and
        `self.batch_shape`, i.e.,
        `concentration1.shape = [N1, N2, ..., Nm] = self.batch_shape`.
      concentration0: Positive floating-point `Tensor` indicating mean
        number of failures; aka "beta". Otherwise has same semantics as
        `concentration1`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    with ops.name_scope(name, values=[concentration1, concentration0]) as name:
      concentration1 = ops.convert_to_tensor(
          concentration1, name="concentration1")
      concentration0 = ops.convert_to_tensor(
          concentration0, name="concentration0")
    super(distributions_Kumaraswamy, self).__init__(
        distribution=uniform.Uniform(
            low=array_ops.zeros([], dtype=concentration1.dtype),
            high=array_ops.ones([], dtype=concentration1.dtype),
            allow_nan_stats=allow_nan_stats),
        bijector=bijectors_Kumaraswamy(
            concentration1=concentration1, concentration0=concentration0,
            validate_args=validate_args),
        batch_shape=distribution_util.get_broadcast_shape(
            concentration1, concentration0),
        name=name)
    self._reparameterization_type = distribution.FULLY_REPARAMETERIZED

  @property
  def concentration1(self):
    """Concentration parameter associated with a `1` outcome."""
    return self.bijector.concentration1

  @property
  def concentration0(self):
    """Concentration parameter associated with a `0` outcome."""
    return self.bijector.concentration0

  def _entropy(self):
    a = self.concentration1
    b = self.concentration0
    return (1 - 1. / a) + (
        1 - 1. / b) * _harmonic_number(b) + math_ops.log(a) + math_ops.log(b)

  def _moment(self, n):
    """Compute the n'th (uncentered) moment."""
    total_concentration = self.concentration1 + self.concentration0
    expanded_concentration1 = array_ops.ones_like(
        total_concentration, dtype=self.dtype) * self.concentration1
    expanded_concentration0 = array_ops.ones_like(
        total_concentration, dtype=self.dtype) * self.concentration0
    beta_arg0 = 1 + n / expanded_concentration1
    beta_arg = array_ops.stack([beta_arg0, expanded_concentration0], -1)
    log_moment = math_ops.log(expanded_concentration0) + special_math_ops.lbeta(
        beta_arg)
    return math_ops.exp(log_moment)

  def _mean(self):
    return self._moment(1)

  def _variance(self):
    # TODO(b/72696533): Investigate a more numerically stable version.
    return self._moment(2) - math_ops.square(self._moment(1))

  @distribution_util.AppendDocstring(
      """Note: The mode is undefined when `concentration1 <= 1` or
      `concentration0 <= 1`. If `self.allow_nan_stats` is `True`, `NaN`
      is used for undefined modes. If `self.allow_nan_stats` is `False` an
      exception is raised when one or more modes are undefined.""")
  def _mode(self):
    a = self.concentration1
    b = self.concentration0
    mode = ((a - 1) / (a * b - 1))**(1. / a)
    if self.allow_nan_stats:
      nan = array_ops.fill(
          self.batch_shape_tensor(),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype),
          name="nan")
      is_defined = (self.concentration1 > 1.) & (self.concentration0 > 1.)
      return array_ops.where(is_defined, mode, nan)

    return control_flow_ops.with_dependencies([
        check_ops.assert_less(
            array_ops.ones([], dtype=self.concentration1.dtype),
            self.concentration1,
            message="Mode undefined for concentration1 <= 1."),
        check_ops.assert_less(
            array_ops.ones([], dtype=self.concentration0.dtype),
            self.concentration0,
            message="Mode undefined for concentration0 <= 1.")
    ], mode)
    
    
def __init__(self, *args, **kwargs):
  RandomVariable.__init__(self, *args, **kwargs)

_name = 'Kumaraswamy'
_candidate = distributions_Kumaraswamy
__init__.__doc__ = _candidate.__init__.__doc__
_globals = globals()
_params = {'__doc__': _candidate.__doc__,
           '__init__': __init__}
_globals[_name] = type(_name, (RandomVariable, _candidate), _params)

Kumaraswamy.support = '01'