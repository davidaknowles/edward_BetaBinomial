# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Modified by David A. Knowles, knowles84@gmail.com
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
"""The BetaBinomial distribution class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation

from edward.models.random_variable import RandomVariable


class distributions_BetaBinomial(distribution.Distribution):
  
  def __init__(self,
               total_count,
               logits=None,
               probs=None,
               concentrations=None,
               validate_args=False,
               allow_nan_stats=True,
               name="BetaBinomial"):
    """Initialize a batch of BetaBinomial distributions.

    Args:
      total_count: Non-negative floating point tensor with shape broadcastable
        to `[N1,..., Nm]` with `m >= 0` and the same dtype as `probs` or
        `logits`. Defines this as a batch of `N1 x ...  x Nm` different Binomial
        distributions. Its components should be equal to integer values.
      logits: Floating point tensor representing the log-odds of a
        positive event with shape broadcastable to `[N1,..., Nm]` `m >= 0`, and
        the same dtype as `total_count`. Each entry represents logits for the
        probability of success for independent Binomial distributions. Only one
        of `logits` or `probs` should be passed in.
      probs: Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm]` `m >= 0`, `probs in [0, 1]`. Each entry represents the
        probability of success for independent Binomial distributions. Only one
        of `logits` or `probs` should be passed in.
      concentrations: Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm]` `m >= 0`. 
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
    parameters = dict(locals())
    with ops.name_scope(name, values=[total_count, logits, probs, concentrations]) as name:
      self._total_count = self._maybe_assert_valid_total_count(
          ops.convert_to_tensor(total_count, name="total_count"),
          validate_args)
      self._logits, self._probs = distribution_util.get_logits_and_probs(
          logits=logits,
          probs=probs,
          validate_args=validate_args,
          name=name)
      self._concentrations = concentrations
    super(distributions_BetaBinomial, self).__init__(
        dtype=self._probs.dtype,
        reparameterization_type=distribution.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._total_count,
                       self._logits,
                       self._probs,
                       self._concentrations],
        name=name)

  @property
  def total_count(self):
    """Number of trials."""
    return self._total_count

  @property
  def logits(self):
    """Log-odds of drawing a `1`."""
    return self._logits

  @property
  def probs(self):
    """Probability of drawing a `1`."""
    return self._probs
    
  @property
  def concentrations(self):
    return self._concentrations

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.total_count),
        array_ops.shape(self.probs))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.total_count.get_shape(),
        self.probs.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _log_prob(self, counts):
    counts = self._maybe_assert_valid_sample(counts)
    first_part = math_ops.lgamma(counts + self.probs * self.concentrations) + \
        math_ops.lgamma(self.total_count - counts + (1. - self.probs) * self.concentrations) - \
        math_ops.lgamma(self.total_count + self.concentrations)
    second_part = math_ops.lgamma(self.probs * self.concentrations) + \
        math_ops.lgamma((1. - self.probs) * self.concentrations) - \
        math_ops.lgamma(self.concentrations)
    return(first_part - second_part)

  def _mean(self):
    return self.total_count * self.probs

  def _maybe_assert_valid_total_count(self, total_count, validate_args):
    if not validate_args:
      return total_count
    return control_flow_ops.with_dependencies([
        check_ops.assert_non_negative(
            total_count,
            message="total_count must be non-negative."),
        distribution_util.assert_integer_form(
            total_count,
            message="total_count cannot contain fractional components."),
    ], total_count)

  def _maybe_assert_valid_sample(self, counts):
    """Check counts for proper shape, values, then return tensor version."""
    if not self.validate_args:
      return counts
    counts = distribution_util.embed_check_nonnegative_integer_form(counts)
    return control_flow_ops.with_dependencies([
        check_ops.assert_less_equal(
            counts, self.total_count,
            message="counts are not less than or equal to n."),
    ], counts)
    
def __init__(self, *args, **kwargs):
  RandomVariable.__init__(self, *args, **kwargs)

_name = 'BetaBinomial'
_candidate = distributions_BetaBinomial
__init__.__doc__ = _candidate.__init__.__doc__
_globals = globals()
_params = {'__doc__': _candidate.__doc__,
           '__init__': __init__}
_globals[_name] = type(_name, (RandomVariable, _candidate), _params)

BetaBinomial.support = 'onehot'