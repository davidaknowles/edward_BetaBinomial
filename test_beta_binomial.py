from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.distributions import bijectors

from edward_additions.betabinomial import BetaBinomial

import edward as ed
import edward.models

tf.reset_default_graph()
sess = tf.InteractiveSession()

# GENERATE DATA
total_count=30
n_samples=100
p_true=0.05
true_conc=3.
p_noisy = np.random.beta(p_true * true_conc, 
                         (1.-p_true) * true_conc, 
                         size=n_samples)
x_data = np.random.binomial(p=p_noisy, 
                            n=total_count).astype(np.float32)

# MODEL
p = ed.models.Beta(1.0, 1.0)
#conc_param = tf.get_variable('conc_param', initializer=10.)
conc_param = ed.models.Gamma(2., 1.)
x = BetaBinomial(total_count=tf.to_float(total_count),
                 probs=p, 
                 concentrations=conc_param * tf.ones(n_samples, dtype=np.float32),
                 sample_shape=n_samples, 
                 value=tf.zeros(n_samples, dtype="float32"))

# INFERENCE
#qp = edward.models.BetaWithSoftplusConcentration(tf.Variable(1.), tf.Variable(1.))
qp = ed.models.Normal(loc=tf.get_variable("qp/loc", []), 
                      scale=tf.nn.softplus(tf.get_variable("qp/scale", [])))
qconc = ed.models.Normal(loc=tf.get_variable("qconc/loc", []), 
                         scale=tf.nn.softplus(tf.get_variable("qconc/scale", [])))

inference = ed.KLqp({p: qp, conc_param: qconc}, data={x: x_data})
inference.run()

# PRINT RESULTS
qp_samples = ed.transform(
    qp, 
    bijectors.Invert(
        inference.transformations[p].bijector)).sample(100).eval()

print( "True prob success: {:.2f}, inferred {:.3f} +- {:.2f}".format(
    p_true, 
    qp_samples.mean(),
    np.sqrt(qp_samples.var())) )

qconc_samples = ed.transform(
    qconc, 
    bijectors.Invert(
        inference.transformations[conc_param].bijector)).sample(100).eval()

print("True concentration: {:.2f}, Inferred: {:.3f} +- {:.2f}".format(
    true_conc,
    qconc_samples.mean(),
    np.sqrt(qconc_samples.var())) )