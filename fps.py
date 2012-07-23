
import math
from math import log
import random
import numpy

from posterior import Partition, Posterior

from mpsim.moran import generalized_moran_simulation_transitions as edge_function
from mpsim.math_helpers import dot_product

# https://atpassos.posterous.com/normalizing-log-probabilities-with-numpy
def lognormalize(x):
    a = numpy.logaddexp.reduce(x)
    return numpy.exp(x - a)  
    
"""Functions for inference of FPS distribution for modified Moran process."""

def likelihood(a,b,d):
    """Likelihood functions for a birth-death process based on fitness proportionate reproduction."""
    # a + b = N
    a = float(a)
    b = float(b)
    if d == 1: # A replicated
        def p(r):
            return a / (a + b * r)
    elif d == 0: # B replicated
        def p(r):
            return b * r / (a + b * r)
    return p

# Warning: this function is not normalized!
def distribution(alphas, betas):
    """Computes values for FPS distribution. Warning: not normalized!"""
    N = len(alphas)
    log = math.log
    log_s0 = 0.
    sum_betas = sum(betas) - betas[0]
    alphas_plus_betas = [alphas[i] + betas[i] for i in range(0, len(alphas))]
    for a in range(1, N):
        log_s0 += alphas[a] * log(a)  + betas[a] * log(N-a)
    def p(r):
        if r == 0:
            if sum_betas > 0:
                return 0
        log_s = log_s0
        for a in range(1, N-1):
            #alpha = alphas[a]
            #beta = betas[a]
            #a = float(a)
            #log_s += alpha * log(a) + beta * log(r*(N-a)) - (alpha + beta) * log(a + r * (N - a))
            log_s -= alphas_plus_betas[a] * log(a + r * (N - a))
        if sum_betas > 0:
            log_s += sum_betas * log(r)
        return math.exp(log_s)
    return p

def likelihood_table(N, partition_points):
    """Precompute likelihood functions on partitions for computational efficiency."""
    tables = []
    num_points = len(partition_points)
    for d in [0,1]:
        table = []
        for a in range(1, N):
            #row = []
            b = N - a
            vfunc = numpy.vectorize(likelihood(a,b,d))
            table.append(numpy.log(vfunc(partition_points)))
        tables.append(numpy.array(table))
    return tables    
    
#### Convert to use table and conjugate parameters.    

# This is a much faster version utilizing numpy and log-space probability calculations.
def construct_posterior(N, conjugate_parameters, prior_points, partition, tables):
    posterior = Posterior(partition, prior_points)
    alphas, betas = conjugate_parameters
    alphas = numpy.array([alphas])
    betas = numpy.array([betas])
    beta_table, alpha_table = tables
    #print alphas.transpose().size, alpha_table.size
    t = numpy.dot(numpy.array([1.]*(N-1)), (alphas.transpose() * alpha_table + betas.transpose() * beta_table))
    t = lognormalize(t)
    posterior.update(t)
    return posterior
    
### Simulation Output conversions

def tuples_to_conjugate_parameters(N, tuples):
    """Reduce the run tuples to counts for the parameters alpha and beta."""
    alphas = [0] * (N-1)
    betas = [0] * (N-1)
    for ((a, b), d) in tuples:
        if d == 1:
            alphas[a-1] += 1
        elif d == 0:
            betas[a-1] += 1
        else:
            raise ValueError, "d not in {0,1}, d=%s" % d
    return (alphas, betas)

def convert_run_to_conjugate_parameters(N, run, max_length=None, sample_size=None):
    return tuples_to_conjugate_parameters(N, convert_run_to_tuples(N, run, max_length=max_length, sample_size=sample_size))
    
def convert_run_to_tuples(N, run, max_length=None, sample_size=None):
    """Returns a multiset of tuple instances of transitions in the form of (i, j) with $j \in {i-1, i, i+1}$, optionally sampling randomly. sample_size is the number of indices to select."""
    # Do not use the last transition -- it may cause odd likelihood effects
    indices = xrange(len(run)-1)
    if sample_size:
        indices = random.sample(indices, min(sample_size, len(indices)))
    if max_length:
        if len(indices) > max_length:
            indices = list(indices)[:max_length]
    for i in indices:
        if sum(run[i]) == N:
            d = run[i+1][0] - run[i][0]
            yield (run[i], d) 

def convert_runs(N, runs, min_length=None, max_length=None, sample_size=None):
    """Converts runs from mpsim to format convenient for inference."""
    return_runs = []
    for run in runs:
        if min_length:
            if len(run) < min_length:
                continue
        return_runs.append(list(convert_run_to_tuples(N, run, max_length=max_length, sample_size=sample_size)))
    return return_runs                
                
#### Birth Counting

def reproduction_rate_test(parameters):
    means = []
    return_runs = [[],[]]
    for alphas, betas in parameters:
        N = len(alphas) + 1
        x = 0.
        y = 0.
        for a in range(1, len(alphas)-1):
            x += alphas[a-1] / float(a)
            y += betas[a-1]/ float(N-a)
        if x != 0:
            means.append(y / x)
            return_runs[0].append((alphas, betas))
        else:
            return_runs[1].append((alphas, betas))
    return return_runs, numpy.mean(means), numpy.std(means), len(means)
