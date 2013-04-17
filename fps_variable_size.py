
import math
from math import log
import random
import numpy

from posterior import Partition, Posterior

from mpsim.moran import generalized_moran_simulation_transitions_variable_size as edge_function
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

def likelihood_table(N, partition_points):
    """Precompute likelihood functions on partitions for computational efficiency."""
    all_tables = []
    num_points = len(partition_points)
    for n in range(1, N+1):
        tables = []
        for d in [0,1]:
            table = []
            for a in range(1, n):
                b = n - a
                vfunc = numpy.vectorize(likelihood(a,b,d))
                table.append(numpy.log(vfunc(partition_points)))
            tables.append(numpy.array(table))
        all_tables.append(tables)
    return all_tables    
    
#### Convert to use table and conjugate parameters.    

# This is a much faster version utilizing numpy and log-space probability calculations.
def construct_posterior(N, conjugate_parameters, prior_points, partition, tables):
    posterior = Posterior(partition, prior_points)
    all_alphas, all_betas = conjugate_parameters
    t = numpy.array([0.]*(len(partition.points)))
    for i in range(2, N):
        alphas = numpy.array([all_alphas[i]])
        betas = numpy.array([all_betas[i]])
        beta_table, alpha_table = tables[i]
        s1 = alphas.transpose() * alpha_table
        s2 = betas.transpose() * beta_table
        s3 = numpy.array([1.]*(i))
        t += numpy.dot(numpy.array([1.]*(i)), (alphas.transpose() * alpha_table + betas.transpose() * beta_table))
        t = lognormalize(t)
        posterior.update(t)
    return posterior
    
### Simulation Output conversions

def tuples_to_conjugate_parameters(N, tuples):
    """Reduce the run tuples to counts for the parameters alpha and beta."""
    alphas = []
    betas = []
    for i in range(1, N+1):
        alphas.append([0]*(i-1))
        betas.append([0]*(i-1))
    try:
        for ((a1, b1), (a2, b2)) in tuples:
            if a2 == a1 + 1:
                alphas[a1 + b1 - 1][a1-1] += 1
            elif b2 == b1 + 1:
                betas[a1 + b1 - 1][a1-1] += 1
    except IndexError:
        #print list(tuples)
        print (a1, b1), (a2, b2)
        print len(alphas)
        exit()
    #print len(alphas), len(alphas[1])
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
        yield (run[i], run[i+1])

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
        x = 0.
        y = 0.
        for n in range(len(alphas)):
            N = len(alphas[n]) + 1
            for a in range(1, len(alphas[n])-1):
                x += alphas[n][a-1] / float(a)
                y += betas[n][a-1]/ float(n-a)
        if x != 0:
            means.append(y / x)
            return_runs[0].append((alphas, betas))
        else:
            return_runs[1].append((alphas, betas))
    return return_runs, numpy.mean(means), numpy.std(means), len(means)
