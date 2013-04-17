import random
import numpy

from posterior import Partition, Posterior

from mpsim.moran import moran_simulation_transitions as edge_function

"""Functions for inference of Moran process."""

# https://atpassos.posterous.com/normalizing-log-probabilities-with-numpy
def lognormalize(x):
    a = numpy.logaddexp.reduce(x)
    return numpy.exp(x - a)

def likelihood(a,b,d):
    """Likelihood functions for the Moran process associated to a -> a' events."""
    # a + b = N
    a = float(a)
    b = float(b)
    if d == 1: # A replicated
        def p(r):
            return a * b / (a + b) * 1. / (a + b * r)
    elif d == -1: # B replicated
        def p(r):
            return a * b / (a + b) * r / (a + b * r)
    elif d == 0: # population stationary
        def p(r):
            return (a*a + b*b*r) / (a + b) * 1. / (a + b * r)
    return p
    
def likelihood_table(N, partition_points):
    """Precompute likelihood functions on partitions for computational efficiency."""
    tables = []
    num_points = len(partition_points)
    for d in [1, -1, 0]:
        table = []
        for a in range(1, N):
            b = N - a
            vfunc = numpy.vectorize(likelihood(a,b,d))
            table.append(numpy.log(vfunc(partition_points)))
        tables.append(numpy.array(table))
    return tables      

def construct_posterior(N, conjugate_parameters, prior_points, partition, tables):
    posterior = Posterior(partition, prior_points)
    alphas, betas, gammas = conjugate_parameters
    alphas = numpy.array([alphas])
    betas = numpy.array([betas])
    gammas = numpy.array([gammas])
    alpha_table, beta_table, gamma_table = tables
    t = numpy.dot(numpy.array([1.]*(N-1)), (alphas.transpose() * alpha_table + betas.transpose() * beta_table + gammas.transpose() * gamma_table))
    t = lognormalize(t)
    posterior.update(t)
    return posterior    

def tuples_to_conjugate_parameters(N, tuples):
    """Reduce the run tuples to counts for the parameters alpha and beta."""
    alphas = [0] * (N-1)
    betas  = [0] * (N-1)
    gammas = [0] * (N-1)
    for ((a1, b1), (a2, b2)) in tuples:
        if (a1 == a2) and (b1 == b2):
                gammas[a1-1] += 1
        elif b2 - b1 == 1:
                betas[a1-1] += 1
        else:
            alphas[a1-1] += 1
    return (alphas, betas, gammas)    
    
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
        return_runs.append(list(convert_run_to_tuples(run, max_length=max_length, sample_size=sample_size)))
    return return_runs

### Birth Counting

def reproduction_rate_test(parameters):
    means = []
    return_runs = [[],[]]
    for alphas, betas, gammas in parameters:
        x = sum(alphas)
        y = sum(betas)
        #for a in range(1, len(alphas)-1):
            #x += alphas[a-1] / float(a)
            #y += betas[a-1]/ float(N-a)
        if x != 0:
            means.append(y / x)
            return_runs[0].append((alphas, betas))
        else:
            return_runs[1].append((alphas, betas))
    return return_runs, numpy.mean(means), numpy.std(means), len(means)

