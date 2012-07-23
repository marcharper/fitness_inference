#import itertools
import csv
import os
import math
import multiprocessing
import random
import sys

import numpy
from matplotlib import pyplot
from scipy import stats

import mpsim
import mpsim.main
from mpsim.memoize import Memoize
from mpsim.multiset import Multiset
from mpsim.helpers import ensure_directory, ensure_digits

from posterior import Partition, Posterior

## Initial State Generator examples

#igen = mpsim.generators.random_state_generator(2,N)
#igen = mpsim.generators.constant_generator((N-1, 1))

def systematic_state_generator(N, each=1000):
    for i in range(1, N):
        for _ in range(each):
            yield (i, N-i)

## Multiprocessing functions for inferences. ##

def batch(args):
    N, run, prior_points, partition, table, sample_size = args
    conjugate_parameters = convert_run_to_conjugate_parameters(N, run, sample_size=sample_size)
    posterior = construct_posterior(N, conjugate_parameters, prior_points, partition, table)
    return (conjugate_parameters, posterior.mean()[0], posterior.mode()[0])

def params_gen(N, runs, prior_points, partition, table, sample_size):
    """Generator for batch parameters."""
    for run in runs:
        yield N, run, prior_points, partition, table, sample_size
    
def run_batches(N, runs, prior_points, partition, table, sample_size=None, processes=None):
    """Runs simulations on multiple processing cores in batch sizes dictated by iters_gen, posting data to callbacks to reduce memory footprint."""
    if not processes:
        processes = multiprocessing.cpu_count()
    params = list(params_gen(N, runs, prior_points, partition, table, sample_size))
    pool = multiprocessing.Pool(processes=processes)
    try:
        results = pool.map(batch, params)
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print 'Control-C received. Exiting.'
        pool.terminate()
        exit()
    return results    
    
### Hypothesis Tests

#def conditionals(N, r, num_runs=1000, min_length=1, max_length=100, sample_size=5):
    #prior = stats.gamma(1, scale=1).pdf
    #R = 10
    #bounds = [0, R]
    #fitness_landscape = mpsim.moran.fitness_static(r)
    #igen = mpsim.generators.random_state_generator(2,N)
    #runs = mpsim.main.two_type_moran_process_simulations(N, fitness_landscape, iterations=num_runs, initial_state_generator=igen)
    #runs = [history for (seed, length, history) in runs]
    #runs = runs_gen(runs, min_length=min_length, max_length=max_length, sample_size=sample_size)
    #means = []
    #for run in runs:
        #mean = infer_static_fitness(run, N, prior, bounds)
        #means.append(mean)
    ##print ""
    ##print 100. * len([x for x in means if x >= r]) / len(means)
    ##print 100. * len([x for x in means if x >= 1.]) / len(means)
    ##pyplot.hist(means, bins=max(20, num_runs/20.))
    ##pyplot.show()
    ##return 100. * len([x for x in means if x >= r]) / len(means), 100. * len([x for x in means if x >= 1.]) / len(means)
    #return 100. * len([x for x in means if x >= 1.]) / len(means)

def inference_test(N, r, table, runs, prior, partition):
    means = []
    for run_tuples in runs:
        #posterior = construct_posterior(N, run_tuples, prior, partition, table)
        posterior = construct_posterior(N, run_tuples, partition)
        means.append(posterior.mean()) 
    m = numpy.mean(means)
    s = numpy.std(means)
    return m, s

#def single_run_video(N=30, bounds=(0, 10), steps=1000, num_runs=1000, r=2.0):
    #k = int(r * N / (1 + r))
    ##k = int(r * N / (2*(1 + r)))
    ##initial_state = (N//2,N - N//2)
    #initial_state = (k, N - k)
    #igen = mpsim.generators.constant_generator(initial_state)
    #partition = Partition(bounds[0], bounds[1], steps)
    #table = likelihood_table(N, partition.points)
    #fitness_landscape = mpsim.moran.fitness_static(r)
    #runs = mpsim.main.two_type_moran_process_simulations(N, fitness_landscape=fitness_landscape, edge_function=mpsim.moran.generalized_moran_simulation_transitions, iterations=num_runs, initial_state_generator=igen)
    #runs = [history for (seed, length, history) in runs]
    #runs = convert_runs(N, runs)
    ### Prior setup
    #prior_points = None
    ##prior = stats.gamma(1, scale=0.8678794).pdf
    ##prior = stats.gamma(2, scale=1./1.67835).pdf
    ##partition = Partition(bounds[0], bounds[1], steps)
    ##vfunc = numpy.vectorize(prior)
    ##prior_points = vfunc(partition.points)
    #alphas = [0]*N
    #betas = [0]*N
    ##alphas[0] = 1
    ##betas[-1] = 1
    #prior_density = birth_death_distribution(alphas, betas)
    #prior_points = numpy.array(map(prior_density, partition.points))
    #means = []
    #modes = []
    #medians = []
    #f_probs_a = []
    #f_probs_b = []
    #death_probs = []
    #for tuples in runs:
        #if len(tuples) < 100:
            #continue
        #digits = len(str(len(tuples)))
        #for i in range(0, len(tuples)):
            #posterior = Posterior(partition, prior_points)
            #if i > 0:
                #(a, b), d = tuples[i]
                #alphas, betas = tuples_to_conjugate_parameters(N, tuples[:i])
                ##print alphas
                ##print betas
                #density = birth_death_distribution(alphas, betas)
                #values = numpy.array(map(density, partition.points))
                #posterior.update(values, normalize=True)
            #else:
                #(a, b) = initial_state
            #mean_x, mean_y = posterior.mean()
            #means.append(mean_x)
            #mode_x, mode_y = posterior.mode()
            #modes.append(mode_x)
            #median_x, median_y = posterior.median()
            #medians.append(median_x)
            #f_probs_a.append(a / (a + r * b))
            #f_probs_b.append(b * r / (a + r * b))
            #death_probs.append(b / float(a + b))
            ##print alphas
            ##print betas
            #print mean_x, mean_y, mode_x, mode_y, median_x, median_y
            #pyplot.clf()
            ##fig = pyplot.subplot(223)
            #fig = pyplot.subplot(411)
            #fig.plot(posterior.points, posterior.values)
            #ylim = pyplot.ylim()[1]
            #pyplot.xlim((0, 2*r))
            #fig.axvline(x=mean_x, ymin=0, ymax=mean_y / ylim, c='r')
            #fig.axvline(x=mode_x, ymin=0, ymax=mode_y / ylim, c='g')
            #fig.axvline(x=median_x, ymin=0, ymax=median_y / ylim, c='black')
            #fig.grid(True)
            #pyplot.subplots_adjust(hspace=.5)
            
            ##fig2 = pyplot.subplot(221)
            #fig2 = pyplot.subplot(412)
            #fig2.barh([0.5, 1.5],[a,b], align='center')
            #pyplot.yticks([0.5, 1.5], ('A', 'B'))
            ##pyplot.xlabel('Counts')
            #pyplot.title('Population Distribution | A: fitness = 1,  B: fitness = %s' % r)
            #pyplot.xlim(0, N)
            #fig2.grid(True)
            
            ##fig4 = pyplot.subplot(222)
            #fig4 = pyplot.subplot(413)
            #domain = range(len(medians))
            #fig4.plot(domain, f_probs_a, label="F_A")
            #fig4.plot(domain, f_probs_b, label="F_B")
            #fig4.plot(domain, death_probs, label="Death_B")
            #pyplot.ylabel("Probability")
            ## Shink current axis by 20%
            #box = fig4.get_position()
            #fig4.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ## Put a legend to the right of the current axis
            #fig4.legend(loc='center left', bbox_to_anchor=(1, 0.5))            

            
            ##fig3 = pyplot.subplot(224)
            #fig3 = pyplot.subplot(414)
            #domain = range(len(medians))
            #fig3.plot(domain, means, label="mean", c='r')
            #fig3.plot(domain, modes, label="mode", c='g')
            #fig3.plot(domain, medians, label="median", c='black')
            #fig3.grid(True)
            #pyplot.xlabel('Iterations')

            ## Shink current axis by 20%
            #box = fig3.get_position()
            #fig3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ## Put a legend to the right of the current axis
            #fig3.legend(loc='center left', bbox_to_anchor=(1, 0.5))            
            
            #pyplot.savefig('conjugate/' + str(ensure_digits(digits, str(i))) + ".png", dpi=160, pad_inches=0.5)
        #exit()
#sigmoid_death(N)
def single_run_test(module, N=20, bounds=(0, 20), steps=1000, num_runs=1000, r=1.2, death_probabilities_func=None, sample_size=None):
    """Single run inference test."""
    if death_probabilities_func:
        death_probabilities = death_probabilities_func(N)
    else:
        death_probabilities = None
    i = max(1, int(r*N / (1. + r)))
    igen = mpsim.generators.constant_generator((i,N - i))
    partition = Partition(bounds[0], bounds[1], steps)
    print "Generating likelihood table..."
    table = module.likelihood_table(N, partition.points)
    fitness_landscape = mpsim.moran.fitness_static(r)
    print "Generating %s trajectories..." % (str(num_runs))
    runs = mpsim.main.two_type_moran_process_simulations(N, fitness_landscape=fitness_landscape, edge_function=module.edge_function, iterations=num_runs, initial_state_generator=igen, death_probabilities=death_probabilities)
    runs = [history for (seed, length, history) in runs]
    #print runs[1]
    #exit()
    ## Prior setup
    prior_points = None
    prior = stats.gamma(2, scale=0.5).pdf
    partition = Partition(bounds[0], bounds[1], steps)
    prior_points = partition.map(prior)
    means = []
    print "Computing inferred values of parameter r..."
    global construct_posterior
    construct_posterior = module.construct_posterior
    global convert_run_to_conjugate_parameters    
    convert_run_to_conjugate_parameters = module.convert_run_to_conjugate_parameters

    #for run in runs:
        #conjugate_parameters = convert_run_to_conjugate_parameters(N, run, sample_size=sample_size)
        ##print conjugate_parameters
        #posterior = construct_posterior(N, conjugate_parameters, prior_points, partition, table)
        #print (posterior.mean()[0], posterior.mode()[0])
    #exit()
    batch_results = run_batches(N, runs, prior_points, partition, table, sample_size=sample_size)
    conjugate_parameters = [c for (c, mean, mode) in batch_results]
    _, m, s, c = module.reproduction_rate_test(conjugate_parameters)
    print m, s, c,
    means = [mean for (c, mean, mode) in batch_results]
    print numpy.mean(means), numpy.std(means),
    modes = [mode for (c, mean, mode) in batch_results]
    print numpy.mean(modes), numpy.std(modes)
    
    #for i, posterior in enumerate(posteriors):
        #mean, mode = posterior
        #means.append(mean)
    #del construct_posterior
    #print numpy.mean(means), numpy.std(means)
    #i = max(1, int(r*N / (1. + r)))
    #print float(len([x for x in means if x > 1])) / num_runs, (1. - r**(i-N)) / (1. - r**(-N))
    #pyplot.clf()
    #pyplot.hist(means, bins=30)
    #pyplot.show()
    
def cache_simulation_results(module, filename, Ns, rs, prior=None, bounds=None, num_runs=1000, sample_size=None, steps=1000):
    if not prior:
        prior = stats.gamma(2, scale=0.5).pdf
    if not bounds:
        bounds = (0, 10)
    partition = Partition(bounds[0], bounds[1], steps)
    prior_points = None
    prior_points = partition.map(prior)

    ensure_directory("simulation_results")
    writer = csv.writer(open(os.path.join("simulation_results", filename),'w'))

    # Pull some needed functions into the main namespace to make multiprocessing happy.
    global construct_posterior
    construct_posterior = module.construct_posterior
    global convert_run_to_conjugate_parameters    
    convert_run_to_conjugate_parameters = module.convert_run_to_conjugate_parameters

    # Generate data
    for r in rs:
        fitness_landscape = mpsim.moran.fitness_static(r)
        for N in Ns:
            table = module.likelihood_table(N, partition.points)
            for i in range(1, N):
                # Compute the results for both inference and counting
                row = [r, N, i]
                print module.__name__, sample_size, r, N, i
                igen = mpsim.generators.constant_generator((i, N-i))
                runs = mpsim.main.two_type_moran_process_simulations(N, fitness_landscape=fitness_landscape, edge_function=module.edge_function, iterations=num_runs, initial_state_generator=igen)
                runs = [history for (seed, length, history) in runs]
                batch_results = run_batches(N, runs, prior_points, partition, table, sample_size=sample_size)
                conjugate_parameters = [c for (c, mean, mode) in batch_results]
                #if sample_size:
                    #for c in conjugate_parameters:
                        #s = sum(sum(x) for x in c)
                        #if s > sample_size:
                            #print "sample_size"
                            #exit()
                #print conjugate_parameters
                #means = [mean for (c, mean, mode) in batch_results]

                #all_tuples = []
                #for run in runs:
                    #tuples = list(module.convert_run_to_tuples(N, run, sample_size=sample_size))
                    #all_tuples.append(tuples)
                    #alphas, betas = module.tuples_to_conjugate_parameters(N, tuples)
                    #conjugate_parameters.append((alphas, betas))
                _, m, s, c = module.reproduction_rate_test(conjugate_parameters)
                #import infer2
                row.extend([m, s, c])
                #split_runs, m, s, l = infer2.reproduction_rate_test(N, r, all_tuples)
                #print m, s, l
                #means = []
                #posteriors = run_batches(N, conjugate_parameters, prior_points, partition, table)
                #for i, posterior in enumerate(posteriors):
                    #mean, mode = posterior
                    ##mean = posterior.mean()[0]
                    #means.append(mean)
                #print means
                means = [mean for (c, mean, mode) in batch_results]
                row.extend([numpy.mean(means), numpy.std(means)])
                modes = [mode for (c, mean, mode) in batch_results]
                row.extend([numpy.mean(modes), numpy.std(modes)])
                ## Also, look at estimate if all conjugate_parameters are joined
                # Produces terrible estimates in general.
                #all_conjugates = []
                #for i in range(len(conjugate_parameters[0])):
                    #cons = sum(numpy.array(x[0]) for x in conjugate_parameters)
                    #all_conjugates.append(cons)
                #posterior = construct_posterior(N, all_conjugates, prior_points, partition, table)
                #row.extend([posterior.mean()[0], posterior.mode()[0]])
                #writer.writerow(map(str,row))                
                
                # Check for agreement with previous (much slower) method
                #from infer2 import run_batches as run_batches2
                #means = []
                #partition = infer2.Partition(bounds[0], bounds[1], steps)
                #vfunc = numpy.vectorize(prior)
                #prior_points = vfunc(partition.points)
                #table = infer2.likelihood_table(N, partition.points)
                #for tuples in module.convert_runs(N, runs):
                    #posterior = infer2.construct_posterior(N, tuples, prior_points, partition, table)
                    #means.append(posterior.mean())
                #m = numpy.mean(means)
                #s = numpy.std(means)
                #print m, s
                
                #filename = os.path.join(data_root, "%s_%s_%s" % (r, N, i))
                #with open(filename, 'a') as handle:
                    #handle.write('\n'.join(rows))
                #results = method_comparison(module, N, r, prior_points, partition, table=table, igen=igen, sample_size=sample_size)
                #print >> handle, r, N, i, " ".join(map(str, results))
                #print r,N,i

def generate_heatmap_data():
    rs = numpy.arange(0.1,2.1,0.1)
    Ns = range(3,51,1)
    #for module_name in ['fps', 'moran']:
    for module_name in ['moran']:
        module = __import__(module_name)
        filename = module_name + ".csv"
        cache_simulation_results(module, filename, Ns, rs)
    #for sample_size in [10, 20]:
        #module_name = "fps"
        #module = __import__(module_name)
        #filename = '_'.join([module_name, "sample_size", str(sample_size)]) + ".csv"
        #cache_simulation_results(module, filename, Ns, rs, sample_size=sample_size)
                
if __name__ == '__main__':
    try:
        module_name = sys.argv[1]
    except IndexError:
        generate_heatmap_data()
    else:
        try:
            module = __import__(module_name)
            #death_probabilities_func = mpsim.moran.sigmoid_death
            #death_probabilities_func = mpsim.moran.moran_cascade
            #death_probabilities_func = mpsim.moran.moran_death
            death_probabilities_func = mpsim.moran.linear_death
            single_run_test(module, N=40, death_probabilities_func=death_probabilities_func)
        except ImportError:
            print "%s is not a valid module" % (module_name)
            exit()
    exit()
    
    single_run_test(module)
    exit()
    
    single_run_video(N=100, bounds=[0, 100], r=50.)
    exit()

