import random
import sys

import numpy
from scipy import stats

from mpsim.graph import Graph, RandomGraph
from mpsim.math_helpers import dot_product, multiply_vectors

from fps import distribution, likelihood_table, construct_posterior, tuples_to_conjugate_parameters
from posterior import Partition, Posterior

def cycle(length, directed=False):
    graph = Graph()
    edges = []
    for i in range(length - 1):
        edges.append((i, i+1))
        if not directed:
            edges.append((i+1, i))
    edges.append((length - 1, 0))
    if not directed:
        edges.append((0, length - 1))
    graph.add_edges(edges)
    return graph
    
def double_cycle(length, directed=False):
    graph = Graph()
    edges = []
    for i in range(length):
        edges.append((i, (i+1) % length))
        edges.append((i, (i+2) % length))
        if not directed:
            edges.append(((i+1) % length, i))
            edges.append(((i+2) % length, i))
    graph.add_edges(edges)
    return graph
    
def star(length):
    "Should amplify selection."""
    graph = Graph()
    edges = []
    for i in range(length):
        edges.append((i, 0))
        edges.append((0, i))
    graph.add_edges(edges)
    return graph    

def star_cycle(length):
    """Wheel graph, apparently."""
    graph = Graph()
    edges = []
    for i in range(length):
        edges.append((i, 0))
        edges.append((0, i))
        edges.append((i, (i+1) % length))
        edges.append(((i+1) % length, i))
    graph.add_edges(edges)
    return graph       

def super_star(k, n):
    edges = []
    for i in range(k):
        edges.append(((0, i), (0,0)))        
    for i in range(k):
        for j in range(1, n):
            edges.append(((j, i), (0, i)))
            edges.append(((0,0),(j, i)))        
    graph = Graph()
    graph.add_edges(edges)
    #for vertex in graph.vertices():
        #print vertex, graph.out_vertices(vertex)
    return graph    
    
def grid(rows, columns):
    graph = Graph()
    edges = []
    for i in range(rows):
        for j in range(columns):
            if j + 1 < columns:
                edges.append(((i, j), (i, j+1)))
                edges.append(((i, j+1), (i, j)))
            if i + 1 < rows:
                edges.append(((i, j), (i+1, j)))
                edges.append(((i+1, j), (i, j)))
    graph.add_edges(edges)
    #for vertex in graph.vertices():
        #print vertex, graph.out_vertices(vertex)
    return graph

def complete(N):
    """Make a graph with two complete graphs connected by a single edge."""
    edges = []
    for i in range(0, N):
        for j in range(i):
            edges.append((i, j))
            edges.append((j, i))
    graph = Graph()
    graph.add_edges(edges)
    return graph    
    
def double_complete(N):
    """Make a graph with two complete graphs connected by a single edge."""
    m = N//2
    edges = []
    for i in range(0, m):
        for j in range(i):
            edges.append((i, j))
            edges.append((j, i))
            edges.append((i+m, j+m))
            edges.append((j+m, i+m))
    edges.append((m-1, m))
    edges.append((m, m-1))
    graph = Graph()
    graph.add_edges(edges)
    return graph
    
#def tree(rows, n):
    
class PopulationOnGraph(object):
    def __init__(self, graph, occupation, fitness, seed=None):
        if not seed:
            seed = random.random()
        self.srandom = random.Random()
        self.srandom.seed(seed)
        self.occupation = occupation
        self.graph = graph
        self.N = len(occupation)
        self.fitness = fitness
        count = sum(occupation)
        self.population = [self.N - count, count]
        self.enum = dict(enumerate(graph.vertices()))
        self.inv_enum = dict([(y,x) for (x, y) in enumerate(graph.vertices())])
        # Initialize graph structure with enumerated vertices
        # Initialize occupation of vertices as a binary vector
        # maintain count of types and fitness landscape
        # birth_death or death_birth
        pass
        
    def fitness_proportionate_selection(self):
        #landscape = multiply_vectors(self.population, self.fitness)
        #total = sum(landscape)
        #r = total * self.srandom.random()
        #if r < landscape[0]:
            #return 0
        #return 1
        landscape = [self.fitness[self.occupation[i]] for i in range(self.N)]
        csum = numpy.cumsum(landscape)
        r = csum[-1] * self.srandom.random()
        #for i in range(self.N):
            #if r <= csum[i]:
                #return i
        for j, x in enumerate(csum):
            if x >= r:
                return j

        
    def __iter__(self):
        return self

    def state(self):
        return tuple(self.population)
        
    def next(self):
        if self.population[0] == 0 or self.population[1] == 0:
            raise StopIteration
        birth_index = self.fitness_proportionate_selection()
        out_vertices = self.graph.out_vertices(self.enum[birth_index])
        # This is for the random graph but can hide errors for other graphs. Beware!
        if not len(out_vertices):
            death_index = birth_index
        else:
            death_index = self.inv_enum[self.srandom.choice(out_vertices)]
        #print birth_index, death_index
        self.population[self.occupation[birth_index]] += 1
        self.population[self.occupation[death_index]] -= 1
        self.occupation[death_index] = self.occupation[birth_index]
        if self.occupation[birth_index] == 0:
            d = 1
        else:
            d = 0
        return (self.state(), d)

#def tuples_to_conjugate_parameters(N, tuples):
    #"""Reduce the run tuples to counts for the parameters alpha and beta."""
    #alphas = [0] * (N-1)
    #betas = [0] * (N-1)
    #for ((a, b), d) in tuples:
        #if d == 0:
            #alphas[a-1] += 1
        #elif d == 1:
            #betas[a-1] += 1
        #else:
            #raise ValueError, "d not in {0,1}, d=%s" % d
    #return (alphas, betas)        
        
#def construct_posterior(N, alphas, betas, partition, prior=None):
    #"""Constructs a posterior from a prior and run tuples. Does not have enough precision if alpha or beta values are too large (over ~150)."""
    ##alphas, betas = tuples_to_conjugate_parameters(N, tuples)
    #density = distribution(alphas, betas)
    #values = numpy.array(map(density, partition.points))
    ## This constructor will normalize values.
    #posterior = Posterior(partition, prior)
    #posterior.update(values, normalize=True)
    #return posterior        

def test():
    N = 250
    r = 1.2
    iterations = 100
    #graph = cycle(N, directed=True)
    #graph = cycle(N, directed=False)
    #graph = complete(N)
    #graph = grid(N//2, N//2)
    #graph = double_cycle(N, directed=True)
    #graph = star_cycle(N)
    #graph = star(N)
    #graph = super_star(5, 6)
    #graph = double_complete(N)
    graph = RandomGraph(N, 0.5)
    N = len(graph.vertices())
    print N
    #print occupation
    fitness = [1, r]
    bounds = [0, 60]
    steps = 1000
    prior = stats.gamma(2, scale=0.5).pdf
    partition = Partition(bounds[0], bounds[1], steps)
    prior_points = partition.map(prior)
    table = likelihood_table(N, partition.points)
    means = []
    for i in range(iterations):
        occupation = [0]*N
        #occupation[::3] = [1]*(len(occupation[::3]))
        occupation[:N//2] = [1]*(len(occupation[:N//2]))
        pop = PopulationOnGraph(graph, occupation, fitness)
        tuples = list(pop)
        #print list(convert_run_to_tuples(N, run))
        #alphas, betas = tuples_to_conjugate_parameters(N, tuples[:-1])
        #print alphas, betas
        posterior = construct_posterior(N, tuples[:-1], prior_points, partition, table)
        #return (posterior.mean()[0], posterior.mode()[0])
        #posterior = construct_posterior(N, alphas, betas, partition)
        mean = posterior.mean()[0]
        means.append(mean) 
        print i, mean, tuples[-1]
        #print posterior.mean()
    print numpy.mean(means), numpy.std(means)
    
def variance_test():
    """Random graph figure."""
    N = 12
    r = 1.2
    iterations = 10000
    steps = 100
    fitness = [1, r]
    bounds = [0, 20]
    results = []
    m = int(N - N*r / (1+ r))
    print m
    for q in range(1, steps):        
        p = q / float(2*steps)
        graph = RandomGraph(N, p)
        N2 = len(graph.vertices())
        prior = stats.gamma(2, scale=0.5).pdf
        partition = Partition(bounds[0], bounds[1], steps=1000)
        prior_points = partition.map(prior)
        table = likelihood_table(N, partition.points)
        means = []
        for i in range(iterations):
            occupation = [0]*N2
            #occupation[::2] = [1]*(len(occupation[::2]))
            #occupation[:N2//2] = [1]*(len(occupation[:N2//2]))
            occupation[:m] = [1]*(len(occupation[:m]))
            pop = PopulationOnGraph(graph, occupation, fitness)
            tuples = list(pop)
            #print list(convert_run_to_tuples(N, run))
            #alphas, betas = tuples_to_conjugate_parameters(N, tuples[:-1])
            #print alphas, betas
            posterior = construct_posterior(N, tuples[:-1], prior_points, partition, table)
            #return (posterior.mean()[0], posterior.mode()[0])
            #posterior = construct_posterior(N, alphas, betas, partition)
            mean = posterior.mean()[0]
            means.append(mean) 
            #print i, mean, tuples[-1]
            #print posterior.mean()
            r_hat = numpy.mean(means)
            std = numpy.std(means)
        results.append((p, r_hat, std))
        print p, r_hat, std, (r - r_hat) / std
    
if __name__ == "__main__":
    #variance_test()
    #exit()
    for a in range(2, 10):
        for b in range(2, 10):
            r = 1.2
            iterations = 200
            #graph = cycle(N, directed=True)
            #graph = cycle(N, directed=False)
            #graph = complete(N)
            #graph = grid(N//2, N//2)
            #graph = double_cycle(N, directed=True)
            #graph = star_cycle(N)
            #graph = star(N)
            graph = super_star(a, b)
            #graph = double_complete(N)
            #graph = RandomGraph(N, 0.01)
            N = len(graph.vertices())
            #print N2, len(set(graph._edges)) // 2
            #print N
            #print occupation
            fitness = [1, r]
            bounds = [0, 60]
            steps = 1000
            prior = stats.gamma(2, scale=0.5).pdf
            partition = Partition(bounds[0], bounds[1], steps)
            prior_points = partition.map(prior)
            table = likelihood_table(N, partition.points)
            means = []
            for i in range(iterations):
                occupation = [0]*N
                occupation[::2] = [1]*(len(occupation[::2]))
                #occupation[:N2//2] = [1]*(len(occupation[:N2//2]))
                pop = PopulationOnGraph(graph, occupation, fitness)
                tuples = list(pop)
                #print list(convert_run_to_tuples(N, run))
                #alphas, betas = tuples_to_conjugate_parameters(N, tuples[:-1])
                #print alphas, betas
                posterior = construct_posterior(N, tuples[:-1], prior_points, partition, table)
                #return (posterior.mean()[0], posterior.mode()[0])
                #posterior = construct_posterior(N, alphas, betas, partition)
                mean = posterior.mean()[0]
                means.append(mean) 
                #print i, mean, tuples[-1]
                #print posterior.mean()
            print N, a, b, numpy.mean(means), numpy.std(means)
        
    