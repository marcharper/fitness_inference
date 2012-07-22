import functools
import math
import os
import sys
import numpy

from scipy import stats
from matplotlib import pyplot

def function_values_to_plot(f, a, b, steps=100):
    delta = (b - a) / float(steps)
    xs, ys = [], []
    for i in range(steps):
        x = a + delta * i
        y = f(x)
        xs.append(x)
        ys.append(y)
    return xs, ys

def indefinite_integral(f, a, b, steps=100, right=True):
    delta = (b - a) / float(steps)
    s = 0.
    if right:
        start = 1
    else:
        start = 0
    for i in range(start, steps):
        x = a + delta * i
        s += f(x) * delta
    return s

def mean(f, a, b, steps=1000, right=True):
    delta = (b - a) / float(steps)
    s = 0.
    if right:
        start = 1
    else:
        start = 0
    for i in range(start, steps):
        x = a + delta * i
        s += x * f(x) * delta
    return s
    
# Non-normalized!!
def birth_death_distribution(alphas, betas, N):
    log = math.log
    #N = float(N)
    def p(r):
        log_s = 0.
        for a in range(1, N):
            alpha = alphas[a]
            beta = betas[a]
            a = float(a)
            log_s += alpha * log(a) + beta * log(r*(N-a)) - (alpha + beta) * log(a + r * (N - a))
        return math.exp(log_s)
    return p

#def mode(alphas, betas, N):
    #p = 0.
    #q = 0.
    #for a in range(1, N):
        #p += float(a) * betas[a]
        #q += (N - float(a)) * alphas[a]
    #return p / q

#def mode(alphas, betas, N):
    #A = 0.
    #B = 0.
    #C = 0.
    #for a in range(1, N):
        #A -= (N - float(a)) * (N - float(a)) * (alphas[a] + 1)
        #B -= a * (N - float(a)) * (betas[a] - 1) - (N - float(a)) * (alphas[a] + 1) * (a + 1)
        #C += a * (a + 1) * (betas[a] - 1)
    ##A = -A
    #d = math.sqrt(B*B - 4 * A * C)
    #print d
    #print (-B + d) / (2 * A)
    #print (-B - d) / (2 * A)

def fig_1_priors(functions_labels, N=30, steps=1000):
    a = 0.0000001
    b = N // 3
    pyplot.clf()
    for f, label in functions_labels:
        xs, ys = function_values_to_plot(f, a, b, steps=steps)
        norm = indefinite_integral(f, a, b, steps=steps)
        ys = [y / norm for y in ys]
        print label
        print "mode:", numpy.argmax(ys) / float(steps) * (b - a)
        print "mean:", mean(f, a, b, steps=steps) / norm
        pyplot.plot(xs, ys, label=label)
    pyplot.grid(True)
    pyplot.legend()
    pyplot.xlabel("Relative Fitness r")
    pyplot.show()
    
    
if __name__ == '__main__':
    #alphas = [0] * N
    #betas = [0] * N
    #r = 2.
    #print r * N / (1 + r)
    #k = int(r * N / (1 + r))
    #alphas[k] += 200
    #betas[k] += 200
    #alphas[-1] = 2
    #betas[1] = 2
    #N = 3
    #prior = stats.gamma(1, scale=0.8678794).pdf
    #gamma = stats.gamma(2, scale=1./1.67835).pdf 
    #gamma = stats.gamma(2, scale=2).pdf
    gamma = stats.gamma(2, scale=1./2).pdf
    N = 50
    alphas = [1]*N
    betas = [1]*N
    fps = birth_death_distribution(alphas, betas, N)
    functions = [(fps, 'FPS'), (gamma, 'gamma')]

    alphas = [0] * N
    betas = [0] * N
    #alphas[10] = 50
    #betas[N-10] = 50
    #alphas[-1] = 50
    #betas[1] = 50
    #alphas[-1] = 10
    alphas[10] = 100
    betas[45] = 100
    #r = 2.
    #print r * N / (1 + r)
    #k = int(r * N / (1 + r))
    #alphas[k] += 200
    #betas[k] += 200
    
    s = 0.
    t = 0.
    for a in range(1, N):
        s += a * betas[a]
        t += (N-a) * alphas[a]
    print s / t        
    
    fps2 = birth_death_distribution(alphas, betas, N)
    functions.append((fps2, "FPS2"))
    fig_1_priors(functions)
