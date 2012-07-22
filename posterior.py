import numpy

DEFAULT_PARTITION_STEPS = 1000

class Partition(object):
    """Uniform partition of an interval [a,b] for computation of trapezoidal integration approximations."""
    def __init__(self, a, b, steps=None, left=False, right=True):
        if not steps:
            steps = DEFAULT_PARTITION_STEPS
        self.a = a
        self.b = b
        delta = (b - a) / float(steps)
        self.delta = delta
        self.steps = steps
        points = []
        if left:
            start = 0
        else:
            start = 1
        if right:
            stop = steps + 1
        else:
            stop = steps
        for i in range(start, stop):
            points.append(a + delta*i)
        self.points = numpy.array(points)

    def map(self, func):
        """Computes the values of a function on the points of a partition."""
        # Have to use prior points here to make multiprocessing happy because of pickling issues
        # This is how to extract the points
        vfunc = numpy.vectorize(func)
        values = vfunc(self.points)
        return tuple(values)
        
class Posterior(object):
    """Tracks posterior function updates for approximations on partitions."""
    def __init__(self, partition, prior_points=None):
        self.partition = partition
        self.delta = partition.delta
        self.points = partition.points
        # Weights for trapezoidal integration.
        self.weights = numpy.array([2.]*len(self.points))
        self.weights[0] = 1.
        self.weights[-1] = 1.
        if prior_points:
            self.values = list(tuple(prior_points))
            self.normalize()
        else:
            self.values = [1.]*len(self.points)
        # Normalization
        self.norm_counter = 1
        self.normed_since_update = False

    def update(self, likelihoods, normalize=False):
        """Takes an array of likelihood values and updates the posterior values."""
        self.values *= likelihoods
        # Normalize if requested but...
        # don't normalize all the time; it's unnecessary and expensive.
        if normalize or self.norm_counter == 10:
            self.normalize()
            self.norm_counter = 0.
        else:
            self.norm_counter += 1

    def normalize(self):
        """Normalize the values to a probability distribution via trapezoidal integration."""
        s = sum(self.values * self.weights) * self.delta / 2.
        self.values /= s

    def mean(self):
        """Compute the mean via trapezoidal integration for parameter estimate extraction."""
        if self.norm_counter > 0:
            self.normalize()
        mean = self.delta / 2. * sum(self.points * self.values * self.weights)
        # find approximate y-value of mean
        i = int((mean - self.partition.a) / self.partition.delta)
        if i < 0:
            i = 0
        if i > len(self.values) - 1:
            i = len(self.values) - 1
        return (mean, self.values[i])
        
    def mode(self):
        index = numpy.argmax(self.values)
        return (self.points[index], self.values[index])

    def median(self):
        areas = self.values * self.weights * self.delta / 2.
        csums = numpy.cumsum(areas)
        for i in range(len(csums)):
            if csums[i] > 0.5:
                break
        index = i
        return (self.points[index], self.values[index])
        
    def points_to_plot(self):
        """Return points of the partitition and probability distribution values for plotting."""
        #return tuple(self.points), tuple(self.values)
        return self.points, self.values

#def compute_on_partition(func, partition):
    #"""Computes the values of a function on the points of a partition."""
    ## Have to use prior points here to make multiprocessing happy because of pickling issues
    ## This is how to extract the points
    #vfunc = numpy.vectorize(func)
    #prior_points = vfunc(partition.points)
    #return prior_points
