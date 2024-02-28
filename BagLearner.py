import random

import numpy as np


class BagLearner(object):

    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        self.learners = []    # list to simulate all bags
        for i in range(0,self.bags):
            self.learners.append(self.learner(**self.kwargs))   # create learner object
            x_random = np.empty((0, data_x.shape[1]))      # select random data w/ replacement
            y_random = np.empty(0)
            for j in range(0,data_x.shape[0]):
                random_counter = random.randint(0,data_x.shape[0]-1)
                x_random = np.append(x_random,np.array([data_x[random_counter,:]]),axis=0)
                y_random = np.append(y_random, np.array([data_y[random_counter]]), axis=0)
            self.learners[i].add_evidence(x_random,y_random)             # input random data to learner

    def query(self, points):
        learner_results = np.empty((0,points.shape[0]))                 # get predictions from each learner
        for i in range(0,self.bags):
            learner_results = np.append(learner_results, np.array([self.learners[i].query(points)]), axis=0)
        final_results = np.empty(0)                                    # compile results
        for i in range(0,points.shape[0]):
            final_results = np.append(final_results,np.mean(learner_results[:,i]))
        return final_results
