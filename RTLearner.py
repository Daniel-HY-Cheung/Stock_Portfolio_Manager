import random

import numpy as np


class RTLearner(object):

    def __init__(self, leaf_size):
        """
        Constructor method
        """
        self.leaf_size = leaf_size

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        data_y = np.vstack(data_y)
        data = np.append(data_x,data_y, axis=1)
        self.tree = self.build_tree(data, self.leaf_size)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        results = np.empty(0)
        for x_data in points:
            results = np.append(results,self.predict_y(x_data, 0))
        return(results)

    def build_tree(self, data, leaf_size):
        if(data[:,-1].shape[0] <= leaf_size):
            return np.array([["leaf", np.mean(data[:,-1]), "NA", "NA"]])
        elif(np.all(data[:,-1] == data[0,-1])):
            return np.array([["leaf", data[0,-1], "NA", "NA"]])
        else:
            # determine best x feature to split on
            best_x = random.randint(0,data.shape[1]-2)
            # find split_val
            split_val = np.median(data[:,best_x])

            # corner case: all remaining data fall on one side of the median (no progress)
            if(np.all(data[:,best_x]<=split_val) or np.all(data[:,best_x]>split_val)):
                if(np.all(data[:,best_x] == data[0,best_x])):
                    return np.array([["leaf", np.mean(data[:,-1]), "NA", "NA"]])
                else:
                    split_val = np.mean(data[:,best_x])
            # corner case: all remaining data have the same value for best_x

            # recursive call to solve for left tree & right tree
            lefttree = self.build_tree(data[data[:, best_x] <= split_val], leaf_size)
            righttree = self.build_tree(data[data[:,best_x]>split_val], leaf_size)

            # create note
            root = np.array([[best_x,split_val,1,lefttree.shape[0]+1]])

            # construct this portion of the list, return it
            tree = np.empty((0,4))
            tree = np.append(tree, root, axis=0)
            tree = np.append(tree, lefttree, axis=0)
            tree = np.append(tree, righttree, axis=0)
            return(tree)

    def predict_y(self, x_data, node):
        if(self.tree[node,0] != "leaf"):
            if(x_data[int(float(self.tree[node,0]))] <= float(self.tree[node,1])):
                return self.predict_y(x_data, node + int(float(self.tree[node,2])))
            else:
                return self.predict_y(x_data, node + int(float(self.tree[node, 3])))
        else:
            return float(self.tree[node,1])
