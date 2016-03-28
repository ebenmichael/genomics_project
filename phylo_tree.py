# -*- coding: utf-8 -*-
"""
Phylogeny Tree class
"""
import numpy as np
import scipy as sp


class Node():
    
    def __init__(self,cov_mat =  None, children = None, data = None, observed = False):
        """Constructor
        Input:
            cov_mat: for unobserved node, covariance matrix representing gene 
                        coexpression, optional
            children: array of (Node,float) tuples representing children and
                        branch lengths
            data: for an observed node, the data, optional
            observed: whether node is oberved or not, defaults to False
        """
        
        self.cov_mat = cov_mat
        if children == None:
            self.children = []
        else:
            self.children = children
        self.observed = observed
        self.data = data
        
    def set_covariance(self, cov_mat):
        """Sets covariance matrix to cov_mat"""
        self.cov_mat = cov_mat
        
    def add_data(self,data):
        """For observed nodes, adds the data"""
        self.data = data
        
    def set_children(self, children):
        """Sets children to children"""
        self.children = children
        
    def add_child(self,child,weight):
        """Adds a child with weight to children"""
        self.children.append((child,weight))
        
    def sample(self):
        """Samples covariance matrices according to N(cov_ij,weight) and 
            assigns sampled covairance matrices to children"""
            #size of covariance matrix
        m = self.cov_mat.shape[0]
        for child, weight in self.children:
            #if an interior node sample covariance
            m = self.cov_mat.shape[0]
            if child.observed != True:
                cov_mat_vec= self.cov_mat.reshape((m**2,))
                #sample from normal distribution
                sample = [np.random.normal(cov_mat_vec[i],weight,1)
                            for i in range(m**2)]
                #reshape sample to be a covariance matrix
                s = np.array(sample)
                s = s.reshape((m,m))
                child.set_covariance(s)
                child.sample()
            #if a child is an observed node sample data
            elif child.observed == True:
                #for observed nodes weight is number of data points
                x = np.random.multivariate_normal(np.zeros(m), 
                                                  self.cov_mat,weight)
                child.add_data(x)
            
    def __repr__(self):
        if self.observed:
            return(str(self.data))
        else:
            return(str(self.cov_mat))
        