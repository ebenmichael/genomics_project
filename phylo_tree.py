# -*- coding: utf-8 -*-
"""
Phylogeny Tree class
"""
import numpy as np
import scipy as sp
import random 
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix


class Node():
    
    def __init__(self,cov_mat =  None, children = None, parent = None,
                 data = None, observed = False, index = None):
        """Constructor
        Input:
            cov_mat: for unobserved node, covariance matrix representing gene 
                        coexpression, optional
            children: array of (Node,float) tuples representing children and
                        branch lengths
            parent: parent node
            data: for an observed node, the data, optional
            observed: whether node is oberved or not, defaults to False
            index: a numbering of nodes, used in a tree 
        """
        
        self.cov_mat = cov_mat
        if children == None:
            self.children = []
        else:
            self.children = children
        self.parent = parent
        self.observed = observed
        self.data = data
        self.index = index
        
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
        
    def add_children(self,children):
        """Adds children from a list of (child,weight) tuples"""
        for child,weight in children:
            self.add_child(child,weight)
    
    def set_parent(self, parent):
        """Sets the parent"""
        self.parent = parent
        
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
        return(self.__str__())
            
    def __str__(self):
        if self.index != None:
            string = "Node_" + str(self.index)
            if self.observed:
                string += "_O"
        else:
            string = "Node"
            if self.observed:
                string += "_O"
        return(string)

class PhyloTree():
    
    
    def __init__(self):
        """Empty construtor"""
        #root of tree
        self.root = None
        #number of nodes
        self.n_nodes = 0
        #degrees of freedom for sampling variance parameters

        
    def generate_tree(self,p, dim = 50):
        """Generate a tree where each tree is a leaf node with probability p"""
        
        cov_mat = make_spd_matrix(dim)
        root = Node(cov_mat = cov_mat, index = 0)
        #make the scale of inverse chi squared half the size of the lowest 
        #eigenvalue
        eVals = np.linalg.eigvals(cov_mat)
        self.scale = eVals[-1] / 2
        self.df = 10
        self.root = root
        self.n_nodes = 1
        self.generate_from_node(root,p)
        
        
        
    def generate_from_node(self,root,p):
        """Recursive function for generating a tree"""
        #sample over Uniform(0,1)
        u = random.random()
        #number of children is 1 with probability p
        tmp = (u,p, u > p)
        print(tmp)
        n_child = 1 + int(u > p)
        #if one child, then make that child an observed node with no children
        if n_child == 1:
            child = Node(parent = root, observed = True, index = self.n_nodes)
            self.n_nodes += 1
            #randomly sample a number of data elements
            n = random.randint(10,100)
            root.add_child(child,n)
        #if two children, then build tree recursively 
        else:
            child1 = Node(parent = root, index = self.n_nodes)
            self.n_nodes += 1
            child2 = Node(parent = root, index = self.n_nodes)
            self.n_nodes += 1
            #randomly sample variances from scaled inverse chi squared (df,scale)
            d1 = self.scale * self.df * 1 / np.random.chisquare(self.df)
            d2 = self.scale * self.df * 1 / np.random.chisquare(self.df)
            print(d1)
            root.add_children([(child1,d1), (child2,d2)])
            #recursively do the same with children
            self.generate_from_node(child1,p)
            self.generate_from_node(child2,p)
            
    def bfs_string(self):
        """Breadth First Search to convert the tree to a string"""
        q1 = [self.root]
        q2 = []
        string = ""
        finished = False
        
        while not finished:
            node = q1.pop()
            string += str(node) + " "
            #add children to lower level queue
            for child,weight in node.children:
                q2.insert(0,child)
            #if upper level queue is empty, add a new line character and 
            #switch to lower leverl queue
            if len(q1) == 0:
                if len(q2) != 0:
                    string += "\n"
                    q1 = q2
                    q2 = []
                else:
                    finished = True
        return(string)
            
    def sample(self):
        """samples recursively, returns data"""
        self.root.sample()
        return(self.get_data())
    
    def get_data(self):
        """Uses breadth first search to find observed nodes and get data"""
        q1 = [self.root]
        q2 = []
        finished = False
        data = []
        while not finished:
            node = q1.pop()
            if node.observed == True:
                data.append(node.data)
            #add children to lower level queue
            for child,weight in node.children:
                q2.insert(0,child)
            #if upper level queue is empty, switch to lower level queue
            if len(q1) == 0:
                if len(q2) != 0:
                    q1 = q2
                    q2 = []
                else:
                    finished = True
        return(data)        
        

    
    def __repr__(self):
        return(self.__str__())
        
    def __str__(self):
        return(self.bfs_string())
            
        
    
            