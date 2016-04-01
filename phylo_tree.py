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
                #sample upper triangle of covariance matrix
                sample = np.zeros((m,m))
                for i in range(m):
                    for j in range(i,m):
                        #sample from normal distribution around parent value
                        sample[i,j] = np.random.normal(self.cov_mat[i,j],
                                                        weight,1)
                #go from upper triangular to symmetric
                s = sample + sample.T
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
        if not self.observed:
            if self.index != None:
                string = "Node_" + str(self.index)
            else:
                string = "Node"
        else:
            if self.parent.index != None:
                string = "Node_" + str(self.parent.index) + "_O"
            else:
                string = "Node_O"
        return(string)

class PhyloTreeSample():
    """Class for sampling data from a phylogeny tree"""
    
    def __init__(self):
        """Empty construtor"""
        #root of tree
        self.root = None
        #number of nodes
        self.n_nodes = 0
        #degrees of freedom for sampling variance parameters
        self.leaves = None

        
    def generate_tree(self,p, dim = 50):
        """Generate a tree where each tree is a leaf node with probability p"""
        
        cov_mat = make_spd_matrix(dim)
        root = Node(cov_mat = cov_mat, index = 0)
        #make the scale of inverse chi squared half the size of the lowest 
        #eigenvalue
        eVals = np.linalg.eigvals(cov_mat)
        self.scale = eVals[-1] / 10
        self.df = 4
        self.root = root
        self.n_nodes = 1
        self.generate_from_node(root,p)
        
        
        
    def generate_from_node(self,root,p):
        """Recursive function for generating a tree"""
        #sample over Uniform(0,1)
        u = random.random()
        #number of children is 1 with probability p
        n_child = 1 + int(u > p)
        #if one child, then make that child an observed node with no children
        if n_child == 1:
            child = Node(parent = root, observed = True, index = root.index)
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
        
    def collect_leaves(self):
        """Collect all leaf nodes (above observed nodes) with BFS"""
        q1 = [self.root]
        q2 = []
        finished = False
        self.leaves = []
        while not finished:
            node = q1.pop()
            if len(node.children) == 1:
                self.leaves.append(node)
            else:
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
        
        
    def get_data(self):
        """Get data from leaves"""
        
        #collect leaves if haven't already
        if self.leaves == None:
            self.collect_leaves()
        data = []
        for leaf in self.leaves:
            data.append(leaf.children[0][0].data)
        
        return(data)
        
    def __repr__(self):
        return(self.__str__())
        
    def __str__(self):
        return(self.bfs_string())
            
        
class PhyloTreeFit(PhyloTreeSample):
    """Class for fitting a phylogeny tree from data"""
    
    def __init__(self):
        """Null constructor"""
        self.nodes = []
        self.weights = []
        self.n_nodes = 0

    def fit(self,X):
        """Uses structural EM to fit a phylogeny tree to data
        Input:
            X:list of length n_tissues of matrices of shape n_samples x n_genes
        """
        #first initialize the topology by using neighbor joining
        self.neighbor_join(X)
        #do EM until convergence 
        tol = .0001
        diff = 1
        while diff > tol:
            #fit the ancestors based on given branch lengths and topologies
            self.fit_ancestors()
            #fit the branch_lengths between all internal nodes and all 
            #internal nodes to leaf nodes
            self.fit_branch_lengths()
            #make a minimum spanning tree
            self.mst()
            
    def neighbor_join(self, X):
        """Uses neighbor joining algorithm to make a phylogenic tree used 
            as initialization
        Input:
            X: n_samples x n_genes x n_tissues data tensor
        """
        #use average vectors for each tissue to calulcaute distances
        avgs = [mat.mean(0) for mat in X]
        #D = spatial.distance.squareform(spatial.distance.pdist(X.mean(0).T)
        #initialize n_tissues clusters
        n_tissues  = len(X)
        
        nodes = [Node(index = i) for i in range(n_tissues)]
        #add observed nodes below the nodes
        for i,node in enumerate(nodes):
            obs_node = Node(data = X[i], observed = True, parent = node)
            node.add_child(obs_node, X[i].shape[0])
            
        self.n_nodes = len(nodes)
        #keep track of roots of each cluster
        cluster_roots = [node for node in nodes]
        #make a disctionary where pairs of nodes are keys and their distances 
        #are the values
        d = dict()
        for node1 in nodes:
            for node2 in nodes:
                d[(node1,node2)] = np.linalg.norm(avgs[node1.index] - 
                                                        avgs[node2.index])
        r = dict()
        count = 0
        while len(cluster_roots) > 2:
            #calculate r[k] for each cluster k
            for cluster1 in cluster_roots:
                val = 0
                for cluster2 in cluster_roots:
                    if cluster2 != cluster1:
                        val += d[(cluster2,cluster1)]
                r[cluster1] = val / (len(cluster_roots) - 2)
            #find (k,m) minimizing d[(k,n)] - r[k] - r[m]
            k_min = None
            m_min = None
            min_val = float("inf")
            for k in cluster_roots:
                for m in cluster_roots:
                    if k != m:
                        tmp_val = d[(k,m)] - r[k] - r[m]
                        if tmp_val < min_val:
                            k_min = k
                            m_min = m
                            min_val = tmp_val
            #define a new node which is the parent of the minimizers
            new_node = Node(index = self.n_nodes)
            self.n_nodes += 1
            #remove k and m
            k = k_min
            m = m_min

            cluster_roots.remove(k)
            cluster_roots.remove(m)

            for s in cluster_roots:
                d[(new_node,s)] = .5 * (d[(k,s)] + d[(m,s)] - d[(k,m)])
                d[(s,new_node)] = d[(new_node,s)]
            #set distance to self to zero
            d[(new_node,new_node)] = 0
            #join nodes k and m to new_node
            d1 = .5 * (d[(k,m)] - r[k] - r[m])
            new_node.add_child(k,d1)
            d2 = .5 * (d[(k,m)] + r[m] - r[k])
            new_node.add_child(m,d2)
            k.parent = new_node
            m.parent = new_node
            #add new_node to cluster roots
            cluster_roots.append(new_node)
            
            #add new_node to set of nodes
            nodes.append(new_node)
            count += 1
        #combine last two clusters
        k = cluster_roots[0]
        m = cluster_roots[1]
        new_node = Node(index = self.n_nodes)
        self.n_nodes += 1
        d1 = .5 * d[(k,m)]
        new_node.add_child(k,d1)
        d2 = .5 * d[(k,m)]
        new_node.add_child(m,d2)
        nodes.append(new_node)
        #save the nodes and the root
        self.nodes = nodes
        #reorder the nodes so the root is 0
        for i,node in enumerate(nodes[::-1]):
            node.index = i
        self.root = new_node
        
            
            
            