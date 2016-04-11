# -*- coding: utf-8 -*-
"""
Phylogeny Tree class
"""
import numpy as np
import scipy as sp
import random 
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from unionfind import UnionFind

class Node():
    
    def __init__(self,cov_mat =  None, children = None, parent = None, 
                 parent_weight = None, data = None, observed = False, 
                 index = None):
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
        self.parent_weight = parent_weight
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
    
    def set_parent(self,parent,weight):
        """Sets the parent and parent_weight"""
        self.parent = parent
        self.parent_weight = weight
    
    def add_child(self,child,weight):
        """Adds a child with weight to children"""
        self.children.append((child,weight))
    
    def add_children(self,children):
        """Adds children from a list of (child,weight) tuples"""
        for child,weight in children:
            self.add_child(child,weight)

        
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

            #randomly sample variances from scaled inverse chi squared (df,scale)

            d1 = self.scale * self.df * 1 / np.random.chisquare(self.df)
            d2 = self.scale * self.df * 1 / np.random.chisquare(self.df)
            child1 = Node(parent = root, parent_weight = d1, index = self.n_nodes)
            self.n_nodes += 1
            child2 = Node(parent = root, parent_weight = d2, index = self.n_nodes)
            self.n_nodes += 1
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
        i = 0
        while diff > tol:
            print(i)
            i += 1
            #fit the ancestors based on given branch lengths and topologies
            self.fit_ancestors()
            #fit the branch_lengths between all internal nodes and all 
            #internal nodes to leaf nodes
            b_lengths = self.fit_branch_lengths()
            #make a minimum spanning tree
            self.mst(b_lengths)
            #go from mst to bifurcating tree
            self.to_bifurcating_tree()
            
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
        observed_nodes = []
        for i,node in enumerate(nodes):
            #compute sample covaraince
            S = np.cov(X[i].T)
            obs_node = Node(data = X[i], observed = True, parent = node, 
                            index = node.index, cov_mat = S)
            node.add_child(obs_node, X[i].shape[0])
            observed_nodes.append(obs_node)
        self.observed_nodes = observed_nodes
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
            #add new_node to cluster roots
            cluster_roots.append(new_node)
            
            #add new_node to set of nodes
            nodes.append(new_node)
            #assign parent to the nodes
            i = nodes.index(k)
            nodes[i].set_parent(new_node,d1)
            j = nodes.index(m)
            nodes[j].set_parent(new_node,d2)
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
        
        #assign parent to the nodes
        i = nodes.index(k)
        nodes[i].set_parent(new_node, d1)
        j = nodes.index(m)
        nodes[j].set_parent(new_node, d2)
        
        #save the nodes and the root
        self.nodes = nodes
        #reorder the nodes so the root is 0
        for i,node in enumerate(nodes[::-1]):
            node.index = i
        self.root = new_node
        
        self.normalize_branch_lengths()
        
    def normalize_branch_lengths(self):
        """Find min and max branch lengths with bfs and normalize all lengths 
            so there are no negative branch lengths"""
        q1 = [self.root]
        q2 = []
        finished = False
        min_d = float('inf')
        max_d = float('-inf')
        #pass through once to find max and min
        while not finished:
            node = q1.pop()
            if len(node.children) != 1:
                #add children to lower level queue and chekc branch lengths
                for child,weight in node.children:
                    q2.insert(0,child)
                    if weight > max_d:
                        max_d = weight
                    if weight < min_d:
                        min_d = weight
            #if upper level queue is empty, switch to lower level queue
            if len(q1) == 0:
                if len(q2) != 0:
                    q1 = q2
                    q2 = []
                else:
                    finished = True  
         #pass through a second time to reweight the branch lengths to be postive
        #decrease the minimum a bit to avoid zero length trees 
        min_d -= .1 * (max_d - min_d)
        q1 = [self.root]
        q2 = []
        finished = False
        #pass through once to find max and min
        while not finished:
            node = q1.pop()
            if len(node.children) != 1:
                #add children to lower level queue and change branch lengths
                for i in range(len(node.children)):
                    child = node.children[i][0]
                    #update weight
                    weight = node.children[i][1]
                    weight = (weight - min_d) / (max_d - min_d)
                    node.children[i] = (child,weight)
                    q2.insert(0,child)
            #if upper level queue is empty, switch to lower level queue
            if len(q1) == 0:
                if len(q2) != 0:
                    q1 = q2
                    q2 = []
                else:
                    finished = True         
        
        
            
    def fit_ancestors(self):
        """Find max likelihood ancestors, solves a system of linear equations"""
        #formula A * nodes = y
        #matrix A for inversion
        A = np.zeros((self.n_nodes,self.n_nodes))
        #fill in A
        for node in self.nodes:
            row = node.index          
            #internal node
            if len(node.children) > 1:
                #root node
                if node.parent == None :
                    c1 = node.children[0][0]
                    d1 = node.children[0][1]
                    c2 = node.children[1][0]
                    d2 = node.children[1][1]
                    c1_ind = c1.index
                    c2_ind = c2.index                
                    A[row,c1_ind] = 1 / d1
                    A[row,c2_ind] = 1 / d2
                    A[row,row] = -(1 / d1 + 1 / d2)
                    continue
                c1 = node.children[0][0]
                d1 = node.children[0][1]
                c2 = node.children[1][0]
                d2 = node.children[1][1]
                p = node.parent
                dp = node.parent_weight
                c1_ind = c1.index
                c2_ind = c2.index
                p_ind = p.index
                A[row,c1_ind] = 1 / d1
                A[row, c2_ind] = 1 / d2
                A[row, p_ind] = 1 / dp
                A[row,row] = -(1/d1 + 1/d2 + 1/dp)
            #leaf node
            else:
                #number of samples
                n = node.children[0][1]
                p = node.parent
                dp = node.parent_weight
                p_ind = p.index
                A[row,p_ind] = n / 2 * (1 / dp)
                A[row,row] = -n / 2 * (1 / dp + n / 2)
        #invert matrix
        A_inv = np.linalg.inv(A)
        #get covariance estimates
        for k in range(len(self.nodes)):
            i  = self.nodes[k].index
            dim = self.observed_nodes[0].data.shape[1]
            cov_est = np.zeros((dim,dim))
            #loop over the observed nodes and add with weights
            for obs_node in self.observed_nodes:
                j= obs_node.index
                cov_est += A_inv[i,j] * obs_node.cov_mat
            #update the estimate of the covariance matrix
            self.nodes[k].cov_mat = cov_est
            
    def fit_branch_lengths(self):
        """Fits the branch lengths given the covariance matrices 
        Output:
            b_lengths: dictionary with keys (node,node) and values branch length
        """
        #collect leaves and ancestors
        self.collect_leaves()
        leaves = set(self.leaves)
        nodes = set(self.nodes)
        ancestors = nodes.difference(leaves)
        
        #estimate branch lengths
        b_lengths = {}
        for ancestor in ancestors:
            for node in nodes:
                #estimate variance
                ca = ancestor.cov_mat
                cn = node.cov_mat
                var = np.power(cn - ca,2).mean()
                #add the varaince as the estimated branch length
                b_lengths[(ancestor,node)] = var
        return(b_lengths)
        
    def mst(self,b_lengths):
        """Kruskal's algorithm for minimum spanning tree 
        Input:
            b_lengths: dictionary with keys (node,node) and values branch length
        """
        #go through all the nodes and get rid of parent/child relationships
        #so we can build the tree again
        #except keep leaf nodes and observed data
        for i in range(len(self.nodes)):
            if len(self.nodes[i].children) > 1:
                self.nodes[i].parent = None
                self.nodes[i].parent_weight = None
                self.nodes[i].children = []
            else:
                self.nodes[i].parent = None
                self.nodes[i].parent_weight = None                
            
        #unionfind object
        d_set = UnionFind()
        #sort the edges into non-decreasing order
        edges = [(edge,weight) for edge,weight in 
                    sorted(b_lengths.items(),key = lambda x: x[1])]
        for edge,weight in edges:
            u,v = edge
            if d_set[u] != d_set[v]:
                d_set.union(u,v)
                #add parents and children. u is v's parent
                u_idx = self.nodes.index(u)
                v_idx = self.nodes.index(v)
                self.nodes[u_idx].add_child(v,weight)
                self.nodes[v_idx].set_parent(u,weight)
            
    
    def to_bifurcating_tree(self):
        """Use Propositions 5.3 and 5.4 in ALGORITHM FOR PHYLOGENETIC 
            INTERFERENCE (Friedman, et. al.) to go from a MST to a bifurcating
            tree of (approximately) equal likelihood. Updates nodes with bfs"""
        #keep track of number of unused nodes
        self.n_unused = 0
        q1 = [self.root]
        q2 = []
        finished = False
        #keep a list of available indices
        idxs = []
        while not finished:
            node = q1.pop()
            #if the degree is 1 (Proposition 5.3)
            if len(node.children) == 0:
                #remove the node
                node.parent.children.remove((node,node.parent_weight))
                self.nodes.remove(node)
                idxs.append(node.index)
            #if the degree is 2 (Proposition 5.3)
            elif len(node.children) == 1:
                #either the node is a parent of an observed node, in which case
                #do nothing
                #otherwise remove the node
                if not node.children[0][0].observed:
                    #remove this node
                    node.parent.children.remove((node,node.parent_weight))
                    #add a branch from the child node to the parent node
                    node.parent.add_child(node.children[0][0],
                                          node.children[0][1] + 
                                          node.parent_weight)
                    #add a branch from the child to the new parent
                    node.children[0][0].parent = node.parent
                    node.children[0][0].parent_weight = node.children[0][1] + node.parent_weight
                    #remove the node from the list of nodes
                    self.nodes.remove(node)
                    idxs.append(node.index)
                    #add the child node to the lower level queue
                    q2.insert(0,node.children[0][0])
                    #if the node was the root, make the child the root
                    if self.root == node:
                        self.root = node.children[0][0]
            #if the degree is greater than 3 (Proposition 5.4)
            elif len(node.children) > 2:
                #find the two child nodes which are closest together
                n1 = None
                n2 = None
                min_d = float('inf')
                #all neighbors including parent
                neighbors = node.children
                neighbors.append((node.parent,node.parent_weight))
                for node_1,weight_1 in neighbors:
                    for node_2, weight_2 in neighbors:
                        if node_1 != node_2:
                            dist = np.power(node_1.cov_mat - node_2.cov_mat,2) 
                            if dist < min_d:
                                n1 = node_1
                                n2 = node_2
                                min_d = dist
                #create a new node
                if len(idxs) > 0:
                    index = idxs.pop()
                else:
                    index = self.n_nodes 
                new_node = Node(index = index)
                #place the new node a little differently if one of the closest
                #nodes is a parent
                if n1 == node.parent or n2 == node.parent:
                    #assign node to be new_node's parent, use a small weight
                    new_node.set_parent(node,.0001)
                    #give the rest of the children to new_node
                    for child,weight in node.children:
                        if child != n1 and child != n2:
                            new_node.add_child(child,weight)
                    #assign new node as a child
                    node.add_child(new_node,.0001) 
                    #add children to lower level queue
                    for child,weight in node.children:
                        q2.insert(0,child)
                else:
                    #assign node's parent to be new_node's parent
                    new_node.set_parent(node.parent,node.parent_weight)
                    #assign new_node to be node's parent
                    node.set_parent(new_node,.0001)
                    #if node is the root then made new_node the root
                    if self.root == node:
                        self.root == new_node
                    #add the rest of the children to be new_node's children
                    for child,weight in node.children:
                        if child != n1 and child != n2:
                            new_node.add_child(child,weight)
                    #add new_node to current queue
                    q1.insert(0,new_node)
                #add new_node to list of nodes
                self.nodes.append(new_node)
            #if the node has two children, just add the children to the lower
            #queue and continue
            else:
                for child,weight in node.children:
                    q2.insert(0,child)
            #if upper level queue is empty, switch to lower level queue
            if len(q1) == 0:
                if len(q2) != 0:
                    q1 = q2
                    q2 = []
                else:
                    finished = True         
             
        
            