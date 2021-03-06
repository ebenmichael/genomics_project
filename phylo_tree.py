# -*- coding: utf-8 -*-
"""
Phylogeny Tree class
"""
import numpy as np
import scipy as sp
import random 
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from unionfind import UnionFind
from collections import Counter, defaultdict
import pprint as pp

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
                        p_ij = self.cov_mat[i,j] #parent value
                        #sample from normal distribution around parent value
                        sample[i,j] = np.random.normal(p_ij,
                                                        weight**2 * p_ij**2
                                                        ,1)
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
        #max number of nodes
        self._max_nodes = 50

        
    def generate_tree(self,p, dim = 50):
        """Generate a tree where each tree is a leaf node with probability p"""
        
        cov_mat = make_spd_matrix(dim)
        root = Node(cov_mat = cov_mat, index = 0)
        #make the scale of inverse chi squared half the size of the lowest 
        #eigenvalue
        eVals = np.linalg.eigvals(cov_mat)
        self.scale = eVals[-1] / 100
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
        #or if there are too many nodes
        #if the node is the root, or its parent is a root, guarantee that it is
        #not a child node
        if (n_child == 1 or self.n_nodes > self._max_nodes) and \
                           not (root == self.root or root.parent == self.root):
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

    def collect_nodes(self):
        """Collect all nodes with BFS"""
        q1 = [self.root]
        q2 = []
        finished = False
        self.nodes = []
        while not finished:
            node = q1.pop()
            self.nodes.append(node)
            if len(node.children) > 1:
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

        
    def collect_leaves(self):
        """Collect all leaf nodes by collecting all nodes"""

        self.collect_nodes()
        self.leaves = [node for node in self.nodes if len(node.children) == 1]
        
        
    def get_data(self):
        """Get data from leaves"""
        
        #collect leaves if haven't already
        if self.leaves == None:
            self.collect_leaves()
        data = []
        for leaf in self.leaves:
            data.append(leaf.children[0][0].data)
        
        return(data)
    
    def to_adj_mat(self):
        """Returns the tree as an adjacency matrix"""
        #collect nodes if haven't already
        if self.nodes == None:
            self.collect_nodes()
        n = self.n_nodes
        adj_matrix = np.zeros((n,n))
        for node in self.nodes:
            #add weights for children
            i = node.index
            child_array = node.children #(node, float) list
            for child in child_array:
                j = child[0].index
                adj_matrix[i][j] = child[1]
            #add weight for parent if not the root
            if node != self.root:
                j = node.parent.index
                adj_matrix[i][j] = node.parent_weight 
        
        return(adj_matrix)
    
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

    
    #scoring
    def score_tree(self,other):
        """Uses eigenvalues to give a difference scoring between 0 and inf"""

        ##adjacency matrics
        #self/fitted tree
        adj_matrix = (self.to_adj_mat() != 0) * 1
        #original tree
        adj_matrix_original = (other.to_adj_mat() != 0)*1


        ##diagonal degree matrics (sum of column)
        #self/fitted tree
        deg_matrix = adj_matrix.sum(0)
        #original tree
        deg_matrix_original = adj_matrix_original.sum(0)

        #find laplacian matrix of each graph (l = d-a)
        laplacian_fitted = np.subtract(deg_matrix, adj_matrix)
        laplacian_original = np.subtract(deg_matrix_original,
                                         adj_matrix_original)

        #find eigenvalues of laplacian matrices
        eigvals_fitted = np.sort(np.linalg.eigvals(laplacian_fitted))
        eigvals_original = np.sort(np.linalg.eigvals(laplacian_original))

        #calculate similarity score
        #if there is one less node in self then only consider the ones in self
        if len(eigvals_fitted) == len(eigvals_original) - 1:
            sims = (np.subtract(eigvals_fitted, eigvals_original[:-1]))**2
        else:
            sims = (np.subtract(eigvals_fitted, eigvals_original))**2
        sim_score = np.sum(sims)

        return(sim_score)


    

    def score_leaves(self, other):
        '''compare similarity of leaves of trees'
            Input:
                other: real tree
            Output:
                (fitted tree score, sample covariance score)
        '''
        #collect leaves if they haven't been collected yet
        self.collect_leaves()
        other.collect_leaves()
            
        def covariance_similarity(matrix1, matrix2):
            '''outputs a similarity score for two square leaf matrices'''
            diff = np.subtract(matrix1, matrix2)
            #can use matrix_power here b/c using square matrices
            diff_sq = np.power(diff,2)
            mean = diff_sq.mean()
            return(mean)
            
            
        error = {} #error between fitted cov and real cov
        sample_error = {} #error between sample covariance and real cov
        fit_to_real = {} #maps fitted nodes to their real counterparts
        #find list of leaf nodes per tree
        nodes1 = self.leaves
        nodes2 = other.leaves
        #find corresponding nodes between trees
        for i in nodes1:
            #obersved node
            i_data = i.children[0][0].data
            for j in nodes2:
                j_data = j.children[0][0].data
                #check if corresponding leaves
                if np.all(i_data == j_data):
                    fit_to_real[i] = j
            #calculate average similarity
            #for each node's covariance matrix
            sim = covariance_similarity(i.cov_mat, j.cov_mat)
            #store similarity score as value to start leaf node's index
            error[i] = sim
            sample_cov = np.cov(i_data.T)
            sample_error[i] = covariance_similarity(j.cov_mat,sample_cov)
                    
        scores1 = [v for v in error.values()]
        mean1 = np.mean(scores1)
    
        scores2 = [v for v in sample_error.values()]
        mean2 = np.mean(scores2)
        #return the average similarity between covariance matrices
        #return similarity #would return dictionary = similiarity per node pair
        return(mean1, mean2)

        

    def fit(self,X):
        """Uses structural EM to fit a phylogeny tree to data
        Input:
            X:list of length n_tissues of matrices of shape n_samples x n_genes
        """
        #first initialize the topology by using neighbor joining
        self.neighbor_join(X)
        self.collect_leaves()
        #do EM until convergence 
        tol = .01
        diff = 1
        i = 0
        #store previous covarainces
        prev_covs = {}
        while diff > tol and i < 50:
            
            for node in self.leaves:
                prev_covs[node] = np.copy(node.cov_mat)
            
            #fit the ancestors based on given branch lengths and topologies
            self.fit_ancestors()
            #fit the branch_lengths between all internal nodes and all 
            #internal nodes to leaf nodes
            b_lengths, weights = self.fit_branch_lengths()
            

            #make a minimum spanning tree
            adj_mat = self.mst(b_lengths, weights)
            #go from mst to bifurcating tree
            adj_mat = self.to_bifurcating_tree(adj_mat)
            #calulcate likelihood
            added = defaultdict(bool)
            log_lik = 0
            for edge,weight in weights.items():
                if not added[edge] or not added[edge[::-1]]:
                    log_lik -= weight
                    added[edge] = True
                    added[edge[::-1]] = True
            #make a directed phylogeny from adj_mat
            self.to_directed_phylogeny(adj_mat)
            #calculate differences between all covariance matrices
            if i != 0:
                diff = 0
                
                for node in self.leaves:
                    c_prev = prev_covs[node]
                    c_pres = node.cov_mat
                    #print(c_prev)
                    #print(c_pres)
                    diff += np.sum(np.power(c_pres - c_prev,2))
                
                
                #diff = np.abs(log_lik - prev_log_lik)
                #prev_log_lik = log_lik
            #else:
                #prev_log_lik = log_lik
            i += 1
            print(i,diff)
            
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
        #make a dictionary where pairs of nodes are keys and their distances 
        #are the values
        d = {}
        for node1 in nodes:
            for node2 in nodes:
                d[(node1,node2)] = np.linalg.norm(avgs[node1.index] - 
                                                        avgs[node2.index])
        r = {}
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
        self.root = new_node
        #normalize the branch lengths if more than 2 branchs
        if len(self.nodes) > 3:
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
        #pass through to normalize the lengths
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
                    child.parent = node
                    child.parent_weight = weight
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
                c1 = node.children[0][0]
                d1 = node.children[0][1]
                c2 = node.children[1][0]
                d2 = node.children[1][1]
                p = node.parent
                dp = node.parent_weight
                c1_ind = c1.index
                c2_ind = c2.index
                if not p is None:
                    p_ind = p.index
                    A[row,c1_ind] = 1 / d1
                    A[row, c2_ind] = 1 / d2
                    A[row, p_ind] = 1 / dp
                    A[row,row] = -(1/d1 + 1/d2 + 1/dp)
                else:
                    if len(node.children) == 2:
                        A[row,c1_ind] = 1 / d1
                        A[row, c2_ind] = 1 / d2
                        A[row,row] = -(1/d1 + 1/d2)   
                    else:
                        c3 = node.children[2][0]
                        d3 = node.children[2][1]
                        c3_ind = c3.index
                        A[row,c1_ind] = 1 / d1
                        A[row, c2_ind] = 1 / d2
                        A[row, c3_ind] = 1 / d3
                        A[row,row] = -(1/d1 + 1/d2 + 1/d3)                           
            #leaf node
            else:
                #number of samples
                n = node.children[0][1]
                p = node.parent
                dp = node.parent_weight
                p_ind = p.index
                A[row,p_ind] = 2/n * (1 / dp)
                A[row,row] = -(2 / (n *dp) + 1)
        #invert matrix
        A = -A
        A_inv = np.linalg.pinv(A)
        #print(A)
        #print(A_inv)
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
        
        return(A,A_inv)
            
    def fit_branch_lengths(self):
        """Fits the branch lengths given the covariance matrices 
        Output:
            b_lengths: dictionary with keys (node,node) and values branch length
            weights: dictionary maps (node,node) to negative log likelihood
        """
        #collect leaves and ancestors
        self.collect_leaves()
        leaves = set(self.leaves)
        nodes = set(self.nodes)
        ancestors = nodes.difference(leaves)
        #estimate branch lengths
        b_lengths = {}
        weights = {}
        for ancestor in ancestors:
            for node in nodes:
                if node != ancestor and not (ancestor,node) in weights:
                    #estimate efficiency (var / mean^2)
                    ca = ancestor.cov_mat
                    cn = node.cov_mat
                    n = ca.shape[0]
                    
                    v = np.power(cn - ca,2)
                    
                    eff = v / np.power(ca,2)
                    eff = np.sqrt(eff.mean())
                    #add the efficiency as the estimated branch length
                    b_lengths[(ancestor,node)] = eff
                    #add negative log likelihood
                    weights[(ancestor,node)] = (n**2 / 2) * np.log(2 * np.pi) + \
                                                (n**2/2) *  np.log(v).sum() + \
                                                n**2 / 2
                    
                    """
                    
                    v = v.mean()
                    b_lengths[(ancestor,node)] = v
                    weights[(ancestor,node)] = (n**2 / 2) * np.log(2 * np.pi) + \
                                                (n**2/2) *  np.log(v) + \
                                                n**2 / 2
                    """
        return(b_lengths,weights)
        
    def mst(self,b_lengths, weights):
        """Kruskal's algorithm for minimum spanning tree 
        Input:
            b_lengths: dictionary with keys (node,node) and values branch length
            weights: dictionary maps (node,node) to negative log likelihood
        Output:
            adj_mat: adjacency matrix. dict (node,node) keys weight values
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
                    sorted(weights.items(),key = lambda x: x[1])]
        #keep track of new graph as an adjacency matrix
        adj_mat = {}
        for edge,weight in edges:
            u,v = edge
            if d_set[u] != d_set[v]:
                d_set.union(u,v)
                #update adjacency matrix
                adj_mat[(u,v)] = b_lengths[(u,v)]
                adj_mat[(v,u)] = b_lengths[(u,v)]
        
        return(adj_mat)
            
    def to_bifurcating_tree(self,adj_mat):
        """Use Propositions 5.3 and 5.4 in ALGORITHM FOR PHYLOGENETIC 
            INTERFERENCE (Friedman, et. al.) to go from a MST to a bifurcating
            tree of (approximately) equal likelihood. 
        Input:
            adj_mat: mst as an adjacency matrix. 
                        dict (node,node) keys weight values
        Output:
            adj_mat: phylogenetic tree as adjacency matrix. 
                        dict (node,node) keys weight values
                        
        """        
        #get number of neighbors for each node
        degree = Counter(edge[0] for edge in adj_mat)
        #go through all the nodes
        stack = list(self.nodes)        
        while len(stack) > 0:
            #print(len(self.nodes))
            #print(stack)
            node = stack.pop()

            #print(node,degree[node],len(node.children))
            #go from adjacency matrix to egde list
            adj_list = defaultdict(list)
            for edge in adj_mat:
                adj_list[edge[0]].append(edge[1])
            #pp.pprint(adj_list)    
            #print(degree)
            #if the node has degree 1 and is not the parent of an observed node
            #remove the node (Proposition 5.3)
            if degree[node] == 1 and len(node.children) == 0:
                #print("REMOVED")
                #print(node,degree[node],len(node.children))
                #remove the node from the list of nodes if the node is there
                if node in self.nodes:
                    self.nodes.remove(node)
                #remove node from degree dictionary
                degree.pop(node,None)
                #remove the node from the stack if it is in the stack again
                if node in stack:
                    stack.remove(node)
                #remove the node from the adjecencey matrix and decrease degree
                #of neighbor
                tmp = {}
                for edge, weight in adj_mat.items():
                    if not node in edge:
                        tmp[edge] = weight
                    else:
                        if node == edge[0]:
                            degree[edge[1]] -= 1
                            #add the node which just lost a neighbor back into 
                            #the stack
                            stack.append(edge[1])
                adj_mat = {edge:weight for edge,weight in adj_mat.items() \
                            if not node in edge}

            #if the node has degree 2 and is not the parent of an observed node
            #remove it (Proposition 5.3)
            elif degree[node] == 2 and len(node.children) == 0:
                #print("REMOVED")
                #print(node,degree[node],len(node.children))
                #remove the node from the list of nodes if the node is there
                if node in self.nodes:
                    self.nodes.remove(node)
                #remove node from degree dictionary
                degree.pop(node,None)
                #remove the node from the stack if it is in the stack again
                if node in stack:
                    stack.remove(node)
                #combine the edges (u,node) and (node,v) and remove node from 
                #adj matrix
                tmp = {}
                new_edge = []
                for edge,weight in adj_mat.items():
                    if node in edge:
                        if edge[0] == node:
                            u = edge[1]
                            if not u in new_edge:
                                new_edge.append(u)
                    else:
                        tmp[edge] = weight
                #to tuple so can hash
                new_edge = tuple(new_edge)
                u,v = new_edge
                tmp[new_edge] = adj_mat[(u,node)] + adj_mat[(node,v)]
                tmp[new_edge[::-1]] = tmp[new_edge]
                adj_mat = tmp
            #if  degree is 2 but the node is the parent of an observed node
            #add a new node (Proposition 5.4)
            elif (degree[node] == 2 or degree[node] == 3) and len(node.children) != 0:
                #create a new node
                idx = self.n_nodes
                new_node = Node(index = idx)
                new_node.cov_mat = np.copy(node.cov_mat)
                #send node's neighbors to new_node   
                neighbors = [edge[1] for edge in adj_mat if edge[0] == node]
                for n in neighbors:
                    #add edge (n,new_node) and (new_node,n)
                    adj_mat[(n, new_node)] = adj_mat[(n,node)]
                    adj_mat[(new_node, n)] = adj_mat[(node, n)]
                    #remove the edges (node,n) and (n,node) from adj mat
                    del adj_mat[(n,node)]
                    del adj_mat[(node, n)]                    
                #add an edge between node and new_node
                adj_mat[(node, new_node)] = .0001
                adj_mat[(new_node, node)] = .0001                
                #add new_node and all neighbors not already in the stack
                #into the stack
                stack.append(new_node)
                for n in neighbors:
                    if not n in stack:
                        stack.append(n)
                #add new_node to the list of nodes
                self.nodes.append(new_node)
                #change the degrees
                degree[new_node] = degree[node] + 1
                degree[node] = 1
                #print("NEW NODE: " + str(new_node))
            #if the degree is greater than 3 (Proposition 5.4)
            elif degree[node] > 3:
                #find the two neighbors which are closest together and group them
                neighbors = [edge[1] for edge in adj_mat if edge[0] == node]
                closest = (None,None)
                d_min = float('inf')
                for n1 in neighbors:
                    for n2 in neighbors:
                        if n1 != n2:
                            dist = np.sum(np.power(n1.cov_mat - n2.cov_mat,2)) 
                            if dist < d_min:
                                closest = (n1,n2)
                                d_min = dist
                #create a new node
                idx = self.n_nodes
                new_node = Node(index = idx)
                new_node.cov_mat = np.copy(node.cov_mat)
                #update edges 
                
                for n in neighbors:
                    #keep the closest 2 neighbors with node, rest go to new_node
                    if n not in closest:
                        #add edge (n,new_node) and (new_node,n)
                        adj_mat[(n, new_node)] = adj_mat[(n,node)]
                        adj_mat[(new_node, n)] = adj_mat[(node, n)]
                        #remove the edges (node,n) and (n,node) from adj mat
                        del adj_mat[(n,node)]
                        del adj_mat[(node, n)]
                #add an edge between node and new_node
                adj_mat[(node, new_node)] = .0001
                adj_mat[(new_node, node)] = .0001
                #add new_node and all neighbors not already in the stack
                #into the stack
                stack.append(new_node)
                for n in neighbors:
                    if not n in stack:
                        stack.append(n)
                #add new_node to the list of nodes
                self.nodes.append(new_node)
                #change the degrees
                degree[new_node] = degree[node]  - 1
                degree[node] = 3
                #print("NEW NODE: " + str(new_node))
        #renumber the nodes
        curr_idx = len(self.leaves)
        for i,node in enumerate(self.nodes):
            if not node in self.leaves:
                node.index = curr_idx
                curr_idx += 1
        return(adj_mat)
        
        
    def to_directed_phylogeny(self, adj_mat):
        """Go from an undirected phylogeny to a directed phylogeny by 
            choosing a root and assigning parent/child relationships
        Input:
            adj_mat: phylogenetic tree as adjacency matrix. 
                        dict (node,node) keys weight values
        """

        #get number of neighbors for each node
        degree = Counter(edge[0] for edge in adj_mat)

        #find a node with degree 3
        idx = len(self.nodes) - 1
        while degree[self.nodes[idx]] != 3:
            idx -= 1
        #go from adjacency matrix to egde list
        adj_list = defaultdict(list)
        for edge in adj_mat:
            adj_list[edge[0]].append(edge[1])
        #pp.pprint(adj_list)
        #assign root as the degree 3 node chosen
        self.root = self.nodes[idx]
        q1 = [self.root]
        q2 = []
        checked = defaultdict(bool)
        
        root = self.root
        #Add 1 more node to turn bifurcating tree into binary tree
        new_node = Node(index = len(self.nodes))
        self.n_nodes +=1 
        #give new_node 2 of root's children
        for neighbor in adj_list[root][:-1]:
            adj_mat[(new_node,neighbor)] = adj_mat[(root,neighbor)]
            adj_mat[(neighbor,new_node)] = adj_mat[(root,neighbor)]
            del adj_mat[(root,neighbor)]
            del adj_mat[(neighbor,root)]
        #add edge from root to new_node
        adj_mat[(new_node,root)] = .00001
        adj_mat[(root,new_node)] = .00001
        
        degree[root] -= 1
        degree[new_node] = 3        
        
        #go from adjacency matrix to egde list
        adj_list = defaultdict(list)
        for edge in adj_mat:
            adj_list[edge[0]].append(edge[1])
            
        self.nodes.append(new_node)
        
        
        while len(q1) != 0:
            #print(q1)
            #print(q2)
            node = q1.pop()
            if checked[node]:
                continue
            if degree[node] == 1:
                """
                parent = adj_list[node][0]
                weight = adj_mat[(parent,node)]
                node.set_parent(parent,weight)
                parent.add_child(node,weight)
                checked[node] = True
                #add parent to upper level queue
                q2.insert(0,parent)
                """
            elif degree[node] >= 2:
                
                parent = node.parent
                """
                #if the node isn't the root node, assign a parent
                if not node == self.root:
                    for n in adj_list[node]:
                        #if the node is checked and it is not a leaf node
                        if checked[n] and not len(n.children) == 1:
                            
                            parent = n
                    weight = adj_mat[(parent,node)]
                    #print(node,parent)
                    node.set_parent(parent,weight)
                    parent.add_child(node,weight)
                """
                
                #add rest of the neighbors as children and into lower queue
                checked[node] = True
                for child in adj_list[node]:
                    if not child == parent:
                        weight = adj_mat[(node,child)]
                        node.add_child(child,weight)
                        child.set_parent(node,weight)
                        if not checked[child]:
                            q2.append(child)
                
            if len(q1) == 0:
                if len(q2) != 0:
                    q1 = q2
                    q2 = []
        self.n_nodes = len(self.nodes)