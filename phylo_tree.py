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
        self.collect_leaves()
        #do EM until convergence 
        tol = .0001
        diff = 1
        i = 0
        #store previous covarainces
        prev_covs = {}
        while diff > tol:
            #print("NEXT ITERATION")
            print(i,diff)
            
            for node in self.leaves:
                prev_covs[node] = np.copy(node.cov_mat)
            
            #fit the ancestors based on given branch lengths and topologies
            self.fit_ancestors()
            #fit the branch_lengths between all internal nodes and all 
            #internal nodes to leaf nodes
            b_lengths, weights = self.fit_branch_lengths()
            
            #calulcate likelihood
            added = defaultdict(bool)
            log_lik = 0
            for edge,weight in weights.items():
                if not added[edge] or not added[edge[::-1]]:
                    log_lik -= weight
                    added[edge] = True
                    added[edge[::-1]] = True
            print(log_lik)
            #make a minimum spanning tree
            adj_mat = self.mst(b_lengths, weights)
            #go from mst to bifurcating tree
            adj_mat = self.to_bifurcating_tree(adj_mat)
            #print(adj_mat)
            #make a directed phylogeny from adj_mat
            self.to_directed_phylogeny(adj_mat)
            #calculate differences between all covariance matrices
            if i != 0:
                diff = 0
                #print(prev_log_lik,log_lik)
                
                for node in self.leaves:
                    c_prev = prev_covs[node]
                    c_pres = node.cov_mat
                    #print(c_prev)
                    #print(c_pres)
                    diff += np.sum(np.power(c_pres - c_prev,2))
                
                #diff = log_lik - prev_log_lik
                #prev_log_lik = log_lik
            #else:
                #prev_log_lik = log_lik
            i += 1
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
                    A[row,c1_ind] = 1 / d1
                    A[row, c2_ind] = 1 / d2
                    A[row,row] = -(1/d1 + 1/d2)                    
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
        #print(A)
        A_inv = np.linalg.pinv(A)
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
                    #estimate variance
                    ca = ancestor.cov_mat
                    cn = node.cov_mat
                    n = ca.shape[0] ** 2
                    var = np.power(cn - ca,2).mean()
                    #add the varaince as the estimated branch length
                    b_lengths[(ancestor,node)] = var
                    #add negative log likelihood
                    weights[(ancestor,node)] = np.log(2 * np.pi) + 2 * np.log(var) \
                                                + n / 2
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
        #print(self.nodes)
        stack = list(self.nodes)
        #keep a list of available indices
        idxs = []        
        while len(stack) > 0:
            #print("AAAAAAAAAAAAAAA")
            #print(stack)
            node = stack.pop()
            #print(node,degree[node])
            #print(adj_mat)
            #if the node has degree 1 and is not the parent of an observed node
            #remove the node (Proposition 5.3)
            if degree[node] == 1 and len(node.children) == 0:
                #remove the node from list of nodes
                self.nodes.remove(node)
                idxs.append(node.index)
                #remove the node from the adjecencey matrix and decrease degree
                #of neighbor
                tmp = {}
                for edge, weight in adj_mat.items():
                    if not node in edge:
                        tmp[edge] = weight
                    else:
                        if node == edge[0]:
                            degree[edge[1]] -= 1
                adj_mat = {edge:weight for edge,weight in adj_mat.items() \
                            if not node in edge}
                                
            #if the node has degree 2 remove it
            elif degree[node] == 2:
                #remove the node from the list of nodes
                self.nodes.remove(node)
                idxs.append(node.index)
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
                #use an unused index if there is one
                if len(idxs) > 0:
                    idx = idxs.pop()
                else:
                    idx = self.n_nodes
                new_node = Node(index = idx)
                new_node.cov_mat = np.copy(node.cov_mat)
                #update edges 
                
                #print(closest)
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
                #print("-------------------")
                #print(adj_mat)
                #add new_node and all neighbors not already in the stack
                #into the stack
                stack.append(new_node)
                for n in neighbors:
                    if not n in stack:
                        stack.append(n)
                #add new_node to the list of nodes
                self.nodes.append(new_node)
                #change the degrees depending on which is the parent
                degree[new_node] = degree[node]  - 1
                degree[node] = 3
            #print(self.nodes)
        #renumber the nodes
        for i,node in enumerate(self.nodes):
            if node.index == self.n_nodes:
                self.nodes[i].index = idxs[0]
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
        #print(adj_mat)
        #print(degree)
        #find a node with degree 3
        idx = len(self.nodes) - 1
        while degree[self.nodes[idx]] != 3:
            idx -= 1
        #go from adjacency matrix to egde list
        adj_list = defaultdict(list)
        for edge in adj_mat:
            adj_list[edge[0]].append(edge[1])
        #assign root as the degree 3 node chosen
        self.root = self.nodes[idx]
        q1 = [self.root]
        q2 = []
        checked = defaultdict(bool)
        
        while len(q1) != 0:
            #print(q1)
            #print(self.__str__())
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
            elif degree[node] == 3:
                
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
                q2.extend(adj_list[node])
                
            if len(q1) == 0:
                if len(q2) != 0:
                    q1 = q2
                    q2 = []