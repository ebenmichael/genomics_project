//create original tree
pt = phylo_tree.PhyloTreeSample()
pt.generate_tree(.55) //a node will be the parent of other nodes with given probability 0.55
#create sampled tree
dat = pt.sample() //samples data from leaf nodes
```
Given a list of numpy arrays ``dat``, you can fit a phylogenetic tree and co-expression networks with 
```
pt_fit = phylo_tree.PhyloTreeFit()
pt_fit.fit(dat)
```

```tester.py``` takes in 2 command line variables, the name of the file to write to and the number of trials to run. 
Usage:```python tester.py filename n_trials```
The resulting csv file contains the following information for each trial:
- n_tissues: the number of tissues this trial
- tree_score: the value of the tree dissimilarity between the fitted tree and the real tree sampled (smaller is better)
- nj_score: the value of the tree dissimilarity between a tree fit using the neighbor joining algorithm and the real tree sampled
- cov_score: the mean square error of the co-expression matrices using the algorithm compared to the real co-expression matrices
- sample_cov_score: the mean square error of the sample covariance for each tissue compared to the real co-expression matrices
- time: the time to run the algorithm
