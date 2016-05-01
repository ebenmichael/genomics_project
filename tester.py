import sys
import phylo_tree
import csv
import numpy as np
import time



def run(filename, trial_number):
    
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["n_tissues","tree_score","nj_score", "cov_score",
                         "sample_cov_score", "time"])
    
        for i in range(trial_number):
            try:
                #create original tree
                pt = phylo_tree.PhyloTreeSample()
                pt.generate_tree(.55) #with given probability 0.5
    
                #create sampled tree
                dat = pt.sample() #gives sampled covariance matrix to children
                #time the algorithm
                start_time = time.time()
                #created tree fitted to data
                pt_fit1 = phylo_tree.PhyloTreeFit()
                pt_fit1.fit(dat)
                total_time = time.time() - start_time
                
                #create neighbor-joined tree
                pt_fit2 = phylo_tree.PhyloTreeFit()
                pt_fit2.neighbor_join(dat)
    
                #tree scoring
                sim_score_fit = np.real(pt_fit1.score_tree(pt))
                sim_score_nj = np.real(pt_fit2.score_tree(pt))
    
                #covariance scoring
                cov_score_fit, sample_score = pt_fit1.score_leaves(pt)
                #results.append([sim_score_fit, sim_score_nj,
                #                     cov_score_fit, sample_score])
                
                #write score to csv file
                #with open(filename, "w") as f:
                
                writer.writerow([str(len(pt.leaves)), str(sim_score_fit), 
                                     str(sim_score_nj), str(cov_score_fit), 
                                    str(sample_score), str(total_time)])
            
                
                
                
            except Exception as e:
                print(e)
                
            except RuntimeWarning as e:
                print(e)

if __name__ == "__main__":
    filename = sys.argv[1]
    trial_number = int(sys.argv[2])
    run(filename, trial_number)