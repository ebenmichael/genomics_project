import sys
import phylo_tree
import csv
import numpy as np




def run(filename, trial_number):
    results = []
    for i in range(trial_number):
        try:
            #create original tree
            pt = phylo_tree.PhyloTreeSample()
            pt.generate_tree(.55) #with given probability 0.5

            #create sampled tree
            dat = pt.sample() #gives sampled covariance matrix to children

            #created tree fitted to data
            pt_fit1 = phylo_tree.PhyloTreeFit()
            pt_fit1.fit(dat)

            #create neighbor-joined tree
            pt_fit2 = phylo_tree.PhyloTreeFit()
            pt_fit2.neighbor_join(dat)

            #tree scoring
            sim_score_fit = np.real(pt_fit1.score_tree(pt))
            sim_score_nj = np.real(pt_fit2.score_tree(pt))

            #covariance scoring
            cov_score_fit, sample_score = pt_fit1.score_leaves(pt)
            results.append([sim_score_fit, sim_score_nj,
                                 cov_score_fit, sample_score])
            """
            #write score to csv file
            with open(filename, "wb") as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow([sim_score_fit, sim_score_nj,
                                 cov_score_fit, sample_score])
            """
            
            
            
        except Exception as e:
            print(e)
    results = np.array(results)
    np.savetxt(filename, results, delimiter = ',')
if __name__ == "__main__":
    filename = sys.argv[1]
    trial_number = int(sys.argv[2])
    run(filename, trial_number)