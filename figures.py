# -*- coding: utf-8 -*-
"""
Plots for presentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv("test.csv")

###figure for covariance scoring
cov_diff = data["sample_cov_score"] - data["cov_score"]
#sort and ignore outliers 
cov_diff = np.sort(cov_diff)
sns_plot = sns.distplot(cov_diff[2:80],bins = 30)
sns_plot.set_title("Difference Between Sample Covariance Score and Fitted Covariance Score")
plt.savefig("diff_cov_mse.png", transparent =True, dpi = 400)


#find amount of time fitted covariance is better
cov_pct_better = sum(cov_diff > 0) / len(cov_diff)
#find median difference
cov_median = np.median(cov_diff)


###figure for tree scoring
tree_diff = data["nj_score"] - data["tree_score"]
sns_plot = sns.distplot(tree_diff, bins = 20)
sns_plot.set_title("Difference Between Neighbor Joining Tree Score and Fitted Tree Score")
plt.savefig("diff_tree_score.png", transparent = True, dpi = 400)

#find amount of time fitted covariance is better
tree_pct_better = sum(tree_diff > 0) / len(tree_diff)
#find median difference
tree_median = np.median(tree_diff)


#how number of tissues affects the score
data["tree_diff"] = data["nj_score"] - data["tree_score"]
sns.lmplot("n_tissues","tree_diff",data)
data.columns = ["n_tissues","Fitted Tree Score", "Neighbor Joining Score",
                "cov_score", "sample_cov_score","time","tree_diff"]
melted = pd.melt(data, id_vars = ["n_tissues"], 
                 value_vars = ["Neighbor Joining Score","Fitted Tree Score"])
                 
sns_plot = sns.lmplot("n_tissues","value",melted,hue="variable", legend_out= False)
sns.plt.title("Tree Error vs. Number of Tissues")
sns.plt.xlabel("Number of Tissues")
sns.plt.ylabel("Tree Error")
plt.savefig("tree_score_vs_tissue.png", transparent = True, dpi = 400)
