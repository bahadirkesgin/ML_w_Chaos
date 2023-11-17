import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("ChaosAndLearningData.txt", sep="\t")
data = np.array(data)

acc = data[:,1]
le = data[:,2]

corr = stats.pearsonr(acc,le)
print(corr)

corr2 = stats.ttest_ind(acc,le,equal_var = False)
print(corr2)