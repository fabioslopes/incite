# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:20:17 2018

@author: Fabio
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
#import statsmodels.api as sm
import statsmodels as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
# special matplotlib argument for improved plots
from matplotlib import rcParams

from sklearn.datasets import load_boston
df = load_boston()

print(df.keys())
print(df.data.shape)
print(df.feature_names)

bos = pd.DataFrame(df.data)

bos.columns = df.feature_names
bos['PRICE'] = df.target
bosstatistic = bos.describe().T

plt.figure(figsize=(15,7.5))
bos.boxplot()

bos.hist()

# the histogram of the data
plt.figure(figsize=(10, 5))
n, bins, patches = plt.hist(df.target, 15, facecolor='green', alpha=0.8)
plt.xlabel('Price ($1000s)')
plt.ylabel('Count')
plt.title('$\mathrm{Histogram\ of\ Boston\ Prices}$')
plt.axis([0, 60, 0, 120])
plt.grid(True)
plt.tight_layout()
plt.show()