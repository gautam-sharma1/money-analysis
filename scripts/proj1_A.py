###################################################################################
# @author Gautam Sharma                                                           #
# Project 1 A                                                                     #
# Program to read the database and analyze the data                               #
#                                                                                 #
# pandas is used to read that csv file and statistical analysis is performed      #
# correlation and covariance matrix is printed to the console                     #
###################################################################################

import numpy as np                               # needed for arrays and math
import pandas as pd                              # needed to read the data
import matplotlib.pyplot as plt                  # used for plotting
import seaborn as sns                            # data visualization


bank_note = pd.read_csv('data_banknote_authentication.txt')       # read in the data
names = ["variance","skewness","curtosis","entropy", "class"]
# print(bank_note.head())                                         # to debug is data is read correctly


corr = bank_note.corr().abs()                                     # calculates the correlation matrix

cov = bank_note.cov().abs()                                       # calculates the covariance matrix

##############################################################################################################
#                                           correlation calculation                                          #
##############################################################################################################

# set the correlations on the diagonal or lower triangle to zero and
# clear the diagonal since the correlation with itself is always 1.
corr *= np.tri(*corr.values.shape, k=-1).T
print("correlation = ")
print(corr)                                                     # prints correlation matrix to the console


# now unstack it so we can sort things
# note that zeros indicate no correlation OR that we cleared below the
# diagonal. Note that corr_unstack is a pandas series.
corr_unstack = corr.unstack()


# Sort values in descending order
corr_unstack.sort_values(inplace=True,ascending=False)
#print(corr_unstack)

##############################################################################################################
#                                           covariance calculation                                           #
##############################################################################################################

# set the covariance on the diagonal or lower triangle to zero and
# clear the diagonal since the correlation with itself is always 1.
cov *= np.tri(*cov.values.shape, k=-1).T
print()
print("covariance = ")
print(cov)                                                      # prints correlation matrix to the console


# now unstack it so we can sort things
# note that zeros indicate no covariance OR that we cleared below the
# diagonal. Note that cov_unstack is a pandas series.
cov_unstack = cov.unstack()


# Sort values in descending order
cov_unstack.sort_values(inplace=True,ascending=False)

print(cov_unstack)

##############################################################################################################
#                                           heat map and pair plot                                           #
##############################################################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(bank_note.corr().abs(), vmin=0, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,5,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.title('Correlation heat map')
plt.show()
sns.set(style='whitegrid', context='notebook')                  # set the apearance
sns.pairplot(bank_note,height=1.5)                              # create the pair plots
plt.show()                                                      # and show them


