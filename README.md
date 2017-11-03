# MultinomialNaiveBayes
python code to represent MNB which can be trained and tested using data.

Packages involved: sys, os, string, re, collections, datetime, time, math, numpy

How to run the code:
1. Program has to be run in Eclipse IDE with python plugin added to it. Copy the files and import it in a new project in eclipse.
2. Used Numpy package. For this, download Anaconda software that includes all these packages and make the eclipse interpreter to point to the Anaconda python directory so that Eclipse can access Numpy package as well.
3. Main1.py file implements the MNB algorithm and gives the folders used and the accuracy as output.
4. Program takes training and test data folder paths as parameters.

Assumptions
1. Tokenized the words by using a regular expression that handles special characters and extra characters like “>>>Atoms” to “Atoms”. The following characters are excluded from such kind of words < > ? _ , ! : ; ( ) " = - $ \ /
2. Algorithm uses Laplace smoothing
3. Done the calculations in log scale to avoid decimal underflow.
