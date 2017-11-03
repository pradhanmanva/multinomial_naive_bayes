**This is a ML Assignment 4**
# Multinomial Naive Bayes Classifier
This is python code to represent MNB which can be trained and tested using data.
Packages involved: math, os, re, sys, collections, nltk, numpy
Dataset Used:  20 Newsgroup Dataset

*Note : There were no ML libraries used for the making for the classifier.*

How to run the code:
1. Keep the folder intact just as is. 
2. Install the above packages using pip. 
3. For windows : Open Command Line & type "python -m pip install _package_name_ "
4. For Linux users: Open Terminal & type "pip install _package_name_"
3. MNB.py outputs the accuracy of the classifier after iterating over 5 classes, irrespective of their name. If you keep 10 classes, then they will iterate over 10 classes and output accuracy for them.
4. Program takes training and test data folder paths as parameters.

Assumptions
1. Tokenized the words by using a regular expression that handles special characters and extra characters like “>>>Atoms” to “Atoms”. The following characters are excluded from such kind of words < > ? _ , ! : ; ( ) " = - $ \ /
2. Each word is thought to be independent of each other.
3. _nltk_ library is used to remove the stopwords like _is, and, are, etc._
2. Algorithm uses Laplace smoothing
3. Done the calculations in log scale to avoid decimal underflow.