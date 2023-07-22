import sys


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

print("You're running python %s" % sys.version.split(" ")[0])

"""
Create vector e.g 128 length,  to store hash value of specific name. 
Split name as prefix and suffix and set the bit to 1
 example peterson change to<
prefix
 p>   = hash("p>") % d  
 pe>
 pet>
 
 suffix 
 n<
 on<
 son 
"""


def setNameasFeature(baby, d, FIX, debug=False):
    """
    Input:
        baby : a string representing the baby's name to be hashed
        d: the number of dimensions to be in the feature vector
        FIX: the number of chunks to extract and hash from each string
        debug: a bool for printing debug values (default False)

    Output:
        v: a feature vector representing the input string
    """
    v = np.zeros(d)
    for m in range(1, FIX + 1):
        prefix = baby[:m] + ">"
        P = hash(prefix) % d
        v[P] = 1

        suffix = "<" + baby[-m:]
        S = hash(suffix) % d
        v[S] = 1

        if debug:
            print(f"Split {m}/{FIX}:\t({prefix}, {suffix}),\t1s at indices [{P}, {S}]")
    if debug:
        print(f"Feature vector for {baby}:\n{v.astype(int)}\n")
    return v


"""
Read file or pass name and create features values
default feature vector length is 128
default value of iteration for each prefix/suffix is 3
"""


def name2features(filename, d=128, FIX=3, LoadFile=True, debug=False):
    """
    Output:
        X : n feature vectors of dimension d, (nxd)
    """
    # read in baby names
    if LoadFile:
        with open(filename, "r") as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split("\n")
    n = len(babynames)
    print(f"length of baby name {n}")
    X = np.zeros((n, d))
    for i in range(n):
        X[i, :] = setNameasFeature(babynames[i], d, FIX)
        # print(X[i])
    return (X, babynames) if debug else X


"""
Xboys, namesBoys = name2features("boys.txt", d=128, FIX=3, debug=True)
Xgirls, namesGirls = name2features("girls.txt", d=128, FIX=3, debug=True)
X = np.concatenate([Xboys[:20], Xgirls[:20]], axis=0)

plt.figure(figsize=(20, 8))
ax = sns.heatmap(X.astype(int), cbar=False)
ax.set_xlabel("feature indices")
ax.set_ylabel("baby names")
ticks = ax.set_yticks(np.arange(40, dtype=int))
ticklabels = ax.set_yticklabels(namesBoys[:20] + namesGirls[:20])
plt.show()
"""


"""
Create Boy and Girl files feature
combine them
randomize the row of each name
"""


def genTrainFeatures(dimension=128):
    """
    Input:
        dimension: desired dimension of the features
    Output:
        X: n feature vectors of dimensionality d (nxd)
        Y: n labels (-1 = girl, +1 = boy) (n)
    """

    # Load in the data
    #    Xgirls = name2features("girls_trim.txt", d=dimension)
    #    Xboys = name2features("boys_trim.txt", d=dimension)
    Xgirls = name2features("girls.txt", d=dimension)
    Xboys = name2features("boys.txt", d=dimension)
    X = np.concatenate([Xgirls, Xboys])

    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    print(Y)
    print(Y.shape)
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    # print(X)
    # print(Y)
    # print(ii)
    return X[ii, :], Y[ii]


"""
X, Y = genTrainFeatures(128)
print(f"Shape of training data: {X.shape}")
print(f"X:\n{X.astype(int)}")
print(f"Y:\n{Y.astype(int)}")
"""


"""
Create probability of each male and female.  count all boy name and divide by total count (boy count + girl count)
"""


def naivebayesPY(X, Y):
    """
    naivebayesPY(X, Y) returns [pos,neg]

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """

    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1, 1]])
    n = len(Y)
    # boys = np.where(Y>0, Y, 0).sum()
    boys = (Y == 1).sum()
    # girls1 =np.where(Y< 0, Y, 0).sum()
    girls = (Y == -1).sum()
    print("Boys:", boys, " girls: " + str(girls))
    pos = np.divide(boys, n)
    neg = np.divide(girls, n)
    return pos, neg


"""
Naive P(X | Y)

here  we sum all individual feature from all names and divide by total count of boy or girl

"""


def naivebayesPXY(X, Y):
    """
    naivebayesPXY(X, Y) returns [posprob,negprob]

    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (n)

    Output:
        posprob: probability vector of p(x_alpha = 1|y=1)  (d)
        negprob: probability vector of p(x_alpha = 1|y=-1) (d)
    """

    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape

    print(f"initial shape is {X.shape}")
    print(Y.shape)

    X = np.concatenate([X, np.ones((2, d)), np.zeros((2, d))])
    Y = np.concatenate([Y, [-1, 1, -1, 1]])
    print(f"shape is {X.shape}")
    print(Y.shape)

    boys = (Y == 1).sum()
    print("Boys Sum: ", str(boys))
    girls = (Y == -1).sum()
    print("girls Sum: ", str(girls))

    # boys = np.sum(X[Y==1],axis=0) /len(X[Y==1])
    # girls = np.sum(X[Y==-1],axis=0) /len(X[Y==-1])

    # print("Brian calc boys", str(boys))
    # print("Brian calc girls", str(girls))

    # create d X 1 vector and divide by number of boys
    # print(X)
    # print(Y)
    posprob = np.dot(X.T, (Y == 1).T) / boys
    # print(np.dot(X.T, (Y == 1).T))

    print(posprob)

    # create d X 1 vector and divide by number of girls
    negprob = np.dot(X.T, (Y == -1).T) / girls
    print(negprob)

    return posprob, negprob


# X, Y = genTrainFeatures(128)
# p, n = naivebayesPY(X, Y)


"""
Use for test name

"""


def loglikelihood(posprob, negprob, X_test, Y_test):
    """
    loglikelihood(posprob, negprob, X_test, Y_test) returns loglikelihood of each point in X_te

    Input:
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)
        Y_test : labels (-1 or +1) (n)

    Output:
        loglikelihood of each point in X_test (n)
    """
    n, d = X_test.shape
    posprob_l4 = np.empty([n])

    # print("postprob: ", posprob)
    # print("negprob: ", negprob)
    # print("X_TEST: ", X_test)
    # print("Y_TEST: ", Y_test)

    # Likelihood of posprob
    posprob_l2 = np.dot(X_test[Y_test == 1], np.log(posprob).T)

    posprob_l3 = np.dot((1 - X_test[Y_test == 1]), np.log(1 - posprob).T)

    posprob_l4[Y_test == 1] = posprob_l2 + posprob_l3
    print("POSprob: ", str(posprob_l4))

    # End Likelihood of posprob

    # Likelihood of negprob
    negprob_l2 = np.dot(X_test[Y_test == -1], np.log(negprob).T)

    negprob_l3 = np.dot((1 - X_test[Y_test == -1]), np.log(1 - negprob).T)

    posprob_l4[Y_test == -1] = negprob_l2 + negprob_l3
    print("POSprob: ", str(posprob_l4))

    # End Likelihood of negprob
    # print("negprob: ", str(negprob_l4))

    # L = np.where((Y_test == 1), posprob_l4, negprob_l4)

    return posprob_l4


"""
X, Y = genTrainFeatures(128)
posprob, negprob = naivebayesPXY(X, Y)
probs = pd.DataFrame(
    {"feature": np.arange(128, dtype=int), "boys": posprob, "girls": negprob}
)

plt.figure(figsize=(20, 4))
ax = sns.lineplot(
    x="feature", y="value", hue="variable", data=pd.melt(probs, ["feature"])
)
ax.set_xlabel("feature indices")
ax.set_ylabel("probability")
plt.show()
print(f"Shape of training data: {X.shape}")
# print(f"positive Negative training data: {p} {n}")
P = name2features("mukesh", d=128, FIX=3, LoadFile=False, debug=False)
print(P)
pos, neg = naivebayesPY(X, Y)
posprob, negprob = naivebayesPXY(X, Y)
print(pos)
print(neg)
print(posprob)
print(np.log(posprob))

print(np.log(posprob).T)
print(posprob.shape)
print((posprob.T).shape)
"""
"""
X, Y = genTrainFeatures(128)
posprob, negprob = naivebayesPXY(X, Y)
yourname = "mukesh"
X_test = name2features(yourname, d=128, LoadFile=False)

Y_boy = np.ones(1)
Y_girl = np.full_like(Y_boy, -1)
postprob = loglikelihood(posprob, negprob, X_test, Y_boy)
print(postprob)
postprob = loglikelihood(posprob, negprob, X_test, Y_girl)
print(postprob)
"""


def naivebayes_pred(pos, neg, posprob, negprob, X_test):
    """
    naivebayes_pred(pos, neg, posprob, negprob, X_test) returns the prediction of each point in X_test

    Input:
        pos: class probability for the negative class
        neg: class probability for the positive class
        posprob: conditional probabilities for the positive class (d)
        negprob: conditional probabilities for the negative class (d)
        X_test : features (nxd)

    Output:
        prediction of each point in X_test (n)
    """
    # Likelihood of posprob using MLE
    n, d = X_test.shape

    Y_boy = np.ones(n)
    Y_girl = np.full_like(Y_boy, -1)

    # print("Y_girl = ", str(Y_girl))

    pos_likelihood = loglikelihood(posprob, negprob, X_test, Y_boy)
    # print("pos likelihood", str(pos_likelihood))
    log_pos_likelihood = pos_likelihood + np.log(pos)
    # print("log pos:", str(log_pos_likelihood))

    neg_likelihood = loglikelihood(posprob, negprob, X_test, Y_girl)
    # print("neg likelihood", str(neg_likelihood))

    log_neg_likelihood = neg_likelihood + np.log(neg)

    # print("log neg", str(log_neg_likelihood))

    result_pred = log_pos_likelihood - log_neg_likelihood
    # print(result)

    result_pred[result_pred < 0] = -1
    result_pred[result_pred > 0] = 1
    # print(result)

    return result_pred


DIMS = 128
print("Loading data ...")
X, Y = genTrainFeatures(DIMS)
print("Training classifier ...")
pos, neg = naivebayesPY(X, Y)
posprob, negprob = naivebayesPXY(X, Y)
error = np.mean(naivebayes_pred(pos, neg, posprob, negprob, X) != Y)
print("Training error: %.2f%%" % (100 * error))

while True:
    print("Please enter a baby name (press enter with empty box to stop prompt)>")
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features(yourname, d=DIMS, LoadFile=False)
    pred = naivebayes_pred(pos, neg, posprob, negprob, xtest)
    if pred > 0:
        print("%s, I am sure you are a baby boy.\n" % yourname)
    else:
        print("%s, I am sure you are a baby girl.\n" % yourname)
