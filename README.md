# LinearRegwithPCA
Linear Regression with PCA Scratch Coding by Python

The classification task is to classify the 10 digit classes, using the original benchmark
set-up. That is, from each of the 200 examples given in the dataset for each digit, take
the first 100 as training data and the second 100 as testing data. Throughout the
training process, never touch the testing data – pretend they don't exist. Only after you
have a trained a classifier solely using the training data, you may test it on the testing
data.

Selected Method: PCA

In this assignment, we choosed PCA in order to determine how many features will be used for linear regression estimation.
In each iteration we tried to predict a model with different number of eigenvectors of
data covariance matrix ( matrix obtained by normalization of original data) and we estimated the test data based on our models.
