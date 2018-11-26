# -*- coding: utf-8 -*-
"""
Created on Friday 13 April 2018

@author: diego and yasin
"""
"Machine Learning 8.th assignment"

import numpy as np
import matplotlib.pyplot as plt
#Loading the dataset
X=np.loadtxt('/home/yasin/mfeat-pix.txt')
X
#Plot the five first digits
def plot_database(database):
    for i in range (5):
        for j in range (5):
            plt.subplot(1, 5, j+1) #make a 1 by 5 grid, and paste in each grid the jth element, note that j starts with zero, therefore jth+1 for the first element
            databases=database[j + 200 * i,].reshape(16, 15) #select the first row,240 digits, of the digits array, and reshape them in a matrix 16x15 
            plt.imshow(databases, cmap='gray') #plot a grid, where the number is the "gray" value
            plt.axis('off') #Turn plot axis off
        plt.show() # show the graph
plot_database(database=X)
#Choosing first 100 elements from each digit for train data
A=np.vstack((X[0:100], X[200:300],X[400:500],X[600:700],X[800:900],X[1000:1100],X[1200:1300],X[1400:1500],X[1600:1700],X[1800:1900]))
sum(sum(A))
#Choosing second 100 elements from each digit for test data
T=np.vstack((X[100:200], X[300:400],X[500:600],X[700:800],X[900:1000],X[1100:1200],X[1300:1400],X[1500:1600],X[1700:1800],X[1900:2000]))
#Calculating the mean of matrix A for columns
M=np.mean(A, axis=0)
print(M)
#Subtract Mean from every vector
AdjMt = A-M
print(AdjMt)
#Data Covariance matrix of Ones
R  = np.dot(AdjMt.T,AdjMt) / (A.shape[0])
#Calculating AdjT for test data with the similar procedure like AdjMt
AdjT = T-np.mean(T, axis=0)
#SVD of Covariance Matrix
U, s, V = np.linalg.svd(R, full_matrices=True)
#Creating Ones vector for bias
n = 1000
m = 1
ones = [1] * n
for i in range(n):
    a[i] = [1] * m
print(ones)

# create class labels 
nb_classes = 10
y_vector = np.array([i for i in range(10) for j in range(100)])
y_matrix = np.eye(nb_classes)[y_vector]
#creating list for mse/classification for test and train data
mse_train=[]
mse_test=[]
class_train=[]
class_test=[]
for k in range (0,240):#For loop for calculation of eigenvectors
    # Select first k eigenvectors of U
    U_new = U[:,:k+1]
    #Calculate the product of AdjMt with U_new
    C=np.dot(AdjMt,U_new)
    C_test=np.dot(AdjT,U_new)    
    #Combining Ones with C
    C_new= np.column_stack((ones,C))
    C_new_test= np.column_stack((ones,C_test))
    #Finding W optimal
    W_opt= np.dot(np.linalg.pinv(C_new),y_matrix)
    #Calculating estimated values for training data
    y_hat = np.dot(C_new,W_opt)
    #Caluclating correctly classified rate of train data by classification
    y_hat_index=np.argmax(y_hat, axis = 1)
    count=0
    for ind in range (0,1000):
        if y_hat_index[ind]==y_vector[ind]:
            count+=1
    class_train.append(1-count/A.shape[0])
    #Calculating estimated values for test data
    y_hat_test=np.dot(C_new_test,W_opt)    
    #Caluclating correctly classified rate of test data by classification
    y_hat_test_index=np.argmax(y_hat_test, axis = 1)
    count1=0
    for ind1 in range (0,1000):
        if y_hat_test_index[ind1]==y_vector[ind1]:
            count1+=1
    class_test.append(1-count1/T.shape[0])
    #Calculating Mean Squared Residual Error for train test data
    mse_train.append(sum(sum((y_matrix-y_hat)**2))/1000)
    mse_test.append(sum(sum((y_matrix-y_hat_test)**2))/1000)
#Obeserving the results of MSE
plt.plot(mse_train,label= 'train')
plt.plot(mse_test, label= 'test')
plt.legend()
plt.ylabel('MSE')
plt.xlabel('Number of Eigenvectors')
nrfeatures_mse=mse_test.index(min(mse_test))+1 #Adding one because of list index properties
print(nrfeatures_mse)
min(mse_test)
mse_train[77]

#Obeserving the results of classification
plt.plot(class_train, label= 'train')
plt.plot(class_test, label= 'test')
plt.legend()
plt.ylabel('MISS')
plt.xlabel('Number of Eigenvectors')

nrfeatures_class=class_test.index(min(class_test))+1 #Adding one because of list index properties
min(class_test)
class_train[43]

