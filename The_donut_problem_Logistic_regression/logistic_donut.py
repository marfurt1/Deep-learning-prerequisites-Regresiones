import numpy as np
import matplotlib.pyplot as plt

#CREATING DATA FOR DEMOSTRATION OF THE DONUT PROBLEM
#in this example we are gonna use a lot more data points so that we can see something substantial
N=1000 # number of samples
D = 2 # number of features or dimensions

# so here we have two radiuses
#we have inner radius
R_inner = 5
# we have outer radius
R_outer = 10

#INNER RADIUS
# we are gonna set a uniformly distributed variable for half the data that depends on the inner radius so it's spread around 5
R1 = np.random.randn(int(N/2)) + R_inner
# we are gonna generate some angles so these are polar coordinates that are uniformly distributed
# formula used here is 2piR here R is N/2 which is half of the data randomly selected by numpy
theta = 2*np.pi*np.random.random(int(N/2)) # polar coordinates
#here we are converting the polar coordinates into xy coordinates
# {cos(Theta) * R and sin(Theta) * R} tranpose this entire matrix that goes along the rows
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

#OUTER RADIUS
# we are gonna do the samething for the outer radius
# we are gonna set a uniformly distributed variable for half the data that depends on the outer radius so it's spread around 10
R2 = np.random.randn(int(N/2)) + R_outer
# we are gonna generate some angles so these are polar coordinates that are uniformly distributed
# formula used here is 2piR here R is N/2 which is half of the data randomly selected by numpy
theta = 2*np.pi*np.random.random(int(N/2)) # polar coordinates
#here we are converting the polar coordinates into xy coordinates
# {cos(Theta) * R and sin(Theta) * R} tranpose this entire matrix that goes along the rows
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

# here we are gonna calculate our entire X by concatenating X_inner and X_outer arrays into one
X = np.concatenate([X_inner , X_outer]) # X input data matrix
#setting up the targets array
#we say the first set is 0 and the second set is 1 for half the data in N (Number of smaples) i.e for 500 samples
T = np.array([0]*(int(N/2)) +[1]*(int(N/2))) #Targets

#Now plotting these data points so we can have a look how these look like
#X[:,0] selecting all the rows in x matrix and selecting 0th column in the X matrix
#X[:,1] selecting all the rows in the x matrix and selecting the 1st column in the X matrix
plt.scatter(X[:,0], X[:,1], c=T)
plt.show()
