from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy.linalg import pinv
'''
	Part 1: Loading the data
'''
print('Loading the data...')
data = np.loadtxt('ex1data1.txt', delimiter=',',usecols=(0,1),unpack=True)
X = np.transpose(np.array(data[:-1]))
y = np.transpose(np.array(data[-1:]))
m = y.size

'''
	Part 2: Plotting the data
'''
print('Plotting the data...')
plt.figure(figsize=(10,6))
plt.plot(X[:,0],y[:,0],'rx',markersize=10)
plt.grid(True)
plt.ylabel("Profit in $10,000s")
plt.xlabel("Population of City in 10,000s")
plt.savefig('data.png', bbox_inches='tight')
plt.close()
print('Plot saved at data.png')

'''
	Part 3: Cost Function
'''
print('Computing the initial cost function...')
X = np.insert(X,0,1,axis=1)
def hypothesis(X,theta):
	return np.dot(X, theta)
def computeCost(X,y, theta):
	m = y.size
	predictions = hypothesis(X,theta) - y
	J = 1/(2*m) * sum(predictions**2)
	return float(J)

initial_theta = np.zeros((X.shape[1],1))
print(computeCost(X,y,initial_theta))

'''
	Part 4: Gradient Descent
'''
print('Computing the gradient descent...')
def gradientdescent(X,y,theta,alpha,iterations):
	J_history = []
	theta_history = []
	for iteration in range(iterations):
		h_theta = hypothesis(X,theta)
		prediction = h_theta - y
		theta[0] = theta[0] - alpha/m * sum(prediction)
		for x in range(1,len(theta)):
			theta[x] = theta[x] - alpha/m * sum(np.dot(prediction.T,X[:,x]))
		J_history.append(computeCost(X,y,theta))
		theta_history.append(list(theta[:,0]))
	return theta,theta_history,J_history

alpha = 0.01
iterations = 15000
final_theta, thetahistory,J_history =gradientdescent(X,y,initial_theta,alpha,iterations)
print('y_hat = ',final_theta[0][0],'+',final_theta[1][0],'* x')

'''
	Part 5: Plotting Convergence of Cost Function
'''
print('Plotting Convergence of Cost Function...')

def plotConvergence(J_history,fname):
	plt.figure(figsize=(10,6))
	plt.plot(range(len(J_history)), J_history,'ro')
	plt.grid(True)
	plt.title('Convergence of Cost Function')
	plt.xlabel("Iteration")
	plt.ylabel("Cost function")
	plt.xlim([-0.05*iterations,1.05*iterations])
	plt.savefig(fname+'.png',bbox_inches='tight')
	plt.close()
	print('Plot saved at',fname+'.png')
plotConvergence(J_history,'convergence single variable')
'''
	Part 6: Fitting Data to a Line
'''
print('Fitting Data to a Line...')
y_hat = final_theta[0] + final_theta[1]*X[:,1]
plt.figure(figsize=(10,6))
plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='Training Data')
plt.plot(X[:,1],y_hat,'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(final_theta[0][0],final_theta[1][0]))
plt.grid(True)
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.legend()
plt.savefig('linear_fit.png',bbox_inches='tight')
plt.close()
print('Plot saved at linear_fit.png')

print('For population = 35,000, we predict a profit of $%f'%((final_theta[0][0]+final_theta[1][0]*3.5)*10000))
print('For population = 70,000, we predict a profit of $%f'%((final_theta[0][0]+final_theta[1][0]*7)*10000))

'''
	Part 7: Visualizing Cost Function
'''
print("Visualizing Cost Function...")

def countourPlot(X,y,J_history,thetahistory):	
	fig = plt.figure(figsize=(12,12))
	ax = fig.gca(projection='3d')

	xvals = np.arange(-10,10,.5)
	yvals = np.arange(-1,4,.1)
	myxs, myys, myzs = [], [], []
	for theta_0 in xvals:
		for theta_1 in yvals:
			myxs.append(theta_0)
			myys.append(theta_1)
			theta = np.array([[theta_0], [theta_1]])
			myzs.append(computeCost(X,y,theta))

	scat = ax.scatter(myxs,myys,myzs,c=np.abs(myzs),cmap=plt.get_cmap('YlOrRd'))
	plt.plot([x[0] for x in thetahistory],[x[1] for x in thetahistory],J_history,'bo-')
	plt.xlabel(r'$\theta_0$',fontsize=30)
	plt.ylabel(r'$\theta_1$',fontsize=30)
	plt.title('Cost Function',fontsize=30)
	plt.show()

'''
	Part 8: Multivariate Linear Regression
'''
print("Multivariate Linear Regression...")
# Loading the data
datafile = 'ex1data2.txt'
data = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
X = np.transpose(np.array(data[:-1]))
y = np.transpose(np.array(data[-1:]))
m = y.size
X = np.insert(X,0,1,axis=1)

#Feature Normalization(x = (x-u)/s)
stored_feature_means, stored_feature_stds = [], []
X_norm = X.copy()
for icol in range(X_norm.shape[1]):
	mean = np.mean(X_norm[:,icol])
	std = np.std(X_norm[:,icol])
	stored_feature_means.append(mean)
	stored_feature_stds.append(std)
	if not icol: continue
	X_norm[:,icol] = (X_norm[:,icol] - mean)/std

#Gradient Descent
initial_theta = np.zeros((X_norm.shape[1],1))
final_theta, thetahistory, J_history = gradientdescent(X_norm,y,initial_theta,alpha,iterations)
print('y_hat =',final_theta[0][0],end = '')
for i in range(1,len(final_theta)):
	print(' +',final_theta[i][0],'* x'+str(i),end= ' ')
#Testing
print ("\nWhat is price of house with 1500 square feet and 2 bedrooms?")
ytest = np.array([1500.,2.])
ytestscaled = [(ytest[x]-stored_feature_means[x+1])/stored_feature_stds[x+1] for x in range(len(ytest))]	#To "undo" feature normalization, we "undo" 1500 and 2, then plug it into our hypothesis
ytestscaled.insert(0,1)
print ("$%0.2f" % float(hypothesis(ytestscaled,final_theta)))

#Convergence Plot
plotConvergence(J_history,'convergence multivariate')

'''
	Part 9: Normal Equations Method
'''

def normalEquations(X,y):
	inverse = pinv(np.dot(X.T,X))
	inverse_product = np.dot(inverse, X.T)
	return np.dot(inverse_product,y)
print ("Normal equation prediction for price of house with 1500 square feet and 2 bedrooms:")
print ("$%0.2f" % float(hypothesis([1,1500.,2],normalEquations(X,y))))
	
