import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sam_size = 2048
sam_size_test = 2048

X_train = - np.pi + np.multiply((np.pi + np.pi),np.random.rand(sam_size,1))
X_test = - np.pi + np.multiply((np.pi + np.pi),np.random.rand(sam_size_test,1))
y_train = np.sinc(X_train)
y_test = np.sinc(X_test)

def lorenz_transformer(x0, y0, z0,sigma=10, beta=2.667, rho=97, tmax=20.0, h=0.01,gamma=10):
    import numpy as np
    # Define the Attractors used in the paper
    def f(t,x): #Lorenz system
        return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]

    #def f(t,x): #Burke-Shaw system 
        #return [-10*x[0] -sigma*x[1], -10*x[0]*x[2]-x[1], 10*x[0]*x[1]+4.272] #LOOK AT MINUS

    #def f(t,x): #Halvorsen attractor 
        #return [-1.27*x[0]-4*x[1]-4*x[2]-x[1]**2, -1.27*x[1]-4*x[2]-4*x[0]-x[2]**2, -1.27*x[2]-4*x[0]-4*x[1]-x[0]**2]

    #def f(t,x): #Chen attractor 
        #return [60*(x[1]-x[0]), (2.667-60)*x[0]-x[0]*x[2]+97*x[1], x[0]*x[1]-x[0]*x[2]-2.667*x[2]]

    #def f(t,x): #Chua's circuit 
        #return [9*(x[1]-x[0]+gamma*x[0]+1/2*((8/7)-(5/7))*(abs(x[0]+1)-abs(x[0]-1))), x[0]-x[1]+x[2], -(100/7)*x[1]]

    #def f(t,x): #Rossler attractor 
        #return [-x[1]-x[2],x[0]+0.2*x[1],0.2+x[2]*(x[0]-5.7)]
    
    #def f(t,x): #Sprott attractor
        #return [x[1]+2.07*x[0]*x[1]+x[0]*x[2], 1-1.79*x[0]**2+x[1]*x[2], x[0]-x[0]**2-x[1]**2]

    #def f(t,x): # Hyperchaotic Lorenz
        #return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1] + x[3],x[0] * x[1] - beta * x[2],-gamma*x[1]]

    # Determine the number of steps based on the time step size
    nsteps = round((tmax - 0) / h)

    # Initialize arrays to hold the solution
    t = np.linspace(0, tmax, nsteps + 1)
    x = np.zeros(nsteps + 1)
    y = np.zeros(nsteps + 1)
    z = np.zeros(nsteps + 1)
    x[0] = x0
    y[0] = y0
    z[0] = z0

    # Perform the time stepping using the fourth-order Runge-Kutta method
    for i in range(nsteps):
        k1 = h * np.array(f(t[i], [x[i], y[i], z[i]]))
        k2 = h * np.array(f(t[i] + h / 2, [x[i], y[i], z[i]] + k1 / 2))
        k3 = h * np.array(f(t[i] + h / 2, [x[i], y[i], z[i]] + k2 / 2))
        k4 = h * np.array(f(t[i] + h, [x[i], y[i], z[i]] + k3))
        x[i + 1] = x[i] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
        y[i + 1] = y[i] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        z[i + 1] = z[i] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6

    a = np.column_stack((x, y, z))
    # a = np.delete(a, 0, axis=0)
    a = a[-1, :]
    return a

print(f"{50*'-'}")

#One Lorenz
trainmtx = np.zeros((sam_size,3))
testmtx = np.zeros((sam_size,3))
RMSE_one = np.zeros((100,1),dtype = float)

for ii in range(0,100):
    for i in range(sam_size):
        trainmtx[i,:] = lorenz_transformer(X_train[i],1.05,-X_train[i],tmax = ii*0.01)
        testmtx[i,:] = lorenz_transformer(X_test[i],1.05,-X_test[i], tmax = ii*0.01)
    reg = LinearRegression()
    reg.fit(trainmtx, y_train)
    y_pred = reg.predict(testmtx)
    RMSE_one[ii] = mean_squared_error(y_test, y_pred, squared=False)
    print(f'Iteration: {ii}, One Lorenz RMSE: {float(RMSE_one[ii]):.6f}')
    if ii == 0 or ii == 1 :
        min_iter_one = 1
    elif RMSE_one[ii] == min(RMSE_one[:ii+1]):
        min_iter_one = ii
   
#Get Best Iteration
for i in range(sam_size):
    trainmtx[i,:] = lorenz_transformer(X_train[i],1.05,-X_train[i],tmax = min_iter_one*0.01)
    testmtx[i,:] = lorenz_transformer(X_test[i],1.05,-X_test[i], tmax = min_iter_one*0.01)
    
reg = LinearRegression()
reg.fit(trainmtx, y_train)
y_pred = reg.predict(testmtx)

#Plot baseline and prediction
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, c='b', label='Baseline')
plt.scatter(X_test, y_pred, c='r', label='Prediction')
plt.title("Sinc Regression with Single Lorenz Transformer (Scaled) \n" +
          f"Iteration: {min_iter_one} RMSE:{float(RMSE_one[min_iter_one]):.6f}")
plt.legend()
plt.show()
plt.savefig('One Lorenz Scaled.png', dpi=300)

print(f"{50*'-'}")

#Two Parallel Lorenz
trainmtx = np.zeros((sam_size,6))
testmtx = np.zeros((sam_size,6))
RMSE_two = np.zeros((100,1),dtype = float)
for ii in range(0,100):
    for i in range(sam_size):
        trainmtx[i,:] = np.concatenate((lorenz_transformer(X_train[i],1.05,-X_train[i],rho = 94.087,
                                                           tmax = ii*0.01),
                                        lorenz_transformer(X_train[i],1.05,-X_train[i],rho = 36.867,
                                                           tmax = ii*0.01)),axis=0)
        testmtx[i,:] = np.concatenate((lorenz_transformer(X_test[i],1.05,-X_test[i],rho = 94.087,
                                                          tmax = ii*0.01),
                                       lorenz_transformer(X_test[i],1.05,-X_test[i],rho = 36.867,
                                                          tmax = ii*0.01)),axis=0)
        
    reg = LinearRegression()
    reg.fit(trainmtx, y_train)
    y_pred = reg.predict(testmtx)
    RMSE_two[ii] = mean_squared_error(y_test, y_pred, squared=False)
    print(f'Iteration: {ii}, Two Lorenz RMSE: {float(RMSE_two[ii]):.6f}')
    if ii == 0 or ii == 1 :
        min_iter_two = 1
    elif RMSE_two[ii] == min(RMSE_two[:ii+1]):
        min_iter_two = ii
    
        
#Get Best Iteration
for i in range(sam_size):
    trainmtx[i,:] = np.concatenate((lorenz_transformer(X_train[i],1.05,-X_train[i],
                                                       rho = 94.087,tmax = min_iter_two*0.01),
                                    lorenz_transformer(X_train[i],1.05,-X_train[i],rho = 36.867,
                                                       tmax = min_iter_two*0.01)),axis=0)
    testmtx[i,:] = np.concatenate((lorenz_transformer(X_test[i],1.05,-X_test[i],rho = 94.087,
                                                      tmax = min_iter_two*0.01),
                                   lorenz_transformer(X_test[i],1.05,-X_test[i],rho = 36.867,
                                                      tmax = min_iter_two*0.01)),axis=0)
    
reg = LinearRegression()
reg.fit(trainmtx, y_train)
y_pred = reg.predict(testmtx)

#Plot baseline and prediction
print('Parallel Lorenz RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
plt.scatter(X_train, y_train, c='b', label='Baseline')
plt.scatter(X_test, y_pred, c='r', label='Prediction')
plt.title("Sinc Regression with Two Lorenz Transformers (Scaled)\n" +
          f"Iteration: {min_iter_two} RMSE:{float(RMSE_two[min_iter_two]):.6f}")
plt.legend()
plt.savefig('Parallel Lorenz Scaled.png', dpi=300)
plt.show()
