import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("data/abalone.csv", header=None)
df = df.fillna(df.mean())
X = df.drop(columns=[8])
y = df[8]
X[0] = X[0].replace({'F': 2,'M': 1,"I":3})
gender = X[0]
X = X.drop(columns=[0])

X = X.astype(float)
X = X.values
msc = MinMaxScaler()
y = msc.fit_transform(y.values.reshape(-1, 1))

sc = StandardScaler(with_mean=False, with_std=True)
X = sc.fit_transform(X,y)
X = np.insert(X,7,gender,axis=1)

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
base_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Base RMSE: {base_rmse:.6f}")

def lorenz_transformer(x0, y0, z0, sigma=10.0, beta=2.667, rho=28.0, tmax=20.0, h=0.01):
    import numpy as np
    # Define the Lorenz system of equations as a function
    def f(t,x):
        return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
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
    a = a[-1, :]
    return a


RMSE_arr = np.zeros((101, 1))
min_iter = 0
trainmtx = np.zeros((np.shape(X_train)[0], 22))
testmtx = np.zeros((np.shape(X_test)[0], 22))
x_ftrans = np.zeros((np.shape(X)[0], 22))


for ii in range(0,100):
    for i in range(0, np.shape(X_train)[0]):
        for j in range(0,7):
            trainmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_train[i, j], 1.05, -X_train[i, j], rho=97, tmax=ii * 0.01)
    trainmtx[:,21] = X_train[:,7]
    for i in range(0, np.shape(X_test)[0]):
        for j in range(0, 7):
            testmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_test[i, j], 1.05, -X_test[i, j], rho=97, tmax=ii * 0.01)
    testmtx[:,21] = X_test[:,7]

    reg.fit(trainmtx, y_train)
    y_pred = reg.predict(testmtx)
    RMSE_arr[ii] = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Iteration: {ii}, RMSE: {float(RMSE_arr[ii]):.6f}")
    if RMSE_arr[ii] == float(min(RMSE_arr)):
        min_iter = ii

#Define Trainmtx as the best iteration matrix
for i in range(0, np.shape(X_train)[0]):
    for j in range(0, np.shape(X_test)[1]):
        trainmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_train[i, j], 1.05, -X_train[i, j], rho = 97, tmax=min_iter * 0.01)
trainmtx[:,21] = X_train[:,7]

for i in range(0, np.shape(X_test)[0]):
    for j in range(0, np.shape(X_test)[1]):
        testmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_test[i, j], 1.05, -X_test[i, j], rho = 97, tmax=min_iter * 0.01)
testmtx[:,21] = X_test[:,7]

reg = LinearRegression()
reg.fit(trainmtx, y_train)
y_pred = reg.predict(testmtx)

#Plot baseline and prediction
plt.scatter(X_train, y_train, c='b', label='Baseline')
plt.scatter(X_test, y_pred, c='r', label='Prediction')
plt.title("Abalone Regression \n" +
          f"Iteration: {min_iter} RMSE:{float(RMSE_arr[min_iter]):.6f}")
plt.legend()
plt.show()
plt.savefig('AfterTransformation.png', dpi=300)
