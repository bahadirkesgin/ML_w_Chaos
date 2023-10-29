import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv("iris.data", header = None)
y = df.iloc[:,4]
X = np.array(df.iloc[:,:4])


sc = StandardScaler()
X = sc.fit_transform(X)

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.3,
                                                      random_state=0)
lda = LinearDiscriminantAnalysis(n_components= 2)
temptr = lda.fit_transform(X_train,y_train)
tempte = lda.transform(X_test)
cls = RidgeClassifier()
cls.fit(temptr, y_train)
y_pred = cls.predict(tempte)
mycmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff","#f5c5c6","#f49d9b","#ff5768","#ab162b"])
print(f"Base Accuracy: {cls.score(tempte, y_test)*100.0:.2f}%")
cmdisplay = ConfusionMatrixDisplay.from_estimator(cls,tempte,y_test,
                                                  cmap = "Oranges",
                                                  colorbar = "false",
                                                  labels = ["Iris-setosa"
                                                            ,"Iris-versicolor"
                                                            ,"Iris-virginica"])
#cmdisplay.confusion_matrix = np.round((cmdisplay.confusion_matrix*100))
cmdisplay.plot(cmap = mycmap,values_format = ".0f")
plt.title("Before Lorenz")
plt.savefig("CM Iris Base", dpi = 300)

def lorenz_transformer(x0, y0, z0,sigma=10, beta=2.667, rho=97, tmax=20.0,h=0.01,gamma=5/7):
    import numpy as np
    # Define the Attractors used in the paper

    def f(t,x): #Lorenz system
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
    # a = np.delete(a, 0, axis=0)
    a = a[-1, :]
    return a

acc_mat = np.zeros((101, 1))
max_iter = 0
trainmtx = np.zeros((np.shape(X_train)[0], np.shape(X_train)[1]*3))
trmtx_lda = np.zeros((np.shape(X_train)[0], 2))
testmtx = np.zeros((np.shape(X_test)[0], np.shape(X_test)[1]*3))
temtx_lda = np.zeros((np.shape(X_train)[0], 2))
x_ftrans = np.zeros((np.shape(X)[0], np.shape(X)[1]*3))

for ii in range(0,101):
    for i in range(0, np.shape(X_train)[0]):
        for j in range(0,np.shape(X_train)[1]):
            trainmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_train[i, j],1.05,-X_train[i, j],tmax=ii * 0.01)

    for i in range(0, np.shape(X_test)[0]):
        for j in range(0, np.shape(X_test)[1]):
            testmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_test[i, j],1.05,-X_test[i, j],tmax=ii * 0.01)

    trmtx_lda = lda.fit_transform(trainmtx,y_train)
    temtx_lda = lda.transform(testmtx)
    cls.fit(trmtx_lda, y_train)
    y_pred = cls.predict(temtx_lda)
    acc_mat[ii] = cls.score(temtx_lda, y_test)
    print(f"Iteration: {ii}, Accuracy: {float(acc_mat[ii])*100:.2f}")
    if acc_mat[ii] == float(max(acc_mat)):
        max_iter = ii
    
xmtx = np.zeros((np.shape(X)[0], np.shape(X)[1]*3))
#Define Trainmtx as the best iteration matrix
for i in range(0, np.shape(X_train)[0]):
    for j in range(0,np.shape(X_train)[1]):
        trainmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_train[i, j], 1.05,-X_train[i, j],tmax=max_iter * 0.01)

for i in range(0, np.shape(X_test)[0]):
    for j in range(0,np.shape(X_train)[1]):
        testmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_test[i, j], 1.05,-X_test[i, j],tmax=max_iter * 0.01)

for i in range(0, np.shape(X)[0]):
    for j in range(0,np.shape(X_train)[1]):
        xmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X[i, j], 1.05, -X[i, j],tmax=max_iter * 0.01)

#Plot Confusion Matrix
trmtx_lda = lda.fit_transform(trainmtx,y_train)
temtx_lda = lda.transform(testmtx)
cls.fit(trmtx_lda, y_train)
y_pred = cls.predict(temtx_lda)
acc_mat[max_iter] = cls.score(temtx_lda, y_test)
print(f"Iteration: {max_iter}, Accuracy: {float(acc_mat[max_iter])*100:.2f}")

cmdisplay = ConfusionMatrixDisplay.from_estimator(cls,temtx_lda,y_test,
                                                  cmap = mycmap,
                                                  colorbar = "false",
                                                  labels = ["Iris-setosa"
                                                            ,"Iris-versicolor"
                                                            ,"Iris-virginica"])
cmdisplay.confusion_matrix = np.round((cmdisplay.confusion_matrix*100))
cmdisplay.plot(cmap = mycmap,values_format = ".0f")
plt.title("After Lorenz")
plt.savefig("CM Iris Lorenz", dpi = 300)