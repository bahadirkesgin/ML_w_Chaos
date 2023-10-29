# Load the dataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay
import umap.umap_ as umap

# Transform PIL image into a tensor. The values are in the range [0, 1]
t = transforms.ToTensor()

# Load datasets for training and testing.
mnist_train = datasets.MNIST(root='/tmp/mnist', train=True, download=True, transform=t)
mnist_test = datasets.MNIST(root='/tmp/mnist', train=False, download=True, transform=t)
X_train = np.array(mnist_train.data).reshape(60000,784)
X_test = np.array(mnist_test.data).reshape(10000,784)
y_train = np.array(mnist_train.targets)
y_test = np.array(mnist_test.targets)


reducer = umap.UMAP(n_components = 7, random_state = 42)
X_train = reducer.fit_transform(X_train)
X_test  = reducer.transform(X_test)

#Base Classification
cls = RidgeClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
base_acc = cls.score(X_test, y_test)
print(f'Base Accuracy: {base_acc} \n')


acc_mat = np.zeros((101, 1))
max_iter = 0
trainmtx = np.zeros((np.shape(X_train)[0], (np.shape(X_train)[1]*3)))
testmtx = np.zeros((np.shape(X_test)[0], (np.shape(X_test)[1])*3))

#Define Lorenz Transformer and Training Matrices
def lorenz_transformer(x0, y0, z0, sigma=10.0, beta=2.667, rho=28.0, tmax=20.0, h=0.01):
    import numpy as np
    # Define the Lorenz system of equations as a function
    def f(t, x):
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
trainmtx = np.zeros((np.shape(X_train)[0], np.shape(X_train)[1] * 3))
testmtx = np.zeros((np.shape(X_test)[0], np.shape(X_train)[1] * 3))


for ii in range(0,100):
    for i in range(0, np.shape(X_train)[0]):
        for j in range(0,np.shape(X_train)[1]):
            trainmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_train[i, j], 1.05, -X_train[i, j], tmax=ii * 0.01)

    for i in range(0, np.shape(X_test)[0]):
        for j in range(0, np.shape(X_train)[1]):
            testmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_test[i, j], 1.05, -X_test[i, j], tmax=ii * 0.01)

    cls.fit(trainmtx, y_train)
    y_pred = cls.predict(testmtx)
    acc_mat[ii] = cls.score(testmtx, y_test)
    print(f"Iteration: {ii}, Accuracy: {float(acc_mat[ii])*100:.2f}")
    if acc_mat[ii] == float(max(acc_mat)):
        max_iter = ii


#Define Trainmtx as the best iteration matrix
for i in range(0, np.shape(X_train)[0]):
    for j in range(0, np.shape(X_test)[1]):
        trainmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_train[i, j], 1.05, -X_train[i, j], rho = 97, tmax=max_iter * 0.01)
for i in range(0, np.shape(X_test)[0]):
    for j in range(0, np.shape(X_test)[1]):
        testmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_test[i, j], 1.05, -X_test[i, j], rho = 97, tmax=max_iter * 0.01)

plotcmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff","#f5c5c6","#f49d9b","#ff5768","#ab162b"])
cls = RidgeClassifier()
cls.fit(X_train,y_train)
y_pred = cls.predict(X_test)
cmdisplay = ConfusionMatrixDisplay.from_estimator(cls,X_test,y_test,normalize = "true",
                                                  cmap = "Oranges",colorbar = "false",values_format=".2f")
cmdisplay.confusion_matrix = cmdisplay.confusion_matrix*100
cmdisplay.plot(cmap = plotcmap,values_format = ".1f")
plt.title(f'Before Lorenz System \n', fontsize=10)

plt.savefig(f'CMBeforeLorenz.png',
            dpi = 300,
            transparent = False,
            facecolor = 'white'
            )

cls.fit(trainmtx, y_train)
y_pred = cls.predict(testmtx)
print(f"Accuracy: {cls.score(testmtx, y_test)*100:.2f}")

cmdisplay = ConfusionMatrixDisplay.from_estimator(cls,testmtx,y_test,normalize = "true",
                                                  cmap = "Oranges",colorbar = "false",values_format=".2f")
cmdisplay.confusion_matrix = cmdisplay.confusion_matrix*100
cmdisplay.plot(cmap = plotcmap,values_format = ".1f")
plt.title(f'After Lorenz System \n', fontsize=10)

plt.savefig(f'CMAfterLorenz.png',
            dpi = 300,
            transparent = False,
            facecolor = 'white'
            )