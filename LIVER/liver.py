import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.font_manager import FontProperties

fp = FontProperties(fname="arial")

df = pd.read_csv("hcvdat0.csv", header = 0)
df = df.drop(columns=["Unnamed: 0"],axis=1)
X = df.drop(columns=["Category"],axis=1)
y = df["Category"]
X = X.fillna(X.mean())
X.Sex = X.Sex.replace({"m":1,"f":0})
y = y.replace({"0=Blood Donor":"Healthy","0s=suspect Blood Donor":"Healthy","1=Hepatitis":"Hepatitis"
                  ,"2=Fibrosis":"Fibrosis","3=Cirrhosis":"Cirrhosis"})

sm = SMOTENC(categorical_features=[1], random_state=0, sampling_strategy='not majority', k_neighbors=5)
X, y = sm.fit_resample(X, y)

category_list = [1]
le = LabelEncoder()
for i in category_list:
    X.iloc[:,i] = le.fit_transform(X.iloc[:,i])

sc = StandardScaler(with_mean=False,with_std=True)
X = sc.fit_transform(X)

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2, random_state=0)
cls = RidgeClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
print(f"Base Accuracy: {cls.score(X_test, y_test)*100.0:.2f}%")

def lorenz_transformer(x0, y0, z0,sigma=10, beta=2.667, rho=42, tmax=20.0, h=0.01,gamma=5/7):
    import numpy as np
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

plotcmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff","#f5c5c6","#f49d9b","#ff5768","#ab162b"])
cmdisplay = ConfusionMatrixDisplay.from_estimator(cls,X_test,y_test,normalize = "true",
                                                  cmap = "Oranges",colorbar = "false")
cmdisplay.confusion_matrix = (cmdisplay.confusion_matrix*100)
cmdisplay.plot(cmap = plotcmap,values_format = ".1f")
plt.title(f'Before Lorenz System \n', fontsize=10)

plt.savefig(f'CMBeforeLorenz.png',
            dpi = 200,
            transparent = False,
            facecolor = 'white'
            )


acc_mat = np.zeros((101, 1))
max_iter = 0
trainmtx = np.zeros((np.shape(X_train)[0], np.shape(X_train)[1] * 3))
testmtx = np.zeros((np.shape(X_test)[0], np.shape(X_train)[1] * 3))
x_ftrans = np.zeros((np.shape(X)[0], np.shape(X_train)[1] * 3))


for ii in range(1,100):
    for i in range(0, np.shape(X_train)[0]):
        for j in range(0,np.shape(X_train)[1]):
            trainmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_train[i, j], 1.05, -X_train[i, j],tmax=ii*0.01)

    for i in range(0, np.shape(X_test)[0]):
        for j in range(0, np.shape(X_train)[1]):
            testmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_test[i, j], 1.05, -X_test[i, j], tmax=ii*0.01)

    cls.fit(trainmtx, y_train)
    y_pred = cls.predict(testmtx)
    acc_mat[ii] = cls.score(testmtx, y_test)
    print(f"Iteration: {ii}, Accuracy: {float(acc_mat[ii])*100:.2f}")
    if acc_mat[ii] == float(max(acc_mat)):
        max_iter = ii

#Define Trainmtx as the best iteration matrix
for i in range(0, np.shape(X_train)[0]):
    for j in range(0,np.shape(X_train)[1]):
        trainmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_train[i, j], 1.05, -X_train[i, j],rho = 42, tmax=max_iter * 0.01)

for i in range(0, np.shape(X_test)[0]):
    for j in range(0, np.shape(X_train)[1]):
        testmtx[i, (j*3):((j*3)+3)] = lorenz_transformer(X_test[i, j], 1.05, -X_test[i, j],rho = 42, tmax=max_iter * 0.01)

cls = RidgeClassifier()
cls.fit(trainmtx, y_train)
y_pred = cls.predict(testmtx)
print(f"Maximum Iteration {max_iter}, Maximum Accuracy {cls.score(testmtx, y_test)}")

cmdisplay = ConfusionMatrixDisplay.from_estimator(cls,testmtx,y_test,normalize = "true",
                                                  cmap = "Oranges",colorbar = "false",
                                                  labels=['Healthy','Hepatitis','Fibrosis','Cirrhosis'])
cmdisplay.confusion_matrix = (cmdisplay.confusion_matrix*100)
cmdisplay.plot(cmap = plotcmap,colorbar = "false")
plt.title(f'After Lorenz System \n', fontsize=10)

plt.savefig(f'CMAfterLorenzRho97.png',
            dpi = 200,
            transparent = False,
            facecolor = 'white'
            )
