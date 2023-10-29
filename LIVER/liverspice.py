import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.2,
                                                      random_state=0)

cls = RidgeClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
print(f"Base Accuracy: {cls.score(X_test, y_test)*100.0:.2f}%")


#LT Spice Simulations
def run_simulation(cir_name,X_train,X_test,x_node,y_node,z_node,rho_node,train_dir,test_dir,rho_val):
    rho_input = (1000-2.7*rho_val)/rho_val
    from PyLTSpice import SimRunner, SpiceEditor
    from PyLTSpice.sim.ltspice_simulator import LTspice
    import numpy as np

    runner_train = SimRunner(output_folder=train_dir, simulator=LTspice,
                             parallel_sims=5)
    runner_test = SimRunner(output_folder=test_dir, simulator=LTspice,
                            parallel_sims=5)
    attr = SpiceEditor(cir_name)

    for i in range(0, np.shape(X_train)[0]):
        if i == np.shape(X_train)[0]:
            break
        else:
            for j in range(0, np.shape(X_train)[1]):
                attr.set_component_value(x_node, f"{X_train[i, j]}")
                attr.set_component_value(y_node, "1.05")
                attr.set_component_value(z_node, f"{X_train[i, j]}")
                attr.set_component_value(rho_node, f"{rho_input}k") 
                filename = f"train_{i}_{j}.net"
                runner_train.run(attr, run_filename=filename)

    runner_train.wait_completion()
    print("Train simulations completed")
    print(f'Successful/Total Simulations: {runner_train.okSim}/{runner_train.runno}')

    for i in range(0, np.shape(X_test)[0]):
        if i == np.shape(X_test)[0]:
            break
        else:
            for j in range(0, np.shape(X_test)[1]):
                attr.set_component_value(x_node, f"{X_test[i, j]}")
                attr.set_component_value(y_node, "1.05")
                attr.set_component_value(z_node, f"{X_test[i, j]}")
                attr.set_component_value(rho_node, f"{rho_input}k")
                filename = f"test_{i}_{j}"
                runner_test.run(attr, run_filename=filename)

    runner_test.wait_completion()
    print("Test simulations completed")
    print(f'Successful/Total Simulations: {runner_test.okSim}/{runner_test.runno}')

def create_array(train_dir,test_dir,train_size,test_size,n_predictors):
    import numpy as np
    import os
    from PyLTSpice import RawRead

    X_train = np.zeros((train_size,n_predictors*2400),dtype=float)
    X_test = np.zeros((test_size,n_predictors*2400),dtype=float)

    for i in range(0,train_size):
        for j in range(0,n_predictors):
            raw_file = os.path.join(train_dir,f"train_{i}_{j}.raw")
            data = RawRead(raw_file)
            for iter in range(1,800):
                X_train[i,((iter*n_predictors*3)+(j*3))] = data.get_trace("V(x)").get_point(iter)
                X_train[i,((iter*n_predictors*3)+(j*3)+1)] = data.get_trace("V(y)").get_point(iter)
                X_train[i,((iter*n_predictors*3)+(j*3)+2)] = data.get_trace("V(z)").get_point(iter)

    for i in range(0,test_size):
        for j in range(0,n_predictors):
            raw_file = os.path.join(test_dir,f"test_{i}_{j}.raw")
            data = RawRead(raw_file)
            for iter in range(1,800):
                X_test[i,((iter*n_predictors*3)+(j*3))] = data.get_trace("V(x)").get_point(iter)
                X_test[i,((iter*n_predictors*3)+(j*3)+1)] = data.get_trace("V(y)").get_point(iter)
                X_test[i,((iter*n_predictors*3)+(j*3)+2)] = data.get_trace("V(z)").get_point(iter)

    print("Data retrieveal completed!")
    return X_train, X_test



train_dir_1 = "C:\\Users\\Excalibur\\Desktop\\Lab Docs\\Paper Codes\\2023Lorenz\\LIVER\\train"
test_dir_1 = "C:\\Users\\Excalibur\\Desktop\\Lab Docs\\Paper Codes\\2023Lorenz\\LIVER\\test"

run_simulation("lorenz.net", X_train, X_test, "V2", "V4", "V6", "R9", train_dir_1, test_dir_1,42)

trainmtx,testmtx = np.zeros((np.shape(X_train)[0],800*36)),np.zeros((np.shape(X_test)[0],800*36))


trainmtx,testmtx = create_array(train_dir_1, test_dir_1, np.shape(X_train)[0],
                                np.shape(X_test)[0],
                                np.shape(X_test)[1])

acc_spice = np.zeros((800,1))

cls = RidgeClassifier()
print(f"{50*'-'}")
for iter in range(1,800):
    n_predictors = np.shape(X_train)[1]
    temptrain = trainmtx[:,(iter*n_predictors*3):((iter*n_predictors*3)+(n_predictors*3))]
    temptest = testmtx[:,(iter*n_predictors*3):((iter*n_predictors*3)+(n_predictors*3))]
    cls.fit(temptrain, y_train)
    y_pred = cls.predict(temptest)
    acc_spice[iter] = cls.score(temptest, y_test)
    print(f"Iteration: {iter}, Accuracy: {float(acc_spice[iter])*100:.2f}")
    if acc_spice[iter] == float(max(acc_spice)):
        max_iter_spice = iter
        
#Define and plot best iteration
temptrain = trainmtx[:,(max_iter_spice*n_predictors*3):((max_iter_spice*n_predictors*3)+(n_predictors*3))]
temptest = testmtx[:,(max_iter_spice*n_predictors*3):((max_iter_spice*n_predictors*3)+(n_predictors*3))]


cls.fit(temptrain, y_train)
y_pred = cls.predict(temptest)
acc_spice[max_iter_spice] = cls.score(temptest, y_test)
print(f"Iteration: {max_iter_spice}, Accuracy: {float(acc_spice[max_iter_spice])*100:.2f}")


#Plot Confusion Matrix
cmdisplay = ConfusionMatrixDisplay.from_estimator(cls,temptest,y_test,
                                                  normalize = "true",
                                                  cmap = "Oranges",
                                                  colorbar = "false",
                                                  labels=["Healthy", 
                                                          "Hepatitis", 
                                                          "Fibrosis", 
                                                          "Cirrhosis"])
cmdisplay.confusion_matrix = np.round((cmdisplay.confusion_matrix*100))
cmdisplay.plot(cmap = "Oranges",values_format = ".0f")
plt.title("After Circuit Simulation")
plt.savefig("CM Iris Spice", dpi = 300)


xmtx = np.concatenate((temptrain,temptest))
ymtx = np.concatenate((y_train,y_test))

af_lor = np.zeros((6,20))
bef_lor = np.zeros((6,1))

print("BEFORE LORENZ TRANSFORMATION")
#Ridge Classifier
cls = RidgeClassifier()
cls.fit(temptrain,y_train)
y_pred = cls.predict(temptest)
print(f"Ridge Classifier Accuracy: {cls.score(temptest, y_test)*100.0:.2f}%")
bef_lor[0,:] = cls.score(temptest, y_test)

#Linear SVM
cls = SVC(kernel="linear")
cls.fit(temptrain,y_train)
y_pred = cls.predict(temptest)
print(f"Linear SVM Accuracy: {cls.score(temptest, y_test)*100.0:.2f}%")
bef_lor[1,:] = cls.score(temptest, y_test)


#Polynomial SVM
cls = SVC(kernel="poly")
cls.fit(temptrain,y_train)
y_pred = cls.predict(temptest)
print(f"Polynomial SVM Accuracy: {cls.score(temptest, y_test)*100.0:.2f}%")
bef_lor[2,:] = cls.score(temptest, y_test)

#Gaussian SVM
cls = SVC()
cls.fit(temptrain,y_train)
y_pred = cls.predict(temptest)
print(f"Gaussian SVM Accuracy: {cls.score(temptest, y_test)*100.0:.2f}%")
bef_lor[3,:] = cls.score(temptest, y_test)

#KNN
cls = KNeighborsClassifier()
cls.fit(temptrain,y_train)
y_pred = cls.predict(temptest)
print(f"KNN Accuracy: {cls.score(temptest, y_test)*100.0:.2f}%")
bef_lor[4,:] = cls.score(temptest, y_test)

#MLP
cls = MLPClassifier(hidden_layer_sizes=(300,),activation="tanh")
cls.fit(temptrain,y_train)
y_pred = cls.predict(temptest)
print(f"Multilayer Perceptron Accuracy: {cls.score(temptest, y_test)*100.0:.2f}%")
bef_lor[5,:] = cls.score(temptest, y_test)

for nn in range(0,20):
    [trainmtx,testmtx,y_train,y_test] = train_test_split(xmtx, ymtx, test_size=0.2)
    
    #Ridge Classifier
    cls = RidgeClassifier()
    cls.fit(trainmtx, y_train)
    y_pred = cls.predict(testmtx)
    af_lor[0,nn] = cls.score(testmtx, y_test)
    
    
    #Linear SVM
    cls = SVC(kernel="linear")
    cls.fit(trainmtx, y_train)
    y_pred = cls.predict(testmtx)
    af_lor[1,nn] = cls.score(testmtx, y_test)
    
    #Quadratic SVM
    cls = SVC(kernel="poly")
    cls.fit(trainmtx, y_train)
    y_pred = cls.predict(testmtx)
    af_lor[2,nn] = cls.score(testmtx, y_test)
    
    #Gaussian SVM
    cls = SVC()
    cls.fit(trainmtx, y_train)
    y_pred = cls.predict(testmtx)
    af_lor[3,nn] = cls.score(testmtx, y_test)
    
    #KNN
    cls = KNeighborsClassifier()
    cls.fit(trainmtx, y_train)
    y_pred = cls.predict(testmtx)
    af_lor[4,nn] = cls.score(testmtx, y_test)
    
    #MLP
    cls = MLPClassifier(hidden_layer_sizes=(300,),activation="tanh")
    cls.fit(trainmtx, y_train)
    y_pred = cls.predict(testmtx)
    af_lor[5,nn] = cls.score(testmtx, y_test)
    
af_lor_std = np.zeros((6,1))
for i in range(0,6):
    af_lor_std[i,:] = np.std(af_lor[i,:],dtype=np.float64)*100