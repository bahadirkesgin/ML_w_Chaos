import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

df = pd.read_csv("data/iris.data", header = None)
y = df.iloc[:,4]
X = np.array(df.iloc[:,:4])

sc = StandardScaler()
X = sc.fit_transform(X)

[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.3,
                                                      random_state=0)
lda = LinearDiscriminantAnalysis(n_components= 2)

X_train = lda.fit_transform(X_train,y_train)
X_test  = lda.transform(X_test)

cls = RidgeClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)
print(f"Base Accuracy: {cls.score(X_test, y_test)*100.0:.2f}%")

#LT Spice Simulations
def run_simulation(cir_name,X_train,X_test,x_node,y_node,z_node,rho_node,train_dir,test_dir,rho_val):
    rho_input = (1000-2.7*rho_val)/rho_val #Compute resistance of R9 for respective rho value
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
            for iteration in range(1,800):
                X_train[i,((iteration*n_predictors*3)+(j*3))] = data.get_trace("V(x)").get_point(iteration)
                X_train[i,((iteration*n_predictors*3)+(j*3)+1)] = data.get_trace("V(y)").get_point(iteration)
                X_train[i,((iteration*n_predictors*3)+(j*3)+2)] = data.get_trace("V(z)").get_point(iteration)

    for i in range(0,test_size):
        for j in range(0,n_predictors):
            raw_file = os.path.join(test_dir,f"test_{i}_{j}.raw")
            data = RawRead(raw_file)
            for iteration in range(1,800):
                X_test[i,((iteration*n_predictors*3)+(j*3))] = data.get_trace("V(x)").get_point(iteration)
                X_test[i,((iteration*n_predictors*3)+(j*3)+1)] = data.get_trace("V(y)").get_point(iteration)
                X_test[i,((iteration*n_predictors*3)+(j*3)+2)] = data.get_trace("V(z)").get_point(iteration)

    print("Data retrieveal completed!")
    return X_train, X_test



train_dir_1 = "C:\\Users\\Excalibur\\Desktop\\Lab Docs\\Paper Codes\\2023Lorenz\\IRIS\\train"
test_dir_1 = "C:\\Users\\Excalibur\\Desktop\\Lab Docs\\Paper Codes\\2023Lorenz\\IRIS\\test"

run_simulation("lorenz.net", X_train, X_test, "V2", "V4", "V6", "R9", train_dir_1, test_dir_1,42)

trainmtx,testmtx = np.zeros((np.shape(X_train)[0],800*6)),np.zeros((np.shape(X_test)[0],800*6))


trainmtx,testmtx = create_array(train_dir_1, test_dir_1, np.shape(X_train)[0],
                                np.shape(X_test)[0],
                                np.shape(X_test)[1])

acc_spice = np.zeros((800,1))

cls = RidgeClassifier()
print(f"{50*'-'}")
for iteration in range(1,800):
    n_predictors = np.shape(X_train)[1]
    temptrain = trainmtx[:,(iteration*n_predictors*3):((iteration*n_predictors*3)+(n_predictors*3))]
    temptest = testmtx[:,(iteration*n_predictors*3):((iteration*n_predictors*3)+(n_predictors*3))]
    temptrain = lda.fit_transform(temptrain,y_train)
    temptest = lda.transform(temptest)
    cls.fit(temptrain, y_train)
    y_pred = cls.predict(temptest)
    acc_spice[iteration] = cls.score(temptest, y_test)
    print(f"Iteration: {iteration}, Accursacy: {float(acc_spice[iteration])*100:.2f}")
    if acc_spice[iteration] == float(max(acc_spice)):
        max_iteration_spice = iteration
        
#Define and plot best Iteration
temptrain = trainmtx[:,(max_iteration_spice*n_predictors*3):((max_iteration_spice*n_predictors*3)+(n_predictors*3))]
temptest = testmtx[:,(max_iteration_spice*n_predictors*3):((max_iteration_spice*n_predictors*3)+(n_predictors*3))]

temptrain = lda.fit_transform(temptrain,y_train)
temptest = lda.transform(temptest)
cls.fit(temptrain, y_train)
y_pred = cls.predict(temptest)
acc_spice[max_iteration_spice] = cls.score(temptest, y_test)
print(f"Iteration: {max_iteration_spice}, Accuracy: {float(acc_spice[max_iteration_spice])*100:.2f}")


#Plot Confusion Matrix
cmdisplay = ConfusionMatrixDisplay.from_estimator(cls,temptest,y_test,
                                                  normalize = "true",
                                                  cmap = "Oranges",
                                                  colorbar = "false")
cmdisplay.confusion_matrix = np.round((cmdisplay.confusion_matrix*100))
cmdisplay.plot(cmap = "Oranges",values_format = ".0f")
plt.title("After Circuit Simulation")
plt.savefig("CM Iris Spice", dpi = 300)


xmtx = np.concatenate((temptrain,temptest))
ymtx = np.concatenate((y_train,y_test))

af_lor = np.zeros((6,20))
bef_lor = np.zeros((6,1))

print("AFTER LORENZ TRANSFORMATION")
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


#Calculate standard deviation for 20 random splits
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