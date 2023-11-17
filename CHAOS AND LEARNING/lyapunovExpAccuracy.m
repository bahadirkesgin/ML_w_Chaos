%% Preprocessing
clear all; close all; clc;
opts = delimitedTextImportOptions("NumVariables", 14);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14"];
opts.SelectedVariableNames = ["VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10", "VarName11", "VarName12", "VarName13", "VarName14"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "VarName14"], "EmptyFieldRule", "auto");
tbl = readtable("SMOTLIVER.csv", opts);
VarName2 = tbl.VarName2;
VarName3 = tbl.VarName3;
VarName4 = tbl.VarName4;
VarName5 = tbl.VarName5;
VarName6 = tbl.VarName6;
VarName7 = tbl.VarName7;
VarName8 = tbl.VarName8;
VarName9 = tbl.VarName9;
VarName10 = tbl.VarName10;
VarName11 = tbl.VarName11;
VarName12 = tbl.VarName12;
VarName13 = tbl.VarName13;
VarName14 = tbl.VarName14;
clear opts tbl

x = horzcat(VarName13,VarName12,VarName11,VarName10,VarName9,VarName8,VarName7,VarName6, ...
    VarName5,VarName4,VarName3,VarName2);
y = VarName14;

clear("VarName2","VarName3","VarName4","VarName5","VarName6","VarName7","VarName8" ...
    ,"VarName9","VarName10","VarName11","VarName12","VarName13","VarName14")
[x_train,x_test,y_train,y_test] = train_test_split(x,y,0.2,0);

size_train = size(x_train);
size_test = size(x_test);

true_count = 0;
t = templateSVM("KernelFunction","linear");
mdl = fitcecoc(x_train,y_train,"Learners",t); 
y_pred = mdl.predict(x_test);
for i = 1:size_test(1)
    if y_pred(i) == y_test(i)
        true_count = true_count + 1;
    end
end
base_acc = true_count / size_test(1);
fprintf("Linear Baseline Accuracy: %f percent\n",base_acc*100);
%% ITERATE RHO FROM 1 TO 100
for i = 1:100
    index = i;
    lyapdata(index,1) = i;
    lyapdata(index,2) = liverAcc(x_train,x_test,y_train,y_test,i,100)*-1;
    lyapdata(index,3) = lyapExpLorenz(i);
    fprintf("Iteration: %d \n",i)
end