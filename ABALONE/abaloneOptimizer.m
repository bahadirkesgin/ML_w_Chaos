%% Preprocessing
clear all;close all;clc;
opts = delimitedTextImportOptions("NumVariables", 9);
opts.DataLines = [1, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["Var1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9"];
opts.SelectedVariableNames = ["VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, "Var1", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "Var1", "EmptyFieldRule", "auto");
df = readtable("C:\Users\Excalibur\Desktop\Lab Docs\Chaotic Learning Systems\LORENZ REAL DATA TESTS\Regession\ABALONE\abalone.csv", opts);
df = table2array(df);
clear opts
opts = delimitedTextImportOptions("NumVariables", 9);
opts.DataLines = [1, Inf];
opts.Delimiter = ",";
opts.VariableNames = ["M", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9"];
opts.SelectedVariableNames = "M";
opts.VariableTypes = ["string", "string", "string", "string", "string", "string", "string", "string", "string"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts = setvaropts(opts, ["M", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["M", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9"], "EmptyFieldRule", "auto");
gender = readmatrix("C:\Users\Excalibur\Desktop\Lab Docs\Chaotic Learning Systems\LORENZ REAL DATA TESTS\Regession\ABALONE\abalone.csv", opts);
clear opts

gender_numeric = zeros(length(gender), 1);
gender_numeric(gender == "M") = 1;
gender_numeric(gender == "F") = 2;
gender_numeric(gender == "I") = 3;

x = horzcat(gender_numeric,df);
y = df(:,end);
x(:,end) = [];

clear("gender","gender_numeric","df");

y(ismember(y, 1:8)) = 1;
y(ismember(y, 9:10)) = 2;
y(ismember(y, 11:30)) = 3;

sc = StandardScaler();
x = sc.fit_transform(x,y);

[x_train,x_test,y_train,y_test] = train_test_split(x,y,0.2);
%% Optimize rho value
num1 = optimizableVariable('n1',[0,200],'Type','real');
fun = @(x)abaloneforoptim(x_train,y_train,x_test,y_test,x.n1,100);
results = bayesopt(fun,[num1]);