function [max_output] = liverAcc(x_train,x_test,y_train,y_test,rho1,max_iter)
    size_train = size(x_train);
    size_test = size(x_test);
    acc_test = ones(1,max_iter);
    for ii = 1:max_iter
        true_count = 0;
        trainmtx = zeros(size_train(1), 36);
        parfor i = 1:size_train(1)
            x = x_train(i,:);
            trainmtx(i,:) = lorenzliver(x, rho1, 0.01*ii);
        end
        testmtx = zeros(size_test(1), 36);
        parfor i = 1:size_test(1)
            x = x_test(i,:);
            testmtx(i,:) = lorenzliver(x, rho1, 0.01*ii);
        end
        t = templateSVM("KernelFunction","linear");
        mdl = fitcecoc(trainmtx,y_train,"Learners",t); 
        y_pred = mdl.predict(testmtx);
        for i = 1:size_test(1)
            if y_pred(i) == y_test(i)
                true_count = true_count + 1;
            end
        end
        acc_test(ii) = true_count / size_test(1);
    end
    max_output = max(acc_test) * -1;
end