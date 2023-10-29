function [max_val] = abaloneforoptim(x_train,y_train,x_test,y_test,rho,n_iter)
    acc_test_SVM = ones(1,n_iter);
    size_train = size(x_train);
    size_test = size(x_test);
    for ii = 1:n_iter
        true_count = 0;
        trainmtx = zeros(size_train(1), 24);
        parfor i = 1:size_train(1)
            x = x_train(i,:);
            amat = lorenzabalone(x, rho, 0.01*ii);
            trainmtx(i,:) = reshape(amat', 1, []);
        end
        testmtx = zeros(size_test(1), 24);
        parfor i = 1:size_test(1)
            x = x_test(i,:);
            amat = lorenzabalone(x, rho, 0.01*ii);
            testmtx(i,:) = reshape(amat', 1, []);
        end
    
        mdl = fitcdiscr(trainmtx,y_train,"DiscrimType","linear");
        y_pred = mdl.predict(testmtx);
        for i= 1:length(y_pred)
            if y_pred(i) == y_test(i)
                true_count = true_count + 1;
            end
        end
        acc_test_SVM(1,ii) = true_count * 100 / length(x_test);
    end
    max_val = max(acc_test_SVM);
end