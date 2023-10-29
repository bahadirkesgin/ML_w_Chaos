function [output_mat] = lorenzliver(input_x,rho,tmax)
    % Solve over time interval [0,100] with initial conditions [input_x,1,-input_x]
    % ''f'' is set of differential equations
    % ''a'' is array containing x, y, and z variables
    % ''t'' is time variable     
    sigma = 10;
    beta = 2.667;
    [t,a1] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,1), 1.05, -input_x(:,1));
    [t,a2] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,2), 1.05, -input_x(:,2));
    [t,a3] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,3), 1.05, -input_x(:,3));
    [t,a4] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,4), 1.05, -input_x(:,4));
    [t,a5] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,5), 1.05, -input_x(:,5));
    [t,a6] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,6), 1.05, -input_x(:,6));
    [t,a7] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,7), 1.05, -input_x(:,7));
    [t,a8] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,8), 1.05, -input_x(:,8));
    [t,a9] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,9), 1.05, -input_x(:,9));
    [t,a10] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,10), 1.05, -input_x(:,10));
    [t,a11] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,11), 1.05, -input_x(:,11));
    [t,a12] = computelorenz(sigma, beta, rho,[0,tmax], 0.01, input_x(:,12), 1.05, -input_x(:,12));
    output_mat = horzcat(a1(end,:),a2(end,:),a3(end,:),a4(end,:),a5(end,:), ...
        a6(end,:),a7(end,:),a8(end,:),a9(end,:),a10(end,:),a11(end,:),a12(end,:));
end