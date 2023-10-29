function [output_mat] = lorenzabalone(inp_mtx,rho,tmax)
    % Solve over time interval [0,100] with initial conditions [input_x,1,-input_x]
    % ''f'' is set of differential equations
    % ''a'' is array containing x, y, and z variables
    % ''t'' is time variable     
    sigma = 10;
    beta = 8/3;
    parfor i = 1:8
        [t,a] = normallorenz(sigma, beta, rho,[0,tmax], 0.01, inp_mtx(:,i), 1.05, -inp_mtx(:,i));
        output_mat(i,:) = reshape(a(end,:)', 1, []);
    end
    output_mat = reshape(output_mat.',1,[]);
end