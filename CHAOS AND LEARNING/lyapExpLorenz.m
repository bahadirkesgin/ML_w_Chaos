function [mle_output] = lyapExpLorenz(rho)
    [t,a] = normallorenz(10, 2.667, rho ,[0,30], 0.01, 1,1,1);
    fs = 100;
    xvals = a(:,1);
    [~,lag,dim] = phaseSpaceReconstruction(xvals);
    mle_output = lyapunovExponent(xvals,fs,lag,dim,'ExpansionRange',200);
end
    