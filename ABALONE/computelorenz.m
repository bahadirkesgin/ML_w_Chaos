function [t, a] = computelorenz(sigma, beta, rho,tspan, h, x0, y0, z0)
% sigma, beta, rho: Lorenz attractor parameters
% tspan: a 2-element vector [t0, tf] specifying the initial and final time
% h: the time step size
% x0, y0, z0: the initial values of x, y, z at t0
% t: a column vector containing the time steps
% x, y, z: column vectors containing the solution at each time step

% Define the Lorenz system of equations as a function handle
f = @(t, x) [sigma*(x(2) - x(1)); x(1)*(rho - x(3)) - x(2); x(1)*x(2) - beta*x(3)];

% Determine the number of steps based on the time step size
nsteps = round((tspan(2) - tspan(1)) / h);

% Initialize arrays to hold the solution
t = linspace(tspan(1), tspan(2), nsteps+1)';
x = zeros(nsteps+1, 1);
y = zeros(nsteps+1, 1);
z = zeros(nsteps+1, 1);
x(1) = x0;
y(1) = y0;
z(1) = z0;

% Perform the time stepping using the fourth-order Runge-Kutta method
for i = 1:nsteps
    k1 = h * f(t(i), [x(i); y(i); z(i)]);
    k2 = h * f(t(i) + h/2, [x(i); y(i); z(i)] + k1/2);
    k3 = h * f(t(i) + h/2, [x(i); y(i); z(i)] + k2/2);
    k4 = h * f(t(i) + h, [x(i); y(i); z(i)] + k3);
    x(i+1) = x(i) + (k1(1) + 2*k2(1) + 2*k3(1) + k4(1)) / 6;
    y(i+1) = y(i) + (k1(2) + 2*k2(2) + 2*k3(2) + k4(2)) / 6;
    z(i+1) = z(i) + (k1(3) + 2*k2(3) + 2*k3(3) + k4(3)) / 6;
end
a = [x y z];
end