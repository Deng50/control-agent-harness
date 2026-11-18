function dx = f01_derivative(x,u,d,dt,parameters) %#ok<INUSD>
% Synthetic F01 RHS in seconds; use only with declared Euler/RK4 integration.
dx = [x(2); -2*x(1)-.8*x(2)+u+d];
end
