function next_x = f01_exact_step(x,u,d,dt,parameters) %#ok<INUSD>
% Synthetic F01 spring fixture; exact ZOH, not measured engineering data.
A = [0 1; -2 -.8]; B = [0;1];
discrete = expm([A B; zeros(1,3)]*dt);
next_x = discrete(1:2,1:2)*x + discrete(1:2,3)*(u+d);
end
