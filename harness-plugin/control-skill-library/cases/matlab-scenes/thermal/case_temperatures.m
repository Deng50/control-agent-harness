function [surface,core]=case_temperatures(heat,ambient,x,dt)
% 与原文件的双节点前向欧拉递推等价，filter 用于整段 GA 反复评估。
rc=x(1);cc=x(2);rs=x(3);cs=x(4);
A=[1-dt/(rc*cc),dt/(rc*cc);dt/(rc*cs),1-dt/(rc*cs)-dt/(rs*cs)];
den=[1,-trace(A),det(A)]; g=dt/cc;
core=ambient+filter([0 g -A(2,2)*g],den,heat(:));
surface=ambient+filter([0 0 A(2,1)*g],den,heat(:));
end
