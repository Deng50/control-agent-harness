function report=verify_increment_matrix(model,P,x0,last,du,d,atol,rtol)
% 核对离散 MPC 的控制增量矩阵。P 按预测步堆叠，每步内按输出/输入通道排列。
% x(k+1)=A*x(k)+B*u(k)+E*d(k)，本拍末读数 y=C*x(k+1)+D*u(k)。
% du、d 每行一拍；last 是前一实际动作。返回首次失败的预测步、通道及两种值。
if nargin<7,atol=1e-10;end
if nargin<8,rtol=1e-9;end
A=model.A;B=model.B;C=model.C;D=model.D;E=model.E; x=x0(:); u=last(:);
[nx,nu]=size(B); ny=size(C,1); N=size(du,1);u0=u;freeState=x;
assert(N>0 && isequal(size(A),[nx nx]) && size(C,2)==nx && isequal(size(D),[ny nu]));
assert(isequal(size(du),[N nu]) && isequal(size(P),[N*ny N*nu]) && numel(x)==nx && numel(u)==nu);
assert(size(E,1)==nx && isequal(size(d),[N size(E,2)]) && atol>=0 && rtol>=0);
for arg={A,B,C,D,E,P,x,u,du,d,atol,rtol}
    assert(isreal(arg{1}) && all(isfinite(arg{1}),'all'),'参数中有复数或非有限值');
end
free=zeros(N,ny); oracle=zeros(N,ny);
for k=1:N
    % 自由响应沿用上一实际动作，真实比较路径逐拍累加本次指定的增量。
    freeState=A*freeState+B*u0+E*d(k,:)';free(k,:)=(C*freeState+D*u0)';
    u=u+du(k,:)';x=A*x+B*u+E*d(k,:)';oracle(k,:)=(C*x+D*u)';
end
lifted=free+reshape(P*reshape(du.',[],1),ny,N).';error=lifted-oracle;
assert(all(isfinite([lifted oracle error]),'all'),'预测递推出现非有限值');
bad=abs(error)>atol+rtol*abs(oracle);
[channel,step]=find(bad.',1);
report=struct('passed',isempty(step),'matrix',lifted,'recursion',oracle,'max_abs',max(abs(error),[],'all'),...
    'first_step',step,'first_channel',channel,'matrix_value',[],'recursion_value',[],'signed_error',[]);
if ~isempty(step)
    report.matrix_value=lifted(step,channel);report.recursion_value=oracle(step,channel);
    report.signed_error=error(step,channel);
end
end
