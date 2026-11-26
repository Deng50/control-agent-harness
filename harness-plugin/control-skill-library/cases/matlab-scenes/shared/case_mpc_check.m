function metrics=case_mpc_check(script,nx,augmented)
% 复制当前实际运行的函数，只增加可观察的矩阵输出。逐步递推独立于矩阵构造。
N=5;dt=.1;
if nx==1,A=-2;B=3;else,A=[0 1;-9 -4.2];B=[0;9];end
Ad=eye(nx)+dt*A;Bd=dt*B;
if augmented
    probe=regexprep(script,'function\s+delta_u_k\s*=\s*MPC_QP_1st\(', ...
        'function [delta_u_k,observed]=scene_matrix_probe(');
    probe=regexprep(probe,'(?m)^\s*delta_u_k = Delta_U_opt\(1:inputDim\);',...
        'delta_u_k=Delta_U_opt(1:inputDim);observed=Phi_y;');
else
    probe=regexprep(script,'function\s+controlInput\s*=\s*MPC_QP\(', ...
        'function [controlInput,observed]=scene_matrix_probe(');
    probe=regexprep(probe,'(?m)^\s*controlInput\s*=\s*deltaControl\(1:inputDim\);',...
        'controlInput=deltaControl(1:inputDim);observed=inputMatrix;');
end
assert(contains(probe,'observed=') && contains(probe,'=scene_matrix_probe('));
writelines(string(probe),'scene_matrix_probe.m',Encoding='UTF-8');clear scene_matrix_probe
if augmented
    [~,P]=scene_matrix_probe(N,1,.1,dt,A,B,1,0,1,0);
else
    [~,P]=scene_matrix_probe(N,eye(nx),.1,dt,A,B,B,zeros(nx,1),zeros(nx,1),ones(nx,1),0);
end
% 对每列施加一个单位增量；此后保持已改变的控制值。
expected=zeros(N*nx,N);
for j=1:N
    x=zeros(nx,1);u=0;
    for k=1:N
        u=u+(k==j);x=Ad*x+Bd*u;expected((k-1)*nx+(1:nx),j)=x;
    end
end
err=abs(P-expected); [i,j]=find(err>1e-11);
% 用可移植的 Skill 入口另查混合增量、非零初态、上一动作及扰动。
model=struct('A',Ad,'B',Bd,'C',eye(nx),'D',zeros(nx,1),'E',Bd);
mixed=verify_increment_matrix(model,P,ones(nx,1)*.1,.2,[0;1;-.25;.1;.3],(.01:.01:.05)',1e-11,0);
metrics=struct('horizon',N,'max_abs',max(err,[],'all'),'failed_increment_columns',unique(j)',...
    'first_prediction_step',[],'first_channel',[],'matrix_value',[],'recurrence_value',[]);
metrics.mixed_increments=mixed;
if ~isempty(i)
    row=min(i);col=j(find(i==row,1));metrics.first_prediction_step=ceil(row/nx);
    metrics.first_channel=mod(row-1,nx)+1;metrics.matrix_value=P(row,col);metrics.recurrence_value=expected(row,col);
end
end
