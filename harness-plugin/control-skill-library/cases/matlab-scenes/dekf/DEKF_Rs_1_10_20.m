% 双 EKF 同步估计两节点温度和表面热阻。核心温度观测只参加最终误差评分。
d=case_thermal_input(1);load tempParameters.mat; Tf=8; n=3500;
Ts=d.ts(1:n)';Tc=d.tc(1:n)'; results=zeros(3,3);
for j=1:3
    initials=[20 10 1]; Rs0=initials(j); theta_est=zeros(1,n);theta_est(1)=Rs0;
    x=zeros(2,n);x(:,1)=[Ts(1);Ts(1)]-Tf+3;P=1e-8*eye(2);Pt=.02;
    sens=zeros(2,1);H=[0 1];q=1e-6*eye(2);rv=10;rt=.5;
    for k=2:n
        rs=theta_est(k-1); A=[1-1/(Rc*Cc) 1/(Rc*Cc);1/(Rc*Cs) 1-1/(Rc*Cs)-1/(rs*Cs)];
        % 可逆热用上一拍估计核心绝对温度；状态敏感度随离散动力学传播。
        heat=d.irrev(k-1)+d.ent(k-1)*(x(1,k-1)+Tf+273.15);
        xn=A*x(:,k-1)+[heat/Cc;0];Pp=A*P*A'+q;innov=Ts(k)-Tf-H*xn;
        G=Pp*H'/(H*Pp*H'+rv);x(:,k)=xn+G*innov; M=eye(2)-G*H;P=M*Pp*M'+G*rv*G';
        sens=(A+[d.ent(k-1)/Cc 0;0 0])*sens+[0;x(2,k-1)/(Cs*rs^2)];
        h=H*sens;prior=Pt+.001;gain=prior*h/(h^2*prior+rt);
        theta_est(k)=max(.01,rs+gain*innov); Pt=(1-gain*h)^2*prior+gain^2*rt;sens=M*sens;
    end
    Ts_pre=x(2,:)+Tf;Tc_pre=x(1,:)+Tf;
    history(j)=struct('Rs',theta_est,'surface',Ts_pre,'core',Tc_pre); %#ok<SAGROW>
    results(j,:)=[theta_est(end),sqrt(mean((Ts_pre-Ts).^2)),sqrt(mean((Tc_pre-Tc).^2))];
end
assert(all(isfinite(results),'all'));scene_metrics=struct('initial_Rs_K_W',initials,'final_Rs_surface_core_rmse',results);
