% 按排气压力、进气压力、排温和阀位的列序计算高压 EGR 质量流量。
   
    clc;
    clear;
    Data = readmatrix('HP_EGR valve.xls');
Data=Data(all(isfinite(Data(:,1:5)),2),:);
% data reading    
    W_HP_EGR = Data(:,5);
    pem = Data(:,1);
    pim_af_thr = Data(:,2);
    Tem = Data(:,3);
    Uhpegr = Data(:,4);
 % inputs and outputs set up
    ydata=W_HP_EGR;% 
    xdata=[pem,pim_af_thr,Tem,Uhpegr];
    x0=[7.4e-3,2.2,287,0,1,0];%带拟参数赋初值，多尝试。
 % NLS algorithm config
    options=optimset('TolFun',1e-12,'TolX',1e-12,'MaxFunEvals',90000,'MaxIter',500, 'Algorithm','trust-region-reflective', 'display', 'iter');
    [x_HpEGR_MassFlow,~,~,~,~,~,jac]=lsqcurvefit(@HpEGR_MassFlow,x0,xdata,ydata,[0,2,286,0,1e-8,-inf],[inf,2.8,288,inf,inf,inf],options);%中括号为对待拟参数取值范围进行限定，前者为下限值，后者为上限值。
% results validation
    HpEGR_MassFlow_est=HpEGR_MassFlow(x_HpEGR_MassFlow,xdata);
    R_HpEGR_MassFlow=corrcoef(W_HP_EGR,HpEGR_MassFlow_est).^2;

    figure(2);
    plot(W_HP_EGR,'bs'); title('HP EGR flow rate model validation');
    hold on;
    plot(HpEGR_MassFlow_est,'-r')
    hold off;
    
    xlabel('采样点数 ');
    ylabel('EGR流量估计值 （kg/s）');
    grid;
    
    legend('实验测量值','模型估计值');
    
    figure(3);
    plot(W_HP_EGR,HpEGR_MassFlow_est,'rs'); title('HP EGR flow rate model validation');
    xlabel('measured HP EGR flow rate ');
    ylabel('estimated HP EGR flow rate');
    grid;
    
    
    
    
    
    Ahp_EGR_max=x_HpEGR_MassFlow(1);
    a1_HP_EGR=x_HpEGR_MassFlow(4);
    a2_HP_EGR=x_HpEGR_MassFlow(5);
    a3_HP_EGR=x_HpEGR_MassFlow(6);
    k1_HP_EGR=x_HpEGR_MassFlow(2);
    Re_HP_EGR=x_HpEGR_MassFlow(3);

    abcc=0:0.01:1;
    Ahp_EGR=(log(a1_HP_EGR.*abcc+a2_HP_EGR)+a3_HP_EGR);

% 隔行留出用于检查对未参与拟合记录的插值，不等同于独立发动机工况。
holdout=mod((1:numel(ydata))',5)==0;
xf=lsqcurvefit(@HpEGR_MassFlow,x0,xdata(~holdout,:),ydata(~holdout),...
    [0 2 286 0 1e-8 -inf],[inf 2.8 288 inf inf inf],optimset(options,'Display','off'));
pred=HpEGR_MassFlow(xf,xdata(holdout,:));
assert(all(pred>=0) && all(HpEGR_MassFlow_est>=0),'正向流量不应为负');
scene_metrics=struct('total',numel(ydata),'heldout_rows',nnz(holdout),'heldout_rmse_kg_s',sqrt(mean((pred-ydata(holdout)).^2)),...
    'min_flow_kg_s',min(HpEGR_MassFlow_est));
