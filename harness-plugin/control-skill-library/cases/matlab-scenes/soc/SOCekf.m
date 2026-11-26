% 状态顺序为 SOC、快极化电压、慢极化电压；检查放电符号与输入向量时标。
%Cn=2*3600;%电池容量，单位As
Cn=38*3600;
delta_t=1;%采样时间
%rp1=-9.015*socc.^6+20.64*socc.^5-15*socc.^4+2.196*socc.^3+1.747*socc.^2-0.7141*socc+0.09635;
%rp2=38.17*socc.^6-106.5*socc.^5+115.7*socc.^4-61.8*socc.^3+16.93*socc.^2-2.263*socc+0.1285;
%cp1=-6.037e+06*socc.^6+1.697e+07*socc.^5-1.81e+07*socc.^4+9.097e+06*socc.^3-2.189e+06*socc.^2+2.367e+05*socc-7963;
%cp2=1.82e+08*socc.^6-4.243e+08*socc.^5+3.83e+08*socc.^4-1.75e+08*socc.^3+4.456e+07*socc.^2-5.862e+06*socc+3.205e+05;
ro=0.00125;%电池内阻
%rp1=5.18E-04;cp1=1.97E+05;
%rp2=6.40E-04;cp2=1.61E+06;%RC环节的内阻和电容值，单位欧姆和法,拟合数据1
rp1=0.000206;cp1=89.958;
rp2=0.0161;cp2=85.17;%RC环节的内阻和电容值，单位欧姆和法，拟合数据2
%rp1=1000000000000;cp1=10000000000000000000000;
%rp2=100000000000000000000000;cp2=100000000000000000000000;
%ro=10000000000000000000000;
%nita=2200/2228;%放电系数
load('DSTI.mat','dstI'); rng(876,'twister'); i=dstI;
i(1)=0;%放电电流矩阵
fn=@(x) 22.19*x^5-60.16*x^4+61.11*x^3-28.73*x^2+6.311*x+2.738;%拟合soc-ocv表达式
%fn=@(x) -189.5*x^6+696.7*x^5-1007*x^4+727.8*x^3-275.4*x^2+51.63*x+9.562;%拟合soc-ocv表达式
%fn=@(x) (-5.678e-14)*(x.^7)+(1.581e-11)*(x.^6)+ (-2.663e-10)*(x.^5)+...
%(-3.208e-7)*(x.^4) + (4.088e-05)*(x.^3) + (-2.004e-3)*(x.^2)+...
%(4.85e-2)*x + 3.093;%拟合soc-ocv表达式
 
g=@(s) polyval(polyder([22.19 -60.16 61.11 -28.73 6.311 2.738]),s);
 
A=[1 0 0;0 rp1*cp1/(delta_t+rp1*cp1) 0;0 0 rp2*cp2/(delta_t+rp2*cp2)];
B=[-delta_t/Cn;delta_t*rp1/(delta_t+rp1*cp1);delta_t*rp2/(delta_t+rp2*cp2)];%A，B矩阵
t=(0:numel(i)-1)*delta_t;%仿真序列时间
N=length(t);
size=[3,N];%数组大小
Q=4e-9*eye(3);%状态方程中的误差
R=1e-2; %观测方程中的误差
 
X=zeros(size);%状态变量
Soc=zeros(1,N);%soc
Up1=zeros(1,N);
Up2=zeros(1,N);%RC回路的电压
E=zeros(1,N);%电池电压
Uoc=zeros(1,N);%电池端电压
X(:,1)=[1;0;0];%状态变量初值
 
for k=2:N
    X(:,k)=A*X(:,k-1)+B*i(k)+sqrtm(Q)*randn(3,1);
end
for k=1:N
    Soc(k)=X(1,k);
    Up1(k)=X(2,k);
    Up2(k)=X(3,k);
    E(k)=fn(Soc(k));
    Uoc(k)=E(k)-i(k)*ro-Up1(k)-Up2(k)+sqrt(R)*randn(1);
end%求取仿真所用的真值
%Uoc=a; 
Xekf=zeros(size);%EKF下的状态变量
P0=eye(3);%协方差矩阵初值
Xekf(:,1)=[1;0;0];%EKF状态变量初值，可更改
 
for k=2:N
    Xn=A*Xekf(:,k-1)+B*i(k);%计算先验估计
    P1=A*P0*A'+Q;%更新协方差矩阵
    % 滤波仅使用先验状态和带噪端电压；真值只用于离线打分。
    Zm=fn(Xn(1))-Xn(2)-Xn(3)-i(k)*ro;
    H=[g(Xn(1)) -1 -1];%雅克比矩阵计算
    K=P1*H'/(H*P1*H'+R);%计算卡尔曼增益
    Xekf(:,k)=Xn+K*(Uoc(k)-Zm);%更新状态
    M=eye(3)-K*H; P0=M*P1*M'+K*R*K';%更新协方差方程
end
 
Socekf=zeros(1,N);%Ekf下的soc
Uocekf=zeros(1,N);%Ekf下的端电压
Eekf=zeros(1,N);%EKF下的电池电压
 
for k=1:N
    Socekf(k)=Xekf(1,k);
    Eekf(k)=fn(Socekf(k));
    Uocekf(k)=fn(Socekf(k))-i(k)*ro-Xekf(2,k)-Xekf(3,k);
end%计算三个值
 
%依次绘制先验数据，后验数据和实验数据做比较
%这里绘制的是soc-ocv曲线kkkk
figure;
ans=Soc-Socekf;
ans=abs(ans);
plot(ans);
figure;
%plot(X(1,:),Uoc,'b');
plot(Uoc,'b');
hold on;
%plot(Xekf(1,:),Uocekf,'g');
plot(Uocekf,'g');
hold on;
figure;
plot(Socekf);
hold on;
plot(Soc);
