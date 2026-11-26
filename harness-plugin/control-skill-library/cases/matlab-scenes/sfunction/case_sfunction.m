function err=case_sfunction()
% 以已知连续状态矩阵构造可核对的台架；执行 S-function 的 Update 与 Outputs 回调。
x_model=[0 1 -9 -4.2 0 9]; assignin('base','x_model',x_model);
mdl='scene_sfun_host'; new_system(mdl);
add_block('simulink/Sources/Constant',mdl+"/u",'Value','1');
add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function',mdl+"/plant",'FunctionName','Plant_Model');
add_block('simulink/Sinks/To Workspace',mdl+"/save",'VariableName','pos','SaveFormat','Timeseries');
add_line(mdl,'u/1','plant/1');add_line(mdl,'plant/1','save/1');
in=Simulink.SimulationInput(mdl);in=in.setModelParameter('StopTime','1','Solver','FixedStepDiscrete','FixedStep','.01','ReturnWorkspaceOutputs','on');
out=sim(in); ref=zeros(2,numel(out.pos.Time)); A=[0 1;-9 -4.2]; B=[0;9];
for k=2:size(ref,2),ref(:,k)=ref(:,k-1)+.01*(A*ref(:,k-1)+B);end
err=max(abs(ref(1,:)'-out.pos.Data)); assert(err<1e-10);
end
