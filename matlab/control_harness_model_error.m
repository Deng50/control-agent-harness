function [recoverable, diagnostic] = control_harness_model_error(stage, err)
% Eligibility for the Host's independent same-session quiescence check.
% The caller sets stage only around the corresponding synchronous operation.
diagnostic = struct('identifier',err.identifier,'message',err.message,'causes',{{}});
isCompile = ismember(stage,{'compile','simulate'});
leaf = isempty(err.cause);
recoverable = leaf && ((strcmp(stage,'configure_solver') && ...
    strcmp(err.identifier,'Simulink:Commands:BadSetParamValue')) || (isCompile && ...
    (ismember(err.identifier,{'Simulink:Engine:InvalidSolver','Simulink:blocks:MuxInValidPortWidths', ...
        'Simulink:blocks:DemuxInValidPortWidths'}) || ...
    ~isempty(regexp(err.identifier,'^Simulink:Engine:PortDimsMismatch[0-9]+$','once')))));
if ~leaf
    allKnown = true;
    for k = 1:numel(err.cause)
        [known, child] = control_harness_model_error(stage,err.cause{k});
        diagnostic.causes{end+1} = child;
        allKnown = allKnown && known;
    end
    recoverable = isCompile && strcmp(err.identifier,'MATLAB:MException:MultipleErrors') && allKnown;
end
end
