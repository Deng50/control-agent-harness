function control_harness_calculation(code, nonce, variables, functions)
% Evaluate the original numerical statements in the MCP evaluator's base
% workspace. Keep the receipt and exception in a separate function frame.
% Eligibility is limited to Host-recognized synchronous numerical programs.
receipt = struct('nonce',nonce,'matlab_pid',feature('getpid'), ...
    'session_id',getenv('CONTROL_HARNESS_SESSION_ID'),'helper',mfilename('fullpath'), ...
    'certifiable',true,'status','returned','identifier','','message','');
existing = evalin('base','whos');
mathClasses = {'double','single','logical','char','string','ss','tf','zpk', ...
    'uint8','uint16','uint32','uint64','int8','int16','int32','int64'};
for k = 1:numel(existing)
    if ismember(existing(k).name,functions) || ...
            (ismember(existing(k).name,variables) && ~ismember(existing(k).class,mathClasses))
        receipt.certifiable = false;
    end
end
for k = 1:numel(functions)
    % Class methods may be resolved only after dispatch on a numeric LTI
    % object. Detect workspace files shadowing the standard MATLAB functions.
    source = which(functions{k});
    if ~isempty(source) && ~startsWith(source,'built-in (') && ...
            ~startsWith(source,[matlabroot filesep],'IgnoreCase',ispc)
        receipt.certifiable = false;
    end
end
try
    evalin('base',code);
catch exception
    receipt.status = 'failed';
    receipt.identifier = exception.identifier;
    receipt.message = exception.message;
end
fprintf('\nCONTROL_HARNESS_CALC_%s:%s\n',nonce,jsonencode(receipt));
if receipt.certifiable
    setappdata(groot,'ControlHarnessCompletedCall',nonce);
end
end
