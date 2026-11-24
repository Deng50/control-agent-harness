function control_harness_evaluate(code, completionNonce)
% The native evaluator uses the base workspace. Only this function owns the
% completion marker; a queued request cannot publish it before its body runs.
try
    evalin('base',code);
catch exception
    setappdata(groot,'ControlHarnessCompletedCall',struct('nonce',completionNonce, ...
        'outcome','failed','identifier',exception.identifier,'message',exception.message));
    rethrow(exception);
end
setappdata(groot,'ControlHarnessCompletedCall',struct('nonce',completionNonce, ...
    'outcome','returned','identifier','','message',''));
end
