"""Execute a real MATLAB comparison and independently finalize its frozen tracking gate in Python."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from control_harness.artifacts import atomic_json, file_hash, read_json
from control_harness.external_pgac_evaluation import evaluate_external_pgac
from tests.test_external_pgac_evaluation import ExternalPgacEvaluationTests


def quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def main():
    executable = shutil.which('matlab')
    if not executable:
        raise SystemExit('MATLAB executable required')
    output = ROOT / 'test-output' / 'pgac-matlab-tracking'
    output.mkdir(parents=True, exist_ok=True)
    fixture = ExternalPgacEvaluationTests()
    objective = dict(q=1., r=.015, s=.01, primary_metric='tracking_rmse',
                     max_action_rms_relative_increase=.15, max_increment_rms_relative_increase=.25,
                     min_relative_improvement=.01, max_worst_case_relative_regression=.05)
    try:
        fixture.setUpClass()
        fixture.setUp()
        candidate, config, context, env, exchange, _ = fixture.native_prepare(objective)
        manifest = exchange / 'manifest.json'
        command = (f"addpath({quote(ROOT/'matlab')}); "
                   f"control_evaluate_pgac_batch({quote(manifest)},{quote(env['CONTROL_MATLAB_NONCE'])});")
        args = [executable, *(['-wait'] if os.name=='nt' else []), '-sd', str(exchange), '-batch', command]
        with (output/'matlab.log').open('w', encoding='utf-8') as log:
            subprocess.run(args, check=True, timeout=300, stdout=log, stderr=subprocess.STDOUT)
        raw = exchange / 'result.json'
        env.update(CONTROL_MATLAB_PHASE='finalize', CONTROL_MATLAB_RESULT_PATH=str(raw),
                   CONTROL_MATLAB_RESULT_SHA256=file_hash(raw))
        with patch.dict(os.environ,env):
            published = evaluate_external_pgac(fixture.workspace,candidate,config,context)
        root,_ = fixture.workspace.resolve(published)
        report = read_json(root/'report.json')
        native = read_json(raw)
        assert report['selected_version']=='imitation' and report['final_acceptance']['approved']
        evidence = dict(matlab_release=native['matlab_release'], selected=report['selected_version'],
                        objective=report['engineering_objective'], test_started_after_selection=native['test_started_after_selection'])
        atomic_json(output/'result.json',evidence)
        print(json.dumps(evidence))
    finally:
        fixture.doCleanups()
        ExternalPgacEvaluationTests.doClassCleanups()


if __name__=='__main__':
    main()
