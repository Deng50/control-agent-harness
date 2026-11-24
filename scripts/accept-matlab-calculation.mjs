/** Keyless R2024b regression for the original MPC calculation in session #166. */
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { openLayer, project } from './matlab-layer-harness.mjs';

const fixtureArgument = process.argv[2];
assert.ok(fixtureArgument, 'Pass a dedicated test fixture containing its private session and workspace.');
const fixture = resolve(fixtureArgument);
const workspace = join(fixture, 'workspace');
const reportPath = join(fixture, 'calculation.json');
const original = String.raw`m=1; c=0.8; k=2; dt=0.1;
Ac=[0 1; -k/m -c/m]; Bc=[0; 1/m]; Cc=[1 0];
sys=d2d(ss(Ac,Bc,Cc,0),dt,'zoh');
Ad=sys.A, Bd=sys.B,
ev=eig(Ad); abs(ev),
wn=sqrt(k/m); zeta=c/(2*sqrt(k*m));
fprintf('wn=%.4f rad/s zeta=%.4f wd=%.4f Td=%.3f s tau=%.3f s\n',wn,zeta,wn*sqrt(1-zeta^2),2*pi/(wn*sqrt(1-zeta^2)),1/(zeta*wn));
fprintf('static gain p/u = %.4f m/N ; u needed for 0.5 m = %.4f N ; for 2 N -> %.4f m\n',1/k,0.5/k,2/k);
Bd*1000`;
const corrected = original.replace('sys=d2d(', 'sys=c2d(')
  + String.raw`
fprintf('AD_JSON=%s\n',jsonencode(Ad)); fprintf('BD_JSON=%s\n',jsonencode(Bd));`;
const layer = await openLayer({fixture, workspace, managed:true});
const report = {workspace, release:layer.marker.matlab_release, pid:layer.marker.matlab_pid,
  session_id:layer.marker.session_id, stages:[]};
const native = 'mcp__matlab__evaluate_matlab_code';
async function evaluate(stage, code) {
  const result = await layer.call(native, {code, project_path:workspace});
  const state = JSON.parse(await readFile(join(layer.resource, 'resource.json'), 'utf8'));
  report.stages.push({stage, code, result, broker_completion:state.last_completion, quarantine:state.quarantine});
  assert.equal(state.owner, null);
  assert.equal(state.quarantine, null);
  assert.equal(state.last_completion.evidence.matlab_pid, report.pid);
  assert.equal(state.last_completion.evidence.session_id, report.session_id);
  return result;
}
function verifiedFailure(result) {
  assert.equal(result.isError, true);
  const receipt = result.meta?.matlab_calculation;
  assert.ok(receipt?.identifier, 'Missing MATLAB exception identifier');
  assert.ok(receipt.message.length > 0);
  assert.equal(receipt.matlab_pid, report.pid);
  assert.equal(receipt.session_verified, true);
  assert.equal(receipt.retryable, true);
  return receipt;
}
try {
  assert.equal(report.release, '2024b');
  const invalid = verifiedFailure(await evaluate('invalid-d2d', original));
  assert.match(invalid.message, /d2d/i);
  const success = await evaluate('corrected-c2d', corrected);
  assert.equal(success.isError, false, success.error?.message);
  const lines = success.value.content.filter(item => item.type === 'text').flatMap(item => item.text.split(/\r?\n/));
  function matrix(label) {
    const data = lines.filter(value => value.startsWith(label + '='));
    assert.equal(data.length, 1, label + ' missing from original MATLAB output');
    return JSON.parse(data[0].slice(label.length+1));
  }
  const Ad = matrix('AD_JSON'), Bd = matrix('BD_JSON');
  const scipy = [
    'import json, numpy as np', 'from scipy.linalg import expm',
    'zoh=expm(0.1*np.array([[0,1,0],[-2,-0.8,1],[0,0,0]]))',
    'print(json.dumps([zoh[:2,:2].tolist(),zoh[:2,2].tolist()]))'
  ].join('\n');
  const expected = JSON.parse(execFileSync(join(project, '.venv/Scripts/python.exe'), ['-c', scipy], {encoding:'utf8'}));
  const actual = [Ad, Bd].flat(Infinity);
  assert.equal(actual.length, expected.flat(Infinity).length);
  const maxAbsError = Math.max(...expected.flat(Infinity).map((value,i) => Math.abs(actual[i]-value)));
  report.comparison = {Ad, Bd, scipy_augmented_expm:expected, max_abs_error:maxAbsError, atol:1e-12};
  assert.ok(maxAbsError <= report.comparison.atol);
  const dimension = verifiedFailure(await evaluate('incompatible-matrix-product', 'X=[1 2;3 4]; Y=X*ones(3,1);'));
  assert.notEqual(dimension.identifier, invalid.identifier);
  report.passed = true;
} catch (error) {
  report.passed = false;
  report.error = error.message;
  throw error;
} finally {
  await writeFile(reportPath, JSON.stringify(report,null,2));
  console.log(JSON.stringify({passed:report.passed, reportPath}));
  await layer.close();
}
