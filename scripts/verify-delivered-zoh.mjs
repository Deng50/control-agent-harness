/** Start a dedicated cold R2024b verifier on byte-identical copies.
 * Arguments: saved slx, original reviewed spring upload, physical_parameters.json, saved x0 JSON.
 * The optional --load-original opens the named SLX directly while all verification output stays in a fresh directory.
 */
import assert from 'node:assert/strict';
import {createHash,randomUUID} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {basename,join,resolve} from 'node:path';
import {copyFile,mkdir,readFile,writeFile} from 'node:fs/promises';
const project=resolve(import.meta.dirname,'..');
const [slx,upload,parameters,initialJson,...options] = process.argv.slice(2);
assert.ok(initialJson,'Specify .slx, original Python, parameter JSON and explicit saved x0.');
assert.ok(!options.length || options.length===1 && options[0]==='--load-original','Unknown verifier argument.');
const folder=join(project,'test-output','delivered-zoh-verify-'+randomUUID());
await mkdir(folder);
const hash=async p=>createHash('sha256').update(await readFile(p)).digest('hex');
const copyDirectory=options.length ? join(folder,'byte-snapshot') : folder;
if(options.length) await mkdir(copyDirectory);
const target=join(copyDirectory,basename(slx)),source=join(folder,'plant_numpy.py');
const sourceHash=await hash(upload),slxHash=await hash(slx);
assert.equal(sourceHash,'c4f054989498454dd25c2f4f78668fcba29bad2231e51fba57d7ae737dbea7b9');
await copyFile(slx,target);await copyFile(upload,source);
assert.equal(await hash(target),slxHash);assert.equal(await hash(source),sourceHash);
const p=JSON.parse(await readFile(parameters,'utf8')),dt=p.sample_time;
const req={sampleTime:dt,parameters:{mass:p.mass,damping:p.damping,stiffness:p.stiffness},absTolerance:1e-12,
  trajectories:[JSON.parse(initialJson),[0.2,-0.11],[-0.35,0.24]].map((initialCondition,i)=>({initialCondition,
    controls:Array.from({length:31},(_,k)=>[k<3 ? 0.4*i : Math.cos(k/5+i)]),
    disturbances:Array.from({length:31},(_,k)=>[(i+k)%3===0?0.18:-0.11])}))};
await writeFile(join(folder,'request.json'),JSON.stringify(req));
const selectedModel=options.length ? resolve(slx) : target;
await writeFile(join(folder,'verification-request.json'),JSON.stringify({model:selectedModel,original:resolve(slx),slx_sha256:slxHash}));
execFileSync(join(project,'.venv/Scripts/python.exe'),['-I','-B',join(project,'scripts/plant-step-reference.py'),
  '--source',source,'--request',join(folder,'request.json'),'--output',join(folder,'python-reference.json')],{cwd:folder,windowsHide:true});
const quote=s=>"'"+s.replaceAll("'","''")+"'";
const verifier=join(project,'tests/fixtures/matlab-scope');
const matlabExe=process.env.CONTROL_MATLAB_EXECUTABLE??'matlab.exe';
await mkdir(join(folder,'prefs'));
execFileSync(matlabExe,['-wait','-sd',folder,'-batch','addpath('+quote(verifier)+'); control_replay_delivered_plant('+quote(folder)+');'],
  {cwd:folder,env:{...process.env,MATLAB_PREFDIR:join(folder,'prefs')},timeout:180000,windowsHide:true,stdio:'inherit'});
assert.equal(await hash(slx),slxHash);assert.equal(await hash(target),slxHash);
const result=JSON.parse(await readFile(join(folder,'independent-simulation.json'),'utf8'));
assert.equal(result.status,'passed');
assert.equal(result.model,selectedModel);
console.log(JSON.stringify({report:join(folder,'independent-simulation.json'),slx_sha256:slxHash,
  sampling_points:result.trajectories.reduce((n,t)=>n+t.time.length,0),max_abs_error:result.max_abs_error}));
