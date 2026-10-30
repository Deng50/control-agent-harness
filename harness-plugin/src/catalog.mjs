/** Read-on-demand operation documentation. Numerical validation remains Python-owned. */
import { agents } from './roles.mjs';

const str = description => ({ type: 'string', description });
const num = (description, extra = {}) => ({ type: 'number', description, ...extra });
const integer = (description, minimum = 1) => ({ type: 'integer', description, minimum });
const array = (items, description) => ({ type: 'array', items, ...(description ? { description } : {}) });
const object = (properties = {}, required = [], extra = {}) => ({ type: 'object', properties, required, additionalProperties: false, ...extra });
const vector = array(num('Finite value in the declared variable unit'), 'Length must match the corresponding model interface');
const matrix = array(vector, 'Finite rectangular matrix; dimensions come from the model interface');
const artifact = str('An exact artifact_id in this workspace manifest; use inspect_workspace/inspect_artifact, never a guessed id');
const path = str('Absolute local file path readable by the Session sandbox');
const seed = integer('Deterministic random seed', 0);
const scenario = object({ name: str('Unique condition label'), initial_state: vector, reference: vector,
  disturbance: vector, steps: integer('Closed-loop simulation steps') }, ['name', 'initial_state', 'reference', 'disturbance', 'steps']);
const scenarios = array(scenario, 'Independent operating conditions; do not reuse held-out controller conditions');
const source = kinds => object({ kind: { type: 'string', enum: kinds }, description: str('Who/what produced the model or data and the engineering meaning') }, ['kind', 'description']);
const training = {
  epochs: integer('Target total epochs, including epochs already saved in a resumed checkpoint'),
  learning_rate: num('Positive optimizer learning rate', { exclusiveMinimum: 0 }),
  batch_size: integer('Mini-batch size'), seed,
  resume_checkpoint: path,
  resume_checkpoint_sha256: str('Optional expected SHA256; the saved checkpoint and sidecar must agree'),
};
const supervised = { ...training, patience: integer('Early-stopping patience in epochs'),
  hidden_dims: array(integer('Hidden layer width'), 'MLP hidden layer widths') };
const affineBalance = object({ coefficients: vector, target: num('Constant affine balance target') }, ['coefficients','target']);
const workflowConfig = object({ schema_version: { const: 1 }, name: str('Workflow label'),
  steps: array(object({ id: str('Unique stage id, starting with a letter, at most 64 characters'),
    operation: str('A stage operation from the operation catalog'),
    inputs: { type: 'object', additionalProperties: { oneOf: [str('Existing artifact id or absolute source path'), object({ stage: str('An earlier stage id') }, ['stage'])] } },
    config: { type: 'object', additionalProperties: true } }, ['id', 'operation']), '1–50 ordered stages; references can only name earlier stages'),
  resume_run: str('Recorded workflow_run UUID; definition and snapshotted external resources must match'),
});
const spec = (summary, inputs, config, prerequisites, outputs, example = {}) => ({ summary, inputsSchema: inputs,
  configSchema: config, prerequisites, outputs, example });

export const operations = {
  inspect_workspace: spec('Read the manifest and actual artifact ids/statuses.', object(), object(),
    ['workspace is the explicit absolute project path. Reading a new workspace initializes its manifest.'],
    'Workspace manifest: project_id, active, artifacts with kind/status/parents/metadata_hash.'),
  inspect_artifact: spec('Inspect an artifact, verified dependencies, metadata and report.', object({ artifact_id: artifact }, ['artifact_id']), object(),
    ['Rejected and stale outputs can be inspected; file and dependency integrity must still pass.'],
    '{artifact_id,kind,status,artifact_path,metadata,report,files}.', { inputs: { artifact_id: '<selected-id>' } }),
  read_artifact_file: spec('Read a bounded artifact text/JSON file or NPZ array summary.', object({ artifact_id: artifact, file: str('A filename listed by inspect_artifact, relative to that artifact') }, ['artifact_id', 'file']),
    object({ max_chars: integer('Maximum returned text characters, at most 32000') }), ['The requested file must be in the verified artifact inventory; no arbitrary file execution.'],
    '{artifact_id,kind,status,file_preview:{name,format,bytes,sha256,text?,arrays?,truncated?}}.', { inputs: { artifact_id: '<selected-id>', file: 'report.json' }, config: { max_chars: 12000 } }),
  prepare_handoff: spec('Derive the next operation from the selected artifact and its exact dependency chain.', object({ artifact_id: artifact }, ['artifact_id']), object(),
    ['Selected artifact must be verified and current. Missing dependencies are returned explicitly; unrelated active artifacts are never substituted.'],
    '{artifact_id,kind,status:ready|needs_input,handoff:{agent,operation,inputs,missing_inputs,missing_config,interface,config_suggestions},source_versions:[{id,metadata_hash}]}.', { inputs: { artifact_id: '<selected-id>' } }),
  recover_interrupted_stage: spec('Archive a verified unpublished training checkpoint left by an OS-confirmed dead worker.', object({ stage_id: str('Exact directory name under workspace/.staging, e.g. corrected_model-12hex') }, ['stage_id']), object(),
    ['The original worker must be confirmed dead by the OS. Live or unknown owners, published stages, changed dependencies and corrupt checkpoints are refused.', 'This archives one stage into failed/. It does not start training. Copy the returned checkpoint and checkpoint_sha256 into the original matching training request.'],
    '{status:archived,stage_id,checkpoint,checkpoint_sha256,algorithm,epoch,optimizer_updates,failed_path}.', { inputs: { stage_id: '<exact-unpublished-stage-id>' } }),
  import_artifact: spec('Import a verified artifact and its whole dependency closure from another workspace.', object({ artifact_path: path }, ['artifact_path']), object(),
    ['artifact_path is a directory containing artifact.json, inside a source workspace with manifest.json. All parents must verify; conflicting ids are rejected.'],
    'An independent immutable local artifact with the same id; full dependency closure is copied transactionally.', { inputs: { artifact_path: '<absolute-source-workspace>/artifacts/<id>' } }),
  inspect_model_source: spec('Read source structure and propose model metadata without executing uploaded code.', object({ model_path: path }, ['model_path']), object(),
    ['Use before registering unfamiliar JSON/Python/MATLAB/Simulink source. Candidate fields and diagnostics require user confirmation; source inspection is not a simulation adapter.'],
    'Detected format, dimensions/variables/equation candidates, ambiguities and registration readiness.', { inputs: { model_path: '<absolute-upload-path>' } }),
  register_model: spec('Register confirmed discrete equations or explicitly integrated continuous polynomial dynamics.', object({ model_path: path }, ['model_path']),
    object({ source: source(['user_nominal', 'reference_simulator', 'reconstructed_equations']), model_spec: { type: 'object', additionalProperties: true, description: 'Explicit user-confirmed override; see model_spec in sharedSchemas. Without this, model_path must be declarative JSON.' } }, ['source']),
    ['Confirm state/control/output/disturbance order, names, units, finite lower/upper bounds and positive sample_time. Use discrete_polynomial for next-state equations. continuous_polynomial additionally requires continuous_state_derivative source semantics, time_unit:s, derivative_units, and integration:{method:euler|rk4|zoh,substeps,input_hold:zero_order_hold}. Exact zoh is affine-only with substeps=1. Runtime artifacts always expose discrete next-state transitions.', 'A/B/C/E and polynomial powers must match dimensions. Unit labels do not prove coefficient dimensional consistency. model_path remains required with an override so the original source file hash is retained.'],
    'nominal_model, or reference_model when source.kind is reference_simulator.', { inputs: { model_path: '<absolute-model.json>' }, config: { source: { kind: 'user_nominal', description: 'User supplied discretized equations' } } }),
  register_python_model: spec('Explicitly execute, validate and freeze reviewed standalone PyTorch dynamics source.', object({ model_path: path }, ['model_path']),
    object({ source: source(['user_nominal', 'reference_simulator', 'reconstructed_equations']), model_spec: { type:'object',additionalProperties:true,description:'Reviewed override; otherwise read one literal MODEL_SPEC in the .py file. See python_model_spec in sharedSchemas.' } }, ['source']),
    ['Inspect and review source first. This operation executes Python under the native worker permissions; inspect_model_source remains read-only.', 'The standalone file declares python_torch_transition or python_torch_derivative, an explicit interface, entrypoint(x,u,d,parameters), JSON parameters, and exactly one C or output_entrypoint(x,parameters). Return finite CPU float64 Tensors supporting single and batch inputs. Derivatives require explicit Euler/RK4 integration and derivative units.', 'No relative/local-project imports or external resource loading; installed math/Torch imports are version-bound. Registration checks repeatability and autograd against finite differences at three declared points. This is not a proof of physical fidelity or arbitrary-domain stability.'],
    'nominal_model/reference_model with frozen source.py, model.json, python-validation.json, optional transition_adapter.json, and hashed dependencies. Residual/MPC/policy stages consume the executable source through the same load_model API.',
    {inputs:{model_path:'<absolute-reviewed-model.py>'},config:{source:{kind:'user_nominal',description:'Reviewed standalone tensor dynamics'}}}),
  import_transitions: spec('Validate and register trajectory-labelled state transitions from an NPZ file.', object({ nominal_model: artifact, data_path: path, reference_model: artifact }, ['nominal_model', 'data_path']),
    object({ source: source(['measured', 'reference_simulator']), seed }, ['source']),
    ['NPZ arrays x:[N,nx], u:[N,nu], d:[N,nd] (including [N,0] without disturbances), next_x:[N,nx], trajectory_id:[N] integer; all finite, at least five independent trajectory ids.', 'Simulation data additionally requires reference_model identifying the exact simulator. Preserve measurement/simulation attribution.'],
    'transition_dataset with trajectory-level train/validation/test split.', { inputs: { nominal_model: '<nominal-id>', data_path: '<absolute-transitions.npz>' }, config: { source: { kind: 'measured', description: 'User supplied experiment and sensor description' }, seed: 42 } }),
  simulate_transitions: spec('Generate transitions using an explicitly selected reference simulator.', object({ nominal_model: artifact, reference_model: artifact }, ['nominal_model', 'reference_model']),
    object({ trajectories: integer('At least five independent trajectories', 5), steps: integer('Maximum steps per trajectory'), seed }),
    ['Both models must have identical interface order, units and sample time. The reference must be independently supplied and explicitly labelled, not a fabricated plant truth.'],
    'transition_dataset labelled reference_simulator.', { inputs: { nominal_model: '<nominal-id>', reference_model: '<reference-id>' }, config: { trajectories: 20, steps: 24, seed: 42 } }),
  train_residual: spec('Train a supervised discrete residual model with bounds and optional affine physical balances.', object({ nominal_model: artifact, dataset: artifact }, ['nominal_model', 'dataset']),
    object({ ...supervised, residual_semantics: { const: 'next_state_residual_in_state_units' }, weight_decay: num('Nonnegative weight decay'), physics_weight: num('Nonnegative physical penalty weight'),
      minimum_improvement: num('Required held-out improvement fraction, in [0,1)', { minimum: 0, exclusiveMaximum: 1 }),
      affine_balance_constraints: array(affineBalance, 'Physical balance: dot(next_state, coefficients) = target; one coefficient per state') }),
    ['dataset must be a transition_dataset tied to this nominal model; preprocessing is fitted on training trajectories only.', 'This learns next-state residuals; it does not solve generic PDE residual PINNs. Checkpoint resume requires unchanged dependencies/effective settings except target epochs.'],
    'corrected_model plus checkpoint, preprocessing and held-out report; rejection does not replace active output.', { inputs: { nominal_model: '<nominal-id>', dataset: '<transition-dataset-id>' }, config: { epochs: 130, hidden_dims: [24,24], learning_rate: 0.006, physics_weight: 0.1, seed: 42 } }),
  identify_linear_model: spec('Fit linear state-transition matrices with rank and held-out error checks.', object({ nominal_model: artifact, dataset: artifact }, ['nominal_model', 'dataset']),
    object({ fit_bias: { type: 'boolean' }, allow_linear_approximation: { type: 'boolean' }, rank_relative_tolerance: num('Positive singular-value rank tolerance'), max_condition_number: num('Maximum regressor condition number'), max_validation_normalized_rmse: num('Validation acceptance limit'), max_test_normalized_rmse: num('Test acceptance limit'), max_spectral_radius: num('Optional maximum spectral radius') }),
    ['Independent training rows must exceed regression parameters and provide full-rank excitation. Nonlinear nominal equations require explicit allow_linear_approximation:true.'],
    'identified_model with fitted A/B/E/bias and rank, condition number, validation/test errors.', { inputs: { nominal_model: '<nominal-id>', dataset: '<transition-dataset-id>' }, config: { fit_bias: true } }),
  design_controller: spec('Design and tune constrained MPC, then verify independent conditions.', object({ model: artifact, evaluation_model: artifact }, ['model', 'evaluation_model']),
    object({ parameters: object({ horizon: integer('Prediction horizon'), q: num('Tracking weight'), r: num('Control effort weight'), s: num('Control change weight'), input_rate_bounds: matrix, max_iterations: integer('Solver iteration budget'), solver_tolerance: num('Positive solver tolerance') }),
      parameter_space: { type: 'object', additionalProperties: object({ lower: num('Lower search bound'), upper: num('Upper search bound') }, ['lower','upper']) },
      budget: integer('Candidate simulation budget'), seed, minimum_improvement: num('Required verification improvement fraction'),
      tuning_scenarios: scenarios, verification_scenarios: scenarios, test_scenarios: scenarios }, ['tuning_scenarios','verification_scenarios','test_scenarios']),
    ['Models must be interface-compatible. Supply three explicit disjoint condition sets; renamed duplicates are rejected. Input/output/state constraints come from the declared interface; rate bounds need explicit engineering meaning.', 'The search evaluates actual candidate closed-loop simulations. Inspect final feasibility and held-out score; a tuned candidate is not automatically better.'],
    'controller containing chosen parameters, optimizer history and independent verification/test report.', { inputs: { model: '<model-id>', evaluation_model: '<reference-id>' }, config: { parameters: { horizon: 5, q: 1, r: 0.01, s: 0.01 }, budget: 8, tuning_scenarios: ['<scenario A>'], verification_scenarios: ['<different scenario B>'], test_scenarios: ['<different scenario C>'] } }),
  validate_estimator: spec('Validate KF/EKF state estimation in closed-loop simulation.', object({ model: artifact, controller: artifact, evaluation_model: artifact }, ['model','controller','evaluation_model']),
    object({ method: { type: 'string', enum: ['auto','KF','EKF'] }, measurement_output_indices: array(integer('Zero-based measured output index',0)), Q: matrix, R: matrix, P0: matrix,
      initial_estimate: vector, observability_relative_tolerance: num('Positive numerical tolerance'), sensor_noise_covariance: matrix,
      max_normalized_estimation_rmse: num('Acceptance limit'), seed, verification_scenarios: scenarios, test_scenarios: scenarios }, ['verification_scenarios','test_scenarios']),
    ['Use measured output indices, not hidden true-state access. Declare sensor covariance and initial estimate. KF requires linear dynamics; EKF uses Jacobians. Verification/test conditions must be independent and sufficiently long for observability checks.'],
    'estimator artifact with configuration, observability diagnostics and held-out estimation/control reports.', { inputs: { model: '<model-id>', controller: '<controller-id>', evaluation_model: '<reference-id>' }, config: { method: 'auto', verification_scenarios: ['<scenario A>'], test_scenarios: ['<different scenario B>'] } }),
  collect_expert: spec('Simulate verified MPC demonstrations under independent collection conditions.', object({ controller: artifact, evaluation_model: artifact }, ['controller','evaluation_model']),
    object({ scenarios, seed }, ['scenarios']), ['Provide at least five independent conditions; exclude controller verification/test conditions. Controller and evaluation model must have matching interfaces.'],
    'expert_dataset with trajectory splits and exact controller/model dependencies.', { inputs: { controller: '<controller-id>', evaluation_model: '<reference-id>' }, config: { scenarios: ['<at least five explicit independent scenarios>'], seed: 42 } }),
  train_imitation: spec('Train an MLP, RNN or LSTM policy from expert trajectories.', object({ dataset: artifact }, ['dataset']),
    object({ ...supervised, architecture: { type: 'string', enum: ['mlp','rnn','lstm'] }, dropout: num('Dropout in [0,1)'), hidden_size: integer('Recurrent hidden width'), num_layers: integer('Recurrent layer count'), preview_len: integer('Disturbance preview length'), disturbance_forecast: { const: 'declared_constant' } }),
    ['dataset must be an expert_dataset. Network input/output widths are derived from the interface; do not invent dimensions. RNN/LSTM requires nd>0 and disturbance_forecast:declared_constant with explicit preview_len.', 'The policy inherits controller input/rate bounds. Recurrent input is a declared disturbance preview, not an arbitrary history window.'],
    'imitation_policy with network, preprocessing, latest checkpoint and held-out comparison.', { inputs: { dataset: '<expert-dataset-id>' }, config: { architecture: 'mlp', hidden_dims: [48,32], epochs: 160, learning_rate: 0.004, seed: 42 } }),
  optimize_policy: spec('Continue from a neural policy through differentiable model rollouts.', object({ policy: artifact, model: artifact }, ['policy','model']),
    object({ ...training, horizon: integer('Differentiable rollout horizon'), imitation_weight: num('Nonnegative imitation regularizer weight'), constraint_weight: num('Nonnegative state/output constraint penalty') }),
    ['The policy, prediction model and dataset/controller ancestry must agree. Training uses the shared loaded checkpoint and optimizer; held-out conditions determine acceptance.', 'This is model-based differentiable optimization, not PPO/SAC. Initial previous action and input/rate projection are preserved.'],
    'optimized_policy plus expert/imitation/optimized held-out report and checkpoint; improvement is not guaranteed.', { inputs: { policy: '<imitation-policy-id>', model: '<prediction-model-id>' }, config: { epochs: 20, learning_rate: 0.0005, horizon: 4, imitation_weight: 0.2, seed: 42 } }),
  plan_workflow: spec('Validate explicit stage dependencies and known interfaces without training.', object({ workflow_path: path }), workflowConfig,
    ['Choose inputs.workflow_path OR inline config.schema_version:1 and config.steps, not both. File paths in a workflow file resolve relative to that file; inline paths must be absolute.', 'A stage input is an existing artifact id/path or {stage:"earlier-id"}. No cycles or forward references.'],
    'Plan with per-stage checks and errors. Inspect status before running.', { config: { schema_version: 1, steps: [{ id: 'imported', operation: 'import_artifact', inputs: { artifact_path: '<absolute-artifact-path>' } }] } }),
  run_workflow: spec('Run a validated explicit stage chain as one native job.', object({ workflow_path: path }), workflowConfig,
    ['Call plan_workflow first with the same definition. This is deterministic engineering-stage composition inside the existing Harness agent loop.', 'resume_run must come from a recorded workflow_run. Definition/source/dependency hashes must match; only independently verified completed stages are reused. Training checkpoint resume is a separate setting on a training stage.'],
    '{workflow_run,status,stages,...}; artifacts remain immutable and stage failure stops dependent work.', { inputs: { workflow_path: '<absolute-workflow.json>' }, config: {} }),
  demo: spec('Run the explicitly simulated complete example.', object(), object({ case: { type: 'string', enum: ['linear','nonlinear'] } }),
    ['Use a new empty workspace. All data are from declared demonstration simulators and are not experimental evidence.'],
    'Full model/controller/policy example artifact ids and stage statuses.', { config: { case: 'linear' } }),
};

/** Detailed schema lookup is read-only and never opens a workspace or starts a job. */
export function operationHelp({ operation, agent } = {}) {
  if (agent !== undefined && !Object.hasOwn(agents, agent)) throw new Error(`Unknown control agent: ${agent}`);
  if (operation !== undefined) {
    if (!Object.hasOwn(operations, operation)) throw new Error(`Unknown control operation: ${operation}`);
    const entry = operations[operation];
    return { operation, agents: Object.entries(agents).filter(([,spec]) => spec.operations.includes(operation)).map(([name]) => name), ...entry,
      request: { operation, workspace: '<absolute-user-selected-workspace>', inputs: {}, config: {}, ...entry.example },
      sharedSchemas: { scenario, model_spec: { representation: 'discrete_polynomial', interface: { schema_version: 1, name: '<model-name>', time_semantics: 'discrete_state_transition', sample_time: '<positive seconds>', states: '[{name,unit,lower,upper},...]', controls: '[{name,unit,lower,upper},...]', outputs: '[{name,unit,lower,upper},...]', disturbances: '[{name,unit,lower,upper},...]' }, A: '[nx,nx]', B: '[nx,nu]', E: '[nx,nd]', C: '[ny,nx]', terms: '[{state_index,coefficient,powers:[nx+nu+nd nonnegative integer exponents]}]' },
        python_model_spec:{representation:'python_torch_transition | python_torch_derivative',interface:'Same explicit ordered interface; continuous_state_derivative source for derivatives, discrete_state_transition otherwise',entrypoint:'top_level_function(x,u,d,parameters)',parameters:'finite JSON object',output_map:'exactly one C:[ny,nx] or output_entrypoint:top_level_function(x,parameters)',integration:'for derivatives only: {method:euler|rk4,substeps:positive integer,input_hold:zero_order_hold},time_unit:s,derivative_units:[state_unit/s,...]'} },
      jobSemantics: 'The execution tools return jobId/attemptId/workspace. Follow job_output until completion and inspect the result status before handing off. Tool help itself creates no job.' };
  }
  return { agents: Object.fromEntries(Object.entries(agents).map(([name,spec]) => [name, { preset: spec.preset ?? null, description: spec.description, operations: spec.operations }])),
    operations: Object.entries(operations).filter(([name]) => !agent || agents[agent].operations.includes(name)).map(([operation,spec]) => ({ operation, summary: spec.summary })),
    usage: 'Call control_help({operation}) for exact required inputs, schemas, prerequisites and an example. Role operations and control_workflow all use the same native Harness jobs.' };
}
