"""Self-contained CPU/float64 MLP export; inference needs only NumPy and Torch.

Inputs are already assembled raw supervised features. Export never guesses a
physical state dimension or reconstructs history windows from feature width.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import platform
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contracts import ContractError, DataInterface


def validate_network_config(config):
    """The same finite architecture definition is copied into every bundle."""
    if not isinstance(config, dict):
        raise ValueError("network configuration must be an object")
    allowed = {"schema_version", "architecture", "input_dim", "output_dim", "hidden_dims", "activation", "dtype"}
    if set(config) - allowed:
        raise ValueError("unknown network configuration fields")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("network schema_version must be 1")
    if config.get("architecture") != "mlp" or config.get("dtype") != "float64":
        raise ValueError("only float64 MLP networks are supported")
    if config.get("activation") not in {"tanh", "relu", "silu"}:
        raise ValueError("activation must be tanh, relu or silu")
    widths = config.get("hidden_dims")
    if not isinstance(widths, (list, tuple)) or len(widths) > 16:
        raise ValueError("hidden_dims must declare at most 16 layers")
    for name, value in [("input_dim", config.get("input_dim")), ("output_dim", config.get("output_dim"))] + [("hidden_dim", value) for value in widths]:
        if type(value) is not int or not 1 <= value <= 65536:
            raise ValueError(f"{name} must be an integer in 1..65536")
    dimensions = [config["input_dim"], *widths, config["output_dim"]]
    if sum((left + 1) * right for left, right in zip(dimensions, dimensions[1:])) > 5_000_000:
        raise ValueError("network exceeds the five-million-parameter export budget")
    return {"schema_version": 1, "architecture": "mlp", "input_dim": config["input_dim"],
            "output_dim": config["output_dim"], "hidden_dims": list(widths),
            "activation": config["activation"], "dtype": "float64"}


def build_supervised_network(network_config):
    """Linear/activation pairs plus a linear output; state keys are 0/2/4/... ."""
    config = validate_network_config(network_config)
    dimensions = [config["input_dim"], *config["hidden_dims"], config["output_dim"]]
    activation = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[config["activation"]]
    layers = []
    for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(nn.Linear(left, right, dtype=torch.float64))
        if index < len(dimensions) - 2:
            layers.append(activation())
    return nn.Sequential(*layers).double()


_INFERENCE_BODY = r'''
def _finite_vector(value, width, label, positive=False):
    if isinstance(value, list) and any(isinstance(item, bool) for item in value):
        raise ValueError(label + ' cannot contain booleans')
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.shape != (width,):
        raise ValueError(label + ' has the wrong numeric width')
    result = raw.astype(np.float64)
    if not np.isfinite(result).all() or (positive and not (result > 0).all()):
        raise ValueError(label + ' must be finite' + (' and positive' if positive else ''))
    return result

def _scaler_arrays(scaler, input_dim, output_dim):
    if not isinstance(scaler, dict) or scaler.get('method') not in {'standard','minmax','none'} or scaler.get('fit_split') != 'train':
        raise ValueError('scaler must declare training-only standard/minmax/none fitting')
    result = {}
    for key, width in [('X',input_dim),('y',output_dim)]:
        group = scaler.get(key)
        if not isinstance(group,dict): raise ValueError('missing scaler group ' + key)
        location = _finite_vector(group.get('location'),width,key+' location')
        scale = _finite_vector(group.get('scale'),width,key+' scale',positive=True)
        if scaler['method']=='none' and (np.any(location != 0) or np.any(scale != 1)):
            raise ValueError('none scaler must be the identity')
        result[key] = (location,scale)
    return result

def _raw_features(value, width):
    def boolean(item):
        return isinstance(item,bool) or (isinstance(item,(list,tuple)) and any(boolean(child) for child in item))
    raw = np.asarray(value)
    if boolean(value) or raw.dtype.kind not in 'iuf' or raw.ndim != 2 or raw.shape[1] != width or not len(raw):
        raise ValueError('features must be a nonempty numeric [samples,input_dim] matrix, without bool/string values')
    if raw.size > 5_000_000: raise ValueError('inference input exceeds the five-million-value batch budget')
    value = raw.astype(np.float64)
    if not np.isfinite(value).all(): raise ValueError('raw features must be finite')
    return value

class SupervisedPredictor:
    def __init__(self, directory):
        self.root = Path(directory).resolve(strict=True)
        self.manifest = json.loads((self.root/'bundle_manifest.json').read_text(encoding='utf-8'))
        expected = {'model.pt','network.json','scaler.json','data_interface.json','features.json','training_config.json','inference.py','predict.py','README.md'}
        if self.manifest.get('schema_version') != 1 or self.manifest.get('kind') != 'control-supervised-mlp-bundle-v1' or set(self.manifest.get('files',{})) != expected:
            raise ValueError('unsupported/incomplete supervised inference bundle manifest')
        for name,digest in self.manifest['files'].items():
            file = (self.root/name).resolve(strict=True)
            if file.parent != self.root or len(digest)!=64 or file.stat().st_size > 64*1024*1024:
                raise ValueError('invalid bundle member')
            hasher=hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''): hasher.update(block)
            if hasher.hexdigest()!=digest: raise ValueError('bundle file hash mismatch: '+name)
        read = lambda name: json.loads((self.root/name).read_text(encoding='utf-8'))
        self.network_config = validate_network_config(read('network.json'))
        self.features = read('features.json')
        self.interface = read('data_interface.json')
        self.scaler = read('scaler.json')
        self.input_dim,self.output_dim = self.network_config['input_dim'],self.network_config['output_dim']
        if self.interface.get('interface_kind')!='tabular_supervised' or any(key in self.interface for key in ('nx','nu','ny','nd','states')):
            raise ValueError('bundle is a supervised feature interface, not an inferred plant')
        if self.interface.get('input_dim')!=self.input_dim or self.interface.get('output_dim')!=self.output_dim:
            raise ValueError('interface/network dimensions differ')
        if (len(self.features.get('feature_order',[]))!=self.input_dim or len(self.features.get('feature_labels',[]))!=self.input_dim
                or len(self.features.get('target_order',[]))!=self.output_dim or len(self.features.get('output_units',[]))!=self.output_dim):
            raise ValueError('feature/target metadata widths differ')
        if len(set(self.features['feature_labels']))!=self.input_dim:
            raise ValueError('feature labels are ambiguous')
        if self.manifest.get('feature_labels')!=self.features['feature_labels'] or self.manifest.get('target_labels')!=self.features['target_labels']:
            raise ValueError('manifest feature/target order differs')
        self.transforms = _scaler_arrays(self.scaler,self.input_dim,self.output_dim)
        self.network = build_supervised_network(self.network_config)
        state = torch.load(self.root/'model.pt',map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or any(not isinstance(value,torch.Tensor) or value.dtype != torch.float64 or not torch.isfinite(value).all() for value in state.values()):
            raise ValueError('model.pt must contain only finite float64 tensor state_dict entries')
        try: self.network.load_state_dict(state,strict=True)
        except RuntimeError as error: raise ValueError('network/state_dict keys or shapes differ') from error
        self.network.eval()

    def predict(self, raw_features, feature_labels=None):
        if feature_labels is not None:
            if not isinstance(feature_labels,(list,tuple)) or list(feature_labels)!=self.features['feature_labels']:
                raise ValueError('provided feature labels/order differ from the exported assembled-feature contract')
        values = _raw_features(raw_features,self.input_dim)
        layer_widths = [self.input_dim,*self.network_config['hidden_dims'],self.output_dim]
        if len(values)*max(layer_widths)>5_000_000:
            raise ValueError('inference intermediate/output exceeds the five-million-value layer batch budget')
        location,scale=self.transforms['X']
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'): scaled=(values-location)/scale
        if not np.isfinite(scaled).all(): raise ValueError('input scaling overflowed')
        with torch.no_grad(): normalized=self.network(torch.as_tensor(scaled,dtype=torch.float64)).cpu().numpy()
        location,scale=self.transforms['y']
        with np.errstate(over='ignore',invalid='ignore'): result=normalized*scale+location
        if result.shape!=(len(values),self.output_dim) or not np.isfinite(result).all(): raise ValueError('network predictions are nonfinite or dimensionally invalid')
        return result

def load_bundle(directory):
    return SupervisedPredictor(directory)
'''


_PREDICT_SOURCE = r'''"""Predict from already assembled raw features; never infer windows or plant nx."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='JSON matrix/object or NPZ with features/X and optional feature_labels')
    parser.add_argument('--output',required=True,help='JSON or NPZ predictions in declared target units')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('_standalone_supervised',root/'inference.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    predictor=module.load_bundle(root)
    source=Path(args.input)
    if source.stat().st_size>64*1024*1024: raise ValueError('input file exceeds the 64 MiB budget')
    labels=None
    if source.suffix.lower()=='.npz':
        import zipfile, math
        with zipfile.ZipFile(source) as archive:
            members=archive.infolist()
            if len(members)>32 or len({item.filename for item in members})!=len(members) or sum(item.file_size for item in members)>64*1024*1024:
                raise ValueError('expanded NPZ/member inventory exceeds input budget')
            for item in members:
                if not item.filename.endswith('.npy'): raise ValueError('NPZ must contain NPY arrays')
                with archive.open(item) as stream:
                    version=np.lib.format.read_magic(stream)
                    reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None: raise ValueError('unsupported NPY version')
                    shape,_,dtype=reader(stream)
                    count=math.prod(shape); payload=count*dtype.itemsize
                    if dtype.hasobject or count>5_000_000 or payload>64*1024*1024 or stream.tell()+payload!=item.file_size:
                        raise ValueError('NPY shape/dtype exceeds input budget or conflicts with stored bytes')
        with np.load(source,allow_pickle=False) as loaded:
            key='features' if 'features' in loaded.files else 'X'
            values=loaded[key]
            if 'feature_labels' in loaded.files: labels=loaded['feature_labels'].tolist()
    elif source.suffix.lower()=='.json':
        data=json.loads(source.read_text(encoding='utf-8'))
        if isinstance(data,dict): values=data['features']; labels=data.get('feature_labels')
        else: values=data
    else: raise ValueError('input must be JSON or NPZ')
    predictions=predictor.predict(values,labels)
    target=Path(args.output)
    if target.suffix.lower()=='.npz':
        np.savez_compressed(target,predictions=predictions,output_units=np.asarray(predictor.features['output_units']),target_labels=np.asarray(predictor.features['target_labels']))
    elif target.suffix.lower()=='.json':
        target.write_text(json.dumps({'predictions':predictions.tolist(),'output_units':predictor.features['output_units'],
            'target_labels':predictor.features['target_labels'],'input_semantics':'assembled_raw_features'},allow_nan=False,indent=2)+'\n',encoding='utf-8')
    else: raise ValueError('output must be JSON or NPZ')

if __name__=='__main__': main()
'''


def _finite_json(value, label):
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must contain finite JSON values") from error


def _feature_metadata(interface):
    signals = {signal["name"]: signal for signal in interface["signals"]}
    features = []
    for feature in interface["features"]:
        signal = signals[feature["signal"]]
        for history in reversed(range(feature["window"])):
            features.append({"signal": signal["name"], "lag": feature["lag"] + signal["delay_steps"] + history,
                             "unit": signal["unit"], "availability_delay_steps": signal["delay_steps"]})
    targets = [{"signal": target["signal"], "offset": target["offset"], "unit": signals[target["signal"]]["unit"]}
               for target in interface["targets"]]
    return {"schema_version": 1, "input_semantics": "assembled_raw_features", "feature_order": features, "target_order": targets,
            "feature_labels": [f"{item['signal']}@lag={item['lag']}" for item in features],
            "target_labels": [f"{item['signal']}@offset={item['offset']}" for item in targets],
            "input_units": [item["unit"] for item in features], "output_units": [item["unit"] for item in targets],
            "history_order": "feature declaration order; oldest to newest within each window; lag includes availability delay",
            "assembly": "Caller assembles rows at one anchor from the declared same-trajectory history. No automatic time/column/state inference."}


def _standalone_source():
    return ('"""Standalone supervised inference: NumPy + Torch, no project imports."""\n'
            'import hashlib\nimport json\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\n\n'
            + inspect.getsource(validate_network_config) + '\n\n' + inspect.getsource(build_supervised_network) + '\n' + _INFERENCE_BODY)


def _load_inference(directory):
    import sys
    path = Path(directory) / "inference.py"
    spec = importlib.util.spec_from_file_location(f"_supervised_bundle_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.load_bundle(directory)


def export_supervised(root, network, network_config, interface, scaler, config, *, feature_order=None, target_order=None):
    """Write a new bundle directory, preserving any unrelated stage files.

    The training owner should pass ``stage.root / 'bundle'``. This function
    writes a pure state_dict and never overwrites a training checkpoint.
    """
    settings = validate_network_config(network_config)
    interface = DataInterface.parse(interface.to_dict() if hasattr(interface, "to_dict") else interface).to_dict()
    if (settings["input_dim"], settings["output_dim"]) != (interface["input_dim"], interface["output_dim"]):
        raise ContractError("Network dimensions conflict with the explicit supervised feature/target interface")
    scaler, config = _finite_json(scaler, "scaler"), _finite_json(config, "resolved training config")
    features = _feature_metadata(interface)
    if feature_order is not None and _finite_json(feature_order, "feature order") != features["feature_order"]:
        raise ContractError("Provided feature_order differs from the declared windows/lags/delays")
    if target_order is not None and _finite_json(target_order, "target order") != features["target_order"]:
        raise ContractError("Provided target_order differs from the declared target offsets/units")
    with torch.random.fork_rng(devices=[]):
        reference = build_supervised_network(settings)
    if type(network) is not nn.Sequential or [(type(module), repr(module)) for module in network] != [(type(module), repr(module)) for module in reference]:
        raise ContractError("Export requires the shared Sequential factory architecture")
    state = network.state_dict()
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or not torch.isfinite(value).all() for value in state.values()):
        raise ContractError("Network weights must be finite float64 tensors")
    state = {key: value.detach().cpu().contiguous().clone() for key, value in state.items()}
    reference.load_state_dict(state, strict=True)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    names = {"model.pt", "network.json", "scaler.json", "data_interface.json", "features.json", "training_config.json",
             "inference.py", "predict.py", "README.md", "bundle_manifest.json"}
    if any((root / name).exists() for name in names):
        raise ContractError("Bundle filenames already exist; export into a new directory")
    temporary = root / f".export-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        torch.save(state, temporary / "model.pt")
        for name, value in (("network", settings), ("scaler", scaler), ("data_interface", interface),
                            ("features", features), ("training_config", config)):
            (temporary / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (temporary / "inference.py").write_text(_standalone_source(), encoding="utf-8")
        (temporary / "predict.py").write_text(_PREDICT_SOURCE, encoding="utf-8")
        readme = ("# Standalone supervised MLP\n\nDependencies: Python, numpy and torch. CPU float64; linear output layer. No repository, MATLAB, Engine or source-material files are required.\n\n"
                  "Run `python predict.py --input features.json --output predictions.json`. Input JSON is a two-dimensional raw feature matrix, or `{\"features\": [[...]], \"feature_labels\": [...]}`. NPZ uses `features` (or `X`) and optional string `feature_labels`; pickle arrays are refused.\n\n"
                  "Rows must already be assembled in the exact order below, in the declared raw input units. Dynamic windows and availability delays are not generated by the runner. Never assemble a window across trajectories. Without labels, a numeric matrix cannot reveal its column order; the caller owns that ordering. Supplied labels must match exactly. Feature width does not define physical nx/nu/ny.\n\n"
                  "Inference applies `(X-X.location)/X.scale`, then the network, then `normalized_y*y.scale+y.location`. Predictions are in the exported target units. This bundle does not claim stability, extrapolation accuracy, physical fidelity or deployment acceptance.\n\n"
                  f"Feature labels: `{json.dumps(features['feature_labels'], ensure_ascii=False)}`\n\nTarget labels: `{json.dumps(features['target_labels'], ensure_ascii=False)}`\n\n"
                  f"Output units: `{json.dumps(features['output_units'], ensure_ascii=False)}`\n\nSee features.json for ordered windows/lags/delays/units and data_interface.json for role/column mappings. training_config.json is audit metadata, not an inference dependency on original paths.\n")
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in temporary.iterdir()}
        manifest = {"schema_version": 1, "kind": "control-supervised-mlp-bundle-v1", "files": hashes,
                    "input_dim": settings["input_dim"], "output_dim": settings["output_dim"],
                    "feature_labels": features["feature_labels"], "target_labels": features["target_labels"],
                    "output_units": features["output_units"], "input_semantics": "assembled_raw_features",
                    "runtime": {"python": platform.python_version(), "numpy": str(np.__version__), "torch": str(torch.__version__),
                                "device": "cpu", "dtype": "float64"},
                    "resource_boundary": {"max_parameters": 5_000_000, "max_hidden_layers": 16,
                                          "max_layer_batch_values": 5_000_000,
                                          "scope": "one network/batch; not a limit on the overall engineering task"}}
        (temporary / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Load through the copied standalone implementation before publication.
        with torch.random.fork_rng(devices=[]):
            predictor = _load_inference(temporary)
            probe = np.zeros((2, settings["input_dim"]), dtype=np.float64)
            predictor.predict(probe, features["feature_labels"])
        for name in sorted(names - {"bundle_manifest.json"}):
            (temporary / name).rename(root / name)
        (temporary / "bundle_manifest.json").rename(root / "bundle_manifest.json")
        return {**manifest, "bundle_root": str(root), "manifest_path": str(root / "bundle_manifest.json")}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
