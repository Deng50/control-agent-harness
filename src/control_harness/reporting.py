"""Read-only bounded chart data and complete, hash-checked portable bundles.

This module never loads a network, executes uploaded code, recomputes metrics,
or changes artifact activation. Download ZIPs contain only a published bundle.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import stat
import uuid
import zipfile

import numpy as np

from .artifacts import atomic_json,read_json
from .contracts import ContractError

MAX_NPZ_BYTES=64*1024*1024
MAX_ARRAY_VALUES=8_000_000
MAX_PREVIEW_VALUES=12_000
MAX_ARRAY_PREVIEW_VALUES=5_000
MAX_BUNDLE_BYTES=48*1024*1024  # Below the pinned native /api/file default50MiB.
_SHA=re.compile(r'^[0-9a-f]{64}$')
_FILES=re.compile(r'^(?:test_predictions|one_step_predictions|rollout_predictions|transitions|partial-transitions|expert|partial|validation_arrays|(?:validation-|test-)?(?:expert|imitation|candidate)-rollout)\.npz$')
_ARRAYS=set('prediction truth row_index raw_action projected_action expert_action source_row actual_previous_u predicted_state true_state predicted_output true_output trajectory_id condition_index time input_time prediction_time time_s initial_temperature_K power_W predicted_temperature_K reference_temperature_K normalized_physics_residual initial_condition_index predicted_initial_temperature_K normalized_initial_residual x u d next_x y next_y plant_y observation reference previous_u split_id reference_source_row reference_x reference_u reference_d reference_next_x reference_y reference_next_y rollout_source_row rollout_state rollout_output'.split())
_GROUPS={'trajectory_id','condition_index','split_id'}
_MANIFESTS=('policy_manifest.json','pinn_manifest.json','dynamics_manifest.json','bundle_manifest.json')
_EXPORT_KINDS={'supervised_model','learned_model','controller_design','controller','corrected_model','imitation_policy','optimized_policy','external_imitation_policy','external_pgac_candidate','external_optimized_policy','pinn_model'}
_PREFERRED='time prediction_time input_time time_s trajectory_id condition_index source_row row_index prediction truth predicted_temperature_K reference_temperature_K normalized_physics_residual predicted_state true_state predicted_output true_output projected_action expert_action raw_action u previous_u plant_y next_y reference x next_x y'.split()


def _relative(value):
    if not isinstance(value,str) or not value or '\\' in value or ':' in value or '\x00' in value:
        raise ContractError('Select an ordinary relative artifact member')
    parts=PurePosixPath(value)
    if parts.is_absolute() or any(v in {'','..','.'} for v in value.split('/')):
        raise ContractError('Artifact paths cannot escape their verified directory')
    return parts


def _no_alias(root,path):
    root=Path(root);path=Path(path)
    try:parts=path.relative_to(root).parts
    except ValueError as error:raise ContractError('Report path escaped its workspace') from error
    current=root
    for part in ('',*parts):
        if part:current=current/part
        info=current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
            raise ContractError('Report/download does not follow symlinks or Windows reparse points')


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _artifact(workspace,artifact_id):
    """Reject aliases in the full parent closure before generic resolve reads it."""
    visited=set();manifest=workspace.manifest
    def inspect(item):
        if item in visited:return
        if len(visited)>=256:raise ContractError('One report dependency inspection exceeds its256-artifact budget')
        visited.add(item);entry=manifest.get('artifacts',{}).get(item)
        if not isinstance(entry,dict):raise ContractError('Selected report artifact is not indexed')
        root=workspace.path/str(_relative(entry['path']));_no_alias(workspace.path,root)
        meta_path=root/'artifact.json';_no_alias(root,meta_path)
        if meta_path.stat().st_size>8*1024*1024:raise ContractError('Artifact metadata exceeds its report budget')
        payload=meta_path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=entry.get('metadata_file_sha256'):raise ContractError('Artifact metadata version changed')
        meta=json.loads(payload.decode('utf-8'));files=meta.get('files')
        if not isinstance(files,dict) or len(files)>4096:raise ContractError('Unsupported artifact inventory for reporting')
        for name in files:_no_alias(root,root/str(_relative(name)))
        for parent in meta.get('parents',[]):inspect(parent['id'])
    inspect(artifact_id)
    return workspace.resolve(artifact_id,allow_stale=True,allow_rejected=True)


def _archive_headers(payload):
    headers={};total_values=0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries=archive.infolist();names=[v.filename for v in entries]
        if len(entries)>64 or len(names)!=len(set(names)) or sum(v.file_size for v in entries)>MAX_NPZ_BYTES:
            raise ContractError('NPZ inventory/expanded bytes exceed the bounded report budget')
        for entry in entries:
            if '/' in entry.filename or '\\' in entry.filename or not entry.filename.endswith('.npy'):
                raise ContractError('NPZ report accepts flat NPY members only')
            with archive.open(entry) as stream:
                version=np.lib.format.read_magic(stream)
                reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                if reader is None:raise ContractError('Unsupported NPY header')
                shape,fortran,dtype=reader(stream)
                size=math.prod(shape);total_values+=size
                if (dtype.hasobject or dtype.kind not in 'fiubUS' or len(shape)>4 or total_values>MAX_ARRAY_VALUES
                        or stream.tell()+size*dtype.itemsize!=entry.file_size):
                    raise ContractError('NPZ header/type/shape exceeds its real bounded numeric budget')
                headers[entry.filename[:-4]]={'shape':shape,'dtype':dtype,'fortran':fortran}
    return headers


def _json_values(value):
    if value.dtype.kind in 'US':return value.tolist()
    if value.dtype.kind=='b':return value.tolist()
    if np.isfinite(value).all():return value.tolist()
    data=value.astype(object);data[~np.isfinite(value)]=None
    return data.tolist()


def read_report_arrays(workspace,artifact_id,name,selection):
    """Return persisted rows only. ``filter`` isolates a trajectory/condition."""
    root,metadata=_artifact(workspace,artifact_id)
    if not isinstance(name,str) or not _FILES.fullmatch(name) or name not in metadata['files']:
        raise ContractError('Select a registered supported report NPZ file')
    if not isinstance(selection,dict) or set(selection)-{'keys','max_rows','filter'}:raise ContractError('Unknown array report selection')
    keys=selection.get('keys');maximum=selection.get('max_rows',240)
    if keys is not None and (not isinstance(keys,list) or not 1<=len(keys)<=12 or len(set(keys))!=len(keys) or any(not isinstance(k,str) or k not in _ARRAYS for k in keys)):
        raise ContractError('Select1..12 unique supported report variables')
    if type(maximum)is not int or not 1<=maximum<=500:raise ContractError('max_rows must be an integer in1..500')
    path=root/name
    if path.stat().st_size>MAX_NPZ_BYTES:raise ContractError('Report NPZ exceeds its64MiB stored-byte limit')
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=metadata['files'][name]:raise ContractError('Report NPZ version changed')
    try:headers=_archive_headers(payload)
    except (ValueError,zipfile.BadZipFile,OverflowError,EOFError) as error:raise ContractError('Malformed bounded report NPZ') from error
    if keys is None:
        candidates=[k for k in _PREFERRED if k in headers and len(headers[k]['shape']) in {1,2}]
        if not candidates:raise ContractError('No supported aligned chart variables in this report file')
        count=headers[candidates[0]]['shape'][0]
        keys=[k for k in candidates if headers[k]['shape'][0]==count][:12]
    if any(k not in headers or len(headers[k]['shape']) not in {1,2} for k in keys):raise ContractError('Selected report variable is missing or not a row matrix')
    count=headers[keys[0]]['shape'][0]
    if any(headers[k]['shape'][0]!=count for k in keys):raise ContractError('Chart variables must share their actual source row axis')
    filtering=selection.get('filter')
    if filtering is not None:
        if not isinstance(filtering,dict) or set(filtering)!={'key','value'} or filtering['key'] not in _GROUPS or filtering['key'] not in headers:
            raise ContractError('Filter only by an existing trajectory/condition/split identifier')
        value=filtering['value']
        if type(value)not in {str,int,float} or isinstance(value,float) and not math.isfinite(value) or isinstance(value,str) and len(value)>128:
            raise ContractError('Filter identifier must be a finite number or short string')
    with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
        rows=np.arange(count,dtype=np.int64);groups={}
        for key in _GROUPS&headers.keys():
            if headers[key]['shape']!=(count,):continue
            values=arrays[key]
            if values.dtype.kind not in 'fiuUS' or values.dtype.kind in 'fiu' and not np.isfinite(values).all():continue
            unique=np.unique(values);groups[key]={'values':_json_values(unique[:64]),'total_groups':len(unique),'truncated':len(unique)>64}
        if filtering is not None:
            field=arrays[filtering['key']]
            if field.shape!=(count,) or field.dtype.kind not in 'fiuUS':raise ContractError('Filter is not an aligned identifier vector')
            if (field.dtype.kind in 'US')!=(type(filtering['value'])is str):raise ContractError('Filter identifier type differs from actual data')
            rows=rows[field==filtering['value']]
        matched=len(rows)
        if matched>maximum:rows=rows[np.linspace(0,matched-1,maximum,dtype=int)]
        scalars={k:len(rows)*math.prod(headers[k]['shape'][1:]) for k in keys}
        if any(v>MAX_ARRAY_PREVIEW_VALUES for v in scalars.values()) or sum(scalars.values())>MAX_PREVIEW_VALUES:
            raise ContractError('Requested chart slice exceeds per-array/total element budget')
        values={key:_json_values(arrays[key][rows]) for key in keys}
        summaries=[{'name':key,'shape':list(headers[key]['shape']),'dtype':str(headers[key]['dtype'])} for key in keys]
    return {'artifact_id':artifact_id,'kind':metadata['kind'],'status':workspace.manifest['artifacts'][artifact_id]['status'],
        'file_preview':{'name':name,'format':'array_archive','bytes':len(payload),'sha256':metadata['files'][name],'arrays':summaries,'truncated':len(rows)<matched,
            'array_rows':{'values':values,'source_rows':rows.tolist(),'total_rows':count,'matched_rows':matched,'returned_rows':len(rows),
                'truncated':len(rows)<matched,'filter':filtering,'groups':groups,'selection':'evenly_spaced_including_endpoints','metrics_recomputed':False}}}


def _bundle(workspace,artifact_id,maximum):
    if type(maximum)is not int or not 1<=maximum<=MAX_BUNDLE_BYTES:raise ContractError('Bundle byte limit must be explicit and no more than48MiB')
    root,metadata=_artifact(workspace,artifact_id);bundle=root/'bundle'
    if metadata['kind'] not in _EXPORT_KINDS or not bundle.is_dir():raise ContractError('This artifact has no published standalone model/controller bundle; data files are not model exports')
    _no_alias(root,bundle)
    choices=[name for name in _MANIFESTS if 'bundle/'+name in metadata['files']]
    # A dynamics bundle may carry an inner supervised manifest. Its outer
    # dynamics manifest owns the full published runtime inventory.
    preferred=next((name for name in _MANIFESTS if name in choices),None)
    if preferred is None:raise ContractError('No supported published standalone bundle manifest; data files are not model exports')
    manifest=read_json(bundle/preferred);files=manifest.get('files')
    if not isinstance(files,dict) or not files or len(files)>2048:raise ContractError('Unsupported independent bundle file inventory')
    expected=dict(files);expected[preferred]=metadata['files']['bundle/'+preferred]
    for name,digest in expected.items():
        _relative(name)
        if not isinstance(digest,str) or not _SHA.fullmatch(digest) or metadata['files'].get('bundle/'+name)!=digest:
            raise ContractError('Bundle manifest differs from the bound artifact inventory')
    actual=set();pending=[bundle]
    directories={parent.as_posix() for name in expected for parent in PurePosixPath(name).parents if str(parent)!='.'}
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path=Path(entry.path);_no_alias(bundle,path);name=path.relative_to(bundle).as_posix()
                if entry.is_dir(follow_symlinks=False):
                    if name not in directories:raise ContractError('Standalone bundle contains an unregistered extra directory')
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    if name not in expected:raise ContractError('Standalone bundle contains an unregistered extra member')
                    actual.add(name)
                else:raise ContractError('Standalone bundle accepts ordinary registered files only')
    if actual!=set(expected):raise ContractError('Standalone bundle has missing or unregistered extra members')
    total=sum((bundle/name).stat().st_size for name in expected)
    if total>maximum:raise ContractError('Complete standalone bundle exceeds the native download byte limit; it will not be truncated')
    return root,metadata,bundle,preferred,expected,total


def prepare_bundle_archive(workspace,artifact_id,*,max_bytes=MAX_BUNDLE_BYTES):
    """Create a full self-contained ZIP for the native authenticated /api/file."""
    root,metadata,bundle,manifest_name,files,total=_bundle(workspace,artifact_id,max_bytes)
    cache=workspace.path/'.report-downloads'
    if not cache.exists():cache.mkdir()
    _no_alias(workspace.path,cache)
    metadata_sha=_sha(root/'artifact.json');archive_id=hashlib.sha256((artifact_id+'\0'+metadata_sha).encode()).hexdigest()
    filename=artifact_id+'-bundle.zip';target=cache/(archive_id+'.zip');sidecar=cache/(archive_id+'.json')
    expected_entries={'bundle/'+name:digest for name,digest in files.items()}
    download_manifest={'schema_version':1,'kind':'verified-portable-bundle-download','artifact_id':artifact_id,'artifact_metadata_sha256':metadata_sha,
        'bundle_manifest':manifest_name,'files':files,'verification':'SHA256 content integrity; not a digital signature','source':'actual published independent bundle only'}
    manifest_bytes=(json.dumps(download_manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')
    expected_entries['DOWNLOAD_MANIFEST.json']=hashlib.sha256(manifest_bytes).hexdigest()
    if not target.exists():
        temporary=cache/('.'+archive_id+'-'+uuid.uuid4().hex+'.tmp')
        try:
            with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                for name,digest in sorted(files.items()):
                    path=bundle/name;_no_alias(bundle,path)
                    with path.open('rb') as stream:
                        before=os.fstat(stream.fileno());hasher=hashlib.sha256()
                        with archive.open('bundle/'+name,'w') as output:
                            for chunk in iter(lambda:stream.read(1024*1024),b''):hasher.update(chunk);output.write(chunk)
                        after=os.fstat(stream.fileno())
                    if hasher.hexdigest()!=digest or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
                        raise ContractError('Published bundle changed during complete download preparation')
                archive.writestr('DOWNLOAD_MANIFEST.json',manifest_bytes)
            if temporary.stat().st_size>max_bytes:raise ContractError('Complete ZIP exceeds the native file limit; download is not truncated')
            temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    _no_alias(cache,target)
    # Cached downloads are never trusted merely by filename or sidecar.
    with zipfile.ZipFile(target) as archive:
        entries=archive.infolist()
        if len(entries)!=len(expected_entries) or len({v.filename for v in entries})!=len(entries) or {v.filename for v in entries}!=set(expected_entries):
            raise ContractError('Cached download ZIP inventory changed')
        sizes={'bundle/'+name:(bundle/name).stat().st_size for name in files};sizes['DOWNLOAD_MANIFEST.json']=len(manifest_bytes)
        if any(entry.file_size!=sizes[entry.filename] for entry in entries):raise ContractError('Cached ZIP expanded size differs from the complete published inventory')
        for name,digest in expected_entries.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise ContractError('Cached download ZIP member hash changed')
    if target.stat().st_size>max_bytes:raise ContractError('Complete ZIP exceeds the native file byte limit')
    descriptor={'schema_version':1,'artifact_id':artifact_id,'kind':metadata['kind'],'status':workspace.manifest['artifacts'][artifact_id]['status'],
        'archive_id':archive_id,'filename':filename,'host_path':str(target),'bytes':target.stat().st_size,'sha256':_sha(target),
        'bundle_manifest':manifest_name,'bundle_files':len(files),'uncompressed_bundle_bytes':total,'artifact_metadata_sha256':metadata_sha,
        'download_route':'api/file','complete':True,'truncated':False,'member_sha256':expected_entries,'max_bytes':max_bytes}
    atomic_json(sidecar,descriptor)
    return descriptor
