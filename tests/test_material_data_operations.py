"""New entry points with real files, immutable artifacts and explicit fixture authority."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from openpyxl import Workbook

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.case_plans import FIXTURE_KIND, PendingPlanConfirmation, propose_plan
from control_harness.case_requirements import confirmation_requirements
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext


class MaterialDataOperations(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / 'test-output/v2-development/20261117T002006/S03'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='material-data-operation-', dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = Workspace(self.root / 'case')
        atomic_json(self.workspace.path / '.case-plans/test-fixture.json', {
            'kind': FIXTURE_KIND, 'workspace': str(self.workspace.path)})
        self.context = OperationContext.test_fixture(self.workspace)
        self.headers = ['time', 'trip', 'current', 'current', 'temperature', 'y1', 'y2']
        self.rows = [[i, f'trip-{trip}', trip + i / 5, trip - i / 10, 20 + trip + i,
                      2 * (trip + i / 5) + .1 * (20 + trip + i), -(trip - i / 10) + .2 * (20 + trip + i)]
                     for trip in range(6) for i in range(12)]
        self.path = self.root / 'measured-format-synthetic.csv'
        with self.path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(self.headers); writer.writerows(self.rows)
        names = ['requested_current', 'measured_current', 'ambient_temperature', 'response1', 'response2']
        roles = ['control', 'output', 'disturbance', 'output', 'output']
        units = ['A', 'A', 'degC', '1', '1']
        self.interface = {'schema_version': 1, 'interface_kind': 'tabular_supervised', 'task': 'static_regression',
            'sample_time': 1., 'time_column': self.column(0), 'trip_column': self.column(1),
            'signals': [{'name': name, 'role': role, 'unit': unit, 'column': self.column(index + 2),
                         'delay_steps': 0, 'observable': True} for index, (name, role, unit) in enumerate(zip(names, roles, units))],
            'features': [{'signal': name, 'window': 1, 'lag': 0} for name in names[:3]],
            'targets': [{'signal': name, 'offset': 0} for name in names[3:]]}

    def column(self, index, sheet=None):
        return {'index': index, 'header': self.headers[index], 'worksheet': sheet}

    def operation(self, name, inputs=None, config=None, context=None):
        return run_operation({'operation': name, 'workspace': str(self.workspace.path),
            'inputs': inputs or {}, 'config': config or {}}, context or self.context)

    def material(self, path=None, **config):
        return self.operation('register_material', {'local_path': str(path or self.path)}, config)['artifact_id']

    def request(self, material, interface=None):
        return {'operation': 'prepare_supervised_dataset', 'workspace': str(self.workspace.path),
            'inputs': {'material': material}, 'config': {'interface': interface or self.interface,
                'split': {'mode': 'trip', 'train': .6, 'validation': .2, 'test': .2},
                'source': {'kind': 'synthetic', 'description': 'Explicit synthetic test_fixture in measured-data file format'}}}

    def approve_fixture(self, request):
        refs = {}
        for required in confirmation_requirements(self.workspace, request):
            plan = propose_plan(self.workspace, required['scope'], required['expected_payload'], required['bindings'])
            refs[required['scope']] = {key: plan[key] for key in ('plan_id', 'version_hash')}
        request['config']['plan_refs'] = refs

    def test_path_upload_share_registration_bytes_and_keep_independent_materials(self):
        path_id = self.material(source_kind='path')
        upload = self.root / 'attachment.bin'; upload.write_bytes(self.path.read_bytes())
        upload_id = self.material(upload, source_kind='upload', original_name=self.path.name,
            source_metadata={'attachment_id': 'test_fixture-upload'})
        p, pm = self.workspace.resolve(path_id); u, um = self.workspace.resolve(upload_id)
        self.assertEqual(read_json(p / 'material.json')['content_hash'], read_json(u / 'material.json')['content_hash'])
        self.assertEqual(pm['interface']['interface_kind'], 'material')
        self.assertNotIn('states', pm['interface'])
        self.assertNotIn('material', self.workspace.manifest['active'])
        self.assertEqual(um['source']['source_kind'], 'upload')
        self.assertEqual(self.path.read_bytes(), (p / 'source' / self.path.name).read_bytes())

    def test_pending_and_json_flags_cannot_prepare_numeric_arrays(self):
        request = self.request(self.material())
        before = set(self.workspace.manifest['artifacts'])
        with self.assertRaises(PendingPlanConfirmation): run_operation(request, OperationContext())
        self.approve_fixture(request)
        request['config'].update(confirmed=True, confirmation_source='test_fixture')
        with self.assertRaises(PendingPlanConfirmation): run_operation(request, OperationContext())
        self.assertEqual(before, set(self.workspace.manifest['artifacts']))

    def test_confirmed_csv_produces_data_only_artifact_and_train_only_scaler(self):
        request = self.request(self.material()); self.approve_fixture(request)
        result = run_operation(request, self.context)
        root, meta = self.workspace.resolve(result['artifact_id'])
        self.assertEqual(meta['interface']['input_dim'], 3); self.assertEqual(meta['interface']['output_dim'], 2)
        self.assertNotIn('states', meta['interface'])
        with np.load(root / 'supervised.npz', allow_pickle=False) as data:
            self.assertEqual(data['X'].shape, (72, 3)); self.assertEqual(data['y'].shape, (72, 2))
            self.assertEqual(set(data['split_id'].tolist()), {0, 1, 2})
            np.testing.assert_allclose(data['X_scaled'][data['split_id'] == 0].mean(axis=0), 0, atol=1e-12)
        self.assertEqual(read_json(root / 'report.json')['training'], 'not_run')
        handoff = self.operation('prepare_handoff', {'artifact_id': result['artifact_id']})
        self.assertEqual(handoff['status'], 'needs_input'); self.assertEqual(handoff['handoff']['operation'], 'train_supervised')
        other = Workspace(self.root / 'imported'); copied = other.import_artifact(root)
        self.assertEqual(other.resolve(copied)[1], meta)

    def test_changed_semantics_or_material_invalidates_the_corresponding_confirmation(self):
        request = self.request(self.material()); self.approve_fixture(request)
        changed = copy.deepcopy(request); changed['config']['interface']['signals'][0]['unit'] = 'mA'
        with self.assertRaises(PendingPlanConfirmation): run_operation(changed, self.context)
        changed = copy.deepcopy(request); changed['inputs']['material'] = self.material()
        with self.assertRaises(PendingPlanConfirmation): run_operation(changed, self.context)

    def test_real_xlsx_uses_explicit_sheet_and_duplicate_header_positions(self):
        path = self.root / 'duplicate-signals.xlsx'; book = Workbook(); sheet = book.active; sheet.title = '行程信号'
        sheet.append(self.headers)
        for row in self.rows: sheet.append(row)
        book.save(path); book.close()
        interface = copy.deepcopy(self.interface)
        for column in [interface['time_column'], interface['trip_column'], *[s['column'] for s in interface['signals']]]:
            column['worksheet'] = '行程信号'
        request = self.request(self.material(path), interface); self.approve_fixture(request)
        result = run_operation(request, self.context)
        root, _ = self.workspace.resolve(result['artifact_id'])
        with np.load(root / 'supervised.npz', allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays['X'][:, :2], np.asarray(self.rows)[:, 2:4].astype(float))


if __name__ == '__main__': unittest.main()
