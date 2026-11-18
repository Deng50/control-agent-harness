"""External-import full contract review with one existing signals scope.

These checks execute an actual CSV import once and verify stale configurations
are refused before data preparation. Ordinary S03 supervision is unchanged.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.case_plans import FIXTURE_KIND, PendingPlanConfirmation, propose_plan, validate_plan
from control_harness.case_requirements import confirmation_requirements
from control_harness.material_registration import register_material
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext
from test_external_expert_data import HEADERS, config as import_config


ROOT = Path(__file__).resolve().parents[1]


class ExternalImportConfirmationTests(unittest.TestCase):
    def setUp(self):
        output = ROOT / "test-output" / "external-import-confirmation-tests"; output.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=output); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.ws = Workspace(self.root / "case")
        atomic_json(self.ws.path / ".case-plans/test-fixture.json", {"kind": FIXTURE_KIND, "workspace": str(self.ws.path)})
        self.context = OperationContext.test_fixture(self.ws)
        path = self.root / "recorded-expert.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(HEADERS)
            for trip in range(6):
                previous = [.3, -.2]
                for step in range(4):
                    action = [previous[0] + .1, previous[1] - .05]
                    writer.writerow([step * .2, trip, .1 * trip + .01 * step, .2, *previous, *action, step])
                    previous = action
        material = register_material(self.ws, str(path), {}, OperationContext())
        self.request = {"operation": "import_expert_data", "workspace": str(self.ws.path), "inputs": {"material": material}, "config": import_config()}

    def requirements(self, request=None): return confirmation_requirements(self.ws, request or self.request)

    def approve(self):
        requirement = self.requirements()[0]
        self.plan = propose_plan(self.ws, "signals", requirement["expected_payload"], requirement["bindings"])
        self.ref = {key: self.plan[key] for key in ("plan_id", "version_hash")}
        self.request["config"]["plan_refs"] = {"signals": self.ref}

    def variants(self):
        variants = []
        def add(name, change):
            request = copy.deepcopy(self.request); change(request["config"]); variants.append((name, request))
        add("action bounds", lambda c: c["action_constraints"].update(bounds=[[-20., 20.], [-20., 20.]]))
        add("action rate bounds", lambda c: c["action_constraints"].update(rate_bounds=[[-2., 2.], [-2., 2.]]))
        add("previous action target", lambda c: c["source_mapping"]["signals"]["prev_a"].update(action_target="act_b"))
        add("source claim", lambda c: c["source"].update(description="Revised external source claim"))
        add("source kind", lambda c: c["source"].update(kind="declared_simulation"))
        add("split", lambda c: c.update(split={"mode": "trip", "train": 1 / 3, "validation": 1 / 3, "test": 1 / 3}))
        add("scaler", lambda c: c.update(scaler="minmax"))
        add("physical interface", lambda c: c.update(physical_interface={"name": "Changed explicit context"}))
        add("physical mapping", lambda c: c.update(physical_mapping={"observations": ["Changed explicit binding"]}))
        add("worksheet", lambda c: c.update(worksheet="other"))
        add("header row", lambda c: c.update(header_row=2))
        add("delimiter", lambda c: c.update(delimiter=";"))
        add("encoding", lambda c: c.update(encoding="utf-16"))
        return variants

    def test_one_signals_plan_contains_complete_contract_and_unchanged_import_runs(self):
        baseline = self.requirements(); self.assertEqual([row["scope"] for row in baseline], ["signals"])
        self.assertEqual(baseline[0]["expected_payload"], self.request["config"])
        self.approve(); self.assertEqual(self.requirements(), baseline)
        result = run_operation(self.request, self.context)
        self.assertEqual(result["kind"], "external_expert_dataset")
        report = read_json(Path(result["artifact_path"]) / "report.json")
        self.assertEqual([row["scope"] for row in report["confirmation_sources"]], ["signals"])
        self.assertEqual(report["confirmation_sources"][0]["confirmation_source"], "test_fixture")
        # Re-reading the same confirmed version does not request another scope.
        self.context.require_case_confirmation(self.request, self.requirements())

    def test_old_ref_rejects_each_changed_import_contract_before_reading_numeric_rows(self):
        self.approve(); baseline = self.requirements(); before = set(self.ws.manifest["artifacts"])
        # Instrument only the zero-execution boundary, not a numerical oracle.
        with patch("control_harness.external_expert_data.prepare_dataset", side_effect=AssertionError("stale confirmation reached data preparation")) as preparation:
            for name, changed in self.variants():
                with self.subTest(change=name):
                    required = self.requirements(changed)
                    self.assertEqual([row["scope"] for row in required], ["signals"])
                    self.assertNotEqual(required[0]["expected_payload"], baseline[0]["expected_payload"])
                    self.assertEqual(required[0]["bindings"], baseline[0]["bindings"])
                    with self.assertRaises(PendingPlanConfirmation):
                        validate_plan(self.ws, self.ref, "signals", expected_payload=required[0]["expected_payload"], bindings=required[0]["bindings"])
                    with self.assertRaises(PendingPlanConfirmation): run_operation(changed, self.context)
            preparation.assert_not_called()
        self.assertEqual(set(self.ws.manifest["artifacts"]), before)

    def test_refs_and_resume_fields_do_not_recursively_change_engineering_summary(self):
        before = self.requirements(); changed = copy.deepcopy(self.request)
        changed["config"].update(plan_refs={"signals": {"plan_id": "ignored-by-summary"}}, resume_checkpoint="unused.pt",
            resume_checkpoint_sha256="f" * 64, resume_run="unused")
        self.assertEqual(self.requirements(changed), before)

    def test_ordinary_supervised_preparation_keeps_its_existing_interface_only_scope(self):
        ordinary = copy.deepcopy(self.request); ordinary["operation"] = "prepare_supervised_dataset"
        before = self.requirements(ordinary)
        self.assertEqual(before[0]["expected_payload"], ordinary["config"]["interface"])
        ordinary["config"]["scaler"] = "none"; ordinary["config"]["source"]["description"] += " unchanged S03 scope"
        self.assertEqual(self.requirements(ordinary), before)


if __name__ == "__main__": unittest.main()
