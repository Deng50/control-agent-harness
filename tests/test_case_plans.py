"""Plan versions and trusted-context boundaries, without sampling or training."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from control_harness.case_plans import (FIXTURE_KIND, PendingPlanConfirmation,
    propose_plan, read_plan, validate_plan)
from control_harness.contracts import ContractError, canonical_hash
from control_harness.runtime import OperationContext

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", ROOT / "test-output" / "case-plan-tests"))
OUTPUT.mkdir(parents=True, exist_ok=True)


class CasePlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.root = Path(self.temporary.name)
        self.bindings = {"material": {"id": "material-1", "metadata_hash": "a" * 64, "content_hash": "b" * 64}}
        self.payload = {"inputs": ["温度"], "output": "速度", "sample_time": .1, "scale": 1.0}

    def tearDown(self): self.temporary.cleanup()

    def plan(self, scope="signals", payload=None, bindings=None):
        return propose_plan(self.root, scope, payload or self.payload, self.bindings if bindings is None else bindings)

    def ref(self, plan): return {key: plan[key] for key in ("plan_id", "version_hash")}

    def request(self, plan):
        return {"operation": "prepare_supervised_dataset", "workspace": str(self.root),
                "inputs": {"material": "material-1"}, "config": {"interface": self.payload, "plan_refs": {"signals": self.ref(plan)}}}

    def test_existing_canonical_hash_is_authoritative_and_same_version_is_idempotent(self):
        first = self.plan()
        self.assertEqual(canonical_hash(json.loads(first["canonical_payload_json"])), first["version_hash"])
        second = self.plan(payload=dict(reversed(list(self.payload.items()))))
        self.assertEqual(first, second)
        self.assertEqual(first["revision"], 1)

    def test_payload_change_invalidates_downstream_but_training_change_preserves_signals(self):
        signals = self.plan()
        sampling = self.plan("sampling", {"steps": 20})
        training = self.plan("training", {"epochs": 10})
        updated_training = self.plan("training", {"epochs": 12})
        self.assertEqual(read_plan(self.root, scope="signals")["version_hash"], signals["version_hash"])
        self.assertEqual(read_plan(self.root, scope="sampling")["status"], "proposed")
        self.assertNotEqual(training["version_hash"], updated_training["version_hash"])
        self.plan(payload={**self.payload, "sample_time": .2})
        for scope in ("sampling", "training"):
            self.assertEqual(read_plan(self.root, scope=scope)["status"], "stale")
        with self.assertRaises(PendingPlanConfirmation): validate_plan(self.root, self.ref(sampling), "sampling")

    def test_material_dataset_or_dependency_version_change_requires_new_review(self):
        plan = self.plan()
        for field in ("metadata_hash", "content_hash", "id"):
            changed = copy.deepcopy(self.bindings)
            changed["material"][field] += "-changed"
            with self.assertRaises(PendingPlanConfirmation):
                validate_plan(self.root, self.ref(plan), "signals", expected_payload=self.payload, bindings=changed)
        changed = self.plan(bindings={**self.bindings, "dataset": {"id": "new-data", "version": 2}})
        self.assertNotEqual(plan["version_hash"], changed["version_hash"])

    def test_json_confirmation_flags_and_claimed_test_source_are_not_authority(self):
        plan = self.plan()
        request = self.request(plan)
        request["config"].update(confirmed=True, confirmation_source="test_fixture", command_id="invented")
        context = OperationContext()
        context.bind_raw_request(json.dumps(request))
        with self.assertRaisesRegex(PendingPlanConfirmation, "request JSON flags"):
            context.require_case_confirmation(request, [{"scope": "signals", "expectedPayload": self.payload, "bindings": self.bindings}])

    def test_explicit_test_fixture_context_requires_marked_isolated_workspace_and_matching_plan(self):
        with self.assertRaises(ContractError): OperationContext.test_fixture(self.root)
        plan = self.plan()
        marker = self.root / ".case-plans" / "test-fixture.json"
        marker.write_text(json.dumps({"kind": FIXTURE_KIND, "workspace": str(self.root)}), encoding="utf-8")
        context = OperationContext.test_fixture(self.root)
        result = context.require_case_confirmation(self.request(plan), [{"scope": "signals", "expectedPayload": self.payload, "bindings": self.bindings}])
        self.assertEqual(result[0]["version_hash"], plan["version_hash"])
        changed = self.request(plan)
        changed["config"]["interface"] = {**self.payload, "scale": 2.0}
        with self.assertRaises(PendingPlanConfirmation):
            context.require_case_confirmation(changed, [{"scope": "signals", "expectedPayload": changed["config"]["interface"], "bindings": self.bindings}])

    def test_tampered_plan_body_or_stale_explicit_dependency_is_rejected(self):
        signals = self.plan()
        bad = {"signals": {"plan_id": signals["plan_id"], "version_hash": "0" * 64}}
        with self.assertRaises(PendingPlanConfirmation):
            propose_plan(self.root, "sampling", {"steps": 2}, {}, bad)
        path = self.root / ".case-plans" / "plans.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["plans"][signals["plan_id"]]["payload"]["scale"] = 100
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(PendingPlanConfirmation): validate_plan(self.root, self.ref(signals), "signals")

    def test_negative_zero_matches_json_transport_but_boolean_is_not_a_numeric_alias(self):
        payload = {**self.payload, "lower": 0, "enabled": False}
        plan = self.plan(payload=payload)
        validate_plan(self.root, self.ref(plan), "signals", expected_payload={**payload, "lower": -0.0})
        with self.assertRaises(PendingPlanConfirmation):
            validate_plan(self.root, self.ref(plan), "signals", expected_payload={**payload, "enabled": 0})


if __name__ == "__main__": unittest.main()
