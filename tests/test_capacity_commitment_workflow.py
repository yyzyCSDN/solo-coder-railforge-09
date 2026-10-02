import unittest
from datetime import datetime, timezone

from railforge.api import RailForgeService
from railforge.core.time import Window
from railforge.core.versioning import VersionConflict
from railforge.customer.capacity_commitment import (
    ContractTerms, LinePlan, Vehicle, COMMITTED, DOWNGRADED, REJECTED)

U = timezone.utc


def at(hour):
    return datetime(2026, 1, 1, hour, tzinfo=U)


TERM = Window(at(8), at(18))


class CapacityCommitmentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.service = RailForgeService()
        self.features = self.service.features
        self.plans = {"C1": [LinePlan("P1", "A", "B", at(8), at(18), 100)]}
        self.vehicles = [Vehicle("V1", 60, (TERM,)), Vehicle("V2", 40, (TERM,))]

    def commit(self, op, contracts, **kw):
        return self.features.commit_freight_capacity(
            op, contracts, self.plans, self.vehicles, at(8), **kw)

    def test_feature_entrypoint_exists_and_commits_atomically(self):
        self.assertTrue(callable(self.features.commit_freight_capacity))
        contract = ContractTerms("C1", "cust", 100, at(8), at(18), priority=5)
        receipt = self.commit("op-cc-1", [contract])
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.event.payload["control_path"], "atomic")
        self.assertEqual(receipt.topic, "customer.capacity.committed")
        self.assertEqual(receipt.version, 1)
        self.assertEqual(receipt.result["ledger_version"], 1)
        decision = receipt.result["decisions"][0]
        self.assertEqual(decision.status, COMMITTED)
        self.assertTrue(self.service.workflow().stream.verify_chain())
        self.assertTrue(self.service.workflow().trail.verify())

    def test_rejection_is_published_but_locks_nothing(self):
        contract = ContractTerms("C1", "cust", 100, at(8), at(18))
        receipt = self.features.commit_freight_capacity(
            "op-cc-2", [contract],
            {"C1": [LinePlan("P1", "A", "B", at(8), at(18), 100)]},
            [Vehicle("V1", 40, (TERM,))], at(8))
        decision = receipt.result["decisions"][0]
        self.assertEqual(decision.status, REJECTED)
        self.assertEqual(decision.proposal.shortage, 60.0)
        self.assertIsNone(self.service.control.commitments.committed("C1"))
        self.assertEqual(self.service.control.commitments.plan_usage("P1"), 0.0)
        self.assertTrue(self.service.workflow().stream.verify_chain())

    def test_downgrade_is_committed_as_a_locked_version(self):
        contract = ContractTerms("C1", "cust", 100, at(8), at(18), allow_downgrade=True)
        receipt = self.features.commit_freight_capacity(
            "op-cc-3", [contract],
            {"C1": [LinePlan("P1", "A", "B", at(8), at(18), 100)]},
            [Vehicle("V1", 70, (TERM,))], at(8))
        decision = receipt.result["decisions"][0]
        self.assertEqual(decision.status, DOWNGRADED)
        self.assertEqual(decision.offered, 70.0)
        self.assertEqual(self.service.control.commitments.committed("C1").status, DOWNGRADED)

    def test_version_conflict_rolls_back_and_publishes_nothing(self):
        self.commit("op-cc-4", [ContractTerms("C1", "cust", 60, at(8), at(18))])
        head_before = self.service.control.stream.head
        with self.assertRaises(VersionConflict):
            self.commit("op-cc-5",
                        [ContractTerms("C1", "cust", 50, at(8), at(18))],
                        expected_versions={"C1": 42})
        # No event, no audit record, no ledger bump, prior commitment intact.
        self.assertEqual(self.service.control.stream.head, head_before)
        self.assertEqual(self.service.control.commitments.ledger_version, 1)
        self.assertEqual(self.service.control.commitments.committed("C1").offered, 60.0)
        self.assertTrue(self.service.workflow().stream.verify_chain())
        self.assertTrue(self.service.workflow().trail.verify())

    def test_rejected_amendment_leaves_no_intermediate_state(self):
        self.commit("op-cc-6", [ContractTerms("C1", "cust", 100, at(8), at(18))])
        receipt = self.commit(
            "op-cc-7", [ContractTerms("C1", "cust", 900, at(8), at(18))],
            expected_versions={"C1": 1})
        decision = receipt.result["decisions"][0]
        self.assertEqual(decision.status, REJECTED)
        commitments = self.service.control.commitments
        self.assertEqual(commitments.sequence_of("C1"), 1)
        self.assertEqual(commitments.committed("C1").offered, 100.0)
        self.assertEqual(commitments.plan_usage("P1"), 100.0)

    def test_operation_id_is_idempotent(self):
        contract = ContractTerms("C1", "cust", 100, at(8), at(18), priority=5)
        first = self.commit("op-cc-8", [contract])
        second = self.commit("op-cc-8", [contract])
        self.assertIs(first, second)
        self.assertEqual(self.service.control.commitments.ledger_version, 1)


if __name__ == "__main__":
    unittest.main()
