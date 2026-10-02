import unittest
from datetime import datetime, timezone

from railforge.core.time import Window
from railforge.core.versioning import VersionConflict
from railforge.customer.capacity_commitment import (
    CommitmentDecision, CommitmentRegistry, CommitmentVersion, ContractTerms, LinePlan,
    Vehicle, COMMITTED, DOWNGRADED, REJECTED, diff_versions, windows_cover)

U = timezone.utc


def at(hour):
    return datetime(2026, 1, 1, hour, tzinfo=U)


TERM = Window(at(8), at(18))


def plan(pid, capacity=100, start=8, end=18):
    return LinePlan(pid, "A", "B", at(start), at(end), capacity)


def vehicle(vid, capacity, windows=(TERM,)):
    return Vehicle(vid, capacity, windows)


def terms(cid, quantity, priority=0, allow_downgrade=False):
    return ContractTerms(cid, f"cust-{cid}", quantity, at(8), at(18),
                         priority=priority, allow_downgrade=allow_downgrade)


def by_id(decisions):
    return {d.contract_id: d for d in decisions}


class MergeTests(unittest.TestCase):
    def test_contract_plan_vehicles_and_priority_merge_into_version(self):
        reg = CommitmentRegistry()
        decisions = reg.commit([terms("C1", 100, priority=5)], {"C1": [plan("P1")]},
                               [vehicle("V1", 60), vehicle("V2", 40)])
        d = decisions[0]
        self.assertEqual(d.status, COMMITTED)
        v = d.version
        self.assertIsInstance(v, CommitmentVersion)
        self.assertEqual((v.sequence, v.priority, v.plan_id, v.offered), (1, 5, "P1", 100.0))
        self.assertEqual(sorted(a.vehicle_id for a in v.assignments), ["V1", "V2"])
        self.assertTrue(v.fingerprint)
        self.assertTrue(v.digest)

    def test_versions_are_comparable(self):
        reg = CommitmentRegistry()
        reg.commit([terms("C1", 100)], {"C1": [plan("P1")]},
                   [vehicle("V1", 60), vehicle("V2", 40)])
        first = reg.committed("C1")
        reg.commit([terms("C1", 80)], {"C1": [plan("P1")]},
                   [vehicle("V1", 60), vehicle("V2", 40)],
                   expected_versions={"C1": 1})
        second = reg.committed("C1")
        delta = diff_versions(first, second)
        self.assertEqual((delta.from_sequence, delta.to_sequence), (1, 2))
        self.assertIn("requested", delta.changed)
        self.assertIn("offered", delta.changed)
        self.assertFalse(delta.same_fingerprint)

    def test_identical_resubmission_does_not_create_a_new_version(self):
        reg = CommitmentRegistry()
        reg.commit([terms("C1", 100)], {"C1": [plan("P1")]},
                   [vehicle("V1", 60), vehicle("V2", 40)])
        first = reg.committed("C1")
        decisions = reg.commit([terms("C1", 100)], {"C1": [plan("P1")]},
                               [vehicle("V1", 60), vehicle("V2", 40)],
                               expected_versions={"C1": 1})
        self.assertIs(decisions[0].version, first)
        self.assertEqual(reg.sequence_of("C1"), 1)
        self.assertEqual(reg.ledger_version, 2)

    def test_window_coverage_requires_the_full_term(self):
        self.assertTrue(windows_cover((Window(at(8), at(12)), Window(at(12), at(18))),
                                      at(8), at(18)))
        self.assertFalse(windows_cover((Window(at(8), at(11)), Window(at(12), at(18))),
                                       at(8), at(18)))
        self.assertFalse(windows_cover((Window(at(9), at(18)),), at(8), at(18)))


class ShortageTests(unittest.TestCase):
    def test_shortage_rejects_without_downgrade_and_proposes_offer(self):
        reg = CommitmentRegistry()
        decisions = reg.commit([terms("C1", 100)], {"C1": [plan("P1", 100)]},
                               [vehicle("V1", 60)])
        d = decisions[0]
        self.assertEqual(d.status, REJECTED)
        self.assertIsNone(d.version)
        self.assertEqual(d.proposal.offered, 60.0)
        self.assertEqual(d.proposal.shortage, 40.0)
        self.assertIsNone(reg.committed("C1"))
        self.assertEqual(reg.plan_usage("P1"), 0.0)

    def test_shortage_downgrades_when_allowed(self):
        reg = CommitmentRegistry()
        decisions = reg.commit([terms("C1", 100, allow_downgrade=True)],
                               {"C1": [plan("P1", 100)]}, [vehicle("V1", 60)])
        d = decisions[0]
        self.assertEqual(d.status, DOWNGRADED)
        self.assertEqual(d.offered, 60.0)
        self.assertEqual(d.version.status, DOWNGRADED)
        self.assertEqual(reg.committed("C1").offered, 60.0)
        self.assertEqual(reg.plan_usage("P1"), 60.0)

    def test_plan_capacity_shortage_is_reported(self):
        reg = CommitmentRegistry()
        decisions = reg.commit([terms("C1", 100, allow_downgrade=True)],
                               {"C1": [plan("P1", 40)]},
                               [vehicle("V1", 60), vehicle("V2", 60)])
        d = decisions[0]
        self.assertEqual((d.status, d.offered), (DOWNGRADED, 40.0))
        self.assertIn("line plan", d.reason)

    def test_no_plan_window_means_rejection(self):
        reg = CommitmentRegistry()
        decisions = reg.commit([terms("C1", 10)], {"C1": [plan("P1", 40, start=12)]},
                               [vehicle("V1", 60)])
        self.assertEqual(decisions[0].status, REJECTED)
        self.assertIn("window", decisions[0].reason)


class PriorityTests(unittest.TestCase):
    def test_higher_priority_wins_scarce_capacity(self):
        reg = CommitmentRegistry()
        low = terms("LOW", 100, priority=1)
        high = terms("HIGH", 100, priority=9)
        decisions = by_id(reg.commit([low, high], {"LOW": [plan("P1")], "HIGH": [plan("P1")]},
                                     [vehicle("V1", 60), vehicle("V2", 40)]))
        self.assertEqual(decisions["HIGH"].status, COMMITTED)
        self.assertEqual(decisions["LOW"].status, REJECTED)
        self.assertEqual(reg.vehicle_locked_for("V1"), ("HIGH",))

    def test_priority_downgrade_offer_reflects_remaining_capacity(self):
        reg = CommitmentRegistry()
        decisions = by_id(reg.commit(
            [terms("HIGH", 70, priority=9, allow_downgrade=True),
             terms("LOW", 60, priority=1, allow_downgrade=True)],
            {"HIGH": [plan("P1")], "LOW": [plan("P1")]},
            [vehicle("V1", 60), vehicle("V2", 60)]))
        self.assertEqual(decisions["HIGH"].status, COMMITTED)
        self.assertEqual(decisions["LOW"].status, DOWNGRADED)
        self.assertEqual(decisions["LOW"].offered, 30.0)


class AtomicityTests(unittest.TestCase):
    def test_stale_contract_version_aborts_the_whole_batch(self):
        reg = CommitmentRegistry()
        reg.commit([terms("C1", 60)], {"C1": [plan("P1")]}, [vehicle("V1", 100)])
        before = reg.snapshot()
        with self.assertRaises(VersionConflict):
            reg.commit([terms("C1", 50), terms("C2", 50)],
                       {"C1": [plan("P1")], "C2": [plan("P1")]},
                       [vehicle("V1", 100)],
                       expected_versions={"C1": 99, "C2": 0})
        after = reg.snapshot()
        self.assertEqual(before["ledger_version"], after["ledger_version"])
        self.assertEqual(before["versions"], after["versions"])
        self.assertNotIn("C2", after["versions"])
        self.assertEqual(reg.plan_usage("P1"), 60.0)
        self.assertEqual(reg.vehicle_locked_for("V1"), ("C1",))

    def test_stale_ledger_version_aborts_before_planning(self):
        reg = CommitmentRegistry()
        with self.assertRaises(VersionConflict):
            reg.commit([terms("C1", 10)], {"C1": [plan("P1")]}, [vehicle("V1", 100)],
                       expected_ledger_version=7)
        self.assertEqual(reg.ledger_version, 0)

    def test_rejected_amendment_keeps_prior_contract_and_locks_atomically(self):
        reg = CommitmentRegistry()
        reg.commit([terms("C1", 100)], {"C1": [plan("P1")]},
                   [vehicle("V1", 60), vehicle("V2", 40)])
        first = reg.committed("C1")
        decisions = reg.commit([terms("C1", 500)], {"C1": [plan("P1")]},
                               [vehicle("V1", 60), vehicle("V2", 40)],
                               expected_versions={"C1": 1})
        self.assertEqual(decisions[0].status, REJECTED)
        self.assertIs(reg.committed("C1"), first)
        self.assertEqual(reg.sequence_of("C1"), 1)
        self.assertEqual(reg.plan_usage("P1"), 100.0)
        self.assertEqual(sorted(reg.vehicle_locked_for("V1") +
                                 reg.vehicle_locked_for("V2")), ["C1", "C1"])

    def test_successful_amendment_supersedes_in_one_swap(self):
        reg = CommitmentRegistry()
        reg.commit([terms("C1", 100)], {"C1": [plan("P1")]},
                   [vehicle("V1", 60), vehicle("V2", 40)])
        decisions = reg.commit([terms("C1", 60)], {"C1": [plan("P1")]},
                               [vehicle("V1", 60), vehicle("V2", 40)],
                               expected_versions={"C1": 1})
        v = decisions[0].version
        self.assertEqual((v.sequence, v.offered, v.status), (2, 60.0, COMMITTED))
        self.assertEqual(reg.plan_usage("P1"), 60.0)
        self.assertEqual(reg.vehicle_locked_for("V2"), ())

    def test_vehicles_cannot_be_double_booked_on_overlapping_terms(self):
        reg = CommitmentRegistry()
        d1 = reg.commit([terms("C1", 60)], {"C1": [plan("P1")]}, [vehicle("V1", 100)])
        self.assertEqual(d1[0].status, COMMITTED)
        d2 = reg.commit([terms("C2", 60)], {"C2": [plan("P2")]}, [vehicle("V1", 100)])
        self.assertEqual(d2[0].status, REJECTED)
        self.assertEqual(reg.vehicle_locked_for("V1"), ("C1",))

    def test_validation_failure_changes_nothing(self):
        reg = CommitmentRegistry()
        with self.assertRaises(ValueError):
            reg.commit([terms("C1", 0)], {"C1": [plan("P1")]}, [vehicle("V1", 100)])
        with self.assertRaises(ValueError):
            reg.commit([terms("C1", 10), terms("C1", 10)],
                       {"C1": [plan("P1")]}, [vehicle("V1", 100)])
        with self.assertRaises(ValueError):
            reg.commit([terms("C1", 10)], {}, [vehicle("V1", 100)])
        self.assertEqual(reg.ledger_version, 0)


if __name__ == "__main__":
    unittest.main()
