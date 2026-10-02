import unittest
from datetime import datetime, timezone

from railforge.api import RailForgeService
from railforge.commercial.capacity_commitment import (
    COMMITTED, DOWNGRADED, REJECTED,
    CommitmentLedger, ContractDemand, LinePlan, VehicleAvailability,
    diff, equivalent, plan_commitments,
)
from railforge.core.versioning import VersionConflict

U = timezone.utc


def demand(cid, qty, prio, min_qty=0, lane="L1", eq="BOX"):
    return ContractDemand(cid, "CUST", lane, eq, qty, prio, min_qty)


class CapacityCommitmentTests(unittest.TestCase):
    def test_full_commitment_merges_all_inputs(self):
        plan = plan_commitments(
            [demand("A", 6, 9), demand("B", 4, 5)],
            [LinePlan("P1", "L1", 10)],
            [VehicleAvailability("BOX", 12)],
        )
        by_id = {d.contract_id: d for d in plan.decisions}
        self.assertEqual((by_id["A"].status, by_id["A"].committed, by_id["A"].plan_id),
                         (COMMITTED, 6, "P1"))
        self.assertEqual((by_id["B"].status, by_id["B"].committed), (COMMITTED, 4))
        self.assertEqual(plan.plan_locks[0].locked_slots, 10)
        self.assertEqual(plan.vehicles[0].reserved, 10)

    def test_scarcity_downgrades_then_rejects_by_priority(self):
        plan = plan_commitments(
            [demand("A", 6, 9, min_qty=4), demand("B", 6, 5, min_qty=3), demand("C", 4, 1)],
            [LinePlan("P1", "L1", 10)],
            [VehicleAvailability("BOX", 10)],
        )
        by_id = {d.contract_id: d for d in plan.decisions}
        self.assertEqual((by_id["A"].status, by_id["A"].committed), (COMMITTED, 6))
        self.assertEqual((by_id["B"].status, by_id["B"].committed, by_id["B"].reason),
                         (DOWNGRADED, 4, "line capacity"))
        self.assertEqual((by_id["C"].status, by_id["C"].committed, by_id["C"].plan_id),
                         (REJECTED, 0, None))
        self.assertEqual(plan.plan_locks[0].locked_slots, 10)
        self.assertEqual(plan.vehicles[0].reserved, 10)

    def test_reject_reasons_cover_missing_plan_and_vehicles(self):
        plan = plan_commitments(
            [demand("NOPLAN", 4, 9, lane="LX"),
             demand("NOFLEET", 4, 8, eq="FLAT"),
             demand("TOOBIG", 10, 7, min_qty=8)],
            [LinePlan("P1", "L1", 5)],
            [VehicleAvailability("BOX", 10)],
        )
        by_id = {d.contract_id: d for d in plan.decisions}
        self.assertEqual(by_id["NOPLAN"].reason, "no line plan")
        self.assertEqual(by_id["NOFLEET"].reason, "vehicle shortage")
        self.assertEqual(by_id["TOOBIG"].reason, "below minimum quantity")
        self.assertEqual(plan.plan_locks, ())
        self.assertEqual(plan.vehicles, ())

    def test_versions_are_deterministic_and_comparable(self):
        first, second = CommitmentLedger(), CommitmentLedger()
        inputs = ([demand("A", 6, 9), demand("B", 4, 5)],
                  [LinePlan("P1", "L1", 10)], [VehicleAvailability("BOX", 10)])
        v1 = first.commit(0, *inputs)
        self.assertTrue(equivalent(v1, second.commit(0, *inputs)))
        v2 = first.commit(1, [demand("A", 8, 9), demand("C", 1, 1)], *inputs[1:])
        self.assertNotEqual(v1.digest, v2.digest)
        delta = diff(v1, v2)
        self.assertEqual((delta["from_version"], delta["to_version"]), (1, 2))
        self.assertEqual(delta["contracts"]["A"]["from"].committed, 6)
        self.assertEqual(delta["contracts"]["A"]["to"].committed, 8)
        self.assertIsNone(delta["contracts"]["B"]["to"])
        self.assertIsNone(delta["contracts"]["C"]["from"])
        self.assertEqual(delta["plans"]["P1"]["to"].locked_slots, 9)
        self.assertEqual(delta["vehicles"]["BOX"]["to"].reserved, 9)

    def test_version_conflict_rolls_back_atomically(self):
        ledger = CommitmentLedger()
        ledger.commit(0, [demand("A", 6, 9)], [LinePlan("P1", "L1", 10)],
                      [VehicleAvailability("BOX", 10)])
        before = ledger.snapshot()
        with self.assertRaises(VersionConflict):
            ledger.commit(0, [demand("B", 4, 5)], [LinePlan("P1", "L1", 10)],
                          [VehicleAvailability("BOX", 10)])
        self.assertEqual(ledger.version, 1)
        self.assertEqual(ledger.snapshot(), before)
        self.assertEqual(len(ledger.history()), 1)
        self.assertNotIn("B", ledger.snapshot()[1])

    def test_no_committed_contract_without_locked_plan(self):
        ledger = CommitmentLedger()
        ledger.commit(0, [demand("A", 6, 9, min_qty=4), demand("B", 6, 5, min_qty=3),
                          demand("C", 4, 1)],
                      [LinePlan("P1", "L1", 10)], [VehicleAvailability("BOX", 10)])
        _, contracts, plans, vehicles = ledger.snapshot()
        for decision in contracts.values():
            if decision.status in (COMMITTED, DOWNGRADED):
                self.assertIn(decision.plan_id, plans)
            else:
                self.assertIsNone(decision.plan_id)
        for plan_id, lock in plans.items():
            promised = sum(d.committed for d in contracts.values() if d.plan_id == plan_id)
            self.assertEqual(lock.locked_slots, promised)
        for reservation in vehicles.values():
            promised = sum(d.committed for d in contracts.values()
                           if d.status in (COMMITTED, DOWNGRADED))
            self.assertEqual(reservation.reserved, promised)

    def test_validation_rejects_bad_inputs_before_staging(self):
        ledger = CommitmentLedger()
        bad = [
            [demand("A", 0, 1)],
            [demand("A", 4, 1, min_qty=5)],
            [demand("A", 4, 1), demand("A", 4, 2)],
        ]
        for demands in bad:
            with self.assertRaises(ValueError):
                ledger.commit(0, demands, [], [])
        with self.assertRaises(ValueError):
            ledger.commit(0, [demand("A", 4, 1)],
                          [LinePlan("P1", "L1", 5), LinePlan("P1", "L1", 5)], [])
        with self.assertRaises(ValueError):
            ledger.commit(0, [demand("A", 4, 1)], [], [VehicleAvailability("BOX", -1)])
        self.assertEqual(ledger.version, 0)
        self.assertEqual(ledger.history(), ())

    def test_feature_commit_is_atomic_through_control_plane(self):
        service = RailForgeService()
        at = datetime(2026, 1, 1, 8, tzinfo=U)
        receipt = service.features.commit_capacity(
            "op1", [demand("A", 6, 9)], [LinePlan("P1", "L1", 10)],
            [VehicleAvailability("BOX", 10)], 0, at)
        self.assertEqual(receipt.event.payload["status"], "committed")
        self.assertEqual(receipt.event.payload["control_path"], "atomic")
        self.assertTrue(service.workflow().stream.verify_chain())
        self.assertTrue(service.workflow().trail.verify())
        head = service.control.stream.head
        with self.assertRaises(VersionConflict):
            service.features.commit_capacity(
                "op2", [demand("B", 4, 5)], [LinePlan("P1", "L1", 10)],
                [VehicleAvailability("BOX", 10)], 0, at)
        self.assertEqual(service.control.stream.head, head)
        self.assertIsNone(service.control.store.read("op2"))
        self.assertNotIn("op2", service.control.receipts)
        retry = service.features.commit_capacity(
            "op3", [demand("B", 4, 5)], [LinePlan("P1", "L1", 10)],
            [VehicleAvailability("BOX", 10)], 1, at)
        self.assertEqual(retry.result.version, 2)


if __name__ == "__main__":
    unittest.main()
