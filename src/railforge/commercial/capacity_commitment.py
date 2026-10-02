from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from json import dumps
from threading import RLock

from railforge.core.versioning import VersionConflict

COMMITTED = 'committed'
DOWNGRADED = 'downgraded'
REJECTED = 'rejected'


@dataclass(frozen=True)
class ContractDemand:
    """Capacity a customer contract asks for in one planning period."""

    contract_id: str
    customer: str
    lane: str
    equipment: str
    quantity: int
    priority: int
    min_quantity: int = 0


@dataclass(frozen=True)
class LinePlan:
    """Planned train capacity on one lane."""

    plan_id: str
    lane: str
    slots: int


@dataclass(frozen=True)
class VehicleAvailability:
    """Bookable vehicle pool for one equipment type."""

    equipment: str
    available: int


@dataclass(frozen=True)
class CommitmentDecision:
    contract_id: str
    customer: str
    status: str
    committed: int
    plan_id: str | None
    reason: str


@dataclass(frozen=True)
class PlanLock:
    plan_id: str
    lane: str
    locked_slots: int


@dataclass(frozen=True)
class VehicleReservation:
    equipment: str
    reserved: int


@dataclass(frozen=True)
class CommitmentPlan:
    """Merged allocation of contract demand against line plans and vehicles."""

    decisions: tuple[CommitmentDecision, ...]
    plan_locks: tuple[PlanLock, ...]
    vehicles: tuple[VehicleReservation, ...]


@dataclass(frozen=True)
class CommitmentVersion:
    """Immutable, comparable commitment snapshot."""

    version: int
    decisions: tuple[CommitmentDecision, ...]
    plan_locks: tuple[PlanLock, ...]
    vehicles: tuple[VehicleReservation, ...]
    digest: str


def _check_demands(demands):
    seen = set()
    for d in demands:
        if d.contract_id in seen:
            raise ValueError('duplicate contract')
        seen.add(d.contract_id)
        if d.quantity <= 0:
            raise ValueError('quantity')
        if d.min_quantity < 0 or d.min_quantity > d.quantity:
            raise ValueError('min_quantity')


def _check_plans(plans):
    seen = set()
    for p in plans:
        if p.plan_id in seen:
            raise ValueError('duplicate plan')
        seen.add(p.plan_id)
        if p.slots < 0:
            raise ValueError('slots')


def _check_vehicles(vehicles):
    seen = set()
    for v in vehicles:
        if v.equipment in seen:
            raise ValueError('duplicate equipment')
        seen.add(v.equipment)
        if v.available < 0:
            raise ValueError('available')


def plan_commitments(demands: list[ContractDemand], plans: list[LinePlan],
                     vehicles: list[VehicleAvailability]) -> CommitmentPlan:
    """Merge contract demand, line plans, vehicle availability and priority.

    Demands are served highest priority first. A demand is committed in full
    when one plan on its lane and the vehicle pool both cover it; otherwise it
    is downgraded to the largest feasible quantity at or above its minimum, or
    rejected with a reason when resources are insufficient.
    """
    _check_demands(demands)
    _check_plans(plans)
    _check_vehicles(vehicles)
    free_slots = {p.plan_id: p.slots for p in plans}
    by_lane = {}
    for p in plans:
        by_lane.setdefault(p.lane, []).append(p.plan_id)
    for ids in by_lane.values():
        ids.sort()
    pool = {v.equipment: v.available for v in vehicles}
    decisions = []
    for d in sorted(demands, key=lambda x: (-x.priority, -x.quantity, x.contract_id)):
        candidates = by_lane.get(d.lane, [])
        available = pool.get(d.equipment, 0)
        full = [pid for pid in candidates
                if free_slots[pid] >= d.quantity and available >= d.quantity]
        if full:
            pid = min(full, key=lambda p: (free_slots[p] - d.quantity, p))
            take, status, reason = d.quantity, COMMITTED, 'ok'
        else:
            pid = min(candidates, key=lambda p: (-min(free_slots[p], available), p), default=None)
            take = min(free_slots[pid], available) if pid is not None else 0
            if 0 < take >= d.min_quantity:
                status = DOWNGRADED
                reason = 'vehicle shortage' if available < free_slots[pid] else 'line capacity'
            else:
                status = REJECTED
                if not candidates:
                    reason = 'no line plan'
                elif available <= 0:
                    reason = 'vehicle shortage'
                elif take <= 0:
                    reason = 'line capacity'
                else:
                    reason = 'below minimum quantity'
                take, pid = 0, None
        if take:
            free_slots[pid] -= take
            pool[d.equipment] = available - take
        decisions.append(CommitmentDecision(d.contract_id, d.customer, status, take, pid, reason))
    decisions.sort(key=lambda x: x.contract_id)
    locks = tuple(PlanLock(p.plan_id, p.lane, p.slots - free_slots[p.plan_id])
                  for p in sorted(plans, key=lambda x: x.plan_id)
                  if free_slots[p.plan_id] < p.slots)
    reserved = tuple(VehicleReservation(v.equipment, v.available - pool[v.equipment])
                     for v in sorted(vehicles, key=lambda x: x.equipment)
                     if pool[v.equipment] < v.available)
    return CommitmentPlan(tuple(decisions), locks, reserved)


def _digest(version: int, plan: CommitmentPlan) -> str:
    body = {
        'version': version,
        'decisions': [asdict(d) for d in plan.decisions],
        'plan_locks': [asdict(p) for p in plan.plan_locks],
        'vehicles': [asdict(v) for v in plan.vehicles],
    }
    return sha256(dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _changed(before, after):
    out = {}
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old != new:
            out[key] = {'from': old, 'to': new}
    return out


def diff(a: CommitmentVersion, b: CommitmentVersion):
    """Comparable view of what changed between two commitment versions."""
    return {
        'from_version': a.version,
        'to_version': b.version,
        'contracts': _changed({d.contract_id: d for d in a.decisions},
                              {d.contract_id: d for d in b.decisions}),
        'plans': _changed({p.plan_id: p for p in a.plan_locks},
                          {p.plan_id: p for p in b.plan_locks}),
        'vehicles': _changed({v.equipment: v for v in a.vehicles},
                             {v.equipment: v for v in b.vehicles}),
    }


def equivalent(a: CommitmentVersion, b: CommitmentVersion):
    return a.digest == b.digest


class CommitmentLedger:
    """Atomic, versioned book of customer capacity commitments.

    Each commit merges contract demand, line plans and vehicle availability
    into one CommitmentVersion. Contract decisions and operational plan locks
    are applied inside a single critical section, so a reader never observes
    a committed contract without its locked plan. A stale expected_version
    raises VersionConflict before anything is staged, which rolls the
    conflicting commit back atomically.
    """

    def __init__(self):
        self._lock = RLock()
        self._version = 0
        self._contracts: dict[str, CommitmentDecision] = {}
        self._plans: dict[str, PlanLock] = {}
        self._vehicles: dict[str, VehicleReservation] = {}
        self._history: list[CommitmentVersion] = []

    @property
    def version(self):
        with self._lock:
            return self._version

    def snapshot(self):
        with self._lock:
            return (self._version, dict(self._contracts), dict(self._plans), dict(self._vehicles))

    def history(self):
        with self._lock:
            return tuple(self._history)

    def commit(self, expected_version: int, demands: list[ContractDemand],
               plans: list[LinePlan], vehicles: list[VehicleAvailability]) -> CommitmentVersion:
        staged = plan_commitments(demands, plans, vehicles)
        with self._lock:
            if expected_version != self._version:
                raise VersionConflict(f'expected={expected_version} actual={self._version}')
            version = self._version + 1
            record = CommitmentVersion(version, staged.decisions, staged.plan_locks,
                                       staged.vehicles, _digest(version, staged))
            self._contracts = {d.contract_id: d for d in staged.decisions}
            self._plans = {p.plan_id: p for p in staged.plan_locks}
            self._vehicles = {v.equipment: v for v in staged.vehicles}
            self._version = version
            self._history.append(record)
            return record
