from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
from threading import RLock

from railforge.core.time import Window, utc
from railforge.core.versioning import VersionConflict

COMMITTED = 'committed'
DOWNGRADED = 'downgraded'
REJECTED = 'rejected'

_EPS = 1e-9


def _jsonable(value):
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return utc(value).isoformat()
    return value


def _canonical(value) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(',', ':'), default=str)


@dataclass(frozen=True)
class ContractTerms:
    """Customer contract capacity requested for a term window."""
    contract_id: str
    customer_id: str
    quantity: float
    start: datetime
    end: datetime
    priority: int = 0
    allow_downgrade: bool = False

    def window(self) -> Window:
        return Window(utc(self.start), utc(self.end))


@dataclass(frozen=True)
class LinePlan:
    """An operations line plan that may carry a contract's volume."""
    plan_id: str
    origin: str
    destination: str
    start: datetime
    end: datetime
    capacity: float


@dataclass(frozen=True)
class Vehicle:
    """Vehicle availability as the windows in which it can serve a term."""
    vehicle_id: str
    capacity: float
    windows: tuple[Window, ...] = ()


@dataclass(frozen=True)
class VehicleAssignment:
    vehicle_id: str
    capacity: float
    window: Window


@dataclass(frozen=True)
class DowngradeProposal:
    """Best offer operations can make when full contract capacity is unavailable."""
    requested: float
    offered: float
    shortage: float
    plan_id: str | None
    assignments: tuple[VehicleAssignment, ...]
    reason: str | None


@dataclass(frozen=True)
class CommitmentVersion:
    """A comparable, content-fingerprinted merge of contract, plan, vehicles and priority."""
    contract_id: str
    customer_id: str
    sequence: int
    priority: int
    start: datetime
    end: datetime
    requested: float
    offered: float
    plan_id: str
    assignments: tuple[VehicleAssignment, ...]
    status: str
    fingerprint: str
    digest: str


@dataclass(frozen=True)
class CommitmentDecision:
    contract_id: str
    status: str
    requested: float
    offered: float
    reason: str | None
    proposal: DowngradeProposal | None
    version: CommitmentVersion | None


@dataclass(frozen=True)
class CommitmentDiff:
    contract_id: str
    from_sequence: int
    to_sequence: int
    changed: tuple[str, ...]
    same_fingerprint: bool


def windows_cover(windows: tuple[Window, ...], start: datetime, end: datetime) -> bool:
    """True when the (merged) availability windows cover the whole [start, end) term."""
    s, e = utc(start), utc(end)
    if e <= s:
        raise ValueError('positive term required')
    merged: list[Window] = []
    for w in sorted(windows, key=lambda x: x.start):
        if not merged or w.start > merged[-1].end:
            merged.append(Window(w.start, w.end))
        elif w.end > merged[-1].end:
            merged.append(Window(merged[-1].start, w.end))
    if not merged or merged[0].start > s or merged[-1].end < e:
        return False
    cursor = s
    for w in merged:
        if w.start > cursor:
            return False
        if w.end >= e:
            return True
        cursor = w.end
    return False


def content_fingerprint(contract: ContractTerms, plan: LinePlan,
                        assignments: tuple[VehicleAssignment, ...]) -> str:
    body = _canonical({
        'contract': contract,
        'plan': plan,
        'vehicles': sorted(a.vehicle_id for a in assignments),
    })
    return sha256(body.encode('utf-8')).hexdigest()


def _digest(version_fields) -> str:
    return sha256(_canonical(version_fields).encode('utf-8')).hexdigest()


def build_version(contract: ContractTerms, plan: LinePlan,
                  assignments: tuple[VehicleAssignment, ...], sequence: int,
                  status: str) -> CommitmentVersion:
    offered = round(sum(a.capacity for a in assignments), 9)
    fingerprint = content_fingerprint(contract, plan, assignments)
    digest = _digest([
        contract.contract_id, sequence, status, fingerprint, offered,
    ])
    return CommitmentVersion(
        contract_id=contract.contract_id,
        customer_id=contract.customer_id,
        sequence=sequence,
        priority=contract.priority,
        start=utc(contract.start),
        end=utc(contract.end),
        requested=contract.quantity,
        offered=offered,
        plan_id=plan.plan_id,
        assignments=tuple(assignments),
        status=status,
        fingerprint=fingerprint,
        digest=digest,
    )


def rebump(version: CommitmentVersion, sequence: int) -> CommitmentVersion:
    """Assign the final sequence to a provisionally built version; content stays."""
    digest = _digest([
        version.contract_id, sequence, version.status, version.fingerprint,
        version.offered,
    ])
    return replace(version, sequence=sequence, digest=digest)


def diff_versions(old: CommitmentVersion, new: CommitmentVersion) -> CommitmentDiff:
    """Compare two versions of the same contract."""
    fields = ('customer_id', 'priority', 'start', 'end', 'requested', 'offered', 'plan_id')
    changed = tuple(f for f in fields if getattr(old, f) != getattr(new, f))
    old_vehicles = tuple(sorted(a.vehicle_id for a in old.assignments))
    new_vehicles = tuple(sorted(a.vehicle_id for a in new.assignments))
    if old_vehicles != new_vehicles:
        changed += ('assignments',)
    return CommitmentDiff(
        contract_id=new.contract_id,
        from_sequence=old.sequence,
        to_sequence=new.sequence,
        changed=changed,
        same_fingerprint=old.fingerprint == new.fingerprint,
    )


class _PlanClaim:
    __slots__ = ('contract_id', 'quantity')

    def __init__(self, contract_id: str, quantity: float):
        self.contract_id = contract_id
        self.quantity = quantity


class _VehicleClaim:
    __slots__ = ('contract_id', 'window', 'quantity')

    def __init__(self, contract_id: str, window: Window, quantity: float):
        self.contract_id = contract_id
        self.window = window
        self.quantity = quantity


class _WorkingState:
    """Planning state for one batch, built privately and only published whole.

    Live claims stay in place while planning: other contracts' claims consume
    capacity, a contract's own prior claim is visible to it but never blocks it.
    """

    def __init__(self, plan_claims: dict[str, list[_PlanClaim]],
                 vehicle_claims: dict[str, list[_VehicleClaim]]):
        self.plan_claims = {k: list(v) for k, v in plan_claims.items()}
        self.vehicle_claims = {k: list(v) for k, v in vehicle_claims.items()}

    @classmethod
    def live(cls, versions: dict[str, CommitmentVersion]):
        plan_claims: dict[str, list[_PlanClaim]] = {}
        vehicle_claims: dict[str, list[_VehicleClaim]] = {}
        for v in versions.values():
            if v.offered > 0:
                plan_claims.setdefault(v.plan_id, []).append(_PlanClaim(v.contract_id, v.offered))
            for a in v.assignments:
                vehicle_claims.setdefault(a.vehicle_id, []).append(
                    _VehicleClaim(v.contract_id, a.window, a.capacity))
        return cls(plan_claims, vehicle_claims)

    def others_usage(self, plan_id: str, contract_id: str) -> float:
        return sum(c.quantity for c in self.plan_claims.get(plan_id, ())
                   if c.contract_id != contract_id)

    def free_vehicle_capacity(self, vehicle: Vehicle, term: Window, contract_id: str) -> float:
        """Capacity left on a vehicle after other contracts' overlapping claims."""
        used = sum(c.quantity for c in self.vehicle_claims.get(vehicle.vehicle_id, ())
                   if c.contract_id != contract_id and term.overlaps(c.window))
        return max(0.0, vehicle.capacity - used)

    def replace_claim(self, version: CommitmentVersion) -> None:
        """Swap this contract's current claims for the ones in the version."""
        for claims in self.plan_claims.values():
            claims[:] = [c for c in claims if c.contract_id != version.contract_id]
        for rows in self.vehicle_claims.values():
            rows[:] = [c for c in rows if c.contract_id != version.contract_id]
        if version.offered > 0:
            self.plan_claims.setdefault(version.plan_id, []).append(
                _PlanClaim(version.contract_id, version.offered))
            for a in version.assignments:
                self.vehicle_claims.setdefault(a.vehicle_id, []).append(
                    _VehicleClaim(version.contract_id, a.window, a.capacity))


def _choose_plan(contract: ContractTerms, plans: list[LinePlan], state: _WorkingState):
    term = contract.window()
    eligible = [p for p in plans
                if windows_cover((Window(utc(p.start), utc(p.end)),), term.start, term.end)]
    if not eligible:
        return None, 'no line plan covers the contract window'
    # Most remaining capacity (excluding this contract's own prior claim) first;
    # deterministic smallest-id tie-break.
    return min(eligible, key=lambda p: (
        -max(0.0, p.capacity - state.others_usage(p.plan_id, contract.contract_id)),
        p.plan_id)), None


def _select_vehicles(contract: ContractTerms, vehicles: list[Vehicle], state: _WorkingState,
                     target: float):
    """Largest-free-capacity-first selection, clipping the last vehicle."""
    term = contract.window()
    usable = []
    for v in vehicles:
        if not windows_cover(v.windows, term.start, term.end):
            continue
        free = state.free_vehicle_capacity(v, term, contract.contract_id)
        if free > _EPS:
            usable.append((free, v))
    usable.sort(key=lambda pair: (-pair[0], pair[1].vehicle_id))
    assignments: list[VehicleAssignment] = []
    covered = 0.0
    for free, v in usable:
        if covered + _EPS >= target:
            break
        take = round(min(free, target - covered), 9)
        assignments.append(VehicleAssignment(v.vehicle_id, take, term))
        covered += take
    return tuple(assignments), round(covered, 9)


def _propose(contract: ContractTerms, plans: list[LinePlan], vehicles: list[Vehicle],
             state: _WorkingState) -> CommitmentDecision:
    """Evaluate one contract against the working state; mutates nothing."""
    plan, reason = _choose_plan(contract, plans, state)
    if plan is None:
        return CommitmentDecision(contract.contract_id, REJECTED, contract.quantity, 0.0,
                                  reason, None, None)
    plan_remaining = max(0.0, plan.capacity - state.others_usage(
        plan.plan_id, contract.contract_id))
    target = min(contract.quantity, plan_remaining)
    assignments, offered = _select_vehicles(contract, vehicles, state, target)
    if offered + _EPS >= contract.quantity:
        version = build_version(contract, plan, assignments, 0, COMMITTED)
        return CommitmentDecision(contract.contract_id, COMMITTED, contract.quantity,
                                  version.offered, None, None, version)
    reasons = []
    if plan_remaining + _EPS < contract.quantity:
        reasons.append('line plan capacity unavailable')
    if offered + _EPS < target or not assignments:
        reasons.append('vehicle availability insufficient')
    proposal = DowngradeProposal(
        requested=contract.quantity,
        offered=offered,
        shortage=round(contract.quantity - offered, 9),
        plan_id=plan.plan_id,
        assignments=assignments,
        reason='; '.join(reasons) or 'insufficient capacity',
    )
    if contract.allow_downgrade:
        version = build_version(contract, plan, assignments, 0, DOWNGRADED)
        return CommitmentDecision(contract.contract_id, DOWNGRADED, contract.quantity,
                                  version.offered, proposal.reason, proposal, version)
    return CommitmentDecision(contract.contract_id, REJECTED, contract.quantity, offered,
                              proposal.reason, proposal, None)


def _validate(contracts: list[ContractTerms], plans_by_id: dict[str, list[LinePlan]],
              vehicles: list[Vehicle]) -> None:
    ids = set()
    for c in contracts:
        if c.contract_id in ids:
            raise ValueError('duplicate contract in batch')
        ids.add(c.contract_id)
        if c.quantity <= 0:
            raise ValueError('contract quantity must be positive')
        c.window()
        plans = plans_by_id.get(c.contract_id)
        if not plans:
            raise ValueError(f'no line plans for contract {c.contract_id}')
        for p in plans:
            if p.capacity <= 0:
                raise ValueError('line plan capacity must be positive')
            Window(utc(p.start), utc(p.end))
    vids = set()
    for v in vehicles:
        if v.vehicle_id in vids:
            raise ValueError('duplicate vehicle in batch')
        vids.add(v.vehicle_id)
        if v.capacity <= 0:
            raise ValueError('vehicle capacity must be positive')
        for w in v.windows:
            Window(w.start, w.end)


class CommitmentRegistry:
    """Single aggregate holding every locked customer capacity commitment.

    Every ``commit`` is atomic: CAS version checks and all planning run against a
    private working copy; live state (plan claims, vehicle locks and versions) is
    replaced in one swap. A version conflict, invalid input or rejected amendment
    leaves previously committed versions - and their operational locks - intact,
    so a committed contract and its operating plan can never diverge.
    """

    SUBJECT = 'customer-capacity-commitments'

    def __init__(self):
        self._lock = RLock()
        self._ledger_version = 0
        self._sequences: dict[str, int] = {}
        self._versions: dict[str, CommitmentVersion] = {}

    @property
    def ledger_version(self) -> int:
        with self._lock:
            return self._ledger_version

    def committed(self, contract_id: str) -> CommitmentVersion | None:
        with self._lock:
            return self._versions.get(contract_id)

    def sequence_of(self, contract_id: str) -> int:
        with self._lock:
            return self._sequences.get(contract_id, 0)

    def plan_usage(self, plan_id: str) -> float:
        with self._lock:
            return round(sum(v.offered for v in self._versions.values()
                             if v.plan_id == plan_id), 9)

    def vehicle_locked_for(self, vehicle_id: str) -> tuple[str, ...]:
        with self._lock:
            owners = []
            for v in self._versions.values():
                if any(a.vehicle_id == vehicle_id for a in v.assignments):
                    owners.append(v.contract_id)
            return tuple(owners)

    def snapshot(self):
        with self._lock:
            return {
                'ledger_version': self._ledger_version,
                'versions': dict(self._versions),
                'sequences': dict(self._sequences),
            }

    def preview(self, contracts, plans_by_id, vehicles):
        """Evaluate a batch against current locks without changing anything."""
        contracts = list(contracts)
        vehicles = list(vehicles)
        _validate(contracts, plans_by_id, vehicles)
        with self._lock:
            decisions, _state = self._plan(contracts, plans_by_id, vehicles)
            return decisions

    def commit(self, contracts, plans_by_id, vehicles, expected_versions=None,
               expected_ledger_version: int | None = None):
        """Atomically merge contract, line plan, vehicle and priority inputs.

        ``expected_versions`` maps contract id to the sequence the caller last saw
        (omit or 0 for a new contract). Any mismatch raises ``VersionConflict``
        before live state is touched and the whole batch is rolled back.
        """
        contracts = list(contracts)
        vehicles = list(vehicles)
        _validate(contracts, plans_by_id, vehicles)
        expected_versions = dict(expected_versions or {})

        with self._lock:
            # Phase 1: optimistic concurrency checks - fail before any mutation.
            if (expected_ledger_version is not None
                    and expected_ledger_version != self._ledger_version):
                raise VersionConflict(
                    f'ledger expected={expected_ledger_version} actual={self._ledger_version}')
            for c in contracts:
                expected = expected_versions.get(c.contract_id, 0)
                actual = self._sequences.get(c.contract_id, 0)
                if expected != actual:
                    raise VersionConflict(
                        f'{c.contract_id}: expected={expected} actual={actual}')

            # Phase 2: plan on a private copy of all live locks.
            decisions, candidate = self._plan(contracts, plans_by_id, vehicles)
            by_id = {d.contract_id: d for d in decisions}

            # Phase 3: assign final sequence numbers on the private copy. Only
            # genuinely changed versions supersede; rejects/no-ops keep the old
            # locks untouched, so a rejected amendment atomically leaves the
            # prior contract and plan locks in force.
            final_versions = dict(self._versions)
            final_sequences = dict(self._sequences)
            final: dict[str, CommitmentDecision] = {}
            for c in contracts:
                decision = by_id[c.contract_id]
                previous = self._versions.get(c.contract_id)
                if decision.status == REJECTED:
                    if previous is not None:
                        decision = replace(decision, offered=previous.offered)
                    final[c.contract_id] = decision
                    continue
                if (previous is not None
                        and decision.version.fingerprint == previous.fingerprint
                        and decision.version.offered == previous.offered):
                    # Identical re-submission: same sequence, same locks.
                    final[c.contract_id] = replace(decision, version=previous,
                                                   offered=previous.offered)
                    final_versions[c.contract_id] = previous
                    final_sequences[c.contract_id] = previous.sequence
                    continue
                sequence = previous.sequence + 1 if previous is not None else 1
                version = rebump(decision.version, sequence)
                candidate.replace_claim(version)
                final[c.contract_id] = replace(decision, version=version,
                                               offered=version.offered)
                final_versions[c.contract_id] = version
                final_sequences[c.contract_id] = sequence

            # Phase 4: single-swap publication - contract versions and operational
            # locks become visible together, never in an intermediate state.
            self._versions = final_versions
            self._sequences = final_sequences
            self._ledger_version += 1
            return [final[c.contract_id] for c in contracts]

    def _plan(self, contracts, plans_by_id, vehicles):
        """Priority-ordered evaluation on a private copy of the live locks."""
        state = _WorkingState.live(self._versions)
        ordered = sorted(contracts, key=lambda c: (-c.priority, c.contract_id))
        decisions: list[CommitmentDecision] = []
        for c in ordered:
            decision = _propose(c, plans_by_id[c.contract_id], vehicles, state)
            # Tentatively claim accepted proposals so lower-priority contracts in
            # the same batch honor them; the private copy is discarded on failure.
            if decision.version is not None:
                state.replace_claim(decision.version)
            decisions.append(decision)
        return decisions, state
