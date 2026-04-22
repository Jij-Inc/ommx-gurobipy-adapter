"""Benchmark eight Plant Placement Problem formulations through Gurobi.

Ported from ``ommx-pyscipopt-adapter/tests/test_bench_placement.py``. Each
``placement_inputs`` parameterisation is converted to ``ommx.v1.Instance``,
then to ``gurobipy.Model``, in session-scoped fixtures — the OMMX construction
and the OMMX → Gurobi translation are *not* in the measurement. Each benchmark
calls ``model.reset()`` to discard any solution/basis kept from the previous
run, then ``model.optimize()`` to re-run presolve and branch-and-bound. The
reported time is therefore Gurobi's own processing time, isolated from
adapter overhead.

Notes on the restricted (``pip install gurobipy``) license: the larger sizes
in this benchmark (``plants=48, clients=80``) build models with more than
2000 variables and will raise ``GurobiError: Model too large for size-limited
license``. Those sizes are collected but skipped when only the restricted
license is available.
"""

from __future__ import annotations

import random
from typing import Callable, List

import gurobipy as gp
import pytest

from ommx.testing.placement import (
    Input,
    build_bigm,
    build_sos1,
    build_sos1_on_both_with_delta,
    build_sos1_on_both_with_delta_with_card,
    build_sos1_on_c_with_delta,
    build_sos1_on_c_with_delta_with_card,
    build_sos1_on_delta,
    build_sos1_on_delta_with_card,
)
from ommx.v1 import Instance
from ommx_gurobipy_adapter import OMMXGurobipyAdapter

_SIZES = [(6, 10), (12, 20), (24, 40), (48, 80)]
_INSTANCES_PER_SIZE = 3


@pytest.fixture(
    scope="session",
    params=_SIZES,
    ids=lambda pc: f"plants={pc[0]:02d}-clients={pc[1]:03d}",
)
def placement_inputs(request: pytest.FixtureRequest) -> List[Input]:
    num_plants, num_clients = request.param
    random.seed(42)
    return [
        Input.random(num_plants=num_plants, num_clients=num_clients)
        for _ in range(_INSTANCES_PER_SIZE)
    ]


def _is_license_error(e: gp.GurobiError) -> bool:
    msg = str(e).lower()
    return "size-limited" in msg or "license" in msg


def _build_models(
    inputs: List[Input], builder: Callable[[Input], Instance]
) -> List[gp.Model]:
    try:
        return [OMMXGurobipyAdapter(builder(inp)).solver_input for inp in inputs]
    except gp.GurobiError as e:
        if _is_license_error(e):
            pytest.skip(f"Gurobi license limit exceeded for this size: {e}")
        raise


@pytest.fixture(scope="session")
def sos1_models(placement_inputs: List[Input]) -> List[gp.Model]:
    return _build_models(placement_inputs, build_sos1)


@pytest.fixture(scope="session")
def sos1_on_c_with_delta_models(
    placement_inputs: List[Input],
) -> List[gp.Model]:
    return _build_models(placement_inputs, build_sos1_on_c_with_delta)


@pytest.fixture(scope="session")
def sos1_on_c_with_delta_with_card_models(
    placement_inputs: List[Input],
) -> List[gp.Model]:
    return _build_models(placement_inputs, build_sos1_on_c_with_delta_with_card)


@pytest.fixture(scope="session")
def sos1_on_delta_models(placement_inputs: List[Input]) -> List[gp.Model]:
    return _build_models(placement_inputs, build_sos1_on_delta)


@pytest.fixture(scope="session")
def sos1_on_delta_with_card_models(
    placement_inputs: List[Input],
) -> List[gp.Model]:
    return _build_models(placement_inputs, build_sos1_on_delta_with_card)


@pytest.fixture(scope="session")
def sos1_on_both_with_delta_models(
    placement_inputs: List[Input],
) -> List[gp.Model]:
    return _build_models(placement_inputs, build_sos1_on_both_with_delta)


@pytest.fixture(scope="session")
def sos1_on_both_with_delta_with_card_models(
    placement_inputs: List[Input],
) -> List[gp.Model]:
    return _build_models(placement_inputs, build_sos1_on_both_with_delta_with_card)


@pytest.fixture(scope="session")
def bigm_models(placement_inputs: List[Input]) -> List[gp.Model]:
    return _build_models(placement_inputs, build_bigm)


def _optimize_all(models: List[gp.Model]) -> None:
    try:
        for m in models:
            m.reset()
            m.optimize()
    except gp.GurobiError as e:
        if _is_license_error(e):
            pytest.skip(f"Gurobi license limit exceeded for this size: {e}")
        raise


@pytest.mark.benchmark
def test_bench_sos1(benchmark, sos1_models: List[gp.Model]) -> None:
    benchmark(_optimize_all, sos1_models)


@pytest.mark.benchmark
def test_bench_sos1_on_c_with_delta(
    benchmark, sos1_on_c_with_delta_models: List[gp.Model]
) -> None:
    benchmark(_optimize_all, sos1_on_c_with_delta_models)


@pytest.mark.benchmark
def test_bench_sos1_on_c_with_delta_with_card(
    benchmark, sos1_on_c_with_delta_with_card_models: List[gp.Model]
) -> None:
    benchmark(_optimize_all, sos1_on_c_with_delta_with_card_models)


@pytest.mark.benchmark
def test_bench_sos1_on_delta(
    benchmark, sos1_on_delta_models: List[gp.Model]
) -> None:
    benchmark(_optimize_all, sos1_on_delta_models)


@pytest.mark.benchmark
def test_bench_sos1_on_delta_with_card(
    benchmark, sos1_on_delta_with_card_models: List[gp.Model]
) -> None:
    benchmark(_optimize_all, sos1_on_delta_with_card_models)


@pytest.mark.benchmark
def test_bench_sos1_on_both_with_delta(
    benchmark, sos1_on_both_with_delta_models: List[gp.Model]
) -> None:
    benchmark(_optimize_all, sos1_on_both_with_delta_models)


@pytest.mark.benchmark
def test_bench_sos1_on_both_with_delta_with_card(
    benchmark, sos1_on_both_with_delta_with_card_models: List[gp.Model]
) -> None:
    benchmark(_optimize_all, sos1_on_both_with_delta_with_card_models)


@pytest.mark.benchmark
def test_bench_bigm(benchmark, bigm_models: List[gp.Model]) -> None:
    benchmark(_optimize_all, bigm_models)
