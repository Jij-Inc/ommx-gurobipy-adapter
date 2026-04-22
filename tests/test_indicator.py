"""Tests for indicator constraint support in the Gurobi adapter."""

import pytest

from ommx.v1 import Instance, DecisionVariable
from ommx_gurobipy_adapter import OMMXGurobipyAdapter


def test_indicator_constraint_le():
    """b = 1 implies x <= 5, but b is free, so x may reach its upper bound."""
    b = DecisionVariable.binary(0)
    x = DecisionVariable.continuous(1, lower=0, upper=10)

    ic = (x <= 5).with_indicator(b)

    instance = Instance.from_components(
        decision_variables=[b, x],
        objective=x,
        constraints={},
        indicator_constraints={0: ic},
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    assert solution.objective == pytest.approx(10.0)


def test_indicator_constraint_forced_on():
    """When b is forced to 1, the indicator constraint becomes active."""
    b = DecisionVariable.binary(0)
    x = DecisionVariable.continuous(1, lower=0, upper=10)

    ic = (x <= 5).with_indicator(b)

    instance = Instance.from_components(
        decision_variables=[b, x],
        objective=x,
        constraints={0: b >= 1},
        indicator_constraints={0: ic},
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    assert solution.objective == pytest.approx(5.0)


def test_indicator_constraint_eq():
    """b = 1 implies x == 3."""
    b = DecisionVariable.binary(0)
    x = DecisionVariable.continuous(1, lower=0, upper=10)

    ic = (x == 3).with_indicator(b)

    instance = Instance.from_components(
        decision_variables=[b, x],
        objective=x,
        constraints={0: b >= 1},
        indicator_constraints={0: ic},
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    assert solution.objective == pytest.approx(3.0)


def test_indicator_constraint_multiple():
    """Two indicator constraints where at least one must be active."""
    b1 = DecisionVariable.binary(0)
    b2 = DecisionVariable.binary(1)
    x = DecisionVariable.continuous(2, lower=0, upper=100)

    ic1 = (x <= 50).with_indicator(b1)
    ic2 = (x <= 30).with_indicator(b2)

    instance = Instance.from_components(
        decision_variables=[b1, b2, x],
        objective=x,
        constraints={0: b1 + b2 >= 1},
        indicator_constraints={10: ic1, 11: ic2},
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    # Optimal: activate only ic1 (weaker) so x can reach 50.
    assert solution.objective == pytest.approx(50.0)
