"""Tests for indicator constraint support in the Gurobi adapter."""

import pytest

from ommx import Constraint, DecisionVariable, Equality, Instance
from ommx_gurobipy_adapter import OMMXGurobipyAdapter


def test_indicator_constraint_le():
    """b = 1 implies x <= 5, but b is free, so x may reach its upper bound."""
    b = DecisionVariable.binary(0)
    x = DecisionVariable.continuous(1, lower=0, upper=10)

    ic = (x <= 5).with_indicator(b)

    instance = Instance.from_components(
        decision_variables=[b, x],
        objective=x,
        active_constraints={},
        active_indicator_constraints={1: ic},
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    assert solution.objective == pytest.approx(10.0)


@pytest.mark.parametrize("trigger_value", [0, 1])
def test_indicator_constraint_forced_on(trigger_value):
    """When b is forced to 1, the indicator constraint becomes active."""
    b = DecisionVariable.binary(0)
    x = DecisionVariable.continuous(1, lower=0, upper=10)

    ic = (x <= 5).with_indicator(b, trigger_value=trigger_value)

    instance = Instance.from_components(
        decision_variables=[b, x],
        objective=x,
        active_constraints={0: b == trigger_value},
        active_indicator_constraints={1: ic},
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    assert solution.objective == pytest.approx(5.0)


@pytest.mark.parametrize("trigger_value", [0, 1])
def test_indicator_constraint_eq(trigger_value):
    """b = 1 implies x == 3."""
    b = DecisionVariable.binary(0)
    x = DecisionVariable.continuous(1, lower=0, upper=10)

    ic = (x == 3).with_indicator(b, trigger_value=trigger_value)

    instance = Instance.from_components(
        decision_variables=[b, x],
        objective=x,
        active_constraints={0: b == trigger_value},
        active_indicator_constraints={1: ic},
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
        active_constraints={0: b1 + b2 >= 1},
        active_indicator_constraints={10: ic1, 11: ic2},
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    # Optimal: activate only ic1 (weaker) so x can reach 50.
    assert solution.objective == pytest.approx(50.0)


@pytest.mark.parametrize(
    ("equality", "constant", "expected"),
    [
        (Equality.EqualToZero, 0, 1.0),
        (Equality.EqualToZero, 1, 0.0),
        (Equality.LessThanOrEqualToZero, -1, 1.0),
        (Equality.LessThanOrEqualToZero, 1, 0.0),
    ],
)
@pytest.mark.parametrize("trigger_value", [0, 1])
def test_constant_indicator_constraint(equality, constant, expected, trigger_value):
    """The backend enforces constant bodies for either activation value."""
    indicator = DecisionVariable.binary(0)
    constraint = Constraint(
        function=constant,
        equality=equality,
    ).with_indicator(indicator, trigger_value=trigger_value)
    instance = Instance.from_components(
        decision_variables=[indicator],
        objective=indicator if trigger_value else 1 - indicator,
        active_constraints={},
        active_indicator_constraints={0: constraint},
        sense=Instance.MAXIMIZE,
    )

    solution = OMMXGurobipyAdapter.solve(instance)

    assert solution.objective == pytest.approx(expected)
