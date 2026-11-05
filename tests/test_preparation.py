import copy

import pytest

from ommx import (
    DecisionVariable,
    IndicatorConstraintID,
    OneHotConstraintID,
    Sos1ConstraintID,
    PolynomialRequirement,
    Equality,
    Instance,
    Kind,
    OneHotConstraint,
    Sense,
    Sos1Constraint,
    ConstraintKind,
)
from ommx.adapter import AdapterNotApplicableError
from ommx_gurobipy_adapter import OMMXGurobipyAdapter


def test_declares_quadratic_mip_input_class() -> None:
    input_class = OMMXGurobipyAdapter.INPUT_CLASS
    assert input_class is not None
    [clause] = input_class.clauses

    assert clause.label == "gurobi-quadratic-mip"
    assert clause.allowed_variable_kinds == {
        Kind.Binary,
        Kind.Integer,
        Kind.Continuous,
    }
    assert clause.objective_polynomial_requirement == PolynomialRequirement.at_most(2)
    assert clause.regular_constraint_polynomial_requirements == {
        Equality.EqualToZero: PolynomialRequirement.at_most(2),
        Equality.LessThanOrEqualToZero: PolynomialRequirement.at_most(2),
    }
    assert clause.indicator_body_polynomial_requirements == {
        Equality.EqualToZero: PolynomialRequirement.at_most(1),
        Equality.LessThanOrEqualToZero: PolynomialRequirement.at_most(1),
    }
    assert not clause.allows_one_hot
    assert clause.allows_sos1
    assert clause.allowed_senses == {Sense.Minimize, Sense.Maximize}


def test_recommended_preparation_policies_are_independent() -> None:
    first = OMMXGurobipyAdapter.recommended_preparation_policy()
    second = OMMXGurobipyAdapter.recommended_preparation_policy()

    assert first is not second
    first.special_constraints = None
    assert second.special_constraints is not None


def test_recommended_preparation_lowers_only_one_hot() -> None:
    indicator = DecisionVariable.binary(0)
    one_hot_variables = [DecisionVariable.binary(i) for i in range(1, 3)]
    value = DecisionVariable.continuous(3, lower=0, upper=2)
    instance = Instance.from_components(
        decision_variables=[indicator, *one_hot_variables, value],
        objective=value,
        constraints={},
        indicator_constraints={30: (value <= 1).with_indicator(indicator)},
        one_hot_constraints={
            10: OneHotConstraint(variables=one_hot_variables),
        },
        sos1_constraints={20: Sos1Constraint(variables=one_hot_variables)},
        sense=Sense.Maximize,
    )
    before = instance.to_v2_bytes()
    input_class = OMMXGurobipyAdapter.INPUT_CLASS
    assert input_class is not None

    assert not OMMXGurobipyAdapter.check_applicability(instance).is_member
    with pytest.raises(AdapterNotApplicableError):
        OMMXGurobipyAdapter(instance)
    with pytest.raises(AdapterNotApplicableError):
        OMMXGurobipyAdapter.solve_without_preparation(instance)
    solution = OMMXGurobipyAdapter.solve(instance)
    assert solution.feasible
    assert solution.objective == pytest.approx(2)
    assert instance.to_v2_bytes() == before

    prepared = copy.copy(instance)
    prepared.prepare(
        input_class,
        OMMXGurobipyAdapter.recommended_preparation_policy(),
    )

    assert set(instance.constraints.of_kind(ConstraintKind.OneHot)) == {
        OneHotConstraintID(10)
    }
    assert not prepared.constraints.of_kind(ConstraintKind.OneHot)
    assert set(prepared.constraints.of_kind(ConstraintKind.Indicator)) == {
        IndicatorConstraintID(30)
    }
    assert set(prepared.constraints.of_kind(ConstraintKind.Sos1)) == {
        Sos1ConstraintID(20)
    }
    assert prepared.active_special_constraint_kinds == {
        ConstraintKind.Indicator,
        ConstraintKind.Sos1,
    }
    assert input_class.contains(prepared)
    assert OMMXGurobipyAdapter.check_applicability(prepared).is_member

    adapter = OMMXGurobipyAdapter(prepared)
    assert adapter.instance is prepared


def test_penalty_preparation_does_not_claim_original_optimality() -> None:
    from ommx import Optimality, InstanceClass, PreparationPolicy

    x = DecisionVariable.binary(0)
    instance = Instance.from_components(
        decision_variables=[x],
        objective=x,
        constraints={7: x == 1},
        sense=Sense.Maximize,
    )
    instance.prepare(
        InstanceClass.qubo(), PreparationPolicy.for_qubo(uniform_penalty_weight=2)
    )
    solution = OMMXGurobipyAdapter.solve_without_preparation(instance)
    assert solution.feasible
    assert solution.objective == pytest.approx(1)
    assert solution.optimality == Optimality.Unspecified
