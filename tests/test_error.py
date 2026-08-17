import pytest
import gurobipy as gp

from ommx_gurobipy_adapter import (
    OMMXGurobipyAdapterError,
    OMMXGurobipyAdapter,
)

from ommx import (
    Constraint,
    DecisionVariable,
    DegreeBound,
    Function,
    Instance,
    InstanceClassMismatch,
    Kind,
    Polynomial,
)
from ommx.adapter import AdapterNotApplicableError, InfeasibleDetected


def test_error_polynomial_objective():
    """Test error when polynomial objective is used"""
    # Objective function: 2.3 * x * x * x
    ommx_instance = Instance.from_components(
        decision_variables=[DecisionVariable.continuous(1)],
        objective=Polynomial(terms={(1, 1, 1): 2.3}),
        constraints={},
        sense=Instance.MINIMIZE,
    )
    with pytest.raises(AdapterNotApplicableError) as e:
        OMMXGurobipyAdapter(ommx_instance)
    mismatches = e.value.report.input_membership.clause_reports[0].mismatches
    assert len(mismatches) == 1
    mismatch = mismatches[0]
    assert isinstance(mismatch, InstanceClassMismatch.ObjectiveDegreeExceedsBound)
    assert mismatch.actual_degree == 3
    assert mismatch.bound == DegreeBound.at_most(2)


def test_error_nonlinear_constraint():
    """Test error when nonlinear constraint is used"""
    # Objective function: 0
    # Constraint: 2.3 * x * x * x = 0
    ommx_instance = Instance.from_components(
        decision_variables=[DecisionVariable.continuous(1)],
        objective=0,
        constraints={
            0: Constraint(
                function=Polynomial(terms={(1, 1, 1): 2.3}),
                equality=Constraint.EQUAL_TO_ZERO,
            ),
        },
        sense=Instance.MINIMIZE,
    )
    with pytest.raises(AdapterNotApplicableError) as e:
        OMMXGurobipyAdapter(ommx_instance)
    mismatches = e.value.report.input_membership.clause_reports[0].mismatches
    assert len(mismatches) == 1
    mismatch = mismatches[0]
    assert isinstance(
        mismatch, InstanceClassMismatch.RegularConstraintDegreeExceedsBound
    )
    assert mismatch.actual_degrees == {0: 3}
    assert mismatch.bound == DegreeBound.at_most(2)


def test_error_nonlinear_indicator_constraint():
    indicator = DecisionVariable.binary(0)
    x = DecisionVariable.continuous(1)
    ommx_instance = Instance.from_components(
        decision_variables=[indicator, x],
        objective=x,
        constraints={},
        indicator_constraints={7: (x * x <= 1).with_indicator(indicator)},
        sense=Instance.MINIMIZE,
    )

    with pytest.raises(AdapterNotApplicableError) as e:
        OMMXGurobipyAdapter(ommx_instance)
    mismatches = e.value.report.input_membership.clause_reports[0].mismatches
    assert len(mismatches) == 1
    mismatch = mismatches[0]
    assert isinstance(mismatch, InstanceClassMismatch.IndicatorBodyDegreeExceedsBound)
    assert mismatch.actual_degrees == {7: 2}
    assert mismatch.bound == DegreeBound.at_most(1)


@pytest.mark.parametrize(
    ("variable", "kind"),
    [
        (DecisionVariable.semi_integer(0, lower=1, upper=3), Kind.SemiInteger),
        (
            DecisionVariable.semi_continuous(0, lower=1, upper=3),
            Kind.SemiContinuous,
        ),
    ],
)
def test_error_unsupported_variable_kind(variable, kind):
    ommx_instance = Instance.from_components(
        decision_variables=[variable],
        objective=variable,
        constraints={},
        sense=Instance.MINIMIZE,
    )

    with pytest.raises(AdapterNotApplicableError) as e:
        OMMXGurobipyAdapter(ommx_instance)
    mismatches = e.value.report.input_membership.clause_reports[0].mismatches
    assert len(mismatches) == 1
    mismatch = mismatches[0]
    assert isinstance(mismatch, InstanceClassMismatch.VariableKindNotAllowed)
    assert mismatch.kind == kind
    assert mismatch.variable_ids == {0}


def test_quadratic_converter_rejects_internal_invariant_violation():
    adapter = OMMXGurobipyAdapter.__new__(OMMXGurobipyAdapter)
    function = Function(Polynomial(terms={(1, 1, 1): 2.3}))

    with pytest.raises(AssertionError, match="INPUT_CLASS invariant violated"):
        adapter._make_expr(function)


def test_linear_converter_rejects_internal_invariant_violation():
    adapter = OMMXGurobipyAdapter.__new__(OMMXGurobipyAdapter)
    function = Function(Polynomial(terms={(1, 1): 2.3}))

    with pytest.raises(AssertionError, match="INPUT_CLASS invariant violated"):
        adapter._make_linear_expr(function)


def test_error_not_optimized_model():
    """Test error when model is not optimized"""
    model = gp.Model()
    instance = Instance.from_components(
        decision_variables=[],
        objective=0,
        constraints={},
        sense=Instance.MINIMIZE,
    )
    with pytest.raises(OMMXGurobipyAdapterError) as e:
        OMMXGurobipyAdapter(instance).decode_to_state(model)
    assert "The model may not be optimized." in str(e.value)


def test_error_infeasible_model():
    """Test error when model is infeasible"""
    x = DecisionVariable.continuous(1)
    ommx_instance = Instance.from_components(
        decision_variables=[x],
        objective=0,
        constraints={
            0: Constraint(
                function=x,
                equality=Constraint.EQUAL_TO_ZERO,
            ),
            1: Constraint(
                function=x - 1,
                equality=Constraint.EQUAL_TO_ZERO,
            ),
        },
        sense=Instance.MINIMIZE,
    )
    with pytest.raises(InfeasibleDetected):
        OMMXGurobipyAdapter.solve(ommx_instance)


def test_error_infeasible_constant_equality_constraint():
    """Test error when infeasible constant equality constraint is used"""
    ommx_instance = Instance.from_components(
        decision_variables=[],
        objective=0,
        constraints={
            0: Constraint(
                function=-1,
                equality=Constraint.EQUAL_TO_ZERO,
            ),
        },
        sense=Instance.MINIMIZE,
    )
    with pytest.raises(OMMXGurobipyAdapterError) as e:
        OMMXGurobipyAdapter(ommx_instance)
    assert "Infeasible constant constraint was found" in str(e.value)


def test_error_infeasible_constant_inequality_constraint():
    """Test error when infeasible constant inequality constraint is used"""
    ommx_instance = Instance.from_components(
        decision_variables=[],
        objective=0,
        constraints={
            0: Constraint(
                function=1,
                equality=Constraint.LESS_THAN_OR_EQUAL_TO_ZERO,
            ),
        },
        sense=Instance.MINIMIZE,
    )
    with pytest.raises(OMMXGurobipyAdapterError) as e:
        OMMXGurobipyAdapter(ommx_instance)
    assert "Infeasible constant constraint was found" in str(e.value)
