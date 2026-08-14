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
    Instance,
    InstanceClassMismatch,
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
