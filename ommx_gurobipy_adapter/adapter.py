from __future__ import annotations

from typing import ClassVar

import gurobipy as gp
from gurobipy import GRB
from ommx import (
    Constraint,
    DegreeBound,
    Equality,
    Function,
    Instance,
    InstanceClass,
    InstanceClassClause,
    Kind,
    Optimality,
    PreparationPolicy,
    Sense,
    Solution,
    SpecialConstraintKind,
    SpecialConstraintPreparation,
    State,
)
from ommx.adapter import (
    DiagnosticsSink,
    InfeasibleDetected,
    SolverAdapter,
    UnboundedDetected,
)

from .exception import OMMXGurobipyAdapterError

ABSOLUTE_TOLERANCE = 1e-6

_GUROBI_VARIABLE_TYPES: dict[Kind, str] = {
    Kind.Binary: GRB.BINARY,
    Kind.Integer: GRB.INTEGER,
    Kind.Continuous: GRB.CONTINUOUS,
}
_GUROBI_MODEL_SENSES: dict[Sense, int] = {
    Sense.Minimize: GRB.MINIMIZE,
    Sense.Maximize: GRB.MAXIMIZE,
}
_GUROBI_CONSTRAINT_SENSES: dict[Equality, str] = {
    Equality.EqualToZero: GRB.EQUAL,
    Equality.LessThanOrEqualToZero: GRB.LESS_EQUAL,
}
_QUADRATIC_REGULAR_CONSTRAINT_DEGREE_BOUNDS = {
    equality: DegreeBound.at_most(2) for equality in _GUROBI_CONSTRAINT_SENSES
}
_LINEAR_INDICATOR_CONSTRAINT_DEGREE_BOUNDS = {
    equality: DegreeBound.at_most(1) for equality in _GUROBI_CONSTRAINT_SENSES
}


class OMMXGurobipyAdapter(SolverAdapter):
    INPUT_CLASS: ClassVar[InstanceClass | None] = InstanceClass(
        [
            InstanceClassClause(
                label="gurobi-quadratic-mip",
                allowed_variable_kinds=set(_GUROBI_VARIABLE_TYPES),
                objective_degree_bound=DegreeBound.at_most(2),
                regular_constraint_degree_bounds=(
                    _QUADRATIC_REGULAR_CONSTRAINT_DEGREE_BOUNDS
                ),
                indicator_constraint_degree_bounds=(
                    _LINEAR_INDICATOR_CONSTRAINT_DEGREE_BOUNDS
                ),
                allows_sos1=True,
                allowed_senses=set(_GUROBI_MODEL_SENSES),
            )
        ]
    )

    @classmethod
    def recommended_preparation_policy(cls) -> PreparationPolicy:
        """Recommend lowering OneHot constraints before using Gurobi.

        Gurobi accepts Indicator and SOS1 constraints directly, so this
        recommendation preserves those families and lowers only OneHot
        constraints. The returned policy is fresh and caller-editable.
        """
        return PreparationPolicy(
            special_constraints=SpecialConstraintPreparation.lower_special_constraints(
                kinds={SpecialConstraintKind.OneHot}
            )
        )

    def __init__(self, ommx_instance: Instance):
        self.require_applicable(ommx_instance)
        self.instance = ommx_instance
        self.model = gp.Model()
        self.model.setParam("OutputFlag", 0)  # Suppress output

        self._set_decision_variables()
        self._set_objective()
        self._set_constraints()

    @classmethod
    def solve(
        cls,
        ommx_instance: Instance,
        *,
        diagnostics: DiagnosticsSink | None = None,
    ) -> Solution:
        """
        Solve the given ommx.Instance using Gurobi, returning an ommx.Solution.

        :param ommx_instance: The ommx.Instance to solve.
        :param diagnostics: Reserved diagnostics sink; currently unused.
        :return: The solution as an ommx.Solution object
        """
        _ = diagnostics
        adapter = cls(ommx_instance)
        model = adapter.solver_input
        model.optimize()
        return adapter.decode(model)

    @property
    def solver_input(self) -> gp.Model:
        """The Gurobi model generated from this OMMX instance"""
        return self.model

    def decode(self, data: gp.Model) -> Solution:
        """Convert optimized Gurobi Model to ommx.Solution."""

        status = data.Status

        if status == GRB.INFEASIBLE:
            raise InfeasibleDetected("Model was infeasible")

        if status == GRB.UNBOUNDED:
            raise UnboundedDetected("Model was unbounded")

        state = self.decode_to_state(data)
        solution = self.instance.evaluate(state)

        if status == GRB.OPTIMAL:
            solution.optimality = Optimality.Optimal

        return solution

    def decode_to_state(self, data: gp.Model) -> State:
        """Create an ommx.State from an optimized Gurobi Model."""

        if data.Status == GRB.LOADED:
            raise OMMXGurobipyAdapterError(
                "The model may not be optimized. [status: loaded]"
            )

        if data.Status == GRB.INFEASIBLE:
            raise InfeasibleDetected("Model was infeasible")

        if data.Status == GRB.UNBOUNDED:
            raise UnboundedDetected("Model was unbounded")

        try:
            if data.SolCount == 0:
                raise OMMXGurobipyAdapterError(
                    f"There is no feasible solution. [status: {data.Status}]"
                )

            entries = {}
            for var in self.instance.used_decision_variables:
                variable = data.getVarByName(str(var.id))
                if variable:
                    entries[var.id] = variable.X
            return State(entries=entries)
        except Exception as e:
            raise OMMXGurobipyAdapterError(f"Failed to decode solution: {str(e)}")

    def _set_decision_variables(self):
        """Set up decision variables in the Gurobi model."""
        for var in self.instance.used_decision_variables:
            kind = Kind.from_pb(var.kind)
            variable_type = _GUROBI_VARIABLE_TYPES[kind]
            if kind == Kind.Binary:
                self.model.addVar(name=str(var.id), vtype=variable_type)
            else:
                self.model.addVar(
                    name=str(var.id),
                    vtype=variable_type,
                    lb=var.bound.lower,
                    ub=var.bound.upper,
                )

        # Create map of OMMX variable IDs to Gurobi variables and ensure model is updated
        self.model.update()
        self.varname_map = {
            str(id): var
            for var, id in zip(
                self.model.getVars(),
                (var.id for var in self.instance.used_decision_variables),
            )
        }

    def _set_objective(self):
        """Set up the objective function in the Gurobi model."""
        objective = self.instance.objective

        # Set optimization direction
        self.model.ModelSense = _GUROBI_MODEL_SENSES[self.instance.sense]

        # Set objective function
        self.model.setObjective(self._make_expr(objective))

    def _set_constraints(self):
        """Set up constraints in the Gurobi model."""
        # Handle SOS1 constraints (first-class in ommx v3)
        for sos1 in self.instance.sos1_constraints.values():
            vars = [self.varname_map[str(v)] for v in sos1.variables]
            self.model.addSOS(GRB.SOS_TYPE1, vars)

        # Handle regular constraints
        for cid, constraint in self.instance.constraints.items():
            sense = _GUROBI_CONSTRAINT_SENSES[constraint.equality]

            # Only constant case.
            if constraint.function.degree() == 0:
                if constraint.evaluate({}, atol=ABSOLUTE_TOLERANCE).feasible:
                    continue
                raise OMMXGurobipyAdapterError(
                    f"Infeasible constant constraint was found: id {cid}"
                )

            # Create Gurobi expression for the constraint
            expr = self._make_expr(constraint.function)
            self.model.addQConstr(expr, sense, 0.0, name=str(cid))

        # Handle indicator constraints (binvar = 1 => f(x) <= 0 or = 0)
        for ind_id, indicator in self.instance.indicator_constraints.items():
            f = indicator.function
            sense = _GUROBI_CONSTRAINT_SENSES[indicator.equality]

            if f.degree() == 0:
                # When the indicator is active, the constant constraint must hold.
                is_feasible = (
                    Constraint(
                        function=f,
                        equality=indicator.equality,
                    )
                    .evaluate({}, atol=ABSOLUTE_TOLERANCE)
                    .feasible
                )
                if is_feasible:
                    continue
                # Otherwise the indicator must be forced off.
                binvar = self.varname_map[str(indicator.indicator_variable_id)]
                self.model.addConstr(binvar == 0, name=f"ind_{ind_id}_forced_off")
                continue

            binvar = self.varname_map[str(indicator.indicator_variable_id)]
            lhs = self._make_linear_expr(f)

            self.model.addGenConstrIndicator(
                binvar, True, lhs, sense, 0.0, name=f"ind_{ind_id}"
            )

    def _make_expr(self, function: Function) -> gp.QuadExpr:
        """Create a Gurobi expression from an OMMX Function."""
        quadratic = function.as_quadratic()
        if quadratic is None:
            raise AssertionError(
                "INPUT_CLASS invariant violated: expected a quadratic function"
            )

        expr = gp.QuadExpr()
        expr.addConstant(quadratic.constant_term)
        for var_id, coefficient in quadratic.linear_terms.items():
            expr.add(coefficient * self.varname_map[str(var_id)])
        for (row, column), coefficient in quadratic.quadratic_terms.items():
            expr.add(
                coefficient * self.varname_map[str(row)] * self.varname_map[str(column)]
            )

        return expr

    def _make_linear_expr(self, function: Function) -> gp.LinExpr:
        """Create a Gurobi linear expression from a linear/constant OMMX Function."""
        linear = function.as_linear()
        if linear is None:
            raise AssertionError(
                "INPUT_CLASS invariant violated: expected a linear function"
            )

        terms = gp.quicksum(
            coeff * self.varname_map[str(var_id)]
            for var_id, coeff in linear.linear_terms.items()
        )
        return terms + linear.constant_term
