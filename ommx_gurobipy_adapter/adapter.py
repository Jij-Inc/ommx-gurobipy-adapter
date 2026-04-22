from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB
import math

from ommx.adapter import SolverAdapter, InfeasibleDetected, UnboundedDetected
from ommx.v1 import (
    AdditionalCapability,
    Instance,
    Constraint,
    DecisionVariable,
    Function,
    State,
    Solution,
    Optimality,
)


from .exception import OMMXGurobipyAdapterError

ABSOLUTE_TOLERANCE = 1e-6


class OMMXGurobipyAdapter(SolverAdapter):
    ADDITIONAL_CAPABILITIES = frozenset(
        {
            AdditionalCapability.Indicator,
            AdditionalCapability.Sos1,
        }
    )

    def __init__(self, ommx_instance: Instance):
        super().__init__(ommx_instance)
        self.instance = ommx_instance
        self.model = gp.Model()
        self.model.setParam("OutputFlag", 0)  # Suppress output

        self._set_decision_variables()
        self._set_objective()
        self._set_constraints()

    @classmethod
    def solve(cls, ommx_instance: Instance) -> Solution:
        """
        Solve the given ommx.v1.Instance using Gurobi, returning an ommx.v1.Solution.

        :param ommx_instance: The ommx.v1.Instance to solve.
        :return: The solution as an ommx.v1.Solution object
        """
        adapter = cls(ommx_instance)
        model = adapter.solver_input
        model.optimize()
        return adapter.decode(model)

    @property
    def solver_input(self) -> gp.Model:
        """The Gurobi model generated from this OMMX instance"""
        return self.model

    def decode(self, data: gp.Model) -> Solution:
        """Convert optimized Gurobi Model to ommx.v1.Solution."""

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
        """Create an ommx.v1.State from an optimized Gurobi Model."""

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
            if var.kind == DecisionVariable.BINARY:
                self.model.addVar(name=str(var.id), vtype=GRB.BINARY)
            elif var.kind == DecisionVariable.INTEGER:
                self.model.addVar(
                    name=str(var.id),
                    vtype=GRB.INTEGER,
                    lb=var.bound.lower,
                    ub=var.bound.upper,
                )
            elif var.kind == DecisionVariable.CONTINUOUS:
                self.model.addVar(
                    name=str(var.id),
                    vtype=GRB.CONTINUOUS,
                    lb=var.bound.lower,
                    ub=var.bound.upper,
                )
            else:
                raise OMMXGurobipyAdapterError(
                    f"Unsupported decision variable kind: "
                    f"id: {var.id}, kind: {var.kind}"
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
        if self.instance.sense == Instance.MAXIMIZE:
            self.model.ModelSense = GRB.MAXIMIZE
        elif self.instance.sense == Instance.MINIMIZE:
            self.model.ModelSense = GRB.MINIMIZE
        else:
            raise OMMXGurobipyAdapterError(
                f"Sense not supported: {self.instance.sense}"
            )

        # Check if the objective function is non linear
        # Non linear are defined as not linear or quadratic.
        # For more details, refer to https://docs.gurobi.com/projects/optimizer/en/current/reference/python/nlexpr.html
        if objective.degree() >= 3:
            raise OMMXGurobipyAdapterError(
                "The objective function must be either `constant`, `linear` or `quadratic`."
            )

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
            # Check if the constraints is non linear
            # Non linear are defined as not linear or quadratic.
            # For more details, refer to https://docs.gurobi.com/projects/optimizer/en/current/reference/python/nlexpr.html
            if constraint.function.degree() >= 3:
                raise OMMXGurobipyAdapterError(
                    f"The constraints must be either `constant`, `linear` or `quadratic`. "
                    f"Constraint ID: {cid}"
                )

            # Only constant case.
            if constraint.function.degree() == 0:
                if constraint.equality == Constraint.EQUAL_TO_ZERO and math.isclose(
                    constraint.function.constant_term, 0, abs_tol=ABSOLUTE_TOLERANCE
                ):
                    continue
                elif (
                    constraint.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
                    and constraint.function.constant_term <= ABSOLUTE_TOLERANCE
                ):
                    continue
                else:
                    raise OMMXGurobipyAdapterError(
                        f"Infeasible constant constraint was found: id {cid}"
                    )

            # Create Gurobi expression for the constraint
            expr = self._make_expr(constraint.function)

            if constraint.equality == Constraint.EQUAL_TO_ZERO:
                self.model.addConstr(expr == 0, name=str(cid))
            elif constraint.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
                self.model.addConstr(expr <= 0, name=str(cid))
            else:
                raise OMMXGurobipyAdapterError(
                    f"Not supported constraint equality: "
                    f"id: {cid}, equality: {constraint.equality}"
                )

        # Handle indicator constraints (binvar = 1 => f(x) <= 0 or = 0)
        for ind_id, indicator in self.instance.indicator_constraints.items():
            f = indicator.function
            degree = f.degree()
            if degree >= 2:
                raise OMMXGurobipyAdapterError(
                    f"Indicator constraints must be linear. "
                    f"id: {ind_id}, degree: {degree}"
                )

            if degree == 0:
                # When the indicator is active, the constant constraint must hold.
                constant_value = f.constant_term
                is_feasible = (
                    indicator.equality == Constraint.EQUAL_TO_ZERO
                    and math.isclose(constant_value, 0, abs_tol=ABSOLUTE_TOLERANCE)
                ) or (
                    indicator.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO
                    and constant_value <= ABSOLUTE_TOLERANCE
                )
                if is_feasible:
                    continue
                # Otherwise the indicator must be forced off.
                binvar = self.varname_map[str(indicator.indicator_variable_id)]
                self.model.addConstr(binvar == 0, name=f"ind_{ind_id}_forced_off")
                continue

            binvar = self.varname_map[str(indicator.indicator_variable_id)]
            lhs = self._make_linear_expr(f)

            if indicator.equality == Constraint.EQUAL_TO_ZERO:
                sense = GRB.EQUAL
            elif indicator.equality == Constraint.LESS_THAN_OR_EQUAL_TO_ZERO:
                sense = GRB.LESS_EQUAL
            else:
                raise OMMXGurobipyAdapterError(
                    f"Not supported indicator constraint equality: "
                    f"id: {ind_id}, equality: {indicator.equality}"
                )

            self.model.addGenConstrIndicator(
                binvar, True, lhs, sense, 0.0, name=f"ind_{ind_id}"
            )

    def _make_expr(self, function: Function) -> gp.QuadExpr:
        """Create a Gurobi expression from an OMMX Function."""
        # QuadExpr includes constant, linear, and quadratic terms, so expr is initialized as QuadExpr
        expr = gp.QuadExpr()
        for ids, coefficient in function.terms.items():
            # Terms with no IDs represent constant terms.
            if len(ids) == 0:
                expr.addConstant(coefficient)
            # Terms with one ID represent linear terms, and terms with two IDs represent quadratic terms.
            elif len(ids) <= 2:
                term = coefficient
                for id in ids:
                    var = self.varname_map[str(id)]
                    term *= var
                expr.add(term)

        return expr

    def _make_linear_expr(self, function: Function) -> gp.LinExpr:
        """Create a Gurobi linear expression from a linear/constant OMMX Function."""
        terms = gp.quicksum(
            coeff * self.varname_map[str(var_id)]
            for var_id, coeff in function.linear_terms.items()
        )
        return terms + function.constant_term
