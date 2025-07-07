"""This script implements the contact-implicit trajectory optimization algorithm, which was introduced in
`A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact` by Michael Posa et al. 2013."
"""

import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import yaml
from helper_functions.drake_system_helper_functions import DrakeSystem
from pydrake.autodiffutils import AutoDiffXd, ExtractValue
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import GeometryId
from pydrake.math import RotationMatrix
from pydrake.multibody.plant import MultibodyPlant, MultibodyPlant_
from pydrake.multibody.tree import JacobianWrtVariable, RigidBody
from pydrake.solvers import (
    MathematicalProgram,
    MathematicalProgramResult,
    QuadraticConstraint,
)
from pydrake.systems.framework import Context, Context_

from src.core import pyCRISP

ArrayLikeType = Union[npt.NDArray[np.float64], npt.NDArray[AutoDiffXd]]
ScalarType = Union[float, AutoDiffXd]


def angle_axis_to_quaternion(angle: float, axis: np.ndarray) -> np.ndarray:
    """
    Convert angle-axis representation to quaternion using Drake functions.

    Args:
        angle: Rotation angle in radians
        axis: 3D unit vector representing rotation axis

    Returns:
        Quaternion as [w, x, y, z]
    """
    # For z-axis rotation, use Drake's MakeZRotation directly
    if np.allclose(axis, [0, 0, 1]):
        rotation_matrix = RotationMatrix.MakeZRotation(angle)
    else:
        # For other axes, use the general approach
        rotation_matrix = RotationMatrix.MakeFromOneVector(axis, 2)
        rotation_matrix = rotation_matrix.multiply(RotationMatrix.MakeZRotation(angle))
    quaternion = Quaternion(rotation_matrix.matrix())
    return np.array([quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()])


@dataclass
class TrajOptConfigs:
    """Configuration class for trajectory optimization parameters."""

    planning_horizon: int
    complementarity_relaxation: float
    print_math_prog: bool
    q_vec: List[float]
    qf_vec: List[float]
    r_vec: List[float]
    default_initial_state: List[float]
    x_lb: List[float]
    x_ub: List[float]
    u_lb: List[float]
    u_ub: List[float]

    # These will be set in post_init
    Q: Optional[np.ndarray] = None
    Qf: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None

    def __post_init__(self):
        """Convert diagonal vectors to matrices after initialization."""
        # Convert diagonal vectors to diagonal matrices
        self.Q = np.diag(self.q_vec)
        self.Qf = np.diag(self.qf_vec)
        self.R = np.diag(self.r_vec)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "TrajOptConfigs":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Ensure complementarity_relaxation is a float
        if "complementarity_relaxation" in config_dict:
            config_dict["complementarity_relaxation"] = float(
                config_dict["complementarity_relaxation"]
            )

        return cls(**config_dict)


class ContactImplicitTrajOpt:
    def __init__(
        self,
        plant: MultibodyPlant,
        plant_context: Context,
        plant_ad: MultibodyPlant_[AutoDiffXd],
        plant_ad_context: Context_[AutoDiffXd],
        planning_horizon: int,
        contact_pairs: List[Dict[str, str]],
        frictional_coefficients: List[float],
        default_complementarity_relaxation: float,
        default_timestep: float,
    ):
        self.prog = MathematicalProgram()

        self._plant_ad = plant_ad
        self._plant_ad_context = plant_ad_context
        self._plant = plant
        self._plant_context = plant_context

        self._contact_geom_id_pairs, self._contact_body_pairs = (
            self._to_geom_id_and_body_pairs(contact_pairs)
        )
        self._n_contacts = len(self._contact_geom_id_pairs)
        self._mu = frictional_coefficients
        assert len(self._mu) == self._n_contacts

        # Follow Stewart-Trinkle contact modelling, we have 5 basis vectors:
        # one for contact normal and four for frictional force.
        self._n_friction_directions = 2
        self._n_force_basis_per_contact = 2 * self._n_friction_directions + 1

        self._nx = self._plant_ad.num_positions() + self._plant_ad.num_velocities()
        self._nu = self._plant_ad.num_actuators()
        self._nq = self._plant_ad.num_positions()
        self._nv = self._plant_ad.num_velocities()

        self._planning_horizon = planning_horizon

        self._is_setting_up_done = False

        self._epsilon = default_complementarity_relaxation
        self._default_timestep = default_timestep
        self._target_states = None

    def find_traj(
        self,
        x_initial: np.ndarray,
        x_final: np.ndarray,
        warm_up_solution: Optional[np.ndarray] = None,
        craft_initial_guess_fn: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
    ) -> np.ndarray:
        start_time = time.perf_counter()
        if not self._is_setting_up_done:
            raise ValueError(
                "Trajectory optimization problem is not set up. Call `set_up_trajopt_problem` to set it up."
            )

        if warm_up_solution is None:
            if craft_initial_guess_fn is not None:
                initial_guess = craft_initial_guess_fn(x_initial, x_final)
                assert initial_guess.shape == (self.prog.num_vars(),)
            else:
                initial_guess = np.zeros(self.prog.num_vars())
        else:
            initial_guess = warm_up_solution

        # Convert Drake's MathematicalProgram to CRISP's OptimizationProblem
        crisp_problem = pyCRISP.OptimizationProblem(self.prog.num_vars(), "ci-trajopt")
        crisp_problem.parse_drake_mathematical_program(self.prog)

        # Set up the solver
        solver_params = pyCRISP.SolverParameters()
        solver = pyCRISP.SolverInterface(crisp_problem, solver_params)
        solver.set_hyper_parameters("trailTol", np.array([1e-3]))
        solver.set_hyper_parameters("trustRegionTol", np.array([1e-3]))
        solver.set_hyper_parameters("WeightedMode", np.array([1]))
        solver.set_hyper_parameters("verbose", np.array([1]))
        solver.initialize(initial_guess)
        solver.solve()
        result = solver.getSolution()

        print(f"Time taken to solve the problem: {time.perf_counter() - start_time}\n")
        return result

    def _to_geom_id_and_body_pairs(
        self, contact_pairs: List[Dict[str, str]]
    ) -> Tuple[List[Tuple[GeometryId, GeometryId]], List[Tuple[RigidBody, RigidBody]]]:
        geom_id_pairs = []
        body_pairs = []

        for geom_pair in contact_pairs:
            body_1 = self._plant_ad.GetBodyByName(
                geom_pair["link_1"],
                self._plant_ad.GetModelInstanceByName(geom_pair["body_1"]),
            )
            body_2 = self._plant_ad.GetBodyByName(
                geom_pair["link_2"],
                self._plant_ad.GetModelInstanceByName(geom_pair["body_2"]),
            )
            geom_1_id = self._plant_ad.GetCollisionGeometriesForBody(body_1)[0]
            geom_2_id = self._plant_ad.GetCollisionGeometriesForBody(body_2)[0]
            geom_id_pairs.append(
                (
                    geom_1_id,
                    geom_2_id,
                )
            )
            body_pairs.append((body_1, body_2))
        return geom_id_pairs, body_pairs

    def set_up_trajopt_problem(
        self,
        x0: np.ndarray,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
        u_lb: np.ndarray,
        u_ub: np.ndarray,
        Q: np.ndarray,
        Qf: np.ndarray,
        R: np.ndarray,
        target_state: np.ndarray,
    ) -> None:
        if self._is_setting_up_done:
            raise ValueError(
                "Trajectory optimization problem is already set up. Call `find_traj` to solve it."
            )
        self._create_default_containers()

        self._add_decision_variables()
        self._add_constraints(x0, x_lb, x_ub, u_lb, u_ub)
        self._add_costs(Q, Qf, R, target_state)

        self._is_setting_up_done = True

    def _create_default_containers(self) -> None:
        self._create_default_decision_variable_containers()
        self._create_default_constraint_containers()

    def _create_default_decision_variable_containers(self) -> None:
        self._timestep_var = self.prog.NewContinuousVariables(1, "timestep")
        self._x_vars = np.zeros(self._planning_horizon + 1, dtype=object)
        self._u_vars = np.zeros(self._planning_horizon, dtype=object)

        ####### Define variables for contact constraints that follows Coulomb friction model ########
        # - $\lambda$ are the contact forces at each contact point expressed in the contact frame.
        #
        # - $\beta$ are slack variables that are used for the following equality constraints
        #   $\beta^{k, j} = \mu \lambda_{k, j, z} - \sum_i^d \lambda_{k, j, x}^i$
        #   k is timestep index, j is contact index, and i is the index of frictional basis.
        #
        # - $\gamma$ are slack variables that imply the maximum sliding relative velocity
        #    between two bodies in contact.
        #
        # - $\alpha$ are slack variables that are used for the following equality constraints
        #    $\alpha^{k, j} = \gamma_{k, j} + \Psi(x_k}^T D_{k, j}^i$
        #    where D_{k, j}^i is the ith tangential basis vector of the frictional polyhedral at
        #    the contact point j at timestep k.
        #############################################################################################

        contact_related_variable_names = [
            "_lambda_vars",
            "_signed_distance_vars",
            "_relative_velocity_in_contact_frame_vars",
            "_slack_gamma_vars",
            "_slack_beta_vars",
            "_slack_alpha_vars",
        ]
        for var_name in contact_related_variable_names:
            setattr(
                self,
                var_name,
                np.zeros((self._planning_horizon, self._n_contacts), dtype=object),
            )

    def _create_default_constraint_containers(self) -> None:
        constraints_config = {
            "_x_constraints": (self._planning_horizon + 1,),
            "_u_constraints": (self._planning_horizon,),
            "_implicit_euler_constraints": (self._planning_horizon,),
            "_dynamics_constraints": (self._planning_horizon,),
            "_non_negativity_signed_distance_constraints": (
                self._planning_horizon,
                self._n_contacts,
            ),
            "_non_negativity_lambda_constraints": (
                self._planning_horizon,
                self._n_contacts,
            ),
            "_non_negativity_gamma_constraints": (
                self._planning_horizon,
                self._n_contacts,
            ),
            "_contact_force_staying_in_friction_cone_equality_constraints": (
                self._planning_horizon,
                self._n_contacts,
            ),
            "_contact_force_staying_in_friction_cone_inequality_constraints": (
                self._planning_horizon,
                self._n_contacts,
            ),
            "_stick_separation_transition_constraints": (
                self._planning_horizon,
                self._n_contacts,
            ),
            "_stick_slip_transition_constraints": (
                self._planning_horizon,
                self._n_contacts,
            ),
            "_max_sliding_velocity_equality_constraints": (
                self._planning_horizon,
                self._n_contacts,
                self._n_force_basis_per_contact - 1,
            ),
            "_max_sliding_velocity_inequality_constraints": (
                self._planning_horizon,
                self._n_contacts,
                self._n_force_basis_per_contact - 1,
            ),
            "_max_dissipation_constraints": (
                self._planning_horizon,
                self._n_contacts,
                self._n_force_basis_per_contact - 1,
            ),
        }
        for name, shape in constraints_config.items():
            setattr(self, name, np.zeros(shape, dtype=object))

    def _create_default_cost_containers(self) -> None:
        self._running_state_costs = np.zeros(self._planning_horizon, dtype="object")
        self._final_costs = np.zeros(self._planning_horizon, dtype="object")
        self._input_costs = np.zeros(self._planning_horizon, dtype="object")

    def _add_decision_variables(self) -> None:
        self._add_state_input_variables()
        self._add_contact_related_variables()

    def _add_state_input_variables(self) -> None:
        for i in range(self._planning_horizon + 1):
            self._x_vars[i] = self.prog.NewContinuousVariables(self._nx, f"x_{i}")

            # we don't have control input for the last state.
            if i < self._planning_horizon:
                self._u_vars[i] = self.prog.NewContinuousVariables(
                    self._nu, f"u_{i + 1}"
                )

    def _add_contact_related_variables(self) -> None:
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                self._signed_distance_vars[i, j] = self.prog.NewContinuousVariables(
                    1, f"signed_distance_{i}_{j}"
                )
                self._lambda_vars[i, j] = self.prog.NewContinuousVariables(
                    self._n_force_basis_per_contact, f"lambda_{i}_{j}"
                )
                self._slack_gamma_vars[i, j] = self.prog.NewContinuousVariables(
                    1, f"gamma_{i}_{j}"
                )
                self._slack_beta_vars[i, j] = self.prog.NewContinuousVariables(
                    1, f"beta_{i}_{j}"
                )
                self._slack_alpha_vars[i, j] = self.prog.NewContinuousVariables(
                    self._n_force_basis_per_contact - 1, f"alpha_{i}_{j}"
                )

    def _add_constraints(
        self,
        x0: np.ndarray,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
        u_lb: np.ndarray,
        u_ub: np.ndarray,
    ) -> None:
        self._add_state_input_constraints(x0, x_lb, x_ub, u_lb, u_ub)
        self._add_implicit_euler_constraints()
        self._add_dynamics_constraints()
        self._add_contact_constraints()
        self._add_timestep_constraint()

    def _add_timestep_constraint(self) -> None:
        self.prog.AddBoundingBoxConstraint(
            self._default_timestep, self._default_timestep, self._timestep_var
        )

    def _add_state_input_constraints(
        self,
        x0: np.ndarray,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
        u_lb: np.ndarray,
        u_ub: np.ndarray,
    ) -> None:
        self._remove_state_input_constraints_if_any()
        self._add_initial_state_constraints(x0)
        self._add_non_initial_state_constraints(x_lb, x_ub)
        self._add_input_constraints(u_lb, u_ub)

    def _remove_state_input_constraints_if_any(self) -> None:
        for x_constraint in self._x_constraints:
            if x_constraint == 0:
                continue
            self.prog.RemoveConstraint(x_constraint)
        for u_constraint in self._u_constraints:
            if u_constraint == 0:
                continue
            self.prog.RemoveConstraint(u_constraint)

    def _add_initial_state_constraints(self, x0: npt.NDArray[np.float64]) -> None:
        self._x_constraints[0] = self.prog.AddBoundingBoxConstraint(
            x0, x0, self._x_vars[0]
        )
        self._x_constraints[0].evaluator().set_description("initial_state_constraint")

    def _add_non_initial_state_constraints(
        self, x_lb: npt.NDArray[np.float64], x_ub: npt.NDArray[np.float64]
    ) -> None:
        for i in range(1, self._planning_horizon + 1):
            self._x_constraints[i] = self.prog.AddBoundingBoxConstraint(
                x_lb, x_ub, self._x_vars[i]
            )
            self._x_constraints[i].evaluator().set_description(f"x_{i}_bounds")

    def _add_input_constraints(
        self, u_lb: npt.NDArray[np.float64], u_ub: npt.NDArray[np.float64]
    ) -> None:
        for i in range(self._planning_horizon):
            self._u_constraints[i] = self.prog.AddBoundingBoxConstraint(
                u_lb, u_ub, self._u_vars[i]
            )
            self._u_constraints[i].evaluator().set_description(f"u_{i}_bounds")

    def _implicit_euler_constraint_evaluator(
        self,
        x_cur: ArrayLikeType,
        x_next: ArrayLikeType,
        timestep: ScalarType,
    ) -> ArrayLikeType:
        if isinstance(x_cur[0], AutoDiffXd):
            self._plant_ad.SetPositionsAndVelocities(self._plant_ad_context, x_next)
            qdot_next = self._plant_ad.MapVelocityToQDot(
                self._plant_ad_context, x_next[self._nq :]
            )
        else:
            self._plant.SetPositionsAndVelocities(self._plant_context, x_next)
            qdot_next = self._plant.MapVelocityToQDot(
                self._plant_context, x_next[self._nq :]
            )
        return x_cur[: self._nq] - x_next[: self._nq] + timestep * qdot_next

    def _implicit_euler_constraint_helper(self, vars: ArrayLikeType) -> ArrayLikeType:
        timestep = vars[0]
        x_cur = vars[1 : (self._nx + 1)]
        x_next = vars[(self._nx + 1) :]
        return self._implicit_euler_constraint_evaluator(x_cur, x_next, timestep)

    def _add_implicit_euler_constraints(self) -> None:
        for i in range(self._planning_horizon):
            self._implicit_euler_constraints[i] = self.prog.AddConstraint(
                self._implicit_euler_constraint_helper,
                lb=np.zeros(self._nq),
                ub=np.zeros(self._nq),
                vars=np.hstack(
                    [self._timestep_var, self._x_vars[i], self._x_vars[i + 1]]
                ),
                description=f"implicit_euler_constraint_{i}",
            )

    def _dynamics_constraint_evaluator(
        self,
        timestep: ScalarType,
        x_cur: ArrayLikeType,
        u_next: ArrayLikeType,
        x_next: ArrayLikeType,
        lambda_next: ArrayLikeType,
    ) -> ArrayLikeType:
        if isinstance(x_cur[0], AutoDiffXd):
            plant = self._plant_ad
            context = self._plant_ad_context
        else:
            plant = self._plant
            context = self._plant_context

        plant.SetPositionsAndVelocities(context, x_next)
        J = self._calc_stacked_contact_jacobian(
            plant, context, self._contact_geom_id_pairs
        )

        mass_matrix = plant.CalcMassMatrix(context)
        bias_term = plant.CalcBiasTerm(context)
        gravity_term = plant.CalcGravityGeneralizedForces(context)
        actuation_matrix = plant.MakeActuationMatrix()

        return mass_matrix @ (x_next[self._nq :] - x_cur[self._nq :]) + timestep * (
            bias_term - gravity_term - actuation_matrix @ u_next - J.T @ lambda_next
        )

    def _dynamics_constraint_evaluator_helper(
        self, vars: ArrayLikeType
    ) -> ArrayLikeType:
        timestep = vars[0]
        x_cur_var = vars[1 : (self._nx + 1)]
        u_cur_var = vars[(self._nx + 1) : (self._nx + self._nu + 1)]
        x_next_var = vars[(self._nx + self._nu + 1) : (2 * self._nx + self._nu + 1)]
        lambda_next_var = vars[-(self._n_contacts * self._n_force_basis_per_contact) :]

        return self._dynamics_constraint_evaluator(
            timestep, x_cur_var, u_cur_var, x_next_var, lambda_next_var
        )

    def _calc_stacked_contact_jacobian(
        self,
        plant: Union[MultibodyPlant, MultibodyPlant_[AutoDiffXd]],
        context: Union[Context, Context_[AutoDiffXd]],
        geom_id_pairs: List[Tuple[GeometryId, GeometryId]],
        wrt: JacobianWrtVariable = JacobianWrtVariable.kV,
    ) -> ArrayLikeType:
        J = np.zeros(
            (self._n_contacts * self._n_force_basis_per_contact, self._nv),
            dtype=np.float64 if isinstance(plant, MultibodyPlant) else object,
        )
        for i, geom_id_pair in enumerate(geom_id_pairs):
            J[
                i
                * self._n_force_basis_per_contact : (i + 1)
                * self._n_force_basis_per_contact
            ] = self._calc_jacobian_for_single_contact(
                plant, context, geom_id_pair, wrt
            )

        return J

    def _calc_jacobian_for_single_contact(
        self,
        plant: Union[MultibodyPlant, MultibodyPlant_[AutoDiffXd]],
        context: Union[Context, Context_[AutoDiffXd]],
        geom_id_pair: Tuple[GeometryId, GeometryId],
        wrt: JacobianWrtVariable,
    ) -> ArrayLikeType:
        force_basis = self._choose_force_basis(
            self._n_force_basis_per_contact, self._n_friction_directions
        )
        p_ACa, p_BCb, frameA, frameB, contact_normal = self._calc_witness_points(
            plant, context, geom_id_pair
        )

        Jv_WCa = plant.CalcJacobianTranslationalVelocity(
            context,
            wrt,
            frameA,
            p_ACa,
            plant.world_frame(),
            plant.world_frame(),
        )
        Jv_WCb = plant.CalcJacobianTranslationalVelocity(
            context,
            wrt,
            frameB,
            p_BCb,
            plant.world_frame(),
            plant.world_frame(),
        )
        R_WC = RotationMatrix.MakeFromOneVector(
            ExtractValue(contact_normal), 2
        ).matrix()
        J = force_basis @ R_WC.transpose() @ (Jv_WCa - Jv_WCb)
        return J

    def _force_basis_in_world_frame(
        self,
        contact_normal: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        force_basis = self._choose_force_basis(
            self._n_force_basis_per_contact, self._n_friction_directions
        )
        R_WC = RotationMatrix.MakeFromOneVector(
            ExtractValue(contact_normal), 2
        ).matrix()
        return force_basis @ R_WC.transpose()

    @staticmethod
    def _choose_force_basis(
        n_force_basis_per_contact: int, n_friction_directions: int
    ) -> npt.NDArray[np.float64]:
        force_basis = np.zeros((n_force_basis_per_contact, 3))
        force_basis[0, :] = [0, 0, 1]

        for i in range(n_friction_directions):
            theta = (np.pi * i) / n_friction_directions
            force_basis[2 * i + 1, :] = [np.cos(theta), np.sin(theta), 0]
            force_basis[2 * i + 2, :] = -force_basis[2 * i + 1, :]

        return force_basis.copy()

    @staticmethod
    def _calc_witness_points(
        plant: Union[MultibodyPlant, MultibodyPlant_[AutoDiffXd]],
        context: Union[Context, Context_[AutoDiffXd]],
        geom_id_pair: Tuple[GeometryId, GeometryId],
    ) -> Tuple[
        np.ndarray[AutoDiffXd],
        np.ndarray[AutoDiffXd],
        np.ndarray[AutoDiffXd],
        RigidBody,
        RigidBody,
        np.ndarray[AutoDiffXd],
    ]:
        query_port = plant.get_geometry_query_input_port()
        query_object = query_port.Eval(context)

        geom_id_A, geom_id_B = geom_id_pair

        signed_distance_pair = query_object.ComputeSignedDistancePairClosestPoints(
            geom_id_A, geom_id_B
        )

        inspector = query_object.inspector()
        frame_A_id = inspector.GetFrameId(geom_id_A)
        frame_B_id = inspector.GetFrameId(geom_id_B)

        frameA = plant.GetBodyFromFrameId(frame_A_id).body_frame()
        frameB = plant.GetBodyFromFrameId(frame_B_id).body_frame()
        pose_in_frameA = inspector.GetPoseInFrame(geom_id_A)
        pose_in_frameB = inspector.GetPoseInFrame(geom_id_B)

        p_ACa = (
            pose_in_frameA.rotation().matrix() @ signed_distance_pair.p_ACa
            + pose_in_frameA.translation()
        )
        p_BCb = (
            pose_in_frameB.rotation().matrix() @ signed_distance_pair.p_BCb
            + pose_in_frameB.translation()
        )
        contact_normal = signed_distance_pair.nhat_BA_W.copy()

        return (
            p_ACa,
            p_BCb,
            frameA,
            frameB,
            contact_normal,
        )

    @staticmethod
    def _get_signed_distance(
        plant: Union[MultibodyPlant, MultibodyPlant_[AutoDiffXd]],
        context: Union[Context, Context_[AutoDiffXd]],
        geom_id_pair: Tuple[GeometryId, GeometryId],
    ) -> AutoDiffXd:
        query_object = plant.get_geometry_query_input_port().Eval(context)

        return query_object.ComputeSignedDistancePairClosestPoints(
            *geom_id_pair
        ).distance

    def _add_dynamics_constraints(self) -> None:
        for i in range(self._planning_horizon):
            var_list = [
                self._timestep_var,
                self._x_vars[i],
                self._u_vars[i],
                self._x_vars[i + 1],
            ]
            var_list.extend([self._lambda_vars[i, j] for j in range(self._n_contacts)])
            self._dynamics_constraints[i] = self.prog.AddConstraint(
                self._dynamics_constraint_evaluator_helper,
                lb=np.zeros(self._nv),
                ub=np.zeros(self._nv),
                vars=np.hstack(var_list),
                description=f"dynamics_constraint_{i}",
            )

    def _add_contact_constraints(self) -> None:
        self._add_non_negativity_signed_distance_constraints()
        self._add_non_negativity_lambda_constraints()
        self._add_non_negativity_gamma_constraints()

        self._add_stick_separation_transition_constraints()
        self._add_contact_force_staying_in_friction_cone_constraints()
        self._add_max_sliding_velocity_constraints()
        self._add_stick_slip_transition_constraints()
        self._add_max_dissipation_constraints()

    def _signed_distance_equality_constraint_evaluator(
        self,
        x_next: ArrayLikeType,
        signed_distance_var: ArrayLikeType,
        contact_index: ArrayLikeType,
    ) -> ArrayLikeType:
        if isinstance(x_next[0], AutoDiffXd):
            plant = self._plant_ad
            context = self._plant_ad_context
        else:
            plant = self._plant
            context = self._plant_context
        plant.SetPositionsAndVelocities(context, x_next)
        return signed_distance_var - self._get_signed_distance(
            plant, context, self._contact_geom_id_pairs[contact_index]
        )

    def _signed_distance_equality_constraint_helper(
        self, vars: ArrayLikeType, contact_index: int
    ) -> ArrayLikeType:
        x_next, signed_distance_var = np.split(vars, [self._nx])
        return self._signed_distance_equality_constraint_evaluator(
            x_next, signed_distance_var, contact_index
        )

    def _add_non_negativity_signed_distance_constraints(self) -> None:
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                self.prog.AddConstraint(
                    partial(
                        self._signed_distance_equality_constraint_helper,
                        contact_index=j,
                    ),
                    lb=[0],
                    ub=[0],
                    vars=np.hstack(
                        [self._x_vars[i + 1], self._signed_distance_vars[i, j]]
                    ),
                    description=f"slack_var_equal_to_signed_distance_constraint_{i}_{j}",
                )
                self._non_negativity_signed_distance_constraints[i, j] = (
                    self.prog.AddBoundingBoxConstraint(
                        0,
                        np.inf,
                        self._signed_distance_vars[i, j],
                    )
                )
                self._non_negativity_signed_distance_constraints[
                    i, j
                ].evaluator().set_description(f"non_negativity_signed_distance_{i}_{j}")

    def _add_non_negativity_lambda_constraints(self) -> None:
        """Add the following constraints \lambda_{k, j, i} \geq 0."""
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                self._non_negativity_lambda_constraints[i, j] = (
                    self.prog.AddBoundingBoxConstraint(
                        np.zeros(self._n_force_basis_per_contact),
                        np.full(self._n_force_basis_per_contact, np.inf),
                        self._lambda_vars[i, j],
                    )
                )
                self._non_negativity_lambda_constraints[
                    i, j
                ].evaluator().set_description(f"non_negativity_lambda_{i}_{j}")

    def _add_non_negativity_gamma_constraints(self) -> None:
        """Add the following constraints \gamma_{k, j} \geq 0."""
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                self._non_negativity_gamma_constraints[i, j] = (
                    self.prog.AddBoundingBoxConstraint(
                        0,
                        np.inf,
                        self._slack_gamma_vars[i, j],
                    )
                )
                self._non_negativity_gamma_constraints[
                    i, j
                ].evaluator().set_description(f"non_negativity_gamma_{i}_{j}")

    def _add_contact_force_staying_in_friction_cone_constraints(self) -> None:
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                a = -np.ones(self._n_force_basis_per_contact + 1)
                a[0] = self._mu[j]
                beq = 0
                self._contact_force_staying_in_friction_cone_equality_constraints[
                    i, j
                ] = self.prog.AddLinearEqualityConstraint(
                    a,
                    beq,
                    np.hstack([self._lambda_vars[i, j], self._slack_beta_vars[i, j]]),
                )
                self._contact_force_staying_in_friction_cone_equality_constraints[
                    i, j
                ].evaluator().set_description(
                    f"contact_force_staying_in_friction_cone_equality_{i}_{j}"
                )

                self._contact_force_staying_in_friction_cone_inequality_constraints[
                    i, j
                ] = self.prog.AddBoundingBoxConstraint(
                    0,
                    np.inf,
                    self._slack_beta_vars[i, j],
                )

                self._contact_force_staying_in_friction_cone_inequality_constraints[
                    i, j
                ].evaluator().set_description(
                    f"contact_force_staying_in_friction_cone_inequality_{i}_{j}"
                )

    def _add_stick_separation_transition_constraints(self) -> None:
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                if self._stick_separation_transition_constraints[i, j] != 0:
                    self.prog.RemoveConstraint(
                        self._stick_separation_transition_constraints[i, j]
                    )
                self._stick_separation_transition_constraints[i, j] = (
                    self.prog.AddQuadraticConstraint(
                        self._signed_distance_vars[i, j][0]
                        * self._lambda_vars[i, j][0],
                        lb=-np.inf,
                        ub=self._epsilon,
                        hessian_type=QuadraticConstraint.HessianType.kIndefinite,
                    )
                )
                self._stick_separation_transition_constraints[
                    i, j
                ].evaluator().set_description(f"stick_separation_transition_{i}_{j}")

    def _add_stick_slip_transition_constraints(self) -> None:
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                if self._stick_slip_transition_constraints[i, j] != 0:
                    self.prog.RemoveConstraint(
                        self._stick_slip_transition_constraints[i, j]
                    )
                self._stick_slip_transition_constraints[i, j] = (
                    self.prog.AddQuadraticConstraint(
                        self._slack_beta_vars[i, j][0]
                        * self._slack_gamma_vars[i, j][0],
                        lb=-np.inf,
                        ub=self._epsilon,
                        hessian_type=QuadraticConstraint.HessianType.kIndefinite,
                    )
                )
                self._stick_slip_transition_constraints[
                    i, j
                ].evaluator().set_description(f"stick_slip_transition_{i}_{j}")

    def _calc_relative_velocity_in_contact_frame(
        self,
        plant: Union[MultibodyPlant, MultibodyPlant_[AutoDiffXd]],
        context: Union[Context, Context_[AutoDiffXd]],
        x_next: ArrayLikeType,
        geom_id_pair: Tuple[GeometryId, GeometryId],
    ) -> ArrayLikeType:
        contact_jacobian = self._calc_jacobian_for_single_contact(
            plant, context, geom_id_pair, JacobianWrtVariable.kV
        )
        return contact_jacobian @ x_next[self._nq :]

    def _max_sliding_velocity_constraint_evaluator(
        self,
        x_next: ArrayLikeType,
        slack_gamma: ArrayLikeType,
        slack_alpha: ArrayLikeType,
        contact_index: int,
        friction_basis_index: int,
    ):
        if isinstance(x_next[0], AutoDiffXd):
            plant = self._plant_ad
            context = self._plant_ad_context
        else:
            plant = self._plant
            context = self._plant_context
        plant.SetPositionsAndVelocities(context, x_next)
        geom_id_pair = self._contact_geom_id_pairs[contact_index]
        relative_velocity = self._calc_relative_velocity_in_contact_frame(
            plant, context, x_next, geom_id_pair
        )
        return [slack_gamma + relative_velocity[friction_basis_index] - slack_alpha]

    def _max_sliding_velocity_constraint_helper(
        self, vars: ArrayLikeType, contact_index: int, friction_basis_index: int
    ) -> ArrayLikeType:
        x_next, slack_gamma, slack_alpha = np.split(vars, [self._nx, self._nx + 1])
        return self._max_sliding_velocity_constraint_evaluator(
            x_next, slack_gamma, slack_alpha, contact_index, friction_basis_index
        )

    def _add_max_sliding_velocity_constraints(self) -> None:
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                for k in range(1, self._n_force_basis_per_contact):
                    self._max_sliding_velocity_equality_constraints[i, j, k - 1] = (
                        self.prog.AddConstraint(
                            partial(
                                self._max_sliding_velocity_constraint_helper,
                                contact_index=j,
                                friction_basis_index=k,
                            ),
                            lb=[0],
                            ub=[0],
                            vars=np.hstack(
                                [
                                    self._x_vars[i + 1],
                                    self._slack_gamma_vars[i, j],
                                    self._slack_alpha_vars[i, j][k - 1],
                                ]
                            ),
                            description=f"max_sliding_velocity_equality_{i}_{j}_{k}",
                        )
                    )

                    self._max_sliding_velocity_inequality_constraints[i, j, k - 1] = (
                        self.prog.AddBoundingBoxConstraint(
                            0,
                            np.inf,
                            self._slack_alpha_vars[i, j][k - 1],
                        )
                    )
                    self._max_sliding_velocity_inequality_constraints[
                        i, j, k - 1
                    ].evaluator().set_description(
                        f"max_sliding_velocity_inequality_{i}_{j}_{k}"
                    )

    def _add_max_dissipation_constraints(self) -> None:
        for i in range(self._planning_horizon):
            for j in range(self._n_contacts):
                for k in range(1, self._n_force_basis_per_contact):
                    if self._max_dissipation_constraints[i, j, k - 1] != 0:
                        self.prog.RemoveConstraint(
                            self._max_dissipation_constraints[i, j, k - 1]
                        )
                    self._max_dissipation_constraints[i, j, k - 1] = (
                        self.prog.AddQuadraticConstraint(
                            self._slack_alpha_vars[i, j][k - 1]
                            * self._lambda_vars[i, j][k],
                            lb=-np.inf,
                            ub=self._epsilon,
                            hessian_type=QuadraticConstraint.HessianType.kIndefinite,
                        )
                    )
                    self._max_dissipation_constraints[
                        i, j, k - 1
                    ].evaluator().set_description(f"max_dissipation_{i}_{j}_{k}")

    def _add_costs(
        self,
        Q: np.ndarray,
        Qf: np.ndarray,
        R: np.ndarray,
        target_state: np.ndarray,
    ) -> None:
        self._target_states = target_state
        self._create_default_cost_containers()
        for i in range(self._planning_horizon):
            self._running_state_costs[i] = self.prog.AddQuadraticErrorCost(
                Q,
                target_state,
                vars=self._x_vars[i],
            )
            self._input_costs[i] = self.prog.AddQuadraticErrorCost(
                R,
                np.zeros(R.shape[0]),
                vars=self._u_vars[i],
            )

        self._final_costs[-1] = self.prog.AddQuadraticErrorCost(
            Qf,
            target_state,
            self._x_vars[-1],
        )

    def update_complementarity_relaxation(self, new_epsilon: float) -> None:
        self._epsilon = new_epsilon
        self._add_stick_separation_transition_constraints()
        self._add_stick_slip_transition_constraints()
        self._add_max_dissipation_constraints()


# utility functions for planning trajectories for simplified trifinger with cube
def plan_single_traj(
    drake_system: DrakeSystem,
    trajopt_configs: TrajOptConfigs,
    x0: np.ndarray = None,
    target_state: np.ndarray = None,
):
    trajopt = ContactImplicitTrajOpt(
        plant=drake_system.plant,
        plant_context=drake_system.plant_context,
        plant_ad=drake_system.plant_ad,
        plant_ad_context=drake_system.plant_ad_context,
        planning_horizon=trajopt_configs.planning_horizon,
        contact_pairs=drake_system.contact_pairs,
        frictional_coefficients=drake_system.contact_friction_coeffs,
        default_complementarity_relaxation=trajopt_configs.complementarity_relaxation,
        default_timestep=drake_system.plant.time_step(),
    )

    trajopt.set_up_trajopt_problem(
        x0,
        trajopt_configs.x_lb,
        trajopt_configs.x_ub,
        trajopt_configs.u_lb,
        trajopt_configs.u_ub,
        trajopt_configs.Q,
        trajopt_configs.Qf,
        trajopt_configs.R,
        target_state,
    )

    # print out the constructed MathematicalProgram and redirect the output to a file.
    if trajopt_configs.print_math_prog:
        with open("math_prog_printout.txt", "w") as f:
            with redirect_stdout(f):
                print(trajopt.prog)

    epsilons = [1e-4, 1e-6, 0.0]
    warm_up_solution = None
    ret = None

    for epsilon in epsilons:
        print(f"Solve trajopt with the following epsilon: {epsilon}\n")
        trajopt.update_complementarity_relaxation(epsilon)
        ret = trajopt.find_traj(
            x0,
            target_state,
            warm_up_solution=warm_up_solution,
        )
        if ret.is_success():
            warm_up_solution = ret

    if ret.is_success():
        print("Planning trajectory is successful")
    else:
        print("Planning trajectory failed")
    return None


def plan_multiple_trajectories(
    drake_system: DrakeSystem,
    trajopt_configs: TrajOptConfigs,
    n_trajs: int,
) -> None:
    for i in range(n_trajs):
        rand_target_angle = np.random.uniform(low=2.0, high=5.0)
        sign_angle = np.random.choice([-1, 1])
        print(
            f"Planning trajectory {i} for target angle: {rand_target_angle * sign_angle}..."
        )
        initial_state = trajopt_configs.default_initial_state.copy()
        target_state = initial_state.copy()
        target_state[9:13] = angle_axis_to_quaternion(
            rand_target_angle, np.array([0, 0, 1])
        )
        ret = plan_single_traj(
            drake_system=drake_system,
            trajopt_configs=trajopt_configs,
            x0=initial_state,
            target_state=target_state,
        )
        if ret is not None:
            print(f"Planning trajectory {i} is successful")
        else:
            print(f"Planning trajectory {i} failed")


if __name__ == "__main__":
    system_config_path = (
        Path(__file__).parent / "configs" / "trifinger_with_cube_system_configs.yaml"
    )
    trajopt_config_path = Path(__file__).parent / "configs" / "trajopt_configs.yaml"
    drake_system = DrakeSystem.from_config(system_config_path)
    trajopt_configs = TrajOptConfigs.from_yaml(trajopt_config_path)

    plan_multiple_trajectories(drake_system, trajopt_configs, 1)
