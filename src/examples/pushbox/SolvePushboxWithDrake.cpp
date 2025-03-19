#include "math.h"

#include <limits>

#include "solver_core/SolverInterface.h"
#include <common/BasicTypes.h>
#include <eigen3/Eigen/src/Core/Matrix.h>

#include "drake/common/eigen_autodiff_types.h"
#include "drake/common/symbolic/expression.h"

using namespace CRISP;
using drake::AutoDiffVecXd;
using drake::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;

// Define model model parameters for pushbox
const scalar_t a = 0.5;
const scalar_t b = 0.25;
const scalar_t m = 1;
const scalar_t mu = 0.5;
const scalar_t g = 9.8;
const scalar_t r = sqrt(a * a + b * b);
const scalar_t c = 0.4;
const scalar_t dt = 0.02;
const size_t N = 100;  // number of time steps
const size_t num_state = 3;
const size_t num_control = 6;

// define the dynamics constraints
class PushboxDynamicConstraint : public Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PushboxDynamicConstraint);
  PushboxDynamicConstraint()
      : Constraint((N - 1) * num_state, N * (num_state + num_control), Eigen::VectorXd::Zero((N - 1) * num_state),
                   Eigen::VectorXd::Zero((N - 1) * num_state)) {
    std::vector<std::pair<int, int>> gradient_sparsity_pattern;
    for (size_t i = 0; i < N - 1; ++i) {
      size_t idx = i * (num_state + num_control);
      size_t next_idx = idx + (num_state + num_control);

      // y[i * num_state + 0] depends on:
      gradient_sparsity_pattern.emplace_back(i * num_state + 0, next_idx + 0);  // px_next
      gradient_sparsity_pattern.emplace_back(i * num_state + 0, idx + 0);       // px_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 0, idx + 2);       // theta_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 0, idx + 5);       // lambda1_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 0, idx + 6);       // lambda2_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 0, idx + 7);       // lambda3_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 0, idx + 8);       // lambda4_i

      // y[i * num_state + 1] depends on:
      gradient_sparsity_pattern.emplace_back(i * num_state + 1, next_idx + 1);  // py_next
      gradient_sparsity_pattern.emplace_back(i * num_state + 1, idx + 1);       // py_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 1, idx + 2);       // theta_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 1, idx + 5);       // lambda1_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 1, idx + 6);       // lambda2_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 1, idx + 7);       // lambda3_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 1, idx + 8);       // lambda4_i

      // y[i * num_state + 2] depends on:
      gradient_sparsity_pattern.emplace_back(i * num_state + 2, next_idx + 2);  // theta_next
      gradient_sparsity_pattern.emplace_back(i * num_state + 2, idx + 2);       // theta_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 2, idx + 3);       // cx_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 2, idx + 4);       // cy_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 2, idx + 5);       // lambda1_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 2, idx + 6);       // lambda2_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 2, idx + 7);       // lambda3_i
      gradient_sparsity_pattern.emplace_back(i * num_state + 2, idx + 8);       // lambda4_i
    }
    SetGradientSparsityPattern(gradient_sparsity_pattern);
  }

 private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
    y->resize((N - 1) * num_state);
    for (size_t i = 0; i < N - 1; ++i) {
      size_t idx = i * (num_state + num_control);
      // Extract state and control for current and next time steps
      auto px_i = x[idx + 0];
      auto py_i = x[idx + 1];
      auto theta_i = x[idx + 2];
      auto cx_i = x[idx + 3];
      auto cy_i = x[idx + 4];
      auto lambda1_i = x[idx + 5];
      auto lambda2_i = x[idx + 6];
      auto lambda3_i = x[idx + 7];
      auto lambda4_i = x[idx + 8];

      auto px_next = x[idx + (num_state + num_control) + 0];
      auto py_next = x[idx + (num_state + num_control) + 1];
      auto theta_next = x[idx + (num_state + num_control) + 2];

      auto px_dot =
          (1 / (mu * m * g)) * (cos(theta_i) * (lambda2_i + lambda4_i) - sin(theta_i) * (lambda1_i + lambda3_i));
      auto py_dot =
          (1 / (mu * m * g)) * (sin(theta_i) * (lambda2_i + lambda4_i) + cos(theta_i) * (lambda1_i + lambda3_i));
      auto theta_dot = (1 / (mu * m * g * c * r)) * (-cy_i * (lambda2_i + lambda4_i) + cx_i * (lambda1_i + lambda3_i));

      // Explicit State Update
      y->segment(i * num_state, num_state) << px_next - px_i - px_dot * dt, py_next - py_i - py_dot * dt,
          theta_next - theta_i - theta_dot * dt;
    }
  }

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const override { DoEvalGeneric(x, y); }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const override { DoEvalGeneric(x, y); }
  void DoEval(const Eigen::Ref<const VectorX<Variable>>&, VectorX<Expression>*) const override {
    throw std::logic_error("PositionConstraint::DoEval() does not work for symbolic variables.");
  }
};

// contact implicit constraints for pushbox
class PushboxContactConstraint : public Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PushboxContactConstraint);
  PushboxContactConstraint()
      : Constraint((N - 1) * 12, N * (num_state + num_control), Eigen::VectorXd::Zero((N - 1) * 12),
                   Eigen::VectorXd::Constant((N - 1) * 12, std::numeric_limits<double>::infinity())) {
    std::vector<std::pair<int, int>> gradient_sparsity_pattern;
    for (size_t i = 0; i < N - 1; ++i) {
      size_t idx = i * (num_state + num_control);

      // y[i * 12 + 0] depends on lambda1_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 0, idx + 5);

      // y[i * 12 + 1] depends on lambda2_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 1, idx + 6);

      // y[i * 12 + 2] depends on lambda3_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 2, idx + 7);

      // y[i * 12 + 3] depends on lambda4_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 3, idx + 8);

      // y[i * 12 + 4] depends on cy_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 4, idx + 4);

      // y[i * 12 + 5] depends on cx_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 5, idx + 3);

      // y[i * 12 + 6] depends on cy_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 6, idx + 4);

      // y[i * 12 + 7] depends on cx_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 7, idx + 3);

      // y[i * 12 + 8] depends on lambda1_i and cy_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 8, idx + 5);
      gradient_sparsity_pattern.emplace_back(i * 12 + 8, idx + 4);

      // y[i * 12 + 9] depends on lambda2_i and cx_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 9, idx + 6);
      gradient_sparsity_pattern.emplace_back(i * 12 + 9, idx + 3);

      // y[i * 12 + 10] depends on lambda3_i and cy_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 10, idx + 7);
      gradient_sparsity_pattern.emplace_back(i * 12 + 10, idx + 4);

      // y[i * 12 + 11] depends on lambda4_i and cx_i
      gradient_sparsity_pattern.emplace_back(i * 12 + 11, idx + 8);
      gradient_sparsity_pattern.emplace_back(i * 12 + 11, idx + 3);
    }
    SetGradientSparsityPattern(gradient_sparsity_pattern);
  }

 private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
    y->resize((N - 1) * 12);
    for (size_t i = 0; i < N - 1; ++i) {
      size_t idx = i * (num_state + num_control);
      auto cx_i = x[idx + 3];
      auto cy_i = x[idx + 4];
      auto lambda1_i = x[idx + 5];
      auto lambda2_i = x[idx + 6];
      auto lambda3_i = x[idx + 7];
      auto lambda4_i = x[idx + 8];

      y->segment(i * 12, 12) << lambda1_i, lambda2_i, -lambda3_i, -lambda4_i, cy_i + b, cx_i + a, b - cy_i, a - cx_i,
          -(lambda1_i) * (cy_i + b), -(lambda2_i) * (cx_i + a), -(-lambda3_i) * (b - cy_i), -(-lambda4_i) * (a - cx_i);
    }
  }

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const override { DoEvalGeneric(x, y); }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const override { DoEvalGeneric(x, y); }
  void DoEval(const Eigen::Ref<const VectorX<Variable>>&, VectorX<Expression>*) const override {
    throw std::logic_error("PositionConstraint::DoEval() does not work for symbolic variables.");
  }
};

// allow only one contact force at a time
class PushboxContactSingleForceConstraint : public Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PushboxContactSingleForceConstraint);
  PushboxContactSingleForceConstraint()
      : Constraint((N - 1) * num_control, N * (num_state + num_control), Eigen::VectorXd::Zero((N - 1) * num_control),
                   Eigen::VectorXd::Constant((N - 1) * num_control, std::numeric_limits<double>::infinity())) {
    std::vector<std::pair<int, int>> gradient_sparsity_pattern;
    for (size_t i = 0; i < N - 1; ++i) {
      size_t idx = i * (num_state + num_control);

      // y[i * 6 + 0] depends on lambda1_i and lambda2_i
      gradient_sparsity_pattern.emplace_back(i * 6 + 0, idx + 5);
      gradient_sparsity_pattern.emplace_back(i * 6 + 0, idx + 6);

      // y[i * 6 + 1] depends on lambda1_i and lambda3_i
      gradient_sparsity_pattern.emplace_back(i * 6 + 1, idx + 5);
      gradient_sparsity_pattern.emplace_back(i * 6 + 1, idx + 7);

      // y[i * 6 + 2] depends on lambda1_i and lambda4_i
      gradient_sparsity_pattern.emplace_back(i * 6 + 2, idx + 5);
      gradient_sparsity_pattern.emplace_back(i * 6 + 2, idx + 8);

      // y[i * 6 + 3] depends on lambda2_i and lambda3_i
      gradient_sparsity_pattern.emplace_back(i * 6 + 3, idx + 6);
      gradient_sparsity_pattern.emplace_back(i * 6 + 3, idx + 7);

      // y[i * 6 + 4] depends on lambda2_i and lambda4_i
      gradient_sparsity_pattern.emplace_back(i * 6 + 4, idx + 6);
      gradient_sparsity_pattern.emplace_back(i * 6 + 4, idx + 8);

      // y[i * 6 + 5] depends on lambda3_i and lambda4_i
      gradient_sparsity_pattern.emplace_back(i * 6 + 5, idx + 7);
      gradient_sparsity_pattern.emplace_back(i * 6 + 5, idx + 8);
    }
    SetGradientSparsityPattern(gradient_sparsity_pattern);
  }

 private:
  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* y) const {
    y->resize((N - 1) * 6);
    for (size_t i = 0; i < N - 1; ++i) {
      size_t idx = i * (num_state + num_control);
      auto lambda1_i = x[idx + 5];
      auto lambda2_i = x[idx + 6];
      auto lambda3_i = x[idx + 7];
      auto lambda4_i = x[idx + 8];

      y->segment(i * 6, 6) << -(lambda1_i * lambda2_i), -(lambda1_i * (-lambda3_i)), -(lambda1_i * (-lambda4_i)),
          -(lambda2_i * (-lambda3_i)), -(lambda2_i * (-lambda4_i)), -(-lambda3_i * (-lambda4_i));
    }
  }

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x, Eigen::VectorXd* y) const override { DoEvalGeneric(x, y); }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) const override { DoEvalGeneric(x, y); }
  void DoEval(const Eigen::Ref<const VectorX<Variable>>&, VectorX<Expression>*) const override {
    throw std::logic_error("PositionConstraint::DoEval() does not work for symbolic variables.");
  }
};

// define the dynamics constraints
ad_function_t pushboxDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
  y.resize((N - 1) * num_state);
  for (size_t i = 0; i < N - 1; ++i) {
    size_t idx = i * (num_state + num_control);
    // Extract state and control for current and next time steps
    ad_scalar_t px_i = x[idx + 0];
    ad_scalar_t py_i = x[idx + 1];
    ad_scalar_t theta_i = x[idx + 2];
    ad_scalar_t cx_i = x[idx + 3];
    ad_scalar_t cy_i = x[idx + 4];
    ad_scalar_t lambda1_i = x[idx + 5];
    ad_scalar_t lambda2_i = x[idx + 6];
    ad_scalar_t lambda3_i = x[idx + 7];
    ad_scalar_t lambda4_i = x[idx + 8];

    ad_scalar_t px_next = x[idx + (num_state + num_control) + 0];
    ad_scalar_t py_next = x[idx + (num_state + num_control) + 1];
    ad_scalar_t theta_next = x[idx + (num_state + num_control) + 2];

    ad_scalar_t px_dot =
        (1 / (mu * m * g)) * (cos(theta_i) * (lambda2_i + lambda4_i) - sin(theta_i) * (lambda1_i + lambda3_i));
    ad_scalar_t py_dot =
        (1 / (mu * m * g)) * (sin(theta_i) * (lambda2_i + lambda4_i) + cos(theta_i) * (lambda1_i + lambda3_i));
    ad_scalar_t theta_dot =
        (1 / (mu * m * g * c * r)) * (-cy_i * (lambda2_i + lambda4_i) + cx_i * (lambda1_i + lambda3_i));

    // Explicit State Update
    y.segment(i * num_state, num_state) << px_next - px_i - px_dot * dt, py_next - py_i - py_dot * dt,
        theta_next - theta_i - theta_dot * dt;
  }
};

// contact implicit constraints for pushbox

ad_function_t pushboxContactConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
  y.resize((N - 1) * 12);
  for (size_t i = 0; i < N - 1; ++i) {
    size_t idx = i * (num_state + num_control);
    ad_scalar_t px_i = x[idx + 0];
    ad_scalar_t py_i = x[idx + 1];
    ad_scalar_t theta_i = x[idx + 2];
    ad_scalar_t cx_i = x[idx + 3];
    ad_scalar_t cy_i = x[idx + 4];
    ad_scalar_t lambda1_i = x[idx + 5];
    ad_scalar_t lambda2_i = x[idx + 6];
    ad_scalar_t lambda3_i = x[idx + 7];
    ad_scalar_t lambda4_i = x[idx + 8];

    y.segment(i * 12, 12) << lambda1_i, lambda2_i, -lambda3_i, -lambda4_i, cy_i + b, cx_i + a, b - cy_i, a - cx_i,
        -(lambda1_i) * (cy_i + b), -(lambda2_i) * (cx_i + a), -(-lambda3_i) * (b - cy_i), -(-lambda4_i) * (a - cx_i);
  }
};

// allow only one contact force at a time
ad_function_t pushboxContactSingleForceConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
  y.resize((N - 1) * 6);
  for (size_t i = 0; i < N - 1; ++i) {
    size_t idx = i * (num_state + num_control);
    ad_scalar_t lambda1_i = x[idx + 5];
    ad_scalar_t lambda2_i = x[idx + 6];
    ad_scalar_t lambda3_i = x[idx + 7];
    ad_scalar_t lambda4_i = x[idx + 8];

    y.segment(i * 6, 6) << -(lambda1_i * lambda2_i), -(lambda1_i * (-lambda3_i)), -(lambda1_i * (-lambda4_i)),
        -(lambda2_i * (-lambda3_i)), -(lambda2_i * (-lambda4_i)), -(-lambda3_i * (-lambda4_i));
  }
};

// initial constraints
ad_function_with_param_t pushboxInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
  y.resize(3);
  y.segment(0, 3) << x[0] - p[0], x[1] - p[1], x[2] - p[2];
};

// cost function for pushbox
ad_function_with_param_t pushboxObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
  y.resize(1);
  y[0] = 0.0;
  ad_scalar_t tracking_cost(0.0);
  ad_scalar_t control_cost(0.0);
  for (size_t i = 0; i < N; ++i) {
    size_t idx = i * (num_state + num_control);
    ad_scalar_t px_i = x[idx + 0];
    ad_scalar_t py_i = x[idx + 1];
    ad_scalar_t theta_i = x[idx + 2];
    ad_scalar_t cx_i = x[idx + 3];
    ad_scalar_t cy_i = x[idx + 4];
    ad_scalar_t lambda1_i = x[idx + 5];
    ad_scalar_t lambda2_i = x[idx + 6];
    ad_scalar_t lambda3_i = x[idx + 7];
    ad_scalar_t lambda4_i = x[idx + 8];
    ad_matrix_t Q(num_state, num_state);
    Q.setZero();
    Q(0, 0) = 100;
    Q(1, 1) = 100;
    Q(2, 2) = 100;
    ad_matrix_t R(4, 4);
    R.setZero();
    R(0, 0) = 0.001;
    R(1, 1) = 0.001;
    R(2, 2) = 0.001;
    R(3, 3) = 0.001;

    if (i == N - 1) {
      ad_vector_t tracking_error(num_state);

      tracking_error << px_i - p[0], py_i - p[1], theta_i - p[2];
      tracking_cost += tracking_error.transpose() * Q * tracking_error;
    }

    if (i < N - 1) {
      ad_vector_t control_error(4);
      control_error << lambda1_i, lambda2_i, lambda3_i, lambda4_i;
      control_cost += control_error.transpose() * R * control_error;
    }
  }
  y[0] = tracking_cost + control_cost;
};

int main() {
  size_t variableNum = N * (num_state + num_control);
  vector_t xInitialStates(num_state);
  xInitialStates.setZero();
  vector_t xInitialGuess(variableNum);
  xInitialGuess.setZero();

  vector_t xFinalStates(num_state);
  size_t num_segments = 18;
  scalar_t theta = 12 * 2 * M_PI / num_segments;
  xFinalStates << 3 * cos(theta), 3 * sin(theta), theta;

  vector_t xOptimal(variableNum);

  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables(N * (num_state + num_control), "x");
  prog.AddConstraint(std::make_shared<PushboxDynamicConstraint>(), x);
  prog.AddConstraint(std::make_shared<PushboxContactConstraint>(), x);
  prog.AddConstraint(std::make_shared<PushboxContactSingleForceConstraint>(), x);
  prog.AddLinearEqualityConstraint(x(0), xInitialStates(0));
  prog.AddLinearEqualityConstraint(x(1), xInitialStates(1));
  prog.AddLinearEqualityConstraint(x(2), xInitialStates(2));

  // add state tracking cost (only at the final state)
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(num_state, num_state) * 100;
  Eigen::VectorXd b(num_state);
  b = -2 * xFinalStates.transpose() * Q;
  double c = xFinalStates.transpose() * Q * xFinalStates;
  prog.AddQuadraticCost(2 * Q, b, c, x.segment((N - 1) * (num_state + num_control), num_state));

  // add control cost
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(4, 4) * 0.001;
  for (int i = 0; i < N - 1; ++i) {
    size_t idx = i * (num_state + num_control);
    prog.AddQuadraticCost(2 * R, Eigen::VectorXd::Zero(4), x.segment(idx + num_state + 2, 4));
  }

  std::string problemName = "Pushbox";
  std::string folderName = "model";
  OptimizationProblem pushboxProblemWithDrake(variableNum, problemName);
  pushboxProblemWithDrake.parseDrakeMathematicalProgram(prog);

  OptimizationProblem pushboxProblem(variableNum, problemName);
  auto obj = std::make_shared<ObjectiveFunction>(variableNum, num_state, problemName, folderName, "pushboxObjective",
                                                 pushboxObjective);
  auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName,
                                                       "pushboxDynamicConstraints", pushboxDynamicConstraints);
  auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactConstraints",
                                                      pushboxContactConstraints);
  auto initial = std::make_shared<ConstraintFunction>(variableNum, num_state, problemName, folderName,
                                                      "pushboxInitialConstraints", pushboxInitialConstraints);
  auto contactSingleForce =
      std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushboxContactSingleForceConstraints",
                                           pushboxContactSingleForceConstraints);
  pushboxProblem.addObjective(obj);
  pushboxProblem.addEqualityConstraint(dynamics);
  pushboxProblem.addEqualityConstraint(initial);
  pushboxProblem.addInequalityConstraint(contact);
  pushboxProblem.addInequalityConstraint(contactSingleForce);
  pushboxProblem.setParameters("pushboxInitialConstraints", xInitialStates);
  pushboxProblem.setParameters("pushboxObjective", xFinalStates);

  scalar_t objValueDrake = pushboxProblemWithDrake.evaluateObjective(xInitialGuess);
  scalar_t objValue = pushboxProblem.evaluateObjective(xInitialGuess);
  std::cout << "Objective value from Drake is same as CRISP: " << (objValueDrake == objValue) << std::endl;

  sparse_matrix_t objJacobianWithDrake = pushboxProblemWithDrake.evaluateObjectiveGradient(xInitialGuess);
  sparse_matrix_t objJacobian = pushboxProblem.evaluateObjectiveGradient(xInitialGuess);
  std::cout << "Objective Jacobian from Drake is same as CRISP: " << objJacobianWithDrake.isApprox(objJacobian, 1e-8)
            << std::endl;
  // CSRSparseMatrix objJacobianCSRWithDrake = pushboxProblemWithDrake.evaluateObjectiveGradientCSR(xInitialGuess);
  // CSRSparseMatrix objJacobianCSR = pushboxProblem.evaluateObjectiveGradientCSR(xInitialGuess);
  // objJacobianCSRWithDrake.print();
  // objJacobianCSR.print();

  vector_t equalityConstraintsWithDrake = pushboxProblemWithDrake.evaluateEqualityConstraints(xInitialGuess);
  vector_t equalityConstraints = pushboxProblem.evaluateEqualityConstraints(xInitialGuess);
  std::cout << "Equality Constraints from Drake is same as CRISP: "
            << equalityConstraintsWithDrake.isApprox(equalityConstraints, 1e-8) << std::endl;

  sparse_matrix_t equalityJacobianWithDrake =
      pushboxProblemWithDrake.evaluateEqualityConstraintsJacobian(xInitialGuess);
  sparse_matrix_t equalityJacobian = pushboxProblem.evaluateEqualityConstraintsJacobian(xInitialGuess);
  std::cout << "Equality Jacobian from Drake is same as CRISP: "
            << equalityJacobianWithDrake.toDense().isApprox(equalityJacobian.toDense(), 1e-8) << std::endl;

  // CSRSparseMatrix equalityJacobianCSRWithDrake =
  //     pushboxProblemWithDrake.evaluateEqualityConstraintsJacobianCSR(xInitialGuess);
  // CSRSparseMatrix equalityJacobianCSR = pushboxProblem.evaluateEqualityConstraintsJacobianCSR(xInitialGuess);
  // equalityJacobianCSRWithDrake.print();
  // equalityJacobianCSR.print();

  vector_t inequalityConstraintsWithDrake = pushboxProblemWithDrake.evaluateInequalityConstraints(xInitialGuess);
  vector_t inequalityConstraints = pushboxProblem.evaluateInequalityConstraints(xInitialGuess);
  std::cout << "Inequality Constraints from Drake is same as CRISP: "
            << inequalityConstraintsWithDrake.isApprox(inequalityConstraints, 1e-8) << std::endl;

  sparse_matrix_t inequalityJacobianWithDrake =
      pushboxProblemWithDrake.evaluateInequalityConstraintsJacobian(xInitialGuess);
  sparse_matrix_t inequalityJacobian = pushboxProblem.evaluateInequalityConstraintsJacobian(xInitialGuess);
  std::cout << "Inequality Jacobian from Drake is same as CRISP: "
            << inequalityJacobianWithDrake.isApprox(inequalityJacobian, 1e-8) << std::endl;

  SolverParameters params;
  SolverInterface solver(pushboxProblemWithDrake, params);
  solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
  solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
  solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
  solver.setHyperParameters("verbose", vector_t::Constant(1, 1));
  solver.initialize(xInitialGuess);
  solver.solve();
  xOptimal = solver.getSolution();
}
