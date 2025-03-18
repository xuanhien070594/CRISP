#include "math.h"

#include <limits>

#include "solver_core/SolverInterface.h"
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
                   Eigen::VectorXd::Zero((N - 1) * num_state)) {}

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
                   Eigen::VectorXd::Constant((N - 1) * 12, std::numeric_limits<double>::infinity())) {}

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
                   Eigen::VectorXd::Constant((N - 1) * num_control, std::numeric_limits<double>::infinity())) {}

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
  OptimizationProblem pushboxProblem(variableNum, problemName);

  pushboxProblem.parseDrakeMathematicalProgram(prog);

  scalar_t obj = pushboxProblem.evaluateObjective(xInitialGuess);
  std::cout << "Objective value: " << obj << std::endl;

  vector_t equalityConstraints = pushboxProblem.evaluateEqualityConstraints(xInitialGuess);
  std::cout << "Equality constraints: " << equalityConstraints.transpose() << std::endl;

  vector_t inequalityConstraints = pushboxProblem.evaluateInequalityConstraints(xInitialGuess);
  std::cout << "inEquality constraints: " << inequalityConstraints.transpose() << std::endl;

  SolverParameters params;
  SolverInterface solver(pushboxProblem, params);
  solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
  solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
  solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
  solver.initialize(xInitialGuess);
  solver.solve();
  xOptimal = solver.getSolution();
}
