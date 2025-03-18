#include "solver_core/SolverInterface.h"
// #include "common/MatlabHelper.h"
#include "math.h"

#include <chrono>

using namespace CRISP;

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
  std::string problemName = "Pushbox";
  std::string folderName = "model";
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

  // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library
  // for this problem and the usage in python ! ---------------------- //

  pushboxProblem.addObjective(obj);
  pushboxProblem.addEqualityConstraint(dynamics);
  pushboxProblem.addEqualityConstraint(initial);
  pushboxProblem.addInequalityConstraint(contact);
  pushboxProblem.addInequalityConstraint(contactSingleForce);

  // problem parameters
  vector_t xInitialStates(num_state);
  vector_t xFinalStates(num_state);
  vector_t xInitialGuess(variableNum);
  vector_t xOptimal(variableNum);
  // define a theta from 0 to 2pi, and define different final state for the problem with equal interval, for example 20
  // degree
  xInitialStates << 0, 0, 0;
  // set zero initial guess
  xInitialGuess.setZero();
  SolverParameters params;
  SolverInterface solver(pushboxProblem, params);
  // solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
  solver.setProblemParameters("pushboxInitialConstraints", xInitialStates);

  //   vector_t equalityConstraints = pushboxProblem.evaluateEqualityConstraints(xInitialGuess);
  //   std::cout << "Equality constraints: " << equalityConstraints.transpose() << std::endl;

  size_t num_segments = 18;
  scalar_t theta = 12 * 2 * M_PI / num_segments;
  xFinalStates << 3 * cos(theta), 3 * sin(theta), theta;
  std::cout << "Desired state: " << xFinalStates.transpose() << std::endl;
  solver.setProblemParameters("pushboxObjective", xFinalStates);
  scalar_t objValue = pushboxProblem.evaluateObjective(xInitialGuess);
  solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
  solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
  solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
  std::cout << "Objective value: " << objValue << std::endl;
  vector_t equalityConstraints = pushboxProblem.evaluateEqualityConstraints(xInitialGuess);
  std::cout << "Equality constraints: " << equalityConstraints.transpose() << std::endl;
  vector_t inequalityConstraints = pushboxProblem.evaluateInequalityConstraints(xInitialGuess);
  std::cout << "inEquality constraints: " << inequalityConstraints.transpose() << std::endl;
  solver.initialize(xInitialGuess);
  solver.solve();
  xOptimal = solver.getSolution();
}
