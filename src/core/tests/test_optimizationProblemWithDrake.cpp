#include "Eigen/Dense"
#include "problem_core/OptimizationProblem.h"
#include <common/BasicTypes.h>

#include "drake/solvers/mathematical_program.h"

using namespace CRISP;

using drake::solvers::Binding;
using drake::solvers::MathematicalProgram;
using drake::solvers::QuadraticCost;
using Eigen::MatrixXd;

void EvaluateCost(const Binding<QuadraticCost>& cost, const vector_t& x, vector_t& output) {
  cost.evaluator()->Eval(x, &output);
}

int main() {
  MathematicalProgram prog;

  auto x = prog.NewContinuousVariables(3, "x");
  vector_t ref(3);
  ref << 1.0, 2.0, 0.0;
  vector_t x0(1);
  vector_t x1(1);
  x0 << 0.5;
  x1 << 1.5;

  // Add cost function
  MatrixXd Q(3, 3);
  Q << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  vector_t b(3);
  b = -2 * ref.transpose() * Q;
  double c = ref(0) * ref(0) + ref(1) * ref(1) + ref(2) * ref(2);
  Binding<QuadraticCost> objectiveFunction = prog.AddQuadraticCost(2 * Q, b, c, x.head(3), true);

  // Add dynamics constraint
  matrix_t Aeq(1, 3);
  Aeq << -1, 1, -1;
  vector_t beq(1);
  beq << 0;
  Binding<Constraint> dynamicsConstraint = prog.AddLinearEqualityConstraint(Aeq, beq, x.head(3));

  // Add initial and final state constraints
  // matrix_t Aeq_state_constraint(2, 2);
  // Aeq_state_constraint << 1, 0, 0, 1;
  // vector_t beq_state_constraint(2);
  // beq_state_constraint << 0.5, 1.5;
  // Binding<Constraint> stateConstraints =
  //     prog.AddLinearEqualityConstraint(Aeq_state_constraint, beq_state_constraint, x.head(2));

  Binding<Constraint> initialStateConstraint = prog.AddLinearEqualityConstraint(x(0), x0(0));
  Binding<Constraint> finalStateConstraint = prog.AddLinearEqualityConstraint(x(1), x1(0));

  // Add control limit constraints
  Binding<Constraint> controlLimitConstraint = prog.AddBoundingBoxConstraint(-1, 1, x(2));

  /// Transfer Drake MathematicalProgram to CRISP OptimizationProblem
  int variableNum = prog.num_vars();
  std::string problemName = "Test_Tracking_Problem";
  std::string folderName = "model";
  OptimizationProblem trackingProblem(variableNum, problemName);

  // 2. create the objective and constraints functions, register the parameters
  auto obj = std::make_shared<ObjectiveFunction>(variableNum, "quadraticObjective", &objectiveFunction);
  auto dynamics = std::make_shared<ConstraintFunction>(variableNum, "dynamicsConstraint", &dynamicsConstraint);
  auto controlLimit =
      std::make_shared<ConstraintFunction>(variableNum, "controlLimitConstraint", &controlLimitConstraint);
  // auto initialState = std::make_shared<ConstraintFunction>(variableNum, "initialStateConstraint", &stateConstraints);
  auto initialState =
      std::make_shared<ConstraintFunction>(variableNum, "initialStateConstraint", &initialStateConstraint);
  auto finalState = std::make_shared<ConstraintFunction>(variableNum, "initialStateConstraint", &finalStateConstraint);
  obj->setDecisionVariableIndices(prog);
  dynamics->setDecisionVariableIndices(prog);
  initialState->setDecisionVariableIndices(prog);
  finalState->setDecisionVariableIndices(prog);
  controlLimit->setDecisionVariableIndices(prog);

  // 3. Add the objective and constraints to the optimization problem
  trackingProblem.addObjective(obj);
  trackingProblem.addEqualityConstraint(dynamics);
  trackingProblem.addInequalityConstraint(controlLimit);
  trackingProblem.addEqualityConstraint(initialState);
  trackingProblem.addEqualityConstraint(finalState);

  // Evaluate the objective
  vector_t test_x(5);
  test_x << 10.0, 20.0, 1.5;
  scalar_t objValue = trackingProblem.evaluateObjective(test_x);
  std::cout << "Objective value: " << objValue << std::endl;
  sparse_matrix_t objGradient = trackingProblem.evaluateObjectiveGradient(test_x);
  std::cout << "Objective gradient: " << objGradient.toDense() << std::endl;

  CSRSparseMatrix objGradientCSR = trackingProblem.evaluateObjectiveGradientCSR(test_x);
  objGradientCSR.print();

  sparse_matrix_t objHessian = trackingProblem.evaluateObjectiveHessian(test_x);
  std::cout << "Objective hessian: \n" << objHessian.toDense() << std::endl;

  // Evaluate the constraints
  vector_t equalityConstraints = trackingProblem.evaluateEqualityConstraints(test_x);
  std::cout << "Equality constraints: " << equalityConstraints.transpose() << std::endl;
  sparse_matrix_t equalityJacobian = trackingProblem.evaluateEqualityConstraintsJacobian(test_x);
  std::cout << "Equality constraints Jacobian: \n" << equalityJacobian.toDense() << std::endl;
  CSRSparseMatrix equalityGradientCSR = trackingProblem.evaluateEqualityConstraintsJacobianCSR(test_x);
  equalityGradientCSR.print();

  vector_t inequalityConstraints = trackingProblem.evaluateInequalityConstraints(test_x);
  std::cout << "Inequality constraints: " << inequalityConstraints.transpose() << std::endl;
  sparse_matrix_t inequalityJacobian = trackingProblem.evaluateInequalityConstraintsJacobian(test_x);
  std::cout << "Inequality constraints Jacobian: \n" << inequalityJacobian.toDense() << std::endl;
}