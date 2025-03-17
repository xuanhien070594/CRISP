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

  auto x = prog.NewContinuousVariables(5, "x");
  vector_t p(3);
  p << 1.0, 2.0, 0.0;
  MatrixXd Q(3, 3);
  Q << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  vector_t b(3);
  b = -2 * p.transpose() * Q;
  double c = p(0) * p(0) + p(1) * p(1) + p(2) * p(2);
  Binding<QuadraticCost> objectiveFunction = prog.AddQuadraticCost(2 * Q, b, c, x.head(3), true);

  // Binding<QuadraticCost> objectiveFunction =
  //     prog.AddQuadraticCost((x(0) - p(0)) * (x(0) - p(0)) + (x(1) - p(1)) *
  //     (x(1) - p(1)) + x(2) * x(2), true);

  // vector_t test_x(4);
  // test_x << 1.0, 2.0, 2.0, 2.0;

  // auto output = std::make_unique<vector_t>(1);
  // objectiveFunction.evaluator()->Eval(test_x, output.get());
  // std::cout << "Objective value: " << (*output)(0) << std::endl;

  // auto gradient_sparsity = objectiveFunction.evaluator()->gradient_sparsity_pattern();
  // std::cout << gradient_sparsity.has_value() << std::endl;

  // std::cout << prog.decision_variables() << std::endl;

  // 1. Create the optimization problem
  // the cppad library will be generated according to the model name and folder name
  // the model name can be taken as the same with the problem name.
  int variableNum = 3;
  std::string problemName = "Test_Tracking_Problem";
  std::string folderName = "model";
  OptimizationProblem trackingProblem(variableNum, problemName);

  // 2. create the objective and constraints functions, register the parameters
  auto obj = std::make_shared<ObjectiveFunction>(variableNum, "quadraticObjective", &objectiveFunction);
  obj->setDecisionVariableIndices(prog);
  // auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "dynamicsConstraint",
  // dynamicsConstraint); auto controlLimit = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName,
  // "controlLimitConstraint", controlLimitConstraint); auto initialState =
  // std::make_shared<ConstraintFunction>(variableNum, 1, problemName, folderName, "initialStateConstraint",
  // initialStateConstraint); auto finalState = std::make_shared<ConstraintFunction>(variableNum, 1, problemName,
  // folderName, "finalStateConstraint", finalStateConstraint);

  // 3. Add the objective and constraints to the optimization problem
  trackingProblem.addObjective(obj);  // !!parameter used in the objective function is automatically registered with the
                                      // function name "quadraticObjective" by default. And you can also give another
                                      // name as the second parameter. trackingProblem.addEqualityConstraint(dynamics);
                                      // trackingProblem.addInequalityConstraint(controlLimit);
                                      // trackingProblem.addEqualityConstraint(initialState);
                                      // trackingProblem.addEqualityConstraint(finalState);

  // 4. Evaluate the problem.
  // An important thing to note is that, by default, the solver will extract all needed derivatives information for you
  // with the embeded CPPAP interface. But we also support the usage of user-specified sparse gradient or hessian
  // functions, e.g., create objective with user-specified gradient and hessian functions: level 2 auto obj =
  // std::make_shared<ObjectiveFunction>(variableNum, 2, problemName, folderName, "quadraticObjective",
  // quadraticObjective, 2, 2); Then, set the value, gradient and hessian functions for the objective function using
  // obj.setValueFunction, obj.setGradientFunction, obj.setHessianFunction interfaces; set the pa(rameters
  // trackingProblem.setParameters("initialStateConstraint", x0);
  // trackingProblem.setParameters("finalStateConstraint", x1);

  // evaluate the objective
  vector_t test_x(5);
  test_x << 10.0, 20.0, 1.5, 1.0, 1.0;
  scalar_t objValue = trackingProblem.evaluateObjective(test_x);
  std::cout << "Objective value: " << objValue << std::endl;
  sparse_matrix_t objGradient = trackingProblem.evaluateObjectiveGradient(test_x);
  std::cout << "Objective gradient: " << objGradient.toDense() << std::endl;

  CSRSparseMatrix objGradientCSR = trackingProblem.evaluateObjectiveGradientCSR(test_x);
  objGradientCSR.print();

  sparse_matrix_t objHessian = trackingProblem.evaluateObjectiveHessian(test_x);
  std::cout << "Objective hessian: \n" << objHessian.toDense() << std::endl;
  // // printSparseMatrix(objHessian);
  // triplet_vector_t objHessianTriplet = trackingProblem.evaluateObjectiveHessianTriplet(x);
  // // printTripletVector(objHessianTriplet);

  // // evaluate the constraints
  // vector_t equalityConstraints = trackingProblem.evaluateEqualityConstraints(x);
  // std::cout << "Equality constraints: " << equalityConstraints.transpose() << std::endl;
  // sparse_matrix_t equalityJacobian = trackingProblem.evaluateEqualityConstraintsJacobian(x);
  // // printSparseMatrix(equalityJacobian);
  // triplet_vector_t equalityJacobianTriplet = trackingProblem.evaluateEqualityConstraintsJacobianTriplet(x);
  // // printTripletVector(equalityJacobianTriplet);

  // vector_t inequalityConstraints = trackingProblem.evaluateInequalityConstraints(x);
  // std::cout << "Inequality constraints: " << inequalityConstraints.transpose() << std::endl;
  // sparse_matrix_t inequalityJacobian = trackingProblem.evaluateInequalityConstraintsJacobian(x);
  // // printSparseMatrix(inequalityJacobian);
  // triplet_vector_t inequalityJacobianTriplet = trackingProblem.evaluateInequalityConstraintsJacobianTriplet(x);
  // // printTripletVector(inequalityJacobianTriplet);
  // // print name of constraints and objectives
  // std::vector<std::string> eqconstraintNames = trackingProblem.getEqualityParamNames();
  // std::vector<std::string> ineqconstraintNames = trackingProblem.getInequalityParamNames();
  // std::vector<std::string> objNames = trackingProblem.getObjectiveParamNames();
  // std::cout << "Equality constraint names: ";
  // for (const auto& name : eqconstraintNames) {
  //     std::cout << name << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "Inequality constraint names: ";
  // for (const auto& name : ineqconstraintNames) {
  //     std::cout << name << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "Objective names: ";
  // for (const auto& name : objNames) {
  //     std::cout << name << " ";
  // }
  // std::cout << std::endl;
}