#include "problem_core/OptimizationProblem.h"
#include <common/BasicTypes.h>

// test: the construction of a simple tracking problem.
// This also serves as a template example for user to create their own optimization problem.

using namespace CRISP;

// ------------------------ 1. Define the optimization problem ------------------------ //
// min          obj
// s.t. inequalityConstraints >= 0
//      equalityConstraints = 0
// In the following toy example, let us define x(0) and x(1) to be the 1D robot state of
// two time steps, and the x(2) to be the 1D control

// objective to track the reference, where the reference is the input paramater.
// obj = (x(0) - p(0))^2 + (x(1) - p(1))^2 + u^2
ad_function_with_param_t quadraticObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
  y.resize(1);
  y(0) = (x(0) - p(0)) * (x(0) - p(0)) + (x(1) - p(1)) * (x(1) - p(1)) + x(2) * x(2);
};

// constraint to enforce the single step dynamics of the system with no parameter
// x(1) = x(0) + u
ad_function_t dynamicsConstraint = [](const ad_vector_t& x, ad_vector_t& y) {
  y.resize(1);
  y(0) = x(1) - x(0) - x(2);
};

// control limit constraint
// inequality |u| <= 1
ad_function_t controlLimitConstraint = [](const ad_vector_t& x, ad_vector_t& y) {
  y.resize(2);
  y(0) = 1 - x(2);  // 1-u >= 0
  y(1) = 1 + x(2);  // 1+u >= 0
};

// initial state constraint
// equality x(0) = p(0)
ad_function_with_param_t initialStateConstraint = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
  y.resize(1);
  y(0) = x(0) - p(0);
};

// final state constraint
// equality x(1) = p(0)
ad_function_with_param_t finalStateConstraint = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
  y.resize(1);
  y(0) = x(1) - p(0);
};

// ------------------------ 2. Create the optimization problem ------------------------ //
int main() {
  // 1. Create the optimization problem
  // the cppad library will be generated according to the model name and folder name
  // the model name can be taken as the same with the problem name.
  int variableNum = 3;
  std::string problemName = "Test_Tracking_Problem";
  std::string folderName = "model";
  OptimizationProblem trackingProblem(variableNum, problemName);

  // 2. create the objective and constraints functions, register the parameters
  auto obj = std::make_shared<ObjectiveFunction>(variableNum, 2, problemName, folderName, "quadraticObjective",
                                                 quadraticObjective);
  auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "dynamicsConstraint",
                                                       dynamicsConstraint);
  auto controlLimit = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName,
                                                           "controlLimitConstraint", controlLimitConstraint);
  auto initialState = std::make_shared<ConstraintFunction>(variableNum, 1, problemName, folderName,
                                                           "initialStateConstraint", initialStateConstraint);
  auto finalState = std::make_shared<ConstraintFunction>(variableNum, 1, problemName, folderName,
                                                         "finalStateConstraint", finalStateConstraint);

  // 3. Add the objective and constraints to the optimization problem
  trackingProblem.addObjective(
      obj);  // !!parameter used in the objective function is automatically registered with the function name
             // "quadraticObjective" by default. And you can also give another name as the second parameter.
  trackingProblem.addEqualityConstraint(dynamics);
  trackingProblem.addInequalityConstraint(controlLimit);
  trackingProblem.addEqualityConstraint(initialState);
  trackingProblem.addEqualityConstraint(finalState);

  // 4. Evaluate the problem.
  // An important thing to note is that, by default, the solver will extract all needed derivatives information for you
  // with the embeded CPPAP interface. But we also support the usage of user-specified sparse gradient or hessian
  // functions, e.g., create objective with user-specified gradient and hessian functions: level 2 auto obj =
  // std::make_shared<ObjectiveFunction>(variableNum, 2, problemName, folderName, "quadraticObjective",
  // quadraticObjective, 2, 2); Then, set the value, gradient and hessian functions for the objective function using
  // obj.setValueFunction, obj.setGradientFunction, obj.setHessianFunction interfaces; set the pa(rameters
  vector_t ref(2);
  ref << 1.0, 2.0;
  vector_t x0(1);
  vector_t x1(1);
  x0 << 0.5;
  x1 << 1.5;
  trackingProblem.setParameters("quadraticObjective", ref);
  trackingProblem.setParameters("initialStateConstraint", x0);
  trackingProblem.setParameters("finalStateConstraint", x1);

  // evaluate the objective
  vector_t x(3);
  x << 10.0, 20.0, 1.5;
  scalar_t objValue = trackingProblem.evaluateObjective(x);
  std::cout << "Objective value: " << objValue << std::endl;
  sparse_matrix_t objGradient = trackingProblem.evaluateObjectiveGradient(x);
  triplet_vector_t objGradientTriplet = trackingProblem.evaluateObjectiveGradientTriplet(x);
  CSRSparseMatrix objCSRGradient = trackingProblem.evaluateObjectiveGradientCSR(x);
  CSRSparseMatrix convertedCSRGradient(objGradient);
  objCSRGradient.print();
  convertedCSRGradient.print();

  // printTripletVector(objGradientTriplet);
  sparse_matrix_t objHessian = trackingProblem.evaluateObjectiveHessian(x);
  std::cout << "Objective hessian: " << objHessian.toDense() << std::endl;
  // printSparseMatrix(objHessian);
  triplet_vector_t objHessianTriplet = trackingProblem.evaluateObjectiveHessianTriplet(x);
  // printTripletVector(objHessianTriplet);

  // evaluate the constraints
  vector_t equalityConstraints = trackingProblem.evaluateEqualityConstraints(x);
  std::cout << "Equality constraints: " << equalityConstraints.transpose() << std::endl;
  sparse_matrix_t equalityJacobian = trackingProblem.evaluateEqualityConstraintsJacobian(x);
  std::cout << "Equality constraints Jacobian: \n" << equalityJacobian.toDense() << std::endl;
  CSRSparseMatrix equalityJacobianCSR = trackingProblem.evaluateEqualityConstraintsJacobianCSR(x);
  equalityJacobianCSR.print();

  vector_t inequalityConstraints = trackingProblem.evaluateInequalityConstraints(x);
  std::cout << "Inequality constraints: " << inequalityConstraints.transpose() << std::endl;
  sparse_matrix_t inequalityJacobian = trackingProblem.evaluateInequalityConstraintsJacobian(x);
  std::cout << "Inequality constraints Jacobian: \n" << inequalityJacobian.toDense() << std::endl;

  // print name of constraints and objectives
  std::vector<std::string> eqconstraintNames = trackingProblem.getEqualityParamNames();
  std::vector<std::string> ineqconstraintNames = trackingProblem.getInequalityParamNames();
  std::vector<std::string> objNames = trackingProblem.getObjectiveParamNames();
  std::cout << "Equality constraint names: ";
  for (const auto& name : eqconstraintNames) {
    std::cout << name << " ";
  }
  std::cout << std::endl;
  std::cout << "Inequality constraint names: ";
  for (const auto& name : ineqconstraintNames) {
    std::cout << name << " ";
  }
  std::cout << std::endl;
  std::cout << "Objective names: ";
  for (const auto& name : objNames) {
    std::cout << name << " ";
  }
  std::cout << std::endl;

  return 0;
}
