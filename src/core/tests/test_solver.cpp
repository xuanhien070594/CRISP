#include "solver_core/SolverInterface.h"

using namespace CRISP;

// Define the objective and constraints using CppAD
ad_function_t test5Objective = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize(1);
    y(0) = x(0) * x(0) + (x(1) - 2.0) * (x(1) - 2.0);
};

ad_function_t test5EqualityConstraint = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize(1);
    y(0) = ((x(0) - 1) * (x(0) - 1) + x(1) * x(1) - 3) *
           ((x(0) + 1) * (x(0) + 1) + x(1) * x(1) - 3);
};

ad_function_t test5InequalityConstraint = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize(2);
    y(0) = -(x(0) - 1) * (x(0) - 1) - x(1) * x(1) + 3;
    y(1) = -(x(0) + 1) * (x(0) + 1) - x(1) * x(1) + 3;
};

int main() {
    // Initialize the number of variables
    size_t variableNum = 2;

    // Create an optimization problem without parameters (or set zero if required)
    OptimizationProblem problem(variableNum, "Test5Problem");

    // Create objective and constraints (no parameters needed)
    auto objFunc = std::make_shared<ObjectiveFunction>(variableNum, "Test5Problem", "model", "test5Objective", test5Objective);
    auto eqConstraintFunc = std::make_shared<ConstraintFunction>(variableNum, "Test5Problem", "model", "test5EqualityConstraint", test5EqualityConstraint);
    auto ineqConstraintFunc = std::make_shared<ConstraintFunction>(variableNum, "Test5Problem", "model", "test5InequalityConstraint", test5InequalityConstraint);

    // Add the objective and constraints to the problem
    problem.addObjective(objFunc);
    problem.addEqualityConstraint(eqConstraintFunc);
    problem.addInequalityConstraint(ineqConstraintFunc);

    vector_t xInitial(2);
    xInitial << 2.0, 2.0;
    // test the problem
    scalar_t objValue = problem.evaluateObjective(xInitial);
    std::cout << "Objective value: " << objValue << std::endl;
    sparse_matrix_t objGradient = problem.evaluateObjectiveGradient(xInitial);
    std::cout << "Objective Gradient: " << std::endl;
    // printSparseMatrix(objGradient);
    triplet_vector_t objHessianTriplet = problem.evaluateObjectiveHessianTriplet(xInitial);
    std::cout<<"Objective Hessian Triplet: "<<std::endl;
    // printTripletVector(objHessianTriplet);

    vector_t eqValues = problem.evaluateEqualityConstraints(xInitial);
    std::cout << "Equality constraint values: " << eqValues.transpose() << std::endl;
    sparse_matrix_t eqGradient = problem.evaluateEqualityConstraintsJacobian(xInitial);
    std::cout << "Equality constraint gradient: " << std::endl;
    // printSparseMatrix(eqGradient);
    sparse_matrix_t ineqGradient = problem.evaluateInequalityConstraintsJacobian(xInitial);
    std::cout << "Inequality constraint gradient: " << std::endl;
    // printSparseMatrix(ineqGradient);

    // -------------------- initialize the solver interface with the problem ------------------------ // 
    // Prepare solver parameters
    SolverParameters params;
    // Optionally set specific parameters

    // Create the solver interface
    SolverInterface solver(problem, params);
    solver.initialize(xInitial);

    // Solve the problem
    solver.solve();

    // Extract the solution and print the result
    const auto solution = solver.getSolution();
    std::cout << "Solution: " << solution.transpose() << std::endl;

    return 0;
}