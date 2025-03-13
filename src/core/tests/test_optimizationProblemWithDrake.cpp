#include "drake/solvers/mathematical_program.h"
#include "problem_core/OptimizationProblem.h"
#include "Eigen/Dense"

using namespace CRISP;

using drake::solvers::MathematicalProgram;
using drake::solvers::Binding;
using drake::solvers::QuadraticCost;
using Eigen::MatrixXd;

void EvaluateCost(const vector_t& x, vector_t* )
{
    y.resize(1);
    y(0) = x(0) * x(0) + (x(1) - 2.0) * (x(1) - 2.0);
}


int main()
{
    MathematicalProgram prog;
    auto x = prog.NewContinuousVariables(4, "x");
    auto y = prog.NewContinuousVariables(5, "y");
    vector_t p(3);
    p << 1.0, 2.0, 0.0;

    MatrixXd Q(3, 3);
    Q << 1, 0, 0,
         0, 1, 0,
         0, 0, 1;

    vector_t b(3);
    b = -2 * p.transpose() * Q;

    double c = p(0)*p(0) + p(1)*p(1) + p(2)*p(2);

    Binding<QuadraticCost> objectiveFunction =
        prog.AddQuadraticCost(2*Q, b, c, x, true);

    // Binding<QuadraticCost> objectiveFunction =
    //     prog.AddQuadraticCost((x(0) - p(0)) * (x(0) - p(0)) + (x(1) - p(1)) * (x(1) - p(1)) + x(2) * x(2), true);

    vector_t test_x(4);
    test_x << 1.0, 2.0, 2.0, 2.0;

    auto output = std::make_unique<vector_t>(1);
    objectiveFunction.evaluator()->Eval(test_x, output.get());
    std::cout << "Objective value: " << (*output)(0) << std::endl;

    auto gradient_sparsity = objectiveFunction.evaluator()->gradient_sparsity_pattern();
    std::cout << gradient_sparsity.has_value() << std::endl;

    std::cout << prog.decision_variables() << std::endl;
}