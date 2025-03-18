#ifndef OBJECTIVE_FUNCTION_H
#define OBJECTIVE_FUNCTION_H

#include "common/ValueFunction.h"
#include <common/BasicTypes.h>
#include <eigen3/Eigen/src/SparseCore/SparseMatrix.h>

#include "drake/math/autodiff.h"
#include "drake/solvers/mathematical_program.h"

using namespace CRISP;

using drake::math::InitializeAutoDiff;
using drake::solvers::Binding;
using drake::solvers::MathematicalProgram;
using drake::solvers::QuadraticCost;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace CRISP {

inline void EvaluateCost(Binding<QuadraticCost>* cost_binding, int n_vars, std::vector<int> var_indices,
                         const vector_t& x, vector_t* y) {
  VectorXd this_x(n_vars);
  for (int i = 0; i < n_vars; ++i) {
    this_x(i) = x(var_indices[i]);
  }
  cost_binding->evaluator()->Eval(this_x, y);
}

inline void EvaluateCostSparseGradient(Binding<QuadraticCost>* cost_binding, int n_vars, std::vector<int> var_indices,
                                       const vector_t& x, sparse_matrix_t* grad) {
  drake::AutoDiffVecXd ty(1);
  VectorXd this_x(n_vars);
  for (int i = 0; i < n_vars; ++i) {
    this_x(i) = x(var_indices[i]);
  }

  cost_binding->evaluator()->Eval(InitializeAutoDiff(this_x), &ty);

  if (ty(0).derivatives().size() > 0) {
    for (int j = 0; j < n_vars; ++j) {
      (*grad).insert(0, var_indices[j]) = ty(0).derivatives()(j);
    }
  }
  grad->makeCompressed();
}

inline void EvaluateCostSparseHessian(Binding<QuadraticCost>* cost_binding, int n_vars, std::vector<int> var_indices,
                                      const vector_t& x, sparse_matrix_t* hessian) {
  for (int i = 0; i < n_vars; ++i) {
    for (int j = 0; j < n_vars; ++j) {
      hessian->insert(var_indices[i], var_indices[j]) = cost_binding->evaluator()->Q()(i, j);
    }
  }
  hessian->makeCompressed();
}

class ObjectiveFunction : public ValueFunction {
 public:
  enum class SpecifiedFunctionLevel { NONE, VALUE, GRADIENT, HESSIAN };
  ObjectiveFunction(size_t variableDim, const std::string& modelName, const std::string& folderName,
                    const std::string& functionName, const ad_function_t& function, bool regenerateLibrary = false,
                    CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::SECOND_ORDER,
                    SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE)
      : specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName) {
    cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, modelName, folderName, functionName, function,
                                                       infoLevel, regenerateLibrary);
    variableDim_ = variableDim;
    funDim_ = 1;
    isParameterized_ = false;
    nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
    nnzHessian_ = cppadInterface_->getNumNonZerosHessian();
  }
  ObjectiveFunction(size_t variableDim, size_t parameterDim, const std::string& modelName,
                    const std::string& folderName, const std::string& functionName,
                    const ad_function_with_param_t& function, bool regenerateLibrary = false,
                    CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::SECOND_ORDER,
                    SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE)
      : specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName) {
    cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, parameterDim, modelName, folderName, functionName,
                                                       function, infoLevel, regenerateLibrary);
    variableDim_ = variableDim;
    parameterDim_ = parameterDim;
    funDim_ = 1;
    isParameterized_ = true;
    nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
    nnzHessian_ = cppadInterface_->getNumNonZerosHessian();
  }

  // Constructor for the cost functions of Drake MathematicalProgram
  // Costs and Constraints of MathematicalProgram do not accept parameters
  ObjectiveFunction(size_t variableDim, const std::string& functionName, Binding<QuadraticCost>* function,
                    SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::HESSIAN)
      : obj_function_(function),
        specifiedFunctionLevel_(specifiedFunctionLevel),
        variableDim_(variableDim),
        funDim_(1),
        functionName_(functionName),
        isParameterized_(false) {
    valueFunction_ = [this](const vector_t& x) -> vector_t {
      vector_t y(1);
      EvaluateCost(obj_function_, n_bind_vars_, var_indices_, x, &y);
      return y;
    };

    gradientFunction_ = [this](const vector_t& x) -> sparse_matrix_t {
      sparse_matrix_t grad(1, total_n_vars_);
      EvaluateCostSparseGradient(obj_function_, n_bind_vars_, var_indices_, x, &grad);
      return grad;
    };
    hessianFunction_ = [this](const vector_t& x) -> sparse_matrix_t {
      sparse_matrix_t hessian(total_n_vars_, total_n_vars_);
      EvaluateCostSparseHessian(obj_function_, n_bind_vars_, var_indices_, x, &hessian);
      return hessian;
    };
  }

  //  for pybind
  ObjectiveFunction(size_t variableDim, const std::string& modelName, const std::string& folderName,
                    const std::string& functionName, bool regenerateLibrary = false,
                    CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::SECOND_ORDER,
                    SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE)
      : specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName) {
    cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, modelName, folderName, functionName, infoLevel,
                                                       regenerateLibrary);
    variableDim_ = variableDim;
    funDim_ = 1;
    isParameterized_ = false;
    nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
    nnzHessian_ = cppadInterface_->getNumNonZerosHessian();
  }

  ObjectiveFunction(size_t variableDim, size_t parameterDim, const std::string& modelName,
                    const std::string& folderName, const std::string& functionName, bool regenerateLibrary = false,
                    CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::SECOND_ORDER,
                    SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE)
      : specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName) {
    cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, parameterDim, modelName, folderName, functionName,
                                                       infoLevel, regenerateLibrary);
    variableDim_ = variableDim;
    parameterDim_ = parameterDim;
    funDim_ = 1;
    isParameterized_ = true;
    nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
    nnzHessian_ = cppadInterface_->getNumNonZerosHessian();
  }

  //  ------------------------ Get function information from ad or user
  //  defined functions ------------------------ //
  vector_t getValue(const vector_t& x, const vector_t& params) override {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::VALUE) {
      if (valueFunctionWithParam_ != nullptr) {
        return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                                 : valueFunctionWithParam_(x, params);
      } else {
        throw std::runtime_error("No value function with parameters specified.");
      }
    }
    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                             : cppadInterface_->computeFunctionValue(x, params);
  }

  vector_t getValue(const vector_t& x) override {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::VALUE) {
      if (valueFunction_ != nullptr) {
        return isParameterized_ ? throw std::runtime_error("Parameters are required.") : valueFunction_(x);
      } else {
        throw std::runtime_error("No value function specified.");
      }
    }
    return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                            : cppadInterface_->computeFunctionValue(x);
  }

  sparse_matrix_t getGradient(const vector_t& x, const vector_t& params) override {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
      if (gradientFunctionWithParam_ != nullptr) {
        return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                                 : gradientFunctionWithParam_(x, params);
      } else {
        throw std::runtime_error("No gradient function with parameters specified.");
      }
    }
    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                             : cppadInterface_->computeSparseJacobian(x, params);
  }

  sparse_matrix_t getGradient(const vector_t& x) override {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
      if (gradientFunction_ != nullptr) {
        return isParameterized_ ? throw std::runtime_error("Parameters are required.") : gradientFunction_(x);
      } else {
        throw std::runtime_error("No gradient function specified.");
      }
    }
    return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                            : cppadInterface_->computeSparseJacobian(x);
  }

  triplet_vector_t getGradientTriplet(const vector_t& x, const vector_t& params) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
      if (gradientFunctionWithParam_ != nullptr) {
        return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                                 : cppadInterface_->computeSparseJacobianTriplet(x, params);
      } else {
        throw std::runtime_error("No gradient function with parameters specified.");
      }
    }
    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                             : cppadInterface_->computeSparseJacobianTriplet(x, params);
  }

  triplet_vector_t getGradientTriplet(const vector_t& x) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
      if (gradientFunction_ != nullptr) {
        return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                                : cppadInterface_->computeSparseJacobianTriplet(x);
      } else {
        throw std::runtime_error("No gradient function specified.");
      }
    }
    return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                            : cppadInterface_->computeSparseJacobianTriplet(x);
  }

  CSRSparseMatrix getGradientCSR(const vector_t& x, const vector_t& params) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
      if (gradientFunctionWithParam_ != nullptr) {
        return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                                 : cppadInterface_->computeSparseJacobianCSR(x, params);
      } else {
        throw std::runtime_error("No gradient function with parameters specified.");
      }
    }
    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                             : cppadInterface_->computeSparseJacobianCSR(x, params);
  }

  CSRSparseMatrix getGradientCSR(const vector_t& x) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::GRADIENT) {
      if (gradientFunction_ != nullptr) {
        if (isParameterized_) {
          throw std::runtime_error("Parameters are required.");
        }
        CSRSparseMatrix objGradientCSR(gradientFunction_(x));
        return objGradientCSR;
      } else {
        throw std::runtime_error("No gradient function specified.");
      }
    }
    return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                            : cppadInterface_->computeSparseJacobianCSR(x);
  }

  sparse_matrix_t getHessian(const vector_t& x, const vector_t& params) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::HESSIAN) {
      if (hessianFunctionWithParam_ != nullptr) {
        return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                                 : hessianFunctionWithParam_(x, params);
      } else {
        throw std::runtime_error("No hessian function with parameters specified.");
      }
    }
    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                             : cppadInterface_->computeSparseHessian(x, params);
  }

  sparse_matrix_t getHessian(const vector_t& x) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::HESSIAN) {
      if (hessianFunction_ != nullptr) {
        return isParameterized_ ? throw std::runtime_error("Parameters are required.") : hessianFunction_(x);
      } else {
        throw std::runtime_error("No hessian function specified.");
      }
    }
    return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                            : cppadInterface_->computeSparseHessian(x);
  }

  triplet_vector_t getHessianTriplet(const vector_t& x, const vector_t& params) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::HESSIAN) {
      if (hessianFunctionWithParam_ != nullptr) {
        return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                                 : cppadInterface_->computeSparseHessianTriplet(x, params);
      } else {
        throw std::runtime_error("No hessian function with parameters specified.");
      }
    }
    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                             : cppadInterface_->computeSparseHessianTriplet(x, params);
  }

  triplet_vector_t getHessianTriplet(const vector_t& x) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::HESSIAN) {
      if (hessianFunction_ != nullptr) {
        return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                                : cppadInterface_->computeSparseHessianTriplet(x);
      } else {
        throw std::runtime_error("No hessian function specified.");
      }
    }
    return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                            : cppadInterface_->computeSparseHessianTriplet(x);
  }

  CSRSparseMatrix getHessianCSR(const vector_t& x, const vector_t& params) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::HESSIAN) {
      if (hessianFunctionWithParam_ != nullptr) {
        return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                                 : cppadInterface_->computeSparseHessianCSR(x, params);
      } else {
        throw std::runtime_error("No hessian function with parameters specified.");
      }
    }
    return !isParameterized_ ? throw std::runtime_error("Parameters are not expected.")
                             : cppadInterface_->computeSparseHessianCSR(x, params);
  }

  CSRSparseMatrix getHessianCSR(const vector_t& x) {
    if (specifiedFunctionLevel_ >= SpecifiedFunctionLevel::HESSIAN) {
      if (hessianFunction_ != nullptr) {
        if (isParameterized_) {
          throw std::runtime_error("Parameters are required.");
        }
        CSRSparseMatrix objHessianCSR(hessianFunction_(x));
        return objHessianCSR;

      } else {
        throw std::runtime_error("No hessian function specified.");
      }
    }
    return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                            : cppadInterface_->computeSparseHessianCSR(x);
  }

  // ------------------------ Set user specified function information
  // ------------------------ //
  void setHessianFunction(const std::function<sparse_matrix_t(const vector_t&)>& hessianFunction) {
    hessianFunction_ = hessianFunction;
  }

  void setHessianFunctionWithParam(
      const std::function<sparse_matrix_t(const vector_t&, const vector_t&)>& hessianFunctionWithParam) {
    hessianFunctionWithParam_ = hessianFunctionWithParam;
  }

  SpecifiedFunctionLevel getSpecifiedFunctionLevel() const { return specifiedFunctionLevel_; }

  bool isParameterized() const { return isParameterized_; }

  size_t getVariableDim() const { return variableDim_; }

  size_t getParameterDim() const { return parameterDim_; }

  size_t getFunDim() const { return funDim_; }

  size_t getNumNonZerosJacobian() const { return nnzJacobian_; }

  size_t getNumNonZerosHessian() const { return nnzHessian_; }

  const std::string& getFunctionName() const { return functionName_; }

  void setDecisionVariableIndices(const MathematicalProgram& prog) {
    total_n_vars_ = prog.num_vars();
    n_bind_vars_ = obj_function_->GetNumElements();
    nnzJacobian_ = n_bind_vars_;
    nnzHessian_ = n_bind_vars_ * n_bind_vars_;
    var_indices_.resize(n_bind_vars_);
    for (int i = 0; i < n_bind_vars_; ++i) {
      var_indices_.at(i) = prog.FindDecisionVariableIndex(obj_function_->variables()(i));
    }
  }

 protected:
  Binding<QuadraticCost>* obj_function_;
  SpecifiedFunctionLevel specifiedFunctionLevel_;
  // User specified function information.
  std::function<sparse_matrix_t(const vector_t&)> hessianFunction_;
  std::function<sparse_matrix_t(const vector_t&, const vector_t&)> hessianFunctionWithParam_;
  size_t variableDim_ = 0;
  size_t parameterDim_ = 0;
  size_t funDim_ = 1;
  size_t nnzHessian_ = 0;
  size_t nnzJacobian_ = 0;
  std::string functionName_;
  bool isParameterized_ = false;
  std::vector<int> var_indices_;
  int n_bind_vars_ = 0;
  int total_n_vars_ = 0;
};

}  // namespace CRISP

#endif  // OBJECTIVE_FUNCTION_H