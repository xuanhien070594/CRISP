#ifndef CONSTRAINT_FUNCTION_H
#define CONSTRAINT_FUNCTION_H

#include <algorithm>
#include <memory>

#include "common/ValueFunction.h"
#include <common/BasicTypes.h>

#include "drake/math/autodiff.h"
#include "drake/solvers/mathematical_program.h"
using drake::math::InitializeAutoDiff;
using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::MathematicalProgram;

// Only first order information is needed for constraints
namespace CRISP {
enum class ConstraintType { EQUALITY, INEQUALITY_BOUNDED, INEQUALITY_UNBOUNDED_UPPER, INEQUALITY_UNBOUNDED_LOWER };
inline void EvaluateConstraint(const std::shared_ptr<Binding<Constraint>> constraint_binding,
                               ConstraintType constraint_type, int n_vars, std::vector<int> var_indices,
                               const vector_t& x, vector_t* y) {
  vector_t this_x(n_vars);
  for (int i = 0; i < n_vars; ++i) {
    this_x(i) = x(var_indices[i]);
  }

  int num_constraints = constraint_binding->evaluator()->num_constraints();
  vector_t lower_bound = constraint_binding->evaluator()->lower_bound();
  vector_t upper_bound = constraint_binding->evaluator()->upper_bound();

  // for equality constraints like Ax = b, MathematicalProgram only evaluates Ax. However, we'd like
  // to compute Ax - b, so we need to subtract b from the result.

  if (constraint_type == ConstraintType::EQUALITY) {
    constraint_binding->evaluator()->Eval(this_x, y);
    for (int i = 0; i < y->size(); ++i) {
      (*y)(i) = (*y)(i)-lower_bound(i);
    }
  } else if (constraint_type == ConstraintType::INEQUALITY_UNBOUNDED_UPPER) {
    constraint_binding->evaluator()->Eval(this_x, y);
    for (int i = 0; i < num_constraints; ++i) {
      (*y)(i) = (*y)(i)-lower_bound(i);
    }
  } else if (constraint_type == ConstraintType::INEQUALITY_UNBOUNDED_LOWER) {
    constraint_binding->evaluator()->Eval(this_x, y);
    for (int i = 0; i < num_constraints; ++i) {
      (*y)(i) = upper_bound(i) - (*y)(i);
    }
  } else {
    // MathematicalProgram treats min <= x <= max as a single constraint but CRISP counts 2
    // x - min >= 0 and max - x >= 0
    vector_t tmp(num_constraints);
    constraint_binding->evaluator()->Eval(this_x, &tmp);
    for (int i = 0; i < num_constraints; ++i) {
      (*y)(i) = upper_bound(i) - tmp(i);
      (*y)(i + num_constraints) = tmp(i) - lower_bound(i);
    }
  }
}

inline void EvaluateConstraintSparseGradient(const std::shared_ptr<Binding<Constraint>> constraint_binding,
                                             ConstraintType constraint_type, int n_vars, std::vector<int> var_indices,
                                             const vector_t& x, sparse_matrix_t* grad) {
  int num_constraints = constraint_binding->evaluator()->num_constraints();
  drake::AutoDiffVecXd ty(num_constraints);
  vector_t this_x(n_vars);
  for (int i = 0; i < n_vars; ++i) {
    this_x(i) = x(var_indices[i]);
  }
  constraint_binding->evaluator()->Eval(InitializeAutoDiff(this_x), &ty);

  if (constraint_binding->evaluator()->gradient_sparsity_pattern().has_value()) {
    auto gradient_sparsity_pattern = constraint_binding->evaluator()->gradient_sparsity_pattern().value();
    for (int i = 0; i < gradient_sparsity_pattern.size(); ++i) {
      auto row_idx = gradient_sparsity_pattern[i].first;
      auto col_idx = gradient_sparsity_pattern[i].second;

      if (constraint_type == ConstraintType::EQUALITY ||
          constraint_type == ConstraintType::INEQUALITY_UNBOUNDED_UPPER) {
        (*grad).insert(row_idx, col_idx) = ty(row_idx).derivatives()(col_idx);
      } else if (constraint_type == ConstraintType::INEQUALITY_UNBOUNDED_LOWER) {
        (*grad).insert(row_idx, col_idx) = -ty(row_idx).derivatives()(col_idx);
      } else {
        (*grad).insert(row_idx, col_idx) = -ty(row_idx).derivatives()(col_idx);
        (*grad).insert(row_idx + num_constraints, col_idx) = ty(row_idx).derivatives()(col_idx);
      }
    }
  } else {
    if (constraint_type == ConstraintType::EQUALITY || constraint_type == ConstraintType::INEQUALITY_UNBOUNDED_UPPER) {
      for (int i = 0; i < num_constraints; ++i) {
        for (int j = 0; j < n_vars; ++j) {
          (*grad).insert(i, var_indices[j]) = ty(i).derivatives()(j);
        }
      }
    } else if (constraint_type == ConstraintType::INEQUALITY_UNBOUNDED_LOWER) {
      for (int i = 0; i < num_constraints; ++i) {
        for (int j = 0; j < n_vars; ++j) {
          (*grad).insert(i, var_indices[j]) = -ty(i).derivatives()(j);
        }
      }
    } else {
      for (int i = 0; i < num_constraints; ++i) {
        for (int j = 0; j < n_vars; ++j) {
          (*grad).insert(i, var_indices[j]) = -ty(i).derivatives()(j);
          (*grad).insert(i + num_constraints, var_indices[j]) = ty(i).derivatives()(j);
        }
      }
    }
  }
  grad->makeCompressed();
}

class ConstraintFunction : public ValueFunction {
 public:
  enum class SpecifiedFunctionLevel { NONE, VALUE, GRADIENT };
  ConstraintFunction(size_t variableDim, const std::string& modelName, const std::string& folderName,
                     const std::string& functionName, const ad_function_t& function, bool regenerateLibrary = false,
                     CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER,
                     SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE)
      : specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(false) {
    cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, modelName, folderName, functionName, function,
                                                       infoLevel, regenerateLibrary);
    nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
    variableDim_ = variableDim;
    funDim_ = cppadInterface_->getFunDim();
  }

  ConstraintFunction(size_t variableDim, size_t parameterDim, const std::string& modelName,
                     const std::string& folderName, const std::string& functionName,
                     const ad_function_with_param_t& function, bool regenerateLibrary = false,
                     CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER,
                     SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE)
      : specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(true) {
    cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, parameterDim, modelName, folderName, functionName,
                                                       function, infoLevel, regenerateLibrary);
    nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
    variableDim_ = variableDim;
    parameterDim_ = parameterDim;
    funDim_ = cppadInterface_->getFunDim();
  }
  ConstraintFunction(size_t variableDim, const std::string& functionName,
                     const std::shared_ptr<Binding<Constraint>> function,
                     SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::GRADIENT)
      : obj_function_(function),
        specifiedFunctionLevel_(specifiedFunctionLevel),
        variableDim_(variableDim),
        functionName_(functionName),
        isParameterized_(false) {
    valueFunction_ = [this](const vector_t& x) -> vector_t {
      vector_t y(funDim_);
      EvaluateConstraint(obj_function_, constraint_type_, n_bind_vars_, var_indices_, x, &y);
      return y;
    };

    gradientFunction_ = [this](const vector_t& x) -> sparse_matrix_t {
      sparse_matrix_t grad(funDim_, total_n_vars_);
      EvaluateConstraintSparseGradient(obj_function_, constraint_type_, n_bind_vars_, var_indices_, x, &grad);
      return grad;
    };
  }

  //  for pybind
  ConstraintFunction(size_t variableDim, const std::string& modelName, const std::string& folderName,
                     const std::string& functionName, bool regenerateLibrary = false,
                     CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER,
                     SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE)
      : specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(false) {
    cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, modelName, folderName, functionName, infoLevel,
                                                       regenerateLibrary);
    nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
    variableDim_ = variableDim;
    funDim_ = cppadInterface_->getFunDim();
  }

  ConstraintFunction(size_t variableDim, size_t parameterDim, const std::string& modelName,
                     const std::string& folderName, const std::string& functionName, bool regenerateLibrary = false,
                     CppAdInterface::ModelInfoLevel infoLevel = CppAdInterface::ModelInfoLevel::FIRST_ORDER,
                     SpecifiedFunctionLevel specifiedFunctionLevel = SpecifiedFunctionLevel::NONE)
      : specifiedFunctionLevel_(specifiedFunctionLevel), functionName_(functionName), isParameterized_(true) {
    //   convert the eigen vector function to ad function
    cppadInterface_ = std::make_unique<CppAdInterface>(variableDim, parameterDim, modelName, folderName, functionName,
                                                       infoLevel, regenerateLibrary);
    nnzJacobian_ = cppadInterface_->getNumNonZerosJacobian();
    variableDim_ = variableDim;
    parameterDim_ = parameterDim;
    funDim_ = cppadInterface_->getFunDim();
  }

  //  ------------------------ Get function information from ad or user defined functions ------------------------ //
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
        CSRSparseMatrix constraintGradientCSR(gradientFunction_(x));
        return constraintGradientCSR;
      } else {
        throw std::runtime_error("No gradient function specified.");
      }
    }
    return isParameterized_ ? throw std::runtime_error("Parameters are required.")
                            : cppadInterface_->computeSparseJacobianCSR(x);
  }

  SpecifiedFunctionLevel getSpecifiedFunctionLevel() const { return specifiedFunctionLevel_; }

  bool isParameterized() const { return isParameterized_; }

  size_t getVariableDim() const { return variableDim_; }

  size_t getFunDim() const { return funDim_; }

  size_t getParameterDim() const { return parameterDim_; }

  size_t getNumNonZerosJacobian() const { return nnzJacobian_; }

  const std::string& getFunctionName() const { return functionName_; }

  bool isInfinityBound(const vector_t& bound) {
    for (int i = 0; i < bound.size(); ++i) {
      if (bound(i) == std::numeric_limits<double>::infinity()) {
        return true;
      }
    }
    return false;
  }

  void setDecisionVariableIndices(const MathematicalProgram& prog) {
    vector_t lower_bound = obj_function_->evaluator()->lower_bound();
    vector_t upper_bound = obj_function_->evaluator()->upper_bound();
    funDim_ = obj_function_->evaluator()->num_constraints();
    if (std::equal(lower_bound.data(), lower_bound.data() + lower_bound.size(), upper_bound.data())) {
      constraint_type_ = ConstraintType::EQUALITY;
    } else if (isInfinityBound(lower_bound)) {
      constraint_type_ = ConstraintType::INEQUALITY_UNBOUNDED_LOWER;
    } else if (isInfinityBound(upper_bound)) {
      constraint_type_ = ConstraintType::INEQUALITY_UNBOUNDED_UPPER;
    } else {
      constraint_type_ = ConstraintType::INEQUALITY_BOUNDED;
      funDim_ = funDim_ * 2;
    }
    total_n_vars_ = prog.num_vars();
    n_bind_vars_ = obj_function_->GetNumElements();

    if (obj_function_->evaluator()->gradient_sparsity_pattern().has_value()) {
      auto gradient_sparsity_pattern = obj_function_->evaluator()->gradient_sparsity_pattern().value();
      nnzJacobian_ = gradient_sparsity_pattern.size();
    } else {
      nnzJacobian_ = n_bind_vars_ * funDim_;
    }
    var_indices_.resize(n_bind_vars_);
    for (int i = 0; i < n_bind_vars_; ++i) {
      var_indices_.at(i) = prog.FindDecisionVariableIndex(obj_function_->variables()(i));
    }
  }

 private:
  const std::shared_ptr<Binding<Constraint>> obj_function_;
  SpecifiedFunctionLevel specifiedFunctionLevel_;
  ConstraintType constraint_type_;
  size_t variableDim_ = 0;
  size_t parameterDim_ = 0;
  size_t funDim_ = 0;
  size_t nnzJacobian_ = 0;
  std::string functionName_;
  bool isParameterized_ = false;
  std::vector<int> var_indices_;
  int n_bind_vars_ = 0;
  int total_n_vars_ = 0;
};

}  // namespace CRISP

#endif  // CONSTRAINT_FUNCTION_H
