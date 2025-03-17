#ifndef OPTIMIZATION_PROBLEM_H
#define OPTIMIZATION_PROBLEM_H

#include "common/ParametersManager.h"
#include "problem_core/ConstraintFunction.h"
#include "problem_core/ObjectiveFunction.h"

namespace CRISP {

class OptimizationProblem {
 public:
  // Constructor for problems with parameters
  OptimizationProblem(size_t variableDim, std::shared_ptr<ParametersManager> paramManager, const std::string& name = "")
      : variableDim_(variableDim),
        parameterManager_(std::move(paramManager)),
        problemName_(name),
        numEqualityConstraints_(0),
        numInequalityConstraints_(0),
        numNonZerosEqualityJacobian_(0),
        numNonZerosInequalityJacobian_(0),
        numNonZerosObjectiveHessian_(0),
        numNonZerosObjectiveJacobian_(0) {}

  // Non-parametric constructor
  explicit OptimizationProblem(size_t variableDim, const std::string& name = "")
      : variableDim_(variableDim),
        parameterManager_(std::make_shared<ParametersManager>()),
        problemName_(name),
        numEqualityConstraints_(0),
        numInequalityConstraints_(0),
        numNonZerosEqualityJacobian_(0),
        numNonZerosInequalityJacobian_(0),
        numNonZerosObjectiveHessian_(0),
        numNonZerosObjectiveJacobian_(0) {}

  // Add objective and constraint functions
  void addObjective(const std::shared_ptr<ObjectiveFunction>& objective) {
    std::string name = objective->getFunctionName();
    objectives_.emplace_back(objective);
    objectiveParamNames_.emplace_back(name);
    numNonZerosObjectiveJacobian_ += objective->getNumNonZerosJacobian();
    numNonZerosObjectiveHessian_ += objective->getNumNonZerosHessian();
  }

  void addEqualityConstraint(const std::shared_ptr<ConstraintFunction>& constraint) {
    std::string name = constraint->getFunctionName();
    equalityConstraints_.emplace_back(constraint);
    equalityParamNames_.emplace_back(name);
    numEqualityConstraints_ += constraint->getFunDim();
    numNonZerosEqualityJacobian_ += constraint->getNumNonZerosJacobian();
  }

  void addInequalityConstraint(const std::shared_ptr<ConstraintFunction>& constraint) {
    std::string name = constraint->getFunctionName();
    inequalityConstraints_.emplace_back(constraint);
    inequalityParamNames_.emplace_back(name);
    numInequalityConstraints_ += constraint->getFunDim();
    numNonZerosInequalityJacobian_ += constraint->getNumNonZerosJacobian();
  }

  void setParameters(const std::string& name, const vector_t& params) {
    parameterManager_->setParameters(name, params);
  }

  scalar_t evaluateObjective(const vector_t& x) const {
    scalar_t value = 0.0;
    for (size_t i = 0; i < objectives_.size(); ++i) {
      if (objectives_[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(objectives_[i]->getFunctionName());
        value += objectives_[i]->getValue(x, params)(0);
      } else {
        value += objectives_[i]->getValue(x)(0);
      }
    }
    return value;
  }

  vector_t evaluateEqualityConstraints(const vector_t& x) const {
    return evaluateConstraints(x, equalityConstraints_, equalityParamNames_, numEqualityConstraints_);
  }

  vector_t evaluateInequalityConstraints(const vector_t& x) const {
    return evaluateConstraints(x, inequalityConstraints_, inequalityParamNames_, numInequalityConstraints_);
  }

  sparse_matrix_t evaluateEqualityConstraintsJacobian(const vector_t& x) const {
    return evaluateConstraintsJacobian(x, equalityConstraints_, equalityParamNames_, numEqualityConstraints_);
  }

  triplet_vector_t evaluateEqualityConstraintsJacobianTriplet(const vector_t& x) const {
    return evaluateConstraintsJacobianTriplet(x, equalityConstraints_, equalityParamNames_);
  }

  CSRSparseMatrix evaluateEqualityConstraintsJacobianCSR(const vector_t& x) const {
    return evaluateConstraintsJacobianCSR(x, equalityConstraints_, equalityParamNames_, numEqualityConstraints_,
                                          numNonZerosEqualityJacobian_);
  }

  sparse_matrix_t evaluateInequalityConstraintsJacobian(const vector_t& x) const {
    return evaluateConstraintsJacobian(x, inequalityConstraints_, inequalityParamNames_, numInequalityConstraints_);
  }

  triplet_vector_t evaluateInequalityConstraintsJacobianTriplet(const vector_t& x) const {
    return evaluateConstraintsJacobianTriplet(x, inequalityConstraints_, inequalityParamNames_);
  }

  CSRSparseMatrix evaluateInequalityConstraintsJacobianCSR(const vector_t& x) const {
    return evaluateConstraintsJacobianCSR(x, inequalityConstraints_, inequalityParamNames_, numInequalityConstraints_,
                                          numNonZerosInequalityJacobian_);
  }

  sparse_matrix_t evaluateObjectiveGradient(const vector_t& x) const {
    sparse_matrix_t gradient(1, variableDim_);
    for (size_t i = 0; i < objectives_.size(); ++i) {
      if (objectives_[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(objectives_[i]->getFunctionName());
        gradient += objectives_[i]->getGradient(x, params);
      } else {
        gradient += objectives_[i]->getGradient(x);
      }
    }
    return gradient;
  }

  triplet_vector_t evaluateObjectiveGradientTriplet(const vector_t& x) const {
    triplet_vector_t gradient;
    for (size_t i = 0; i < objectives_.size(); ++i) {
      if (objectives_[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(objectives_[i]->getFunctionName());
        auto gradientTripletCurrent = objectives_[i]->getGradientTriplet(x, params);
        gradient.insert(gradient.end(), gradientTripletCurrent.begin(), gradientTripletCurrent.end());
      } else {
        auto gradientTripletCurrent = objectives_[i]->getGradientTriplet(x);
        gradient.insert(gradient.end(), gradientTripletCurrent.begin(), gradientTripletCurrent.end());
      }
    }
    return gradient;
  }

  CSRSparseMatrix evaluateObjectiveGradientCSR(const vector_t& x) const {
    CSRSparseMatrix gradients;
    if (objectives_[0]->isParameterized()) {
      auto params = parameterManager_->getParameters(objectives_[0]->getFunctionName());
      gradients = objectives_[0]->getGradientCSR(x, params);
    } else {
      gradients = objectives_[0]->getGradientCSR(x);
    }

    return gradients;
  }

  sparse_matrix_t evaluateObjectiveHessian(const vector_t& x) const {
    sparse_matrix_t hessian(variableDim_, variableDim_);
    for (size_t i = 0; i < objectives_.size(); ++i) {
      if (objectives_[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(objectives_[i]->getFunctionName());
        hessian += objectives_[i]->getHessian(x, params);
      } else {
        hessian += objectives_[i]->getHessian(x);
      }
    }
    return hessian;
  }

  triplet_vector_t evaluateObjectiveHessianTriplet(const vector_t& x) const {
    triplet_vector_t hessian;
    for (size_t i = 0; i < objectives_.size(); ++i) {
      if (objectives_[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(objectives_[i]->getFunctionName());
        auto hessianTripletCurrent = objectives_[i]->getHessianTriplet(x, params);
        hessian.insert(hessian.end(), hessianTripletCurrent.begin(), hessianTripletCurrent.end());
      } else {
        auto hessianTripletCurrent = objectives_[i]->getHessianTriplet(x);
        hessian.insert(hessian.end(), hessianTripletCurrent.begin(), hessianTripletCurrent.end());
      }
    }
    return hessian;
  }

  CSRSparseMatrix evaluateObjectiveHessianCSR(const vector_t& x) const {
    CSRSparseMatrix hessians;
    if (objectives_[0]->isParameterized()) {
      auto params = parameterManager_->getParameters(objectives_[0]->getFunctionName());
      hessians = objectives_[0]->getHessianCSR(x, params);
    } else {
      hessians = objectives_[0]->getHessianCSR(x);
    }
    return hessians;
  }

  size_t getVariableDim() const { return variableDim_; }

  size_t getNumObjectives() const { return objectives_.size(); }

  size_t getNumEqualityConstraints() const { return numEqualityConstraints_; }

  size_t getNumInequalityConstraints() const { return numInequalityConstraints_; }

  size_t getNumNonZeroEqJacobian() { return numNonZerosEqualityJacobian_; }

  size_t getNumNonZeroIneqJacobian() { return numNonZerosInequalityJacobian_; }

  size_t getNumNonZeroObjHessian() { return numNonZerosObjectiveHessian_; }

  size_t getNumNonZeroObjJacobian() { return numNonZerosObjectiveJacobian_; }

  std::string getProblemName() const { return problemName_; }

  std::vector<std::string> getObjectiveParamNames() const { return objectiveParamNames_; }

  std::vector<std::string> getEqualityParamNames() const { return equalityParamNames_; }

  std::vector<std::string> getInequalityParamNames() const { return inequalityParamNames_; }

 private:
  vector_t evaluateConstraints(const vector_t& x, const std::vector<std::shared_ptr<ConstraintFunction>>& constraints,
                               const std::vector<std::string>& paramNames, size_t totalRows) const {
    vector_t allConstraints(totalRows);
    size_t currentRow = 0;
    for (size_t i = 0; i < constraints.size(); ++i) {
      vector_t constrValues;
      if (constraints[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(constraints[i]->getFunctionName());
        constrValues = constraints[i]->getValue(x, params);
      } else {
        constrValues = constraints[i]->getValue(x);
      }
      std::memcpy(allConstraints.data() + currentRow, constrValues.data(), constrValues.size() * sizeof(scalar_t));
      currentRow += constrValues.size();
    }
    return allConstraints;
  }

  sparse_matrix_t evaluateConstraintsJacobian(const vector_t& x,
                                              const std::vector<std::shared_ptr<ConstraintFunction>>& constraints,
                                              const std::vector<std::string>& paramNames, size_t totalRows) const {
    std::vector<Eigen::Triplet<double>> tripletList;
    size_t currentRow = 0;

    for (size_t i = 0; i < constraints.size(); ++i) {
      sparse_matrix_t constrJacobian;
      if (constraints[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(constraints[i]->getFunctionName());
        constrJacobian = constraints[i]->getGradient(x, params);
      } else {
        constrJacobian = constraints[i]->getGradient(x);
      }

      for (int k = 0; k < constrJacobian.outerSize(); ++k) {
        for (sparse_matrix_t::InnerIterator it(constrJacobian, k); it; ++it) {
          tripletList.emplace_back(currentRow + it.row(), it.col(), it.value());
        }
      }
      currentRow += constraints[i]->getFunDim();
    }

    sparse_matrix_t jacobian(totalRows, variableDim_);
    jacobian.setFromTriplets(tripletList.begin(), tripletList.end());
    return jacobian;
  }

  triplet_vector_t evaluateConstraintsJacobianTriplet(
      const vector_t& x, const std::vector<std::shared_ptr<ConstraintFunction>>& constraints,
      const std::vector<std::string>& paramNames) const {
    triplet_vector_t tripletList;
    size_t currentRow = 0;

    for (size_t i = 0; i < constraints.size(); ++i) {
      triplet_vector_t constrJacobian;
      if (constraints[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(constraints[i]->getFunctionName());
        constrJacobian = constraints[i]->getGradientTriplet(x, params);
      } else {
        constrJacobian = constraints[i]->getGradientTriplet(x);
      }

      for (size_t j = 0; j < constrJacobian.size(); ++j) {
        tripletList.emplace_back(currentRow + constrJacobian[j].row(), constrJacobian[j].col(),
                                 constrJacobian[j].value());
      }
      currentRow += constraints[i]->getFunDim();
    }

    return tripletList;
  }

  // generated eigen CSR format sparse matrix of all constraints to speed up sparsity computation
  CSRSparseMatrix evaluateConstraintsJacobianCSR(const vector_t& x,
                                                 const std::vector<std::shared_ptr<ConstraintFunction>>& constraints,
                                                 const std::vector<std::string>& paramNames, const size_t totalRows,
                                                 const size_t numNonZeros) const {
    CSRSparseMatrix jacobianCSR(totalRows, numNonZeros);
    // fill in the CSR format
    size_t currentRow = 0;
    size_t currentNonZero = 0;
    for (size_t i = 0; i < constraints.size(); ++i) {
      CSRSparseMatrix constrJacobianCSR;
      if (constraints[i]->isParameterized()) {
        auto params = parameterManager_->getParameters(constraints[i]->getFunctionName());
        constrJacobianCSR = constraints[i]->getGradientCSR(x, params);
      } else {
        constrJacobianCSR = constraints[i]->getGradientCSR(x);
      }
      // contatenate the data vertically

      size_t constrJacNonZeros = constraints[i]->getNumNonZerosJacobian();
      size_t constrJacRows = constraints[i]->getFunDim();
      std::memcpy(jacobianCSR.innerIndices.data() + currentNonZero, constrJacobianCSR.innerIndices.data(),
                  constrJacNonZeros * sizeof(size_t));
      std::memcpy(jacobianCSR.values.data() + currentNonZero, constrJacobianCSR.values.data(),
                  constrJacNonZeros * sizeof(scalar_t));

      // adding an offset to the outer index
      for (size_t& val : constrJacobianCSR.outerIndex) {
        val += currentNonZero;
      }
      std::memcpy(jacobianCSR.outerIndex.data() + currentRow + 1, constrJacobianCSR.outerIndex.data() + 1,
                  constrJacRows * sizeof(size_t));
      currentRow += constrJacRows;
      currentNonZero += constrJacNonZeros;
    }

    return jacobianCSR;
  }
  size_t variableDim_;
  size_t numEqualityConstraints_;
  size_t numInequalityConstraints_;
  size_t numNonZerosEqualityJacobian_;
  size_t numNonZerosInequalityJacobian_;
  size_t numNonZerosObjectiveHessian_;
  size_t numNonZerosObjectiveJacobian_;
  std::shared_ptr<ParametersManager> parameterManager_;
  std::vector<std::shared_ptr<ObjectiveFunction>> objectives_;
  std::vector<std::shared_ptr<ConstraintFunction>> equalityConstraints_;
  std::vector<std::shared_ptr<ConstraintFunction>> inequalityConstraints_;
  std::string problemName_;
  std::vector<std::string> objectiveParamNames_;
  std::vector<std::string> equalityParamNames_;
  std::vector<std::string> inequalityParamNames_;
};

}  // namespace CRISP

#endif  // OPTIMIZATION_PROBLEM_H