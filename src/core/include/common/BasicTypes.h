#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <cppad/cg.hpp>
#include <cppad/cppad.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>

namespace CRISP{
// Basic types
using scalar_t = double;
using SizeVector = std::vector<size_t>;
using ValueVector = std::vector<scalar_t>;

// Eigen vector and matrix types
using vector_t = Eigen::VectorXd;
using matrix_t = Eigen::MatrixXd;
using sparse_vector_t = Eigen::SparseVector<scalar_t>;
using sparse_matrix_t = Eigen::SparseMatrix<scalar_t, Eigen::RowMajor>;
// define a structure with outterIndex, innerIndices and values
struct CSRSparseMatrix {
    SizeVector outerIndex;
    SizeVector innerIndices;
    ValueVector values;
    CSRSparseMatrix() = default;
    CSRSparseMatrix(const size_t& rows, const size_t& nnz) {
        outerIndex.resize(rows + 1);
        innerIndices.resize(nnz);
        values.resize(nnz);
    }
    CSRSparseMatrix(const SizeVector& outer, const SizeVector& inner, const ValueVector& vals) : outerIndex(outer), innerIndices(inner), values(vals) {}
    CSRSparseMatrix(const CSRSparseMatrix& other) {
        std::memcpy(outerIndex.data(), other.outerIndex.data(), other.outerIndex.size() * sizeof(int));
        std::memcpy(innerIndices.data(), other.innerIndices.data(), other.innerIndices.size() * sizeof(int));
        std::memcpy(values.data(), other.values.data(), other.values.size() * sizeof(scalar_t));
    }
    
    void toEigenSparseMatrix(sparse_matrix_t &sparseMatrix) {
        std::vector<int> outerIndex_int(outerIndex.begin(), outerIndex.end());
        std::vector<int> innerIndices_int(innerIndices.begin(), innerIndices.end());
        std::memcpy(sparseMatrix.outerIndexPtr(), outerIndex_int.data(), outerIndex_int.size() * sizeof(int));
        std::memcpy(sparseMatrix.innerIndexPtr(), innerIndices_int.data(), innerIndices_int.size() * sizeof(int));
        std::memcpy(sparseMatrix.valuePtr(), values.data(), values.size() * sizeof(scalar_t));
    }
    void print() {
        std::cout << "OuterIndex (row Pointers):" << std::endl;
        for (size_t i = 0; i < outerIndex.size(); ++i) {
            std::cout << outerIndex[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "InnerIndices (colum Indices):" << std::endl;
        for (size_t i = 0; i < innerIndices.size(); ++i) {
            std::cout << innerIndices[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Values:" << std::endl;
        for (size_t i = 0; i < values.size(); ++i) {
            std::cout << values[i] << " ";
        }
    }
};

// Non-AD functions
using function_t = std::function<void(const vector_t&, vector_t&)>;
using function_with_param_t = std::function<void(const vector_t&, const vector_t&, vector_t&)>;


// AD vector types (can use std::vector or Eigen containers)
using cg_scalar_t = CppAD::cg::CG<scalar_t>;
using ad_scalar_t = CppAD::AD<cg_scalar_t>;
using ad_vector_std = std::vector<ad_scalar_t>;
using ad_vector_t = Eigen::Matrix<ad_scalar_t, Eigen::Dynamic, 1>;
using ad_matrix_t = Eigen::Matrix<ad_scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
// AD function types
using ad_function_t = std::function<void(const ad_vector_t&, ad_vector_t&)>;
using ad_function_with_param_t = std::function<void(const ad_vector_t&, const ad_vector_t&, ad_vector_t&)>;
using triplet_vector_t = std::vector<Eigen::Triplet<scalar_t>>;
}   
#endif // BASIC_TYPES_H
