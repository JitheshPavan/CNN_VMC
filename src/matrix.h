#include <Eigen/Core>

#ifndef MATRIX_H
#define MATRIX_H

using Vector = Eigen::VectorXd;
using Vector3i = Eigen::Vector3i;
using Vector3d = Eigen::Vector3d;
using ivector = Eigen::VectorXi;
using RealVector = Eigen::VectorXd;
using ComplexVector = Eigen::VectorXcd;
using IntMatrix = Eigen::MatrixXi;
using RealMatrix = Eigen::MatrixXd;
using ComplexMatrix = Eigen::MatrixXcd;
using ColVector = Eigen::VectorXcd;
using RowVector = Eigen::RowVectorXcd;
using double_Array = Eigen::ArrayXd;
namespace eig {
  using real_vec = Eigen::VectorXd;
  using cmpl_vec = Eigen::VectorXcd;
}
using Matrix = Eigen::MatrixXd;

#endif
