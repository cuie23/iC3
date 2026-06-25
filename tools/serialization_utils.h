#pragma once

#include <vector>
#include <string>
#include <Eigen/Dense>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>

// 1. Tell Cereal how to serialize Eigen matrices.
// These must be templated so they work with any Cereal archive type (Binary, JSON, etc.)
namespace cereal {
    template<class Archive>
    void save(Archive& archive, const Eigen::MatrixXd& matrix) {
        Eigen::Index rows = matrix.rows();
        Eigen::Index cols = matrix.cols();
        archive(rows, cols);
        archive(binary_data(matrix.data(), rows * cols * sizeof(double)));
    }

    template<class Archive>
    void load(Archive& archive, Eigen::MatrixXd& matrix) {
        Eigen::Index rows, cols;
        archive(rows, cols);
        matrix.resize(rows, cols);
        archive(binary_data(matrix.data(), rows * cols * sizeof(double)));
    }
}

// 2. Define your top-level saving/loading function signatures
namespace c3 {
namespace utils {

using NestedMatrixData = std::vector<std::vector<std::vector<Eigen::MatrixXd>>>;

void SaveTrajectoryData(const NestedMatrixData& data, const std::string& filename);
NestedMatrixData LoadTrajectoryData(const std::string& filename);

} // namespace utils
} // namespace c3