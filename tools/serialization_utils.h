#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <Eigen/Dense>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>

// Tell Cereal how to serialize Eigen plain objects, which covers both MatrixXd
// and VectorXd.
namespace cereal {
    template<class Archive, class Derived>
    void save(Archive& archive, const Eigen::PlainObjectBase<Derived>& matrix) {
        static_assert(std::is_same_v<typename Derived::Scalar, double>,
                      "serialization_utils only supports double Eigen types");
        Eigen::Index rows = matrix.rows();
        Eigen::Index cols = matrix.cols();
        archive(rows, cols);
        archive(binary_data(matrix.derived().data(), rows * cols * sizeof(double)));
    }

    template<class Archive, class Derived>
    void load(Archive& archive, Eigen::PlainObjectBase<Derived>& matrix) {
        static_assert(std::is_same_v<typename Derived::Scalar, double>,
                      "serialization_utils only supports double Eigen types");
        Eigen::Index rows, cols;
        archive(rows, cols);
        matrix.derived().resize(rows, cols);
        archive(binary_data(matrix.derived().data(), rows * cols * sizeof(double)));
    }
}

// 2. Define your top-level saving/loading function signatures
namespace c3 {
namespace utils {

using NestedMatrixDataC3Proj = std::vector<std::vector<std::vector<Eigen::MatrixXd>>>;
using NestedVectorDataZSol = std::vector<std::vector<std::vector<Eigen::VectorXd>>>;

void SaveTrajectoryDataProj(const NestedMatrixDataC3Proj& data, const std::string& filename);
NestedMatrixDataC3Proj LoadTrajectoryDataProj(const std::string& filename);

void SaveTrajectoryDataZSol(const NestedVectorDataZSol& data, const std::string& filename);
NestedVectorDataZSol LoadTrajectoryDataZSol(const std::string& filename);

template <typename T>
void SaveTrajectoryData(const T& data, const std::string& filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    cereal::BinaryOutputArchive archive(os);
    archive(data);
}

template <typename T>
T LoadTrajectoryData(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    T data;
    cereal::BinaryInputArchive archive(is);
    archive(data);
    return data;
}

} // namespace utils
} // namespace c3
