#include "serialization_utils.h"
#include <fstream>

namespace c3 {
namespace utils {

void SaveTrajectoryDataProj(const NestedMatrixDataC3Proj& data, const std::string& filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    cereal::BinaryOutputArchive archive(os);
    archive(data); // Automatically cascades into vectors and your custom Eigen save template
}

NestedMatrixDataC3Proj LoadTrajectoryDataProj(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    NestedMatrixDataC3Proj data;
    cereal::BinaryInputArchive archive(is);
    archive(data); // Automatically cascades into vectors and your custom Eigen load template
    
    return data;
}

void SaveTrajectoryDataZSol(const NestedVectorDataZSol& data, const std::string& filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    cereal::BinaryOutputArchive archive(os);
    archive(data); // Automatically cascades into vectors and your custom Eigen save template
}

NestedVectorDataZSol LoadTrajectoryDataZSol(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    NestedVectorDataZSol data;
    cereal::BinaryInputArchive archive(is);
    archive(data); // Automatically cascades into vectors and your custom Eigen load template
    
    return data;
}

} // namespace utils
} // namespace c3