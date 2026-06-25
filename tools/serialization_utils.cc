#include "serialization_utils.h"
#include <fstream>

namespace c3 {
namespace utils {

void SaveTrajectoryData(const NestedMatrixData& data, const std::string& filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    cereal::BinaryOutputArchive archive(os);
    archive(data); // Automatically cascades into vectors and your custom Eigen save template
}

NestedMatrixData LoadTrajectoryData(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    NestedMatrixData data;
    cereal::BinaryInputArchive archive(is);
    archive(data); // Automatically cascades into vectors and your custom Eigen load template
    
    return data;
}

} // namespace utils
} // namespace c3