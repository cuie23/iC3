#include "serialization_utils.h"

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char* argv[]) {
  
  int n_x = 31;
  int n_u = 9;
  int n_lambda = 28;

  std::string filename = "all_delta_projections.bin";
  vector<vector<vector<MatrixXd>>> all_delta_projections = c3::utils::LoadTrajectoryData(filename);


  std::cout << all_delta_projections.size() << std::endl;
  std::cout << all_delta_projections[0].size() << std::endl;
  std::cout << all_delta_projections[0][0].size() << std::endl;
  std::cout << all_delta_projections[0][0][0].rows() << " " << all_delta_projections[0][0][0].cols() << std::endl;


  int iC3_iter = 2;
  int admm_iter = 2;

  vector<vector<MatrixXd>> delta_projection_iter = all_delta_projections[iC3_iter-1];

  // Get projection for final admm iter, first c3 timestep
  for (int i = 0; i < delta_projection_iter.size(); i++) {
    std::cout << "i " << i << std::endl;
    vector<MatrixXd> c3_delta_projection = delta_projection_iter[i];

    VectorXd z = c3_delta_projection[admm_iter-1].col(0);

    std::cout << "x " << z.segment(0, n_x).transpose() << std::endl;
    std::cout << "u " << z.segment(n_x + n_lambda, n_u).transpose() << std::endl;
    std::cout << "lambda " << z.segment(n_x, n_lambda).transpose() << std::endl;
    std::cout << "eta " << z.segment(n_x + n_lambda + n_u, n_lambda).transpose() << std::endl;
    std::cout << std::endl;

  }
  
}