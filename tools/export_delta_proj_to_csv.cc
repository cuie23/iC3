#include "serialization_utils.h"
#include <fstream>  
#include <iostream>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char* argv[]) {
  
  int n_x = 31;
  int n_u = 9;
  int n_lambda = 28;

  std::string filename = 
      "examples/resources/multifinger_hand/delta_projection_data/all_delta_projections_anitescu_drake_with_thresh2.bin";
  vector<vector<vector<MatrixXd>>> all_delta_projections = c3::utils::LoadTrajectoryData(filename);


  std::cout << all_delta_projections.size() << std::endl;
  std::cout << all_delta_projections[0].size() << std::endl;
  std::cout << all_delta_projections[0][0].size() << std::endl;
  std::cout << all_delta_projections[0][0][0].rows() << " " << all_delta_projections[0][0][0].cols() << std::endl;


  int iC3_iter = all_delta_projections.size();
  int admm_iter = all_delta_projections[0][0].size();

  std::cout << "iC3 iter " << iC3_iter << std::endl;
  std::cout << "ADMM iter " << admm_iter << std::endl;

  std::string csv_filename = 
      "examples/resources/multifinger_hand/delta_projection_data/csv/delta_projections_anitescu_drake_with_thresh2.csv";
  std::ofstream outfile(csv_filename);

  outfile << "timestep";
  for (int j = 0; j < n_x; j++) outfile << ",x_" << j;
  for (int j = 0; j < n_u; j++) outfile << ",u_" << j;
  for (int j = 0; j < n_lambda; j++) outfile << ",lambda_" << j;
  for (int j = 0; j < n_lambda; j++) outfile << ",eta_" << j;
  outfile << "\n";
  Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",");

  vector<vector<MatrixXd>> delta_projection_iter = all_delta_projections[iC3_iter-1];

  // Get projection for final admm iter, first c3 timestep
  for (int i = 0; i < delta_projection_iter.size(); i++) {
    // std::cout << "i " << i << std::endl;
    vector<MatrixXd> c3_delta_projection = delta_projection_iter[i];

    VectorXd z = c3_delta_projection[admm_iter-1].col(0);

    VectorXd x = z.segment(0, n_x).transpose();
    VectorXd u = z.segment(n_x + n_lambda, n_u).transpose();
    VectorXd lambda = z.segment(n_x, n_lambda).transpose();
    VectorXd eta = z.segment(n_x + n_lambda + n_u, n_lambda).transpose();

    // Write row to csv
    outfile << i << ","
                << x.transpose().format(CSVFormat) << ","
                << u.transpose().format(CSVFormat) << ","
                << lambda.transpose().format(CSVFormat) << ","
                << eta.transpose().format(CSVFormat) << "\n";
  }
  std::cout << "done" << std::endl;
  outfile.close();

  return 0;
}