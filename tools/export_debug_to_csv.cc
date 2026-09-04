#include "serialization_utils.h"
#include <filesystem>
#include <fstream>  
#include <iostream>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// bazel-bin/tools/export_debug_to_csv

int main(int argc, char* argv[]) {
  
  int n_x = 31;
  int n_u = 9;
  int n_lambda = 28;

  std::string c3_proj_filename = "examples/resources/multifinger_hand/ic3_debug_data/all_delta_projections.bin";
  if (!std::filesystem::exists(c3_proj_filename)) {
    c3_proj_filename = "examples/resources/multifinger_hand/ic3_debug_data/all_delta_projections_180_separate_thresh.bin";
  }
  const c3::utils::NestedMatrixDataC3Proj all_delta_projections =
      c3::utils::LoadTrajectoryData<c3::utils::NestedMatrixDataC3Proj>(
          c3_proj_filename);

  std::string z_sol_filename = "examples/resources/multifinger_hand/ic3_debug_data/all_z_sols.bin";
  if (!std::filesystem::exists(z_sol_filename)) {
    z_sol_filename = "examples/resources/multifinger_hand/ic3_debug_data/all_z_sols_180_separate_thresh.bin";
  }
  const c3::utils::NestedVectorDataZSol all_z_sols =
      c3::utils::LoadTrajectoryData<c3::utils::NestedVectorDataZSol>(
          z_sol_filename);

  std::string z_sol_rollout_filename = "examples/resources/multifinger_hand/ic3_debug_data/rollout_z_sol.bin";
  if (!std::filesystem::exists(z_sol_rollout_filename)) {
    z_sol_rollout_filename = "examples/resources/multifinger_hand/ic3_debug_data/rollout_z_sol_180_separate_thresh.bin";
  }
  vector<MatrixXd> z_sol_rollout;
  if (std::filesystem::exists(z_sol_rollout_filename)) {
    try {
      z_sol_rollout = c3::utils::LoadTrajectoryData<vector<MatrixXd>>(z_sol_rollout_filename);
    } catch (...) {}
  }

  std::string gamma_filename = "examples/resources/multifinger_hand/ic3_debug_data/gammas.bin";
  if (!std::filesystem::exists(gamma_filename)) {
    gamma_filename = "examples/resources/multifinger_hand/ic3_debug_data/gammas_180_separate_thresh.bin";
  }
  vector<MatrixXd> gamma;
  if (std::filesystem::exists(gamma_filename)) {
    try {
      gamma = c3::utils::LoadTrajectoryData<vector<MatrixXd>>(gamma_filename);
    } catch (...) {}
  }

  std::string in_contact_filename = "examples/resources/multifinger_hand/ic3_debug_data/in_contact.bin";
  if (!std::filesystem::exists(in_contact_filename)) {
    in_contact_filename = "examples/resources/multifinger_hand/ic3_debug_data/in_contact_180_separate_thresh.bin";
  }
  vector<MatrixXd> in_contact;
  if (std::filesystem::exists(in_contact_filename)) {
    try {
      in_contact = c3::utils::LoadTrajectoryData<vector<MatrixXd>>(in_contact_filename);
    } catch (...) {}
  }

  int iC3_iter = all_delta_projections.size();
  int admm_iter = all_delta_projections[0][0].size();

  std::cout << "iC3 iter " << iC3_iter << std::endl;
  std::cout << "ADMM iter " << admm_iter << std::endl;

  std::filesystem::create_directories("examples/resources/multifinger_hand/ic3_debug_data/csv");
  std::string c3_proj_csv_filename = 
      "examples/resources/multifinger_hand/ic3_debug_data/csv/delta_projections.csv";
  std::ofstream outfile_proj(c3_proj_csv_filename);

  outfile_proj << "timestep";
  for (int j = 0; j < n_x; j++) outfile_proj << ",x_" << j;
  for (int j = 0; j < n_u; j++) outfile_proj << ",u_" << j;
  for (int j = 0; j < n_lambda; j++) outfile_proj << ",lambda_" << j;
  for (int j = 0; j < n_lambda; j++) outfile_proj << ",eta_" << j;
  outfile_proj << "\n";
  Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",");

  vector<vector<MatrixXd>> delta_projection_iter = all_delta_projections[iC3_iter-1];

  // Get projection for final admm iter, first c3 timestep
  for (int i = 0; i < delta_projection_iter.size(); i++) {
    vector<MatrixXd> c3_delta_projection = delta_projection_iter[i];

    VectorXd z = c3_delta_projection[admm_iter-1].col(0);

    VectorXd x = z.segment(0, n_x).transpose();
    VectorXd u = z.segment(n_x + n_lambda, n_u).transpose();
    VectorXd lambda = z.segment(n_x, n_lambda).transpose();
    VectorXd eta = z.segment(n_x + n_lambda + n_u, n_lambda).transpose();

    // Write row to csv
    outfile_proj << i << ","
                << x.transpose().format(CSVFormat) << ","
                << u.transpose().format(CSVFormat) << ","
                << lambda.transpose().format(CSVFormat) << ","
                << eta.transpose().format(CSVFormat) << "\n";
  }
  outfile_proj.close();
  std::cout << "Saved " << c3_proj_csv_filename << std::endl;

  std::string z_sol_csv_filename = 
      "examples/resources/multifinger_hand/ic3_debug_data/csv/z_sol.csv";
  std::ofstream outfile_z_sol(z_sol_csv_filename);

  outfile_z_sol << "timestep";
  for (int j = 0; j < n_x; j++) outfile_z_sol << ",x_" << j;
  for (int j = 0; j < n_u; j++) outfile_z_sol << ",u_" << j;
  for (int j = 0; j < n_lambda; j++) outfile_z_sol << ",lambda_" << j;
  for (int j = 0; j < n_lambda; j++) outfile_z_sol << ",eta_" << j;
  outfile_z_sol << "\n";

  vector<vector<VectorXd>> z_sol_iter = all_z_sols[iC3_iter-1];

  for (int i = 0; i < z_sol_iter.size(); i++) {
    const vector<VectorXd>& c3_z_solution = z_sol_iter[i];

    VectorXd z = c3_z_solution[0];

    VectorXd x = z.segment(0, n_x).transpose();
    VectorXd u = z.segment(n_x + n_lambda, n_u).transpose();
    VectorXd lambda = z.segment(n_x, n_lambda).transpose();
    VectorXd eta = z.segment(n_x + n_lambda + n_u, n_lambda).transpose();

    // Write row to csv
    outfile_z_sol << i << ","
                << x.transpose().format(CSVFormat) << ","
                << u.transpose().format(CSVFormat) << ","
                << lambda.transpose().format(CSVFormat) << ","
                << eta.transpose().format(CSVFormat) << "\n";
  }
  outfile_z_sol.close();
  std::cout << "Saved " << z_sol_csv_filename << std::endl;

  if (z_sol_rollout.size() >= iC3_iter && iC3_iter > 0) {
    std::string rollout_csv_filename = 
        "examples/resources/multifinger_hand/ic3_debug_data/csv/rollout.csv";
    std::ofstream outfile_rollout(rollout_csv_filename);

    std::string gamma_csv_filename = 
        "examples/resources/multifinger_hand/ic3_debug_data/csv/gammas.csv";
    std::ofstream outfile_gamma(gamma_csv_filename);

    std::string in_contact_csv_filename = 
        "examples/resources/multifinger_hand/ic3_debug_data/csv/in_contact.csv";
    std::ofstream outfile_in_contact(in_contact_csv_filename);

    outfile_rollout << "timestep";
    for (int j = 0; j < n_x; j++) outfile_rollout << ",x_" << j;
    for (int j = 0; j < n_u; j++) outfile_rollout << ",u_" << j;
    for (int j = 0; j < n_lambda; j++) outfile_rollout << ",lambda_" << j;
    outfile_rollout << "\n";

    outfile_gamma << "timestep";
    for (int j = 0; j < n_lambda / 4; j++) outfile_gamma << ",gamma_" << j;
    outfile_gamma << "\n";

    outfile_in_contact << "timestep";
    for (int j = 0; j < n_lambda / 4; j++) outfile_in_contact << ",in_contact_" << j;
    outfile_in_contact << "\n";

    MatrixXd rollout_iter = z_sol_rollout[std::min(iC3_iter, (int)z_sol_rollout.size()-1)];
    for (int i = 0; i < rollout_iter.cols(); i++) {
      VectorXd z = rollout_iter.col(i);
      VectorXd x = z.segment(0, n_x).transpose();
      VectorXd u = z.segment(n_x + n_lambda, n_u).transpose();
      VectorXd lambda = z.segment(n_x, n_lambda).transpose();

      outfile_rollout << i << ","
                  << x.transpose().format(CSVFormat) << ","
                  << u.transpose().format(CSVFormat) << ","
                  << lambda.transpose().format(CSVFormat) << "\n";

      if (gamma.size() >= iC3_iter) {
        MatrixXd gamma_iter = gamma[iC3_iter-1];
        if (i < gamma_iter.cols()) {
          VectorXd gamma_i = gamma_iter.col(i);
          outfile_gamma << i << "," << gamma_i.transpose().format(CSVFormat) << "\n";
        }
      }
      if (in_contact.size() >= iC3_iter) {
        MatrixXd in_contact_iter = in_contact[iC3_iter-1];
        if (i < in_contact_iter.cols()) {
          VectorXd in_contact_i = in_contact_iter.col(i);
          outfile_in_contact << i << "," << in_contact_i.transpose().format(CSVFormat) << "\n";
        }
      }
    }
    outfile_rollout.close();
    outfile_gamma.close();
    outfile_in_contact.close();
  }

  return 0;
}
