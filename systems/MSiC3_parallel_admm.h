#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <optional>
#include <memory>

#include <drake/common/yaml/yaml_io.h>

#include "common/find_resource.h"
#include "core/c3.h"
#include "core/c3_plus.h"
#include "core/lcs.h"
#include "systems/c3_controller_options.h"
#include "systems/MSiC3_options.h"
#include "systems/framework/c3_output.h"
#include "systems/framework/timestamped_vector.h"
#include "systems/PdInputSource.h"
#include "systems/hybrid_mpc_options.h"
#include "systems/hybrid_mpc.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/multibody/plant/contact_results.h"

using std::vector;
using std::pair;
using std::tuple;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::multibody::MultibodyPlant;
using drake::geometry::GeometryId;
using drake::SortedPair;
using drake::multibody::ContactResults;

namespace c3 {
namespace systems {

class MSiC3ParallelAdmm {

public:

  // Example idx
  // 0 = plate
  // 1 = trifinger 180
  // 2 = trifinger pivot
  explicit MSiC3ParallelAdmm(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<vector<SortedPair<GeometryId>>>& contact_groups,
    const vector<vector<SortedPair<GeometryId>>>& contact_groups_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, 
    HybridMpcOptions mpc_options, int example_idx, bool is_optuna = false);

  explicit MSiC3ParallelAdmm(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<vector<SortedPair<GeometryId>>>& contact_groups,
    const vector<vector<SortedPair<GeometryId>>>& contact_groups_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, 
    int example_idx, bool is_optuna = false);

  explicit MSiC3ParallelAdmm(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const vector<SortedPair<GeometryId>>& contact_geoms_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, 
    HybridMpcOptions mpc_options, int example_idx);

  explicit MSiC3ParallelAdmm(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const vector<SortedPair<GeometryId>>& contact_geoms_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, 
    int example_idx);

  // Outputs:
  // 0: x_hat for each ADMM iteration
  // 1: u_hat for each ADMM iteration
  // 2: lambda_hat for each ADMM iteration
  tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>> ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    std::vector<drake::systems::Context<double>*> contexts_rollout,
    drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout,
    double plate_u_torque_bound = -1);

  void SetSolverOptions(const drake::solvers::SolverOptions& solver_options);
  void SetSolverOptions(const std::string& solver_options_file);
  const std::optional<drake::solvers::SolverOptions>& solver_options() const { return solver_options_; }

private:
  
  VectorXd ProjectFeasible(const VectorXd& x_in, drake::systems::Context<double>& context,
                           const MatrixXd& A_x, const VectorXd& lb_x, const VectorXd& ub_x);

  VectorXd ProjectContactVertical(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx,
                                MatrixXd A_x, VectorXd lb_x, VectorXd ub_x);

  VectorXd ProjectContactPlate(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx, int pitch_idx, 
                                MatrixXd A_x, VectorXd lb_x, VectorXd ub_x);

  VectorXd ProjectContact(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                          VectorXd x_init, int start_idx, int q_size, MatrixXd A_x, VectorXd lb_x, VectorXd ub_x); 

  tuple<LCS, MatrixXd, MatrixXd, MatrixXd> DoLCSRollout(VectorXd x0, MatrixXd u_hat, 
                                              LCSFactory factory, LCSFactory rollout_factory, MatrixXd A_constraint_x, 
                                              VectorXd lower_bound_x, VectorXd upper_bound_x, MatrixXd A_constraint_u, 
                                              VectorXd lower_bound_u, VectorXd upper_bound_u, vector<MatrixXd> K, 
                                              vector<VectorXd> k_ff, double alpha);

  struct StepAdmmMetrics {
    std::vector<double> lambda_res;
    std::vector<double> step_changes;
    std::vector<double> comp_slacks;
    Eigen::VectorXd x0;
    Eigen::VectorXd x1;
    Eigen::VectorXd lambda;
    Eigen::VectorXd eta;
    Eigen::VectorXd lambda_last;
    Eigen::VectorXd eta_last;
    std::vector<Eigen::VectorXd> full_x_lookahead;
  };

  // Given initial x0 and ADMM boundary targets, simulate with C3 mpc for one segment
  // returns x_hat, u_hat, lambda_hat, J_policy, admm_metrics
  tuple<MatrixXd, MatrixXd, MatrixXd, vector<MatrixXd>, vector<StepAdmmMetrics>> DoC3Rollout(
    VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, 
    MatrixXd lambda_hat, VectorXd ud, VectorXd x_boundary_target,
    LCSFactory factory, LCSFactory rollout_factory,
    int start_idx, int segment_idx,                                          
    MatrixXd A_x, VectorXd lb_x, VectorXd ub_x,
    MatrixXd A_u, VectorXd lb_u, VectorXd ub_u,
    drake::systems::Context<double>& context, drake::systems::Context<double>& context_rollout,
    c3::C3Plus& c3_tracking, double rho);

  LCS MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, LCSFactory factory);
  LCS MakeTimeVaryingLCSWithEE(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, LCSFactory factory, VectorXd ee_pose, int ee_idx);

  // Returns lambda, gamma, in_contact
  std::tuple<VectorXd, VectorXd, VectorXd> ConstructLambdasFromContactResults(drake::multibody::ContactResults<double> contact_results, std::string contact_model);

  void UpdateQuaternionCosts(MatrixXd x_hat, VectorXd x_des);
  vector<MatrixXd> UpdateQuaternionCosts(VectorXd x_curr, const vector<VectorXd>& x_des,
                                         const vector<VectorXd>& x_admm, const vector<MatrixXd>& Q_in,
                                         double rho);

  Eigen::Quaterniond slerpLong(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1, double t);

  void ResolveContacts(
    const drake::systems::Context<double>& context,
    const drake::systems::Context<double>& context_rollout);

  const drake::multibody::MultibodyPlant<double>& plant_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_;
  const drake::multibody::MultibodyPlant<double>& plant_rollout_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout_;
  drake::systems::Diagram<double>& rollout_diagram_;
  std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context_;

  vector<vector<SortedPair<GeometryId>>> contact_groups_;
  vector<vector<SortedPair<GeometryId>>> contact_groups_rollout_;
  vector<SortedPair<GeometryId>> contact_geoms_;
  vector<SortedPair<GeometryId>> contact_geoms_rollout_;

  bool use_drake_sim_ = false;
  C3ControllerOptions controller_options_;
  MSiC3Options ms_ic3_options_;
  HybridMpcOptions mpc_options_;
  int N_;  // Horizon length (of whole trajectory).
  int example_idx_;
  std::optional<drake::solvers::SolverOptions> solver_options_;

  std::vector<std::unique_ptr<c3::C3Plus>> c3_trackings_;

  // Convenience variables for dimensions.
  int n_q_;       // Number of generalized positions.
  int n_v_;       // Number of generalized velocities.
  int n_x_;       // Total state dimension.
  int n_lambda_;  // Number of Lagrange multipliers.
  int n_u_;       // Number of control inputs.
  double dt_;     // Time step for c3

  std::vector<std::unique_ptr<drake::systems::Simulator<double>>> simulators_;

  // Cost matrices for optimization.
  mutable std::vector<Eigen::MatrixXd> Q_;  ///< State cost matrices.
  mutable std::vector<Eigen::MatrixXd> R_;  ///< Input cost matrices.
  mutable std::vector<Eigen::MatrixXd> G_;  ///< State-input cross-term matrices.
  mutable std::vector<Eigen::MatrixXd> U_;  ///< Constraint matrices.

  int num_segments_;
  int L_;  // segment length

  bool is_optuna_ = false;
};

} // namespace systems
} // namespace c3