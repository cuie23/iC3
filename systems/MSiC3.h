#pragma once

#include <string>
#include <vector>
#include <tuple>

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

class MSiC3 {

public:

  // Example idx
  // 0 = plate
  // 1 = trifinger (point fingers)
  explicit MSiC3(
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

  explicit MSiC3(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const vector<SortedPair<GeometryId>>& contact_geoms_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, 
    int example_idx, bool is_optuna = false);

  // Outputs
  // Note: doesn't store stuff from warmup iterations
  // 0: x_hat for each MSiC3 iteration
  // 1: u_hat for each MSiC3 iteration
  // 2: lambda_hat for each MSiC3 iteration
  // 3: Quadratic terms for LQR value function for each MSiC3 iteration
  // 4: Linear terms for LQR value function for each MSiC3 iteration
  // 5: LQR feedback gains for each MSiC3 iteration
  // 6: LQR feedforward gains for each MSiC3 iteration
  // 7: c3 projections for each MSiC3 iteration
  // 8: c3 z solutions for each MSiC3 iteration
  // 9: gammas for each MSiC3 iteration
  // 10: whether each contact point is in contact for each MSiC3 iteration
  tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>, vector<vector<MatrixXd>>, 
        vector<vector<VectorXd>>, vector<vector<MatrixXd>>, vector<vector<VectorXd>>, 
        vector<vector<vector<MatrixXd>>>, vector<vector<vector<VectorXd>>>,
        vector<MatrixXd>, vector<MatrixXd>> ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    drake::systems::Context<double>& context_rollout,
    drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout,
    double plate_u_torque_bound = -1);

  // 1. x_hat for each x0
  // 2. u_hat for each x0
  // 3. lambda_hat for each x0
  tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>> DoHybridMPCTracking(
    vector<VectorXd> x0s, MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, 
      HybridMpcOptions mpc_options, drake::systems::Context<double>& context,
      drake::systems::Context<drake::AutoDiffXd>& context_ad, drake::systems::Context<double>& context_rollout);

  void SetSolverOptions(const drake::solvers::SolverOptions& solver_options);
  void SetSolverOptions(const std::string& solver_options_file);
  const std::optional<drake::solvers::SolverOptions>& solver_options() const { return solver_options_; }

private:
  
  VectorXd ProjectContactVertical(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx);

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

  // Given an initial x, nominal u, time-varying lcs, simulate with C3 mpc
  // Number of steps simulated = number of cols in u_hat
  // x_hat, lcs, H, g, x_targets are all over the entire iC3 time horizon, indexing done in function
  // start_idx is the timestep w.r.t the entire iC3 time horizon to start from
  // returns x_hat, u_hat, lambda_hat, gamma, in_contact
  tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd> DoC3Rollout(VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, 
                                              MatrixXd lambda_hat, VectorXd ud, VectorXd x_anchor_next,
                                              LCSFactory factory, LCSFactory rollout_factory, vector<MatrixXd> H, 
                                              vector<VectorXd> g, int start_idx,                                          
                                              MatrixXd A_x, VectorXd lb_x, VectorXd ub_x,
                                              MatrixXd A_u, VectorXd lb_u, VectorXd ub_u,
                                              drake::systems::Context<double>& context, drake::systems::Context<double>& context_rollout);


  // For affine time-varying LQR problem get value function
  // min  Σ (x[k]'Q[k]x[k] + u[k]'R[k]u[k]) + x[f]'Q[f]x[f]
  // s.t. x[k+1] = A[k]x[k] + B[k]u[k] + (D[k]λ[k] + d[k])
  // where λ[k] is fixed, denote c[k] = D[k]λ[k] + d[k] 
  //
  // Value function of form
  // V(x,k) = (1/2)(x'H[k]x) + g[k]'x
  //
  // Returns H, g, K, k_ff (gains for debugging)
  std::tuple<vector<MatrixXd>, vector<VectorXd>, vector<MatrixXd>, vector<VectorXd>> 
    ComputeLQRValueFunction(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat,
                            LCS lcs, VectorXd xd, VectorXd ud, MatrixXd defects);

  std::tuple<vector<MatrixXd>, vector<VectorXd>, vector<MatrixXd>, vector<VectorXd>> 
    ComputeBoxDDPValueFunction(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat,
        LCS lcs, VectorXd xd, VectorXd ud, MatrixXd defects,
        VectorXd u_min, VectorXd u_max);
  VectorXd SolveBoxQP(const MatrixXd& Q_uu,
                      const VectorXd& Q_u,
                      const VectorXd& lower_bound,
                      const VectorXd& upper_bound,
                      const VectorXd& k_init,
                      std::vector<int>& free_indices);

  vector<MatrixXd> GetLowRankApproximation(vector<MatrixXd> H_in);

  LCS MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, LCSFactory factory);
  LCS MakeTimeVaryingLCSWithEE(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, LCSFactory factory, VectorXd ee_pose, int ee_idx);

  LCS GetLCSSegment(LCS lcs, int start_idx, int length);

  // ASSUMES ANITESCU AND 2 FRICTION DIRECTIONS
  // Returns lambda, gamma, in_contact
  std::tuple<VectorXd, VectorXd, VectorXd> ConstructLambdasFromContactResults(drake::multibody::ContactResults<double> contact_results, std::string contact_model);

  // x_hat (N by n_x), kth row is x at time k
  void UpdateQuaternionCosts(
    MatrixXd x_hat, VectorXd x_des);

  vector<MatrixXd> UpdateQuaternionCosts(
    VectorXd x_curr, vector<VectorXd> x_des, vector<MatrixXd> Q);

  MatrixXd UpdateQuaternionCostsSlack(
    VectorXd x_curr, VectorXd x_des, MatrixXd Q_in);

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

  // C3 options and solver configuration.
  C3ControllerOptions controller_options_;
  MSiC3Options ms_ic3_options_;
  std::optional<drake::solvers::SolverOptions> solver_options_;

  std::unique_ptr<c3::C3Plus> c3_tracking_;

  // Convenience variables for dimensions.
  int n_q_;       // Number of generalized positions.
  int n_v_;       // Number of generalized velocities.
  int n_x_;       // Total state dimension.
  int n_lambda_;  // Number of Lagrange multipliers.
  int n_u_;       // Number of control inputs.
  double dt_;     // Time step for c3

  bool use_drake_sim_ = false;
  std::unique_ptr<drake::systems::Simulator<double>> simulator_ = nullptr;

  // Cost matrices for optimization.
  mutable std::vector<Eigen::MatrixXd> Q_;  ///< State cost matrices.
  mutable std::vector<Eigen::MatrixXd> R_;  ///< Input cost matrices.
  mutable std::vector<Eigen::MatrixXd> G_;  ///< State-input cross-term matrices.
  mutable std::vector<Eigen::MatrixXd> U_;  ///< Constraint matrices.
  

  int N_;  // Horizon length (of whole trajectory).
  int num_segments_;
  int L_;  // segment length

  int example_idx_;

  // Indexing: ic3 timestep, admm iteration, c3 horizon
  std::vector<std::vector<Eigen::MatrixXd>> delta_projection_iter_;

  // Indexing: ic3_timestep, c3 horizon
  std::vector<std::vector<Eigen::VectorXd>> z_sol_iter_;

  struct LambdaResidualRecord {
    int outer_iter;
    int segment;
    int plan_timestep;
    int admm_iter;
    double lambda_diff_norm;
    double iterate_step_change;
    double complementarity_slack;
  };
  struct C3PlanStepRecord {
    int outer_iter;
    int segment;
    int plan_timestep;
    Eigen::VectorXd x0;
    Eigen::VectorXd x1;
    Eigen::VectorXd lambda;
    Eigen::VectorXd eta;
    Eigen::VectorXd lambda_last;
    Eigen::VectorXd eta_last;
  };
  std::vector<C3PlanStepRecord> c3_plan_step_records_;

  struct C3FullLookaheadRecord {
    int outer_iter;
    int segment;
    int plan_timestep;
    int lookahead_step;
    Eigen::VectorXd x;
  };
  std::vector<C3FullLookaheadRecord> c3_full_lookahead_records_;
  std::vector<LambdaResidualRecord> lambda_residual_records_;
  int current_outer_iter_ = 0;
  bool is_optuna_ = false;
};



} // namespace systems
} // namespace c3