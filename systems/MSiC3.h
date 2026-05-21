#include <string>
#include <vector>
#include <tuple>

#include <drake/common/yaml/yaml_io.h>

#include "common/find_resource.h"
#include "core/c3.h"
#include "core/c3_miqp.h"
#include "core/c3_qp.h"
#include "core/lcs.h"
#include "systems/c3_controller_options.h"
#include "systems/MSiC3_options.h"
#include "systems/framework/c3_output.h"
#include "systems/framework/timestamped_vector.h"
#include "systems/PdInputSource.h"

#include "drake/systems/analysis/simulator.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

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
    MultibodyPlant<double>& plant,
    MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    MultibodyPlant<double>& plant_rollout,
    MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, 
    int example_idx);

  // Outputs
  // 0: x_hat for each iC3 iteration
  // 1: u_hat for each iC3 iteration
  // 2: Quadratic terms for LQR value function for each iC3 iteration
  // 3: Linear terms for LQR value function for each iC3 iteration
  // 4: LQR feedback gains for each iC3 iteration
  // 5: LQR feedforward gains for each iC3 iteration
  tuple<vector<MatrixXd>, vector<MatrixXd>, vector<vector<MatrixXd>>, vector<vector<VectorXd>>, 
      vector<vector<MatrixXd>>, vector<vector<VectorXd>>> ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    drake::systems::Context<double>& context_rollout,
    drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout, 
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const vector<SortedPair<GeometryId>>& contact_geoms_rollout);

private:
  
  VectorXd ProjectContactVertical(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx);

  VectorXd ProjectContact(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                  VectorXd x_init, int start_idx, int q_size); 

  tuple<LCS, MatrixXd, MatrixXd, MatrixXd> DoLCSRollout(VectorXd x0, MatrixXd x_hat_prev, MatrixXd c3_x_hat, MatrixXd u_hat, 
                                              LCSFactory factory, LCSFactory rollout_factory, MatrixXd A_constraint_x, 
                                              VectorXd lower_bound_x, VectorXd upper_bound_x, MatrixXd A_constraint_u, 
                                              VectorXd lower_bound_u, VectorXd upper_bound_u, vector<MatrixXd> K, 
                                              vector<VectorXd> k_ff, double alpha);

  // Given an initial x, nominal u, time-varying lcs, simulate with C3 mpc
  // Number of steps simulated = number of cols in u_hat
  // lcs, H, g, x_targets are all over the entire iC3 time horizon, indexing done in function
  // start_idx is the timestep w.r.t the entire iC3 time horizon to start from
  // returns x_hat, u_hat, lambda_hat
  // TODO: context and contact_geoms only get used for debugging
  tuple<MatrixXd, MatrixXd, MatrixXd> DoC3Rollout(VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, 
                                              LCSFactory factory, LCSFactory rollout_factory, vector<MatrixXd> H, 
                                              vector<VectorXd> g, vector<VectorXd> x_targets, int start_idx,                                          
                                              MatrixXd A_x, VectorXd lb_x, VectorXd ub_x,
                                              MatrixXd A_u, VectorXd lb_u, VectorXd ub_u);


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

  LCS MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory);
  LCS MakeTimeVaryingLCSWithEE(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory, VectorXd ee_pose, int ee_idx);

  LCS GetLCSSegment(LCS lcs, int start_idx, int length);

  // x_hat (N by n_x), kth row is x at time k
  void UpdateQuaternionCosts(
    MatrixXd x_hat, const Eigen::VectorXd& x_des);


  const drake::multibody::MultibodyPlant<double>& plant_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_;
  const drake::multibody::MultibodyPlant<double>& plant_rollout_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout_;

  // C3 options and solver configuration.
  C3ControllerOptions controller_options_;
  MSiC3Options ms_ic3_options_;

  // Convenience variables for dimensions.
  int n_q_;       // Number of generalized positions.
  int n_v_;       // Number of generalized velocities.
  int n_x_;       // Total state dimension.
  int n_lambda_;  // Number of Lagrange multipliers.
  int n_u_;       // Number of control inputs.
  double dt_;     // Time step for c3

  // Cost matrices for optimization.
  mutable std::vector<Eigen::MatrixXd> Q_;  ///< State cost matrices.
  mutable std::vector<Eigen::MatrixXd> R_;  ///< Input cost matrices.
  mutable std::vector<Eigen::MatrixXd> G_;  ///< State-input cross-term matrices.
  mutable std::vector<Eigen::MatrixXd> U_;  ///< Constraint matrices.
  
  mutable std::vector<Eigen::MatrixXd> P_;  ///< Defect cost matrices

  int N_;  // Horizon length (of whole trajectory).
  int num_segments_;
  int L_;  // segment length

  int example_idx_;

};

} // namespace systems
} // namespace c3