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
#include "systems/iC3_options.h"
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

class iC3 : public drake::systems::LeafSystem<double> {

public:

  // Example idx
  // 0 = plate
  // 1 = allegro (joint space)
  // 2 = allegro (point fingers)
 explicit iC3(
    MultibodyPlant<double>& plant,
    MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    MultibodyPlant<double>& plant_rollout,
    MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    C3::CostMatrices& costs, 
    C3ControllerOptions controller_options, iC3Options ic3_options, 
    int example_idx);


  // Outputs
  // 0: x_hat for each iC3 iteration
  // 1: u_hat for each iC3 iteration
  // 2: C3's x solution for each iC3 iteration
  // 3: x's simulated with drake for each iC3 iteration
  // 4: Quadratic terms for LQR value function for each iC3 iteration
  // 5: Linear terms for LQR value function for each iC3 iteration
  // 6: LQR feedback gains for each iC3 iteration
  // 7: LQR feedforward gains for each iC3 iteration
  tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>, vector< vector<MatrixXd>>, 
      vector<vector<VectorXd>>, vector<vector<MatrixXd>>, vector<vector<VectorXd>>> ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    drake::systems::Context<double>& context_rollout,
    drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout, 
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const vector<SortedPair<GeometryId>>& contact_geoms_rollout);

private:
  
  // Given an initial x and u trajectory, return x rollout out using lcs
  // returns LCS, x_hat, u_hat, lambda_hat
  tuple<LCS, MatrixXd, MatrixXd, MatrixXd> DoLCSRollout(VectorXd x0, MatrixXd x_hat_prev, MatrixXd c3_x_hat, MatrixXd u_hat, 
                                              LCSFactory factory, LCSFactory rollout_factory, MatrixXd A_constraint_x, 
                                              VectorXd lower_bound_x, VectorXd upper_bound_x, MatrixXd A_constraint_u, 
                                              VectorXd lower_bound_u, VectorXd upper_bound_u, vector<MatrixXd> K, 
                                              vector<VectorXd> k_ff, double alpha);
 
  MatrixXd RolloutUHatPlate(VectorXd x0, MatrixXd c3_x, MatrixXd c3_u);

  tuple<MatrixXd, MatrixXd> RolloutUHatHand(VectorXd x0, MatrixXd c3_x, MatrixXd c3_u,
       const vector<SortedPair<GeometryId>>& contact_geoms);

  tuple<MatrixXd, MatrixXd> RolloutUHatPointHand(VectorXd x0, MatrixXd c3_x, MatrixXd c3_u,
       const vector<SortedPair<GeometryId>>& contact_geoms);

  // TODO: FIX THIS
  VectorXd GetLambdaFromContacts(ContactResults<double> contact_results,
      const vector<SortedPair<GeometryId>>& contact_geoms);


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
    ComputeLQRValueFunction(MatrixXd x_hat, MatrixXd u_hat, 
      MatrixXd lambda_hat, VectorXd xd, VectorXd ud, LCS lcs);

  LCS MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory);
  LCS MakeTimeVaryingLCSWithEE(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory, VectorXd ee_position, int ee_idx, int num_ee);

  // removes num_timesteps_to_remove timesteps from the front of the LCS
  LCS ShortenLCSFront(LCS lcs, int num_timesteps_to_remove);

  C3::CostMatrices ShortenCostsFront(int num_timesteps_to_remove);

  // x_hat (N by n_x), kth row is x at time k
  void UpdateQuaternionCosts(
    MatrixXd x_hat, const Eigen::VectorXd& x_des, vector<VectorXd> c3_quat_norms);

  // Note: both override (parts of) Q_ and R_, make sure this doesn't conflict with quaternion cost terms
  void UpdateDecouplingCosts(int position_idx, int velocity_idx);
  void UpdateL1Costs(int position_idx, int num_positions, int velocity_idx, int num_velocities);
  void UpdateHuberCosts(int position_idx, int num_positions, int velocity_idx, int num_velocities);

  const drake::multibody::MultibodyPlant<double>& plant_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_;
  const drake::multibody::MultibodyPlant<double>& plant_rollout_;
  const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout_;

  // C3 options and solver configuration.
  C3ControllerOptions controller_options_;
  iC3Options ic3_options_;

  // Convenience variables for dimensions.
  int n_q_;       ///< Number of generalized positions.
  int n_v_;       ///< Number of generalized velocities.
  int n_x_;       ///< Total state dimension.
  int n_lambda_;  ///< Number of Lagrange multipliers.
  int n_u_;       ///< Number of control inputs.
  double dt_;     ///< Time step.

  // C3 solver instance.
  mutable std::unique_ptr<C3> c3_;

  // Cost matrices for optimization.
  mutable std::vector<Eigen::MatrixXd> Q_;  ///< State cost matrices.
  mutable std::vector<Eigen::MatrixXd> R_;  ///< Input cost matrices.
  mutable std::vector<Eigen::MatrixXd> G_;  ///< State-input cross-term matrices.
  mutable std::vector<Eigen::MatrixXd> U_;  ///< Constraint matrices.

  int N_;  ///< Horizon length.
  int example_idx_;

};

} // namespace systems
} // namespace c3