#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/systems/analysis/simulator.h"

#include <Eigen/Dense>
#include <cmath>
#include <iostream> 
#include "core/c3.h"
#include "core/lcs.h"
#include "multibody/lcs_factory.h"
#include "systems/c3_controller_options.h"
#include "systems/MSiC3_options.h"
#include "systems/hybrid_mpc_options.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/osqp_solver.h"

using drake::multibody::MultibodyPlant;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using drake::geometry::GeometryId;
using drake::SortedPair;
using drake::math::RigidTransform;

namespace c3 {

using multibody::LCSFactory;

namespace systems {

// Outputs a manually generated set of inputs
class HybridMPC {
 public:
  explicit HybridMPC(const MultibodyPlant<double>& plant_rollout, LCSFactory lcs_factory,
    drake::systems::Diagram<double>& rollout_diagram, 
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context, 
    const vector<SortedPair<GeometryId>>& contact_geoms_rollout, HybridMpcOptions mpc_options,
    MSiC3Options ms_ic3_options, int example_idx, vector<double> mu_vector,
    MatrixXd A_x, VectorXd lb_x, VectorXd ub_x, MatrixXd A_u, VectorXd lb_u, VectorXd ub_u) ;

  std::tuple<MatrixXd, MatrixXd, MatrixXd> SimulateHybridMPC(
      VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, drake::systems::Context<double>& context_rollout);

private:
  
  int n_q_;
  int n_v_;
  int n_x_;
  int n_u_;
  int n_lambda_;

  MatrixXd Q_;
  MatrixXd R_;
  MatrixXd S_;
  MatrixXd G_;

  const MultibodyPlant<double>& plant_rollout_;
  c3::multibody::LCSFactory lcs_factory_;
  const vector<SortedPair<GeometryId>>& contact_geoms_rollout_;

  std::unique_ptr<drake::systems::Simulator<double>> simulator_;
  drake::systems::Diagram<double>& rollout_diagram_;
  std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context_;
  
  HybridMpcOptions mpc_options_;
  MSiC3Options ms_ic3_options_;
  drake::solvers::SolverOptions solver_options_;
  
  double dt_; 
  int N_;

  int example_idx_;

  vector<double> mu_vector_;
  VectorXd lambda_threshold_;
  VectorXd eta_threshold_;

  drake::math::RigidTransform<double> X_delta_;

  MatrixXd A_x_;
  VectorXd lb_x_;
  VectorXd ub_x_;
  MatrixXd A_u_;
  VectorXd lb_u_;
  VectorXd ub_u_;
  
  drake::solvers::MathematicalProgram prog_;
  drake::solvers::OsqpSolver osqp_;
  std::vector<drake::solvers::VectorXDecisionVariable> x_;
  std::vector<drake::solvers::VectorXDecisionVariable> u_;
  std::vector<drake::solvers::VectorXDecisionVariable> lambda_;
  std::vector<drake::solvers::VectorXDecisionVariable> epsilon_;

  std::vector<drake::solvers::QuadraticCost*> target_costs_;
  std::vector<drake::solvers::QuadraticCost*> input_costs_;
  std::vector<drake::solvers::QuadraticCost*> force_costs_;
  std::vector<drake::solvers::QuadraticCost*> slack_costs_;

  drake::solvers::LinearEqualityConstraint* initial_state_constraint_;
  std::vector<drake::solvers::LinearEqualityConstraint*> dynamics_constraints_;
  std::vector<drake::solvers::LinearConstraint*> lambda_constraints_;
  std::vector<drake::solvers::LinearConstraint*> eta_constraints_;

  void UpdateXDelta(VectorXd x_curr, VectorXd x_nom);

  LCS MakeLCS(VectorXd x_curr, VectorXd u_curr);

  void UpdateQP(VectorXd x_curr, LCS lcs, vector<VectorXd> x_nom, vector<VectorXd> u_nom, vector<VectorXd> lambda_nom);

  void UpdateQuaternionCosts(VectorXd x_curr, VectorXd x_des);

  VectorXd ConstructLambdasFromContactResults(
    drake::multibody::ContactResults<double> contact_results, std::string contact_model);

};

} // namespace systems
} // namespace c3