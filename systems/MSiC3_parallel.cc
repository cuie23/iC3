#include "MSiC3_parallel.h"

#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "core/c3_plus.h"
#include "core/solver_options_io.h"
#include "multibody/lcs_factory.h"
#include "multibody/geom_geom_collider.h"
#include "common/quaternion_error_hessian.h"
#include "systems/hybrid_mpc.h"

#include "drake/common/text_logging.h"
#include "drake/solvers/osqp_solver.h"
#include <drake/multibody/parsing/parser.h>
#include <omp.h>

using drake::multibody::ModelInstanceIndex;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::multibody::MultibodyPlant;
using drake::systems::DiscreteValues;
using drake::systems::DiagramBuilder;
using drake::math::RigidTransform;
using drake::multibody::Parser;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::VectorXf;
using Eigen::Quaterniond;
using Eigen::Matrix3d;
using drake::math::RotationMatrix;
using drake::math::RollPitchYaw;
using drake::multibody::ContactResults;
using drake::geometry::GeometryId;
using drake::SortedPair;
using Eigen::LLT;

namespace c3 {
namespace systems {

MSiC3Parallel::MSiC3Parallel(const MultibodyPlant<double>& plant, const MultibodyPlant<drake::AutoDiffXd>& plant_ad, 
  const MultibodyPlant<double>& plant_rollout, const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout, 
  drake::systems::Diagram<double>& rollout_diagram, std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,    
  const vector<vector<SortedPair<GeometryId>>>& contact_groups, const vector<vector<SortedPair<GeometryId>>>& contact_groups_rollout,
  C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, HybridMpcOptions mpc_options, int example_idx, bool is_optuna)
    : plant_(plant),
      plant_ad_(plant_ad),
      plant_rollout_(plant_rollout),
      plant_ad_rollout_(plant_ad_rollout),
      rollout_diagram_(rollout_diagram),
      rollout_diagram_context_(std::move(rollout_diagram_context)),
      contact_groups_(contact_groups),
      contact_groups_rollout_(contact_groups_rollout),
      use_drake_sim_(ms_ic3_options.use_drake_sim),
      controller_options_(controller_options),
      ms_ic3_options_(ms_ic3_options),
      mpc_options_(mpc_options),
      N_(ms_ic3_options.N),
      example_idx_(example_idx),
      is_optuna_(is_optuna) {

  if (rollout_diagram_context_ == nullptr) {
    std::cout << "NULLPTR SPJOIGNSUIPD" << std::endl;
  }

  for (const auto& group : contact_groups_) {
    contact_geoms_.insert(contact_geoms_.end(), group.begin(), group.end());
  }
  for (const auto& group : contact_groups_rollout_) {
    contact_geoms_rollout_.insert(contact_geoms_rollout_.end(), group.begin(), group.end());
  }

  if (controller_options_.resolve_contacts_to_lists.has_value() ||
      controller_options_.c3_options.resolve_contacts_to_lists.has_value()) {
    const auto& res_lists =
        controller_options_.resolve_contacts_to_lists.has_value()
            ? controller_options_.resolve_contacts_to_lists.value()
            : controller_options_.c3_options.resolve_contacts_to_lists.value();
    int idx = controller_options_.num_contacts_index.value_or(
        controller_options_.c3_options.num_contacts_index.value_or(0));
    const auto& res_list = res_lists[idx];
    int total_resolved = 0;
    for (size_t g = 0; g < contact_groups_.size(); ++g) {
      int limit = (g < res_list.size()) ? res_list[g] : static_cast<int>(contact_groups_[g].size());
      total_resolved += std::min(limit, static_cast<int>(contact_groups_[g].size()));
    }
    controller_options_.lcs_factory_options.num_contacts = total_resolved;

    if (controller_options_.mu_per_pair_type.has_value() ||
        controller_options_.c3_options.mu_per_pair_type.has_value()) {
      const auto& mu_types =
          controller_options_.mu_per_pair_type.has_value()
              ? controller_options_.mu_per_pair_type.value()
              : controller_options_.c3_options.mu_per_pair_type.value();
      std::vector<double> new_mu;
      for (size_t g = 0; g < contact_groups_.size(); ++g) {
        int n_active = (g < res_list.size())
                           ? std::min(res_list[g], static_cast<int>(contact_groups_[g].size()))
                           : contact_groups_[g].size();
        double mu_val = (g < mu_types.size()) ? mu_types[g] : (mu_types.empty() ? 0.3 : mu_types[0]);
        new_mu.insert(new_mu.end(), n_active, mu_val);
      }
      controller_options_.lcs_factory_options.mu = new_mu;
    }
  } else if (controller_options_.mu_per_pair_type.has_value() ||
             controller_options_.c3_options.mu_per_pair_type.has_value()) {
    const auto& mu_types =
        controller_options_.mu_per_pair_type.has_value()
            ? controller_options_.mu_per_pair_type.value()
            : controller_options_.c3_options.mu_per_pair_type.value();
    std::vector<double> new_mu;
    for (size_t g = 0; g < contact_groups_.size(); ++g) {
      double mu_val = (g < mu_types.size()) ? mu_types[g] : (mu_types.empty() ? 0.3 : mu_types[0]);
      new_mu.insert(new_mu.end(), contact_groups_[g].size(), mu_val);
    }
    controller_options_.lcs_factory_options.mu = new_mu;
  }

  // Initialize dimensions
  n_q_ = plant_.num_positions();
  n_v_ = plant_.num_velocities();
  n_u_ = plant_.num_actuators();
  n_x_ = n_q_ + n_v_;
  dt_ = controller_options_.lcs_factory_options.dt;

  // Determine the size of lambda based on the contact model
  n_lambda_ = multibody::LCSFactory::GetNumContactVariables(
      controller_options_.lcs_factory_options);

  std::cout << "n x " << n_x_ << std::endl;
  std::cout << "n u " << n_u_ << std::endl;
  std::cout << "n lambda: " << n_lambda_ << std::endl;

  num_segments_ = ms_ic3_options_.num_segments;
  L_ = N_ / num_segments_;

  // Initialize cost matrices
  for (int i = 0; i < N_+1; i++) {
    Q_.push_back(controller_options_.c3_options.Q);
    if (i < N_) {
      R_.push_back(controller_options_.c3_options.R);
      G_.push_back(controller_options_.c3_options.G);
      U_.push_back(controller_options_.c3_options.U);
    }
  }

  simulators_.resize(num_segments_);
  if (use_drake_sim_) {
    for (int i = 0; i < num_segments_; i++) {
      auto context_clone = rollout_diagram_.CreateDefaultContext();   
      simulators_[i] = std::make_unique<drake::systems::Simulator<double>>(
          rollout_diagram_, std::move(context_clone));
    }
  }
}


MSiC3Parallel::MSiC3Parallel(const MultibodyPlant<double>& plant, const MultibodyPlant<drake::AutoDiffXd>& plant_ad, 
  const MultibodyPlant<double>& plant_rollout, const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout, 
  drake::systems::Diagram<double>& rollout_diagram, std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,    
  const vector<SortedPair<GeometryId>>& contact_geoms, const vector<SortedPair<GeometryId>>& contact_geoms_rollout,
  C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, HybridMpcOptions mpc_options, int example_idx)
    : MSiC3Parallel(plant, plant_ad, plant_rollout, plant_ad_rollout, rollout_diagram,
                    std::move(rollout_diagram_context),
                    vector<vector<SortedPair<GeometryId>>>{contact_geoms},
                    vector<vector<SortedPair<GeometryId>>>{contact_geoms_rollout},
                    controller_options, ms_ic3_options, mpc_options, example_idx) {}

MSiC3Parallel::MSiC3Parallel(const MultibodyPlant<double>& plant, const MultibodyPlant<drake::AutoDiffXd>& plant_ad, 
  const MultibodyPlant<double>& plant_rollout, const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout, 
  drake::systems::Diagram<double>& rollout_diagram, std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,    
  const vector<vector<SortedPair<GeometryId>>>& contact_groups, const vector<vector<SortedPair<GeometryId>>>& contact_groups_rollout,
  C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, int example_idx, bool is_optuna)
    : MSiC3Parallel(plant, plant_ad, plant_rollout, plant_ad_rollout, rollout_diagram,
                    std::move(rollout_diagram_context),
                    contact_groups,
                    contact_groups_rollout,
                    controller_options, ms_ic3_options, HybridMpcOptions{}, example_idx, is_optuna) {}

void MSiC3Parallel::ResolveContacts(
    const drake::systems::Context<double>& context,
    const drake::systems::Context<double>& context_rollout) {
  if (controller_options_.resolve_contacts_to_lists.has_value() ||
      controller_options_.c3_options.resolve_contacts_to_lists.has_value()) {
    const auto& res_lists =
        controller_options_.resolve_contacts_to_lists.has_value()
            ? controller_options_.resolve_contacts_to_lists.value()
            : controller_options_.c3_options.resolve_contacts_to_lists.value();
    int idx = controller_options_.num_contacts_index.value_or(
        controller_options_.c3_options.num_contacts_index.value_or(0));
    const auto& res_list = res_lists[idx];

    contact_geoms_ = multibody::LCSFactory::ResolveContactPairs(
        plant_, context, contact_groups_, res_list);
    contact_geoms_rollout_ = multibody::LCSFactory::ResolveContactPairs(
        plant_rollout_, context_rollout, contact_groups_rollout_, res_list);

    controller_options_.lcs_factory_options.num_contacts = contact_geoms_.size();
    n_lambda_ = multibody::LCSFactory::GetNumContactVariables(
        controller_options_.lcs_factory_options);

    if (controller_options_.mu_per_pair_type.has_value() ||
        controller_options_.c3_options.mu_per_pair_type.has_value()) {
      const auto& mu_types =
          controller_options_.mu_per_pair_type.has_value()
              ? controller_options_.mu_per_pair_type.value()
              : controller_options_.c3_options.mu_per_pair_type.value();
      std::vector<double> new_mu;
      for (size_t g = 0; g < contact_groups_.size(); ++g) {
        int n_active =
            (g < res_list.size())
                ? std::min(res_list[g],
                           static_cast<int>(contact_groups_[g].size()))
                : contact_groups_[g].size();
        double mu_val = (g < mu_types.size()) ? mu_types[g] : (mu_types.empty() ? 0.3 : mu_types[0]);
        new_mu.insert(new_mu.end(), n_active, mu_val);
      }
      controller_options_.lcs_factory_options.mu = new_mu;
    } else {
      if (controller_options_.lcs_factory_options.mu.empty()) {
        controller_options_.lcs_factory_options.mu.resize(contact_geoms_.size(), 0.3);
      } else {
        controller_options_.lcs_factory_options.mu.resize(
            contact_geoms_.size(),
            controller_options_.lcs_factory_options.mu[0]);
      }
    }
  } else {
    contact_geoms_.clear();
    for (const auto& group : contact_groups_) {
      contact_geoms_.insert(contact_geoms_.end(), group.begin(), group.end());
    }
    contact_geoms_rollout_.clear();
    for (const auto& group : contact_groups_rollout_) {
      contact_geoms_rollout_.insert(contact_geoms_rollout_.end(), group.begin(), group.end());
    }
    controller_options_.lcs_factory_options.num_contacts = contact_geoms_.size();
    n_lambda_ = multibody::LCSFactory::GetNumContactVariables(
        controller_options_.lcs_factory_options);
    if (controller_options_.mu_per_pair_type.has_value() ||
        controller_options_.c3_options.mu_per_pair_type.has_value()) {
      const auto& mu_types =
          controller_options_.mu_per_pair_type.has_value()
              ? controller_options_.mu_per_pair_type.value()
              : controller_options_.c3_options.mu_per_pair_type.value();
      std::vector<double> new_mu;
      for (size_t g = 0; g < contact_groups_.size(); ++g) {
        double mu_val = (g < mu_types.size()) ? mu_types[g] : (mu_types.empty() ? 0.3 : mu_types[0]);
        new_mu.insert(new_mu.end(), contact_groups_[g].size(), mu_val);
      }
      controller_options_.lcs_factory_options.mu = new_mu;
    } else if (controller_options_.lcs_factory_options.mu.empty()) {
      controller_options_.lcs_factory_options.mu.resize(contact_geoms_.size(), 0.3);
    }
  }
}

tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>, vector<vector<MatrixXd>>, 
  vector<vector<VectorXd>>, vector<vector<MatrixXd>>, vector<vector<VectorXd>>, 
  vector<vector<vector<MatrixXd>>>, vector<vector<vector<VectorXd>>>, 
  vector<MatrixXd>, vector<MatrixXd>> MSiC3Parallel::ComputeTrajectory(
  drake::systems::Context<double>& context,
  drake::systems::Context<drake::AutoDiffXd>& context_ad, 
  std::vector<drake::systems::Context<double>*> contexts_rollout,
  drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout,
  double plate_u_torque_bound) {

  auto start_total = std::chrono::high_resolution_clock::now();

  struct DefectRecord {
    int iter;
    int segment;
    double ee_defect_norm;
    double quat_defect_deg;
    double object_defect_norm;
  };
  std::vector<DefectRecord> defect_records;
  
  // num segements must divide the entire time horizon
  DRAKE_DEMAND((double)(N_ / num_segments_) == (double)N_ / num_segments_);

  std::vector<double> x_init = *controller_options_.x_init;
  VectorXd x0 = Eigen::Map<VectorXd>(x_init.data(), x_init.size());    

  std::vector<double> x_des = *controller_options_.x_des;
  VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());  

  // Add linear constraints
  MatrixXd A_x(MatrixXd::Zero(n_x_, n_x_));
  MatrixXd A_u(MatrixXd::Zero(n_u_, n_u_));
  VectorXd lower_bound_x(VectorXd::Zero(n_x_));
  VectorXd upper_bound_x(VectorXd::Zero(n_x_));
  VectorXd lower_bound_u(VectorXd::Zero(n_u_));
  VectorXd upper_bound_u(VectorXd::Zero(n_u_));

  MatrixXd A_u_warmup(MatrixXd::Zero(n_u_, n_u_));
  VectorXd lower_bound_u_warmup(VectorXd::Zero(n_u_));
  VectorXd upper_bound_u_warmup(VectorXd::Zero(n_u_));

  // HARDCODED
  if (example_idx_ == 0) { // plate
    A_x(0, 0) = 1;
    A_x(1, 1) = 1;
    A_x(2, 2) = 1;
    A_x(3, 3) = 1;
    A_x(4, 4) = 1;

    lower_bound_x(0) = -0.08;
    lower_bound_x(1) = -0.08;
    lower_bound_x(2) = x0(2) - 0.1; 
    lower_bound_x(3) = -0.8;
    lower_bound_x(4) = -0.8;

    upper_bound_x(0) = 0.08;
    upper_bound_x(1) = 0.08;
    upper_bound_x(2) = x0(2) + 0.1;
    upper_bound_x(3) = 0.8;
    upper_bound_x(4) = 0.8;

    // Actuation limits
    A_u(0, 0) = 1;
    A_u(1, 1) = 1;
    A_u(2, 2) = 1;
    A_u(3, 3) = 1;
    A_u(4, 4) = 1;

    lower_bound_u(0) = -1;
    lower_bound_u(1) = -1;
    lower_bound_u(2) = 9.81 * 0.85 - 3.2;
    lower_bound_u(3) = (plate_u_torque_bound == -1) ? -2.4 : -plate_u_torque_bound;
    lower_bound_u(4) = (plate_u_torque_bound == -1) ? -2.4 : -plate_u_torque_bound;

    upper_bound_u(0) = 1;
    upper_bound_u(1) = 1;
    upper_bound_u(2) = 9.81 * 0.85 + 3.2;
    upper_bound_u(3) = (plate_u_torque_bound == -1) ? 2.4 : plate_u_torque_bound;
    upper_bound_u(4) = (plate_u_torque_bound == -1) ? 2.4 : plate_u_torque_bound;

    A_u_warmup(0, 0) = 1;
    A_u_warmup(1, 1) = 1;
    A_u_warmup(2, 2) = 1;
    A_u_warmup(3, 3) = 1;
    A_u_warmup(4, 4) = 1;

    lower_bound_u_warmup(0) = -1;
    lower_bound_u_warmup(1) = -1;
    lower_bound_u_warmup(2) = 9.81 * 0.85 - 5;
    lower_bound_u_warmup(3) = -2;
    lower_bound_u_warmup(4) = -2;

    upper_bound_u_warmup(0) = 1;
    upper_bound_u_warmup(1) = 1;
    upper_bound_u_warmup(2) = 9.81 * 0.85 + 5;
    upper_bound_u_warmup(3) = 2;
    upper_bound_u_warmup(4) = 2;

  } else if (example_idx_ == 1) { // trifinger 180

    for (int i = 0; i < 3; i++) {
      // Position constraints
      A_x(3*i, 3*i) = 1;
      A_x(3*i+1, 3*i+1) = 1;
      A_x(3*i+2, 3*i+2) = 1;

      // Velocity constraints
      A_x(16 + 3*i, 16 + 3*i) = 1;
      A_x(16 + 3*i + 1, 16 + 3*i + 1) = 1;
      A_x(16 + 3*i + 2, 16 + 3*i + 2) = 1;

      // Offset from initial position
      lower_bound_x(3*i) = xd(3*i) - 0.06;
      lower_bound_x(3*i+1) = xd(3*i+1) - 0.06;
      lower_bound_x(3*i+2) = xd(3*i+2) - 0.01;

      lower_bound_x(16 + 3*i) = -0.1;
      lower_bound_x(16 + 3*i+1) = -0.1;
      lower_bound_x(16 + 3*i+2) = -0.05;

      upper_bound_x(3*i) = xd(3*i) + 0.06;
      upper_bound_x(3*i+1) = xd(3*i+1) + 0.06;
      upper_bound_x(3*i+2) = xd(3*i+2) + 0.01;

      upper_bound_x(16 + 3*i) = 0.1;
      upper_bound_x(16 + 3*i+1) = 0.1;
      upper_bound_x(16 + 3*i+2) = 0.05;

      A_u(3*i, 3*i) = 1;
      A_u(3*i+1, 3*i+1) = 1;
      A_u(3*i+2, 3*i+2) = 1;

      lower_bound_u(3*i) = -0.5;
      lower_bound_u(3*i+1) = -0.5;
      lower_bound_u(3*i+2) = 0.15;
      
      upper_bound_u(3*i) = 0.5;
      upper_bound_u(3*i+1) = 0.5;
      upper_bound_u(3*i+2) = 0.25;
    }

    A_u_warmup = A_u;
    lower_bound_u_warmup = lower_bound_u;
    upper_bound_u_warmup = upper_bound_u;

  } else if (example_idx_ == 2) { // trifinger pivot

    for (int i = 0; i < 3; i++) {
      // Position constraints
      A_x(3*i, 3*i) = 1;
      A_x(3*i+1, 3*i+1) = 1;
      A_x(3*i+2, 3*i+2) = 1;

      // Velocity constraints
      A_x(16 + 3*i, 16 + 3*i) = 1;
      A_x(16 + 3*i + 1, 16 + 3*i + 1) = 1;
      A_x(16 + 3*i + 2, 16 + 3*i + 2) = 1;

      double xy_bound = (i == 0) ? 0.07 : 0.05;
      double z_bound = (i == 0) ? 0.07 : 0.05;

      // Offset from initial position
      lower_bound_x(3*i) = xd(3*i) - xy_bound;
      lower_bound_x(3*i+1) = xd(3*i+1) - xy_bound;
      lower_bound_x(3*i+2) = xd(3*i+2) - 0.01;

      lower_bound_x(16 + 3*i) = -0.1;
      lower_bound_x(16 + 3*i+1) = -0.1;
      lower_bound_x(16 + 3*i+2) = -0.1;

      upper_bound_x(3*i) = xd(3*i) + xy_bound;
      upper_bound_x(3*i+1) = xd(3*i+1) + xy_bound;
      upper_bound_x(3*i+2) = xd(3*i+2) + z_bound;

      upper_bound_x(16 + 3*i) = 0.1;
      upper_bound_x(16 + 3*i+1) = 0.1;
      upper_bound_x(16 + 3*i+2) = 0.1;

      A_u(3*i, 3*i) = 1;
      A_u(3*i+1, 3*i+1) = 1;
      A_u(3*i+2, 3*i+2) = 1;

      double u_bound_xy = (i == 0) ? 2 : 2;
      double u_bound_z = (i == 0) ? 1.5 : 1;

      lower_bound_u(3*i) = -u_bound_xy;
      lower_bound_u(3*i+1) = -u_bound_xy;
      lower_bound_u(3*i+2) = -1;
      
      upper_bound_u(3*i) = u_bound_xy;
      upper_bound_u(3*i+1) = u_bound_xy;
      upper_bound_u(3*i+2) = u_bound_z;
    }
    A_u_warmup = A_u;
    lower_bound_u_warmup = lower_bound_u;
    upper_bound_u_warmup = upper_bound_u;
  }

  vector<VectorXd> x_targets;
  for (int k = 0; k < N_+1; k++) {
    x_targets.push_back(xd);
  }

  // ith column = ith timestep
  MatrixXd x_hat = x0.replicate(1, N_+1);
  MatrixXd u_hat(Eigen::MatrixXd::Zero(n_u_, N_));
  MatrixXd lambda_hat = MatrixXd::Zero(n_lambda_, N_);
  MatrixXd x_anchors = MatrixXd::Zero(n_x_, num_segments_+1);
  MatrixXd defects = MatrixXd::Zero(n_x_, num_segments_+1);
  MatrixXd gamma = MatrixXd::Zero(n_lambda_ / 4, N_);
  MatrixXd in_contact = MatrixXd::Zero(n_lambda_ / 4, N_);

  const VectorXd tau_g = plant_rollout_.CalcGravityGeneralizedForces(*contexts_rollout[0]);
  VectorXd gravity = -1 * plant_rollout_.MakeActuationMatrix().transpose() * tau_g;

  // Object must be supported by plate, so need additional torque
  if (example_idx_ == 0) {
    gravity[4] = (x0(9) * tau_g(10));
  }
  std::cout << "gravity " << gravity.transpose() << std::endl;

  ResolveContacts(context, *contexts_rollout[0]);

  LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
      contact_geoms_, controller_options_.lcs_factory_options);

  LCSFactory lcs_factory_rollout(plant_rollout_, *contexts_rollout[0], plant_ad_rollout_,
      context_ad_rollout, contact_geoms_rollout_, controller_options_.lcs_factory_options);

  std::cout << "plant lcs dt " << plant_.time_step() << std::endl;
  std::cout << "plant rollout dt " << plant_rollout_.time_step() << std::endl;

  int tracking_N = controller_options_.lcs_factory_options.N;
  vector<VectorXd> x_targets_temp(tracking_N + 1, xd);
  MatrixXd x_hat_temp = x0.replicate(1, tracking_N);
  MatrixXd u_hat_temp = MatrixXd::Zero(n_u_, tracking_N);
  vector<MatrixXd> Q_temp(Q_.begin(), Q_.begin() + tracking_N + 1);
  vector<MatrixXd> R_temp(R_.begin(), R_.begin() + tracking_N);
  vector<MatrixXd> G_temp(G_.begin(), G_.begin() + tracking_N);
  vector<MatrixXd> U_temp(U_.begin(), U_.begin() + tracking_N);
  
  C3::CostMatrices costs_temp(Q_temp, R_temp, G_temp, U_temp);
  LCS lcs_temp = MakeTimeVaryingLCS(
      x_hat_temp, u_hat_temp, MatrixXd::Zero(n_lambda_, tracking_N), lcs_factory);

  c3_trackings_.resize(num_segments_);
  for (int s = 0; s < num_segments_; s++) {
    c3_trackings_[s] = std::make_unique<C3Plus>(
        lcs_temp, costs_temp, x_targets_temp, controller_options_.c3_options);

    if (solver_options_.has_value()) {
      c3_trackings_[s]->SetSolverOptions(*solver_options_);
    }

    if (ms_ic3_options_.penalize_acceleration) {
      c3_trackings_[s]->AddAccelerationCost(n_q_, n_v_, ms_ic3_options_.acceleration_cost_weight);
    }

    if (ms_ic3_options_.add_terminal_constraint) {
      c3_trackings_[s]->AddTerminalConstraint();
    }

    if (example_idx_ == 1) {
      std::vector<int> v(12);
      std::iota(v.begin(), v.end(), 0);
      c3_trackings_[s]->AddLambdaBound(2 * 0.5, v);
    } else if (example_idx_ == 2) {
      std::vector<int> v(12);
      std::iota(v.begin(), v.end(), 0);
      c3_trackings_[s]->AddLambdaBound(2 * 1.5, v);
    }
  }

  // Set initial guess to something kinda reasonable
  // Set initial guess for x - linear interpolation (including in quaternion space)
  // But project contacts so that each x is feasible (i.e. no penetration)
  VectorXd x_diff = xd - x0;
  for (int k = 0; k < N_+1; k++) {
    x_hat.col(k) = x0 + k * x_diff / (N_);
    
    // Linearly interpolate quaternions correctly
    for (auto idx : controller_options_.quaternion_indices) {
      double rotation = (double)k / (N_);

      Eigen::Quaterniond q0(x0(idx), x0(idx+1), x0(idx+2), x0(idx+3));
      Eigen::Quaterniond qd(xd(idx), xd(idx+1), xd(idx+2), xd(idx+3));
      VectorXd v0 = x0.segment(idx, 4);
      VectorXd vd = xd.segment(idx, 4);

      // Ensure quaternions are in the same hemisphere 
      if (v0.dot(vd) < 0) {
          vd = -vd;
          qd = Eigen::Quaterniond(vd(0), vd(1), vd(2), vd(3));
      }

      if (-1e-3 < q0.dot(qd) && q0.dot(qd) < 1e-3) { 
        // Fallback for antipodal points, use linear interpolation in R3 to get default axis
        Eigen::Vector4d mid = v0 + vd;
        Eigen::Vector4d tangent = (mid - mid.dot(v0) * v0).normalized();

        double theta = std::acos(std::clamp(v0.dot(vd), -1.0, 1.0)); 
        Eigen::Vector4d v_interpolated = v0 * std::cos(rotation * theta) + tangent * std::sin(rotation * theta);

        x_hat.col(k).segment(idx, 4) = v_interpolated;

      } else {
        if (example_idx_ == 0) {
          Eigen::Quaterniond slerp = slerpLong(q0, qd, rotation); 
          x_hat.col(k).segment(idx, 4) << slerp.w(), slerp.x(), slerp.y(), slerp.z(); 
        } else if (example_idx_ == 1 || example_idx_ == 2) {
          Eigen::Quaterniond slerp = q0.slerp(rotation, qd);
          x_hat.col(k).segment(idx, 4) << slerp.w(), slerp.x(), slerp.y(), slerp.z();
        }
      }
    }

    VectorXd x_projected = x_hat.col(k);
    if (example_idx_ == 0) {  
      drake::geometry::GeometryId plate_collision_geom =
        plant_.GetCollisionGeometriesForBody(
            plant_.GetBodyByName("plate"))[0];
      drake::geometry::GeometryId cube_collision_geom =
        plant_.GetCollisionGeometriesForBody(
            plant_.GetBodyByName("cube"))[0];
      SortedPair<GeometryId>cube_plate_contact(plate_collision_geom, cube_collision_geom);

      x_projected = ProjectContactPlate(context, cube_plate_contact, x_projected, 2, 11,
                A_x, lower_bound_x, upper_bound_x);

    } else if (example_idx_ == 1) {

      // Ensure anchors don't have penetration
      for (int j = 0; j < 3; j++) {
        x_projected = ProjectContact(context, contact_geoms_[j], x_projected, 3*j, 3, 
                                      A_x, lower_bound_x, upper_bound_x);
      }

    } else if (example_idx_ == 2) {
      // Ensure cube not penetrating ground
      drake::geometry::GeometryId cube_collision_geom =
        plant_.GetCollisionGeometriesForBody(
            plant_.GetBodyByName("cube"))[0];
      drake::geometry::GeometryId ground_collision_geom =
        plant_.GetCollisionGeometriesForBody(
            plant_.GetBodyByName("ground"))[0];
      SortedPair<GeometryId>cube_ground_contact(cube_collision_geom, ground_collision_geom);

      x_projected = ProjectContactVertical(context, cube_ground_contact, x_projected, 15);

      // Ensure fingers don't have penetration
      for (int j = 0; j < 3; j++) {
        x_projected = ProjectContact(context, contact_geoms_[j], x_projected, 3*j, 3, 
                                      A_x, lower_bound_x, upper_bound_x);
      }
    }
    x_hat.col(k) = x_projected;

    if (k < N_) {
      u_hat.col(k) = gravity;
    }
  }

  // Get lambda_hat initial guess and defects
  MatrixXd x_hat_init(MatrixXd::Zero(n_x_, N_+1));
  if (use_drake_sim_) {
    Context<double>& root_context = simulators_[0]->get_mutable_context();
    Context<double>& sim_plant_context = rollout_diagram_.GetMutableSubsystemContext(
        plant_rollout_, &root_context);
    VectorXd x_curr = x0;
    for (int i = 0; i < N_; i++) {
      x_hat_init.col(i) = x_curr;
      plant_rollout_.SetPositionsAndVelocities(&sim_plant_context, x_curr);

      plant_rollout_.get_actuation_input_port().FixValue(&sim_plant_context, u_hat.col(i));

      double target_time = root_context.get_time() + dt_;
      simulators_[0]->AdvanceTo(target_time);

      auto abstract_contact_results = drake::AbstractValue::Make<drake::multibody::ContactResults<double>>({});
      plant_rollout_.get_contact_results_output_port().Calc(sim_plant_context, abstract_contact_results.get());
      const auto& contact_results = abstract_contact_results->get_value<drake::multibody::ContactResults<double>>();
        
      x_curr = plant_rollout_.GetPositionsAndVelocities(sim_plant_context);
      if (ms_ic3_options_.use_rollout_lambdas) {
        auto [lambda, gamma_out, in_contact_out] = 
          ConstructLambdasFromContactResults(contact_results, controller_options_.lcs_factory_options.contact_model);
        lambda_hat.col(i) = lambda;
      }
    }  
    x_hat_init.col(N_) = x_curr;
  } else {
    std::vector<MatrixXd> K_init(N_, Eigen::MatrixXd::Identity(n_u_, n_u_));
    std::vector<VectorXd> K_ff_init(N_, VectorXd::Zero(n_u_));
    auto [lcs_init_out, x_hat_init_out, u_hat_init_out, lambda_hat_init_out] = DoLCSRollout(x0, u_hat, 
      lcs_factory, lcs_factory_rollout, MatrixXd::Zero(n_x_, n_x_), VectorXd::Zero(n_x_), VectorXd::Zero(n_x_), 
      MatrixXd::Zero(n_u_, n_u_), VectorXd::Zero(n_u_), VectorXd::Zero(n_u_), 
      K_init, K_ff_init, 0);
    x_hat_init = x_hat_init_out;

    if (ms_ic3_options_.use_rollout_lambdas) {
      lambda_hat = lambda_hat_init_out;
    }
  }

  for (int i = 0; i < num_segments_+1; i++) {
    VectorXd x_projected = x_hat.col(i * L_);

    if (example_idx_ == 0) {  
      drake::geometry::GeometryId plate_collision_geom =
        plant_.GetCollisionGeometriesForBody(
            plant_.GetBodyByName("plate"))[0];
      drake::geometry::GeometryId cube_collision_geom =
        plant_.GetCollisionGeometriesForBody(
            plant_.GetBodyByName("cube"))[0];
      SortedPair<GeometryId>cube_plate_contact(plate_collision_geom, cube_collision_geom);

      x_projected = ProjectContactPlate(context, cube_plate_contact, x_projected, 2, 11,
          A_x, lower_bound_x, upper_bound_x);
    } else if (example_idx_ == 1) {

      // Ensure anchors don't have penetration
      for (int j = 0; j < 3; j++) {
        x_projected = ProjectContact(context, contact_geoms_[j], x_projected, 3*j, 3, 
                                      A_x, lower_bound_x, upper_bound_x);
      }

    } else if (example_idx_ == 2) {
      // Ensure cube not penetrating ground
      drake::geometry::GeometryId cube_collision_geom =
        plant_.GetCollisionGeometriesForBody(
            plant_.GetBodyByName("cube"))[0];
      drake::geometry::GeometryId ground_collision_geom =
        plant_.GetCollisionGeometriesForBody(
            plant_.GetBodyByName("ground"))[0];
      SortedPair<GeometryId>cube_ground_contact(cube_collision_geom, ground_collision_geom);

      x_projected = ProjectContactVertical(context, cube_ground_contact, x_projected, 15);

      // Ensure fingers don't have penetration
      for (int j = 0; j < 3; j++) {
        x_projected = ProjectContact(context, contact_geoms_[j], x_projected, 3*j, 3, 
                                      A_x, lower_bound_x, upper_bound_x);
      }

    }
    x_anchors.col(i) = x_projected;
    defects.col(i) = x_hat_init.col(i * L_) - x_anchors.col(i);
  }

  vector<VectorXd> u_nominal(N_, gravity);

  vector<MatrixXd> all_x_hats;
  vector<MatrixXd> all_u_hats;
  vector<MatrixXd> all_lambda_hats;
  vector<MatrixXd> all_defects;
  vector<MatrixXd> all_x_anchors;
  vector<MatrixXd> all_gammas;
  vector<MatrixXd> all_in_contacts;

  vector<vector<MatrixXd>> Hs;
  vector<vector<VectorXd>> gs;
  vector<vector<MatrixXd>> Ks;
  vector<vector<VectorXd>> k_ffs;

  // ic3 iteration, ic3 timestep, c3 horizon
  vector<vector<vector<MatrixXd>>> all_delta_projections;
  vector<vector<vector<VectorXd>>> all_z_sols;

  all_x_hats.push_back(x_hat);
  all_u_hats.push_back(u_hat);
  all_lambda_hats.push_back(lambda_hat);
  all_defects.push_back(defects);
  all_x_anchors.push_back(x_anchors);

  LCS lcs = MakeTimeVaryingLCS(x_hat, u_hat, lambda_hat, lcs_factory);

  struct LambdaResidualRecord {
    int outer_iter;
    int segment;
    int plan_timestep;
    int admm_iter;
    double lambda_diff_norm;
    double iterate_step_change;
    double complementarity_slack;
  };
  std::vector<LambdaResidualRecord> lambda_residual_records;

  struct C3PlanStepRecord {
    int outer_iter;
    int segment;
    int plan_timestep;
    VectorXd x0;
    VectorXd x1;
    VectorXd lambda;
    VectorXd eta;
    VectorXd lambda_last;
    VectorXd eta_last;
  };
  std::vector<C3PlanStepRecord> c3_plan_step_records;

  struct C3FullLookaheadRecord {
    int outer_iter;
    int segment;
    int plan_timestep;
    int lookahead_step;
    VectorXd x;
  };
  std::vector<C3FullLookaheadRecord> c3_full_lookahead_records;

  int num_iters = ms_ic3_options_.num_iters;
  int num_warmup_iters = ms_ic3_options_.num_warmup_iters;

  for (int iter = 1 - num_warmup_iters; iter <= num_iters; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "iC3 iteration " << iter << std::endl;

    delta_projection_iter_.clear();
    z_sol_iter_.clear();

    bool is_warmup = (iter < 1);

    for (int s = 0; s < num_segments_; s++) {
      c3_trackings_[s]->RemoveConstraints();
      if (ms_ic3_options_.add_position_constraints) {
        c3_trackings_[s]->AddLinearConstraint(A_x, lower_bound_x, upper_bound_x,
                                  ConstraintVariable::STATE);
      }  
      if (is_warmup) {
        if (ms_ic3_options_.add_input_constraints) {
          c3_trackings_[s]->AddLinearConstraint(A_u_warmup, lower_bound_u_warmup, upper_bound_u_warmup,
                                    ConstraintVariable::INPUT);
        }
      } else {
        if (ms_ic3_options_.add_input_constraints) {
          c3_trackings_[s]->AddLinearConstraint(A_u, lower_bound_u, upper_bound_u,
                                    ConstraintVariable::INPUT);
        }
      }
    }

    UpdateQuaternionCosts(x_hat, xd); // Note: this overrides R, G, U as well

    // Backwards Pass - Compute Value Function
    auto [H_out, g, K, k_ff] = ComputeBoxDDPValueFunction(x_hat, u_hat, lambda_hat,
                              lcs, xd, u_nominal[0], defects, lower_bound_u, upper_bound_u);
    vector<MatrixXd> H = GetLowRankApproximation(H_out);

    // Don't store warmup iters
    if (!is_warmup) {
      Hs.push_back(H);
      gs.push_back(g);
      Ks.push_back(K);
      k_ffs.push_back(k_ff);
    }

    MatrixXd new_x_anchors(MatrixXd::Zero(n_x_, num_segments_+1));
    new_x_anchors.col(0) = x0;

    // Update anchor for next segment
    double alpha_ee = std::min(1.0, ms_ic3_options_.alpha_ee + (iter-1) * ms_ic3_options_.alpha_ee_step);
    double alpha_object = std::min(1.0, ms_ic3_options_.alpha_object + (iter-1) * ms_ic3_options_.alpha_object_step);
    if (is_warmup) {
      alpha_ee = ms_ic3_options_.warm_start_alpha; // Don't update alpha if warmup
      alpha_object = 0; // Don't update object anchor if warmup
    } 

    std::cout << "alpha ee " << alpha_ee << " alpha object " << alpha_object << std::endl;
    std::cout << "num segments " << num_segments_ << std::endl;
    std::cout << "L " << L_ << std::endl;

    std::vector<MatrixXd> seg_x_hat_out(num_segments_);
    std::vector<MatrixXd> seg_u_hat_out(num_segments_);
    std::vector<MatrixXd> seg_lambda_hat_out(num_segments_);
    std::vector<MatrixXd> seg_gamma_out(num_segments_);
    std::vector<MatrixXd> seg_in_contact_out(num_segments_);
    std::vector<vector<vector<MatrixXd>>> seg_delta_proj(num_segments_);
    std::vector<vector<vector<VectorXd>>> seg_z_sol(num_segments_);

    // Forwards Pass - Do C3 MPC with value function terminal cost across all segments in parallel
    #pragma omp parallel for num_threads(ms_ic3_options_.num_threads.value_or(1))
    for (int i = 0; i < num_segments_; i++) {
      auto [x_hat_out, u_hat_out, lambda_hat_out, gamma_out, in_contact_out, delta_proj_out, z_sol_out] = 
         DoC3Rollout(x_anchors.col(i), x_hat, u_hat.middleCols(i*L_, L_), 
                      lambda_hat, gravity, x_anchors.col(i+1),
                      lcs_factory, lcs_factory_rollout, H, g, i*L_, i,
                      A_x, lower_bound_x, upper_bound_x, A_u, lower_bound_u, upper_bound_u,
                      context, *contexts_rollout[i], *c3_trackings_[i]);

      seg_x_hat_out[i] = x_hat_out;
      seg_u_hat_out[i] = u_hat_out;
      seg_lambda_hat_out[i] = lambda_hat_out;
      seg_gamma_out[i] = gamma_out;
      seg_in_contact_out[i] = in_contact_out;
      seg_delta_proj[i] = delta_proj_out;
      seg_z_sol[i] = z_sol_out;
    }

    // Assemble results and update anchors
    for (int i = 0; i < num_segments_; i++) {
      gamma.block(0, i*L_, n_lambda_ / 4, L_) = seg_gamma_out[i];
      in_contact.block(0, i*L_, n_lambda_ / 4, L_) = seg_in_contact_out[i];

      for (size_t t = 0; t < seg_delta_proj[i].size(); t++) {
        delta_projection_iter_.push_back(seg_delta_proj[i][t]);
        z_sol_iter_.push_back(seg_z_sol[i][t]);

        int global_t = i * L_ + t;
        const auto& deltas = seg_delta_proj[i][t];
        const auto& z_vec = seg_z_sol[i][t];
        int N_c3 = z_vec.size();
        for (int k = 0; k < static_cast<int>(deltas.size()); ++k) {
          double sum_diff = 0.0;
          double sum_step = 0.0;
          double sum_comp = 0.0;
          for (int stage = 0; stage < N_c3; ++stage) {
            VectorXd diff_stage = z_vec[stage].segment(n_x_, n_lambda_) - deltas[k].col(stage).segment(n_x_, n_lambda_);
            sum_diff += diff_stage.norm();
            if (k > 0) {
              sum_step += (deltas[k].col(stage) - deltas[k - 1].col(stage)).norm();
            } else {
              sum_step += deltas[0].col(stage).norm();
            }
            VectorXd lam = z_vec[stage].segment(n_x_, n_lambda_);
            VectorXd eta = z_vec[stage].segment(n_x_ + n_u_ + n_lambda_, n_lambda_);
            sum_comp += std::abs(lam.dot(eta));
          }
          double avg_norm = (N_c3 > 0) ? (sum_diff / N_c3) : 0.0;
          double avg_step = (N_c3 > 0) ? (sum_step / N_c3) : 0.0;
          double avg_comp = (N_c3 > 0) ? (sum_comp / N_c3) : 0.0;
          lambda_residual_records.push_back({iter, i, global_t, k, avg_norm, avg_step, avg_comp});
        }

        VectorXd x0 = (z_vec.size() > 0) ? VectorXd(z_vec[0].segment(0, n_x_)) : VectorXd::Zero(n_x_);
        VectorXd x1 = (z_vec.size() > 1) ? VectorXd(z_vec[1].segment(0, n_x_)) : x0;
        VectorXd lam0 = (z_vec.size() > 0 && z_vec[0].size() >= n_x_ + n_lambda_) ? VectorXd(z_vec[0].segment(n_x_, n_lambda_)) : VectorXd::Zero(n_lambda_);
        VectorXd eta0 = (z_vec.size() > 0 && z_vec[0].size() >= n_x_ + 2 * n_lambda_ + n_u_) ? VectorXd(z_vec[0].segment(n_x_ + n_lambda_ + n_u_, n_lambda_)) : VectorXd::Zero(n_lambda_);
        size_t last_k = z_vec.size() > 0 ? z_vec.size() - 1 : 0;
        VectorXd lam_last = (z_vec.size() > 0 && z_vec[last_k].size() >= n_x_ + n_lambda_) ? VectorXd(z_vec[last_k].segment(n_x_, n_lambda_)) : VectorXd::Zero(n_lambda_);
        VectorXd eta_last = (z_vec.size() > 0 && z_vec[last_k].size() >= n_x_ + 2 * n_lambda_ + n_u_) ? VectorXd(z_vec[last_k].segment(n_x_ + n_lambda_ + n_u_, n_lambda_)) : VectorXd::Zero(n_lambda_);
        c3_plan_step_records.push_back({iter, i, global_t, x0, x1, lam0, eta0, lam_last, eta_last});
        for (size_t k = 0; k < z_vec.size(); ++k) {
          c3_full_lookahead_records.push_back({iter, i, global_t, static_cast<int>(k), z_vec[k].segment(0, n_x_)});
        }
      }

      VectorXd x_L = seg_x_hat_out[i].col(L_);
      VectorXd x_anchor_next = x_anchors.col(i+1);

      // Update anchors
      // HARDCODED INDICES
      if (example_idx_ == 0) {
        new_x_anchors.col(i+1).segment(0, 5) = x_L.segment(0, 5) - (1-alpha_ee) * (x_L - x_anchor_next).segment(0, 5);
        new_x_anchors.col(i+1).segment(12, 5) = x_L.segment(12, 5) - (1-alpha_ee) * (x_L - x_anchor_next).segment(12, 5);
        
        // Update object anchors
        new_x_anchors.col(i+1).segment(9, 3) = x_L.segment(9, 3) - (1-alpha_object) * (x_L - x_anchor_next).segment(9, 3);
        new_x_anchors.col(i+1).segment(17, 6) = x_L.segment(17, 6) - (1-alpha_object) * (x_L - x_anchor_next).segment(17, 6);

      } else if (example_idx_ == 1 || example_idx_ == 2) {
        // Update ee anchors
        new_x_anchors.col(i+1).segment(0, 9) = x_L.segment(0, 9) - (1-alpha_ee) * (x_L - x_anchor_next).segment(0, 9);
        new_x_anchors.col(i+1).segment(16, 9) = x_L.segment(16, 9) - (1-alpha_ee) * (x_L - x_anchor_next).segment(16, 9);
        
        // Update object anchors
        new_x_anchors.col(i+1).segment(13, 3) = x_L.segment(13, 3) - (1-alpha_object) * (x_L - x_anchor_next).segment(13, 3);
        new_x_anchors.col(i+1).segment(25, 6) = x_L.segment(25, 6) - (1-alpha_object) * (x_L - x_anchor_next).segment(25, 6);
      }
      // Linearly interpolate quaternions correctly for object
      for (auto idx : controller_options_.quaternion_indices) {
        Eigen::Quaterniond q0(x_anchor_next(idx), x_anchor_next(idx+1), x_anchor_next(idx+2), x_anchor_next(idx+3));
        Eigen::Quaterniond qf(x_L(idx), x_L(idx+1), x_L(idx+2), x_L(idx+3));
        VectorXd v0 = x_anchor_next.segment(idx, 4);
        VectorXd vf = x_L.segment(idx, 4);

        // Ensure quaternions are in the same hemisphere 
        if (v0.dot(vf) < 0) {
            vf = -vf;
            qf = Eigen::Quaterniond(vf(0), vf(1), vf(2), vf(3));
        }

        if (-1e-3 < q0.dot(qf) && q0.dot(qf) < 1e-3) { 
          // Fallback for antipodal points, use linear interpolation in R3 to get default axis
          Eigen::Vector4d mid = v0 + vf;
          Eigen::Vector4d tangent = (mid - mid.dot(v0) * v0).normalized();

          double theta = std::acos(std::clamp(v0.dot(vf), -1.0, 1.0)); 
          Eigen::Vector4d v_interpolated = v0 * std::cos(alpha_object * theta) + tangent * std::sin(alpha_object * theta);

          new_x_anchors.col(i+1).segment(idx, 4) = v_interpolated;
        } else {
          Eigen::Quaterniond slerp = q0.slerp(alpha_object, qf);
          new_x_anchors.col(i+1).segment(idx, 4) << slerp.w(), slerp.x(), slerp.y(), slerp.z(); 
        }
        new_x_anchors.col(i+1).segment(idx, 4) = new_x_anchors.col(i+1).segment(idx, 4).normalized();
      }
      // Don't update final anchor
      if (i == num_segments_) {
        new_x_anchors.col(i+1) = xd;
      }
      
      // Ensure anchors don't have penetration
      VectorXd x_projected = new_x_anchors.col(i+1);
      if (example_idx_ == 0) {
        // HARDCODED CONTACT GEOM
        drake::geometry::GeometryId plate_collision_geom =
            plant_.GetCollisionGeometriesForBody(
                plant_.GetBodyByName("plate"))[0];
        drake::geometry::GeometryId cube_collision_geom =
          plant_.GetCollisionGeometriesForBody(
              plant_.GetBodyByName("cube"))[0];
        SortedPair<GeometryId>cube_plate_contact(plate_collision_geom, cube_collision_geom);

        x_projected = ProjectContactPlate(context, cube_plate_contact, x_projected, 2, 11,
                        A_x, lower_bound_x, upper_bound_x);

      } else if (example_idx_ == 1) {
        for (int j = 0; j < 3; j++) {
          x_projected = ProjectContact(context, contact_geoms_[j], x_projected, 3*j, 3, 
                                        A_x, lower_bound_x, upper_bound_x);
        }
        
      } else if (example_idx_ == 2) {
        // Ensure cube not penetrating ground
        drake::geometry::GeometryId cube_collision_geom =
          plant_.GetCollisionGeometriesForBody(
              plant_.GetBodyByName("cube"))[0];
        drake::geometry::GeometryId ground_collision_geom =
          plant_.GetCollisionGeometriesForBody(
              plant_.GetBodyByName("ground"))[0];
        SortedPair<GeometryId>cube_ground_contact(cube_collision_geom, ground_collision_geom);

        x_projected = ProjectContactVertical(context, cube_ground_contact, x_projected, 15);

        // Ensure fingers don't have penetration
        for (int j = 0; j < 3; j++) {
          x_projected = ProjectContact(context, contact_geoms_[j], x_projected, 3*j, 3, 
                                        A_x, lower_bound_x, upper_bound_x);
        }
    
      }
      new_x_anchors.col(i+1) = x_projected;

      defects.col(i+1) = seg_x_hat_out[i].col(L_) - new_x_anchors.col(i+1);

      VectorXd ee_defect;
      VectorXd object_defect;
      if (example_idx_ == 0) {
        ee_defect = defects.col(i+1).segment(0, 5);
        object_defect = defects.col(i+1).segment(9, 3);
      } else if (example_idx_ == 1 || example_idx_ == 2) {
        ee_defect = defects.col(i+1).segment(0, 9);
        object_defect = defects.col(i+1).segment(13, 3);
      }

      std::cout << "Segment " << i << " ee defect: " << ee_defect.transpose() << std::endl;
      double quat_defect_deg = 0.0;
      for (auto quat_idx : controller_options_.quaternion_indices) {
        Eigen::Quaterniond q_anchor(x_anchor_next(quat_idx), x_anchor_next(quat_idx+1), x_anchor_next(quat_idx+2), x_anchor_next(quat_idx+3));
        Eigen::Quaterniond q_L(x_L(quat_idx), x_L(quat_idx+1), x_L(quat_idx+2), x_L(quat_idx+3));
        quat_defect_deg = q_anchor.angularDistance(q_L) * 180.0 / M_PI;
        std::cout << "Segment " << i << " quaternion defect (deg): " << quat_defect_deg << std::endl;
      }
      std::cout << "Segment " << i << " object defect: " << object_defect.transpose() << std::endl;

      if (!is_warmup) {
        defect_records.push_back({iter, i, ee_defect.norm(), quat_defect_deg, object_defect.norm()});
      }

      // Don't update initial x guess during warm start
      if (!is_warmup) {
        x_hat.middleCols(i*L_, L_) = seg_x_hat_out[i].leftCols(L_);
      }
      u_hat.middleCols(i*L_, L_) = seg_u_hat_out[i].leftCols(L_);
      lambda_hat.middleCols(i*L_, L_) = seg_lambda_hat_out[i].leftCols(L_);

      if (i == num_segments_-1) { // Tack on final state if last segment
        x_hat.col(N_) = seg_x_hat_out[i].col(L_);
      }
    } 

    x_anchors = new_x_anchors;

    // Linearize about new nominal trajectory
    lcs = MakeTimeVaryingLCS(x_hat, u_hat, lambda_hat, lcs_factory);

    // Don't store if warmup
    if (!is_warmup) {
      all_x_hats.push_back(x_hat);
      all_u_hats.push_back(u_hat);
      all_lambda_hats.push_back(lambda_hat);
      all_defects.push_back(defects);
      all_x_anchors.push_back(x_anchors);
      all_delta_projections.push_back(delta_projection_iter_);
      all_z_sols.push_back(z_sol_iter_);
      all_gammas.push_back(gamma);
      all_in_contacts.push_back(in_contact);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Iteration runtime: " << duration.count() << " seconds\n\n " << std::endl;

  }

  UpdateQuaternionCosts(x_hat, xd);

  auto [H_out, g, K, k_ff] = ComputeBoxDDPValueFunction(x_hat, u_hat, lambda_hat,
                            lcs, xd, u_nominal[0], defects, lower_bound_u, upper_bound_u);
  vector<MatrixXd> H = GetLowRankApproximation(H_out);

  Hs.push_back(H);
  gs.push_back(g);
  Ks.push_back(K);
  k_ffs.push_back(k_ff);

  // Write defects to CSV
  if (!is_optuna_) {
    std::vector<std::string> output_paths = {
      "defects.csv",
      "examples/resources/multifinger_hand/ic3_debug_data/defects.csv"
    };
    for (const auto& path : output_paths) {
      try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
          std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream file(path, std::ios::trunc);
        if (file.is_open()) {
          file << "iteration,segment,ee_defect_norm,quat_defect_deg,object_defect_norm\n";
          for (const auto& r : defect_records) {
            file << r.iter << "," << r.segment << "," << r.ee_defect_norm << ","
                 << r.quat_defect_deg << "," << r.object_defect_norm << "\n";
          }
          file.close();
        }
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to write defects to " << path << ": " << e.what() << std::endl;
      }
    }

    // Export lambda ADMM residuals to CSV
    std::vector<std::string> residual_output_paths = {
      "lambda_admm_residuals.csv",
      "examples/resources/multifinger_hand/ic3_debug_data/lambda_admm_residuals.csv"
    };
    for (const auto& path : residual_output_paths) {
      try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
          std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream file(path, std::ios::trunc);
        if (file.is_open()) {
          file << "iteration,segment,plan_timestep,admm_iter,lambda_diff_norm,iterate_step_change,complementarity_slack\n";
          for (const auto& rec : lambda_residual_records) {
            file << rec.outer_iter << "," << rec.segment << ","
                 << rec.plan_timestep << "," << rec.admm_iter << ","
                 << rec.lambda_diff_norm << "," << rec.iterate_step_change << ","
                 << rec.complementarity_slack << "\n";
          }
          file.close();
        }
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to write lambda residuals to " << path << ": " << e.what() << std::endl;
      }
    }
    std::cout << "Saved lambda ADMM residuals to lambda_admm_residuals.csv ("
              << lambda_residual_records.size() << " records)\n";

    // Export C3 x0 and x1 plan overlay to CSV
    std::vector<std::string> plan_output_paths = {
      "c3_x0_x1_plan.csv",
      "examples/resources/multifinger_hand/ic3_debug_data/c3_x0_x1_plan.csv"
    };
    for (const auto& path : plan_output_paths) {
      try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
          std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream file(path, std::ios::trunc);
        if (file.is_open()) {
          file << "iteration,segment,plan_timestep";
          for (int j = 0; j < n_x_; ++j) file << ",x0_" << j;
          for (int j = 0; j < n_x_; ++j) file << ",x1_" << j;
          for (int j = 0; j < n_lambda_; ++j) file << ",lambda_" << j;
          for (int j = 0; j < n_lambda_; ++j) file << ",eta_" << j;
          for (int j = 0; j < n_lambda_; ++j) file << ",lambda_last_" << j;
          for (int j = 0; j < n_lambda_; ++j) file << ",eta_last_" << j;
          file << "\n";
          for (const auto& rec : c3_plan_step_records) {
            file << rec.outer_iter << "," << rec.segment << "," << rec.plan_timestep;
            for (int j = 0; j < n_x_; ++j) file << "," << rec.x0(j);
            for (int j = 0; j < n_x_; ++j) file << "," << rec.x1(j);
            for (int j = 0; j < n_lambda_; ++j) file << "," << (j < rec.lambda.size() ? rec.lambda(j) : 0.0);
            for (int j = 0; j < n_lambda_; ++j) file << "," << (j < rec.eta.size() ? rec.eta(j) : 0.0);
            for (int j = 0; j < n_lambda_; ++j) file << "," << (j < rec.lambda_last.size() ? rec.lambda_last(j) : 0.0);
            for (int j = 0; j < n_lambda_; ++j) file << "," << (j < rec.eta_last.size() ? rec.eta_last(j) : 0.0);
            file << "\n";
          }
          file.close();
        }
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to write c3_x0_x1_plan to " << path << ": " << e.what() << std::endl;
      }
    }
    std::cout << "Saved C3 x0/x1 plan to c3_x0_x1_plan.csv ("
              << c3_plan_step_records.size() << " timesteps)\n";

    // Export C3 full lookahead plan to CSV
    std::vector<std::string> lookahead_output_paths = {
      "c3_full_lookahead_plan.csv",
      "examples/resources/multifinger_hand/ic3_debug_data/c3_full_lookahead_plan.csv"
    };
    for (const auto& path : lookahead_output_paths) {
      try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
          std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream file(path, std::ios::trunc);
        if (file.is_open()) {
          file << "iteration,segment,plan_timestep,lookahead_step";
          for (int j = 0; j < n_x_; ++j) file << ",x_" << j;
          file << "\n";
          for (const auto& rec : c3_full_lookahead_records) {
            file << rec.outer_iter << "," << rec.segment << "," << rec.plan_timestep << "," << rec.lookahead_step;
            for (int j = 0; j < n_x_; ++j) file << "," << rec.x(j);
            file << "\n";
          }
          file.close();
        }
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to write c3_full_lookahead_plan to " << path << ": " << e.what() << std::endl;
      }
    }
    std::cout << "Saved C3 full lookahead plan to c3_full_lookahead_plan.csv ("
              << c3_full_lookahead_records.size() << " records)\n";

    (void)std::system("python3 examples/python/plot_lambda_admm_residuals.py --csv lambda_admm_residuals.csv --out lambda_admm_residuals_3d.png > /dev/null 2>&1 &");
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_total - start_total;
  std::cout << "Total runtime: " << duration.count() << " seconds\n\n " << std::endl;

  return std::make_tuple(all_x_hats, all_u_hats, all_lambda_hats, Hs, gs, Ks, k_ffs, 
                        all_delta_projections, all_z_sols, all_gammas, all_in_contacts);
}

tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>> MSiC3Parallel::DoHybridMPCTracking(
    vector<VectorXd> x0s, MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, 
      HybridMpcOptions mpc_options, drake::systems::Context<double>& context,
  drake::systems::Context<drake::AutoDiffXd>& context_ad, drake::systems::Context<double>& context_rollout) {

  std::vector<double> x_des = *controller_options_.x_des;
  VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());  

  // Add linear constraints
  MatrixXd A_x(MatrixXd::Zero(n_x_, n_x_));
  MatrixXd A_u(MatrixXd::Zero(n_u_, n_u_));
  VectorXd lower_bound_x(VectorXd::Zero(n_x_));
  VectorXd upper_bound_x(VectorXd::Zero(n_x_));
  VectorXd lower_bound_u(VectorXd::Zero(n_u_));
  VectorXd upper_bound_u(VectorXd::Zero(n_u_));

  // HARDCODED
  if (example_idx_ == 0) { // plate
    A_x(0, 0) = 1;
    A_x(1, 1) = 1;
    A_x(2, 2) = 1;
    A_x(3, 3) = 1;
    A_x(4, 4) = 1;

    A_x(9, 9) = 1;

    lower_bound_x(0) = -0.1;
    lower_bound_x(1) = -0.1;
    lower_bound_x(2) = -0.15; 
    lower_bound_x(3) = -0.6;
    lower_bound_x(4) = -0.6;

    lower_bound_x(9) = 0.0;
    upper_bound_x(9) = 0.2;

    upper_bound_x(0) = 0.1;
    upper_bound_x(1) = 0.1;
    upper_bound_x(2) = 0.15;
    upper_bound_x(3) = 0.6;
    upper_bound_x(4) = 0.6;

    // Actuation limits
    A_u(0, 0) = 1;
    A_u(1, 1) = 1;
    A_u(2, 2) = 1;
    A_u(3, 3) = 1;
    A_u(4, 4) = 1;

    lower_bound_u(0) = -2;
    lower_bound_u(1) = -2;
    lower_bound_u(2) = 0;
    lower_bound_u(3) = -0.7;
    lower_bound_u(4) = -0.7;

    upper_bound_u(0) = 2;
    upper_bound_u(1) = 2;
    upper_bound_u(2) = 25;
    upper_bound_u(3) = 0.7;
    upper_bound_u(4) = 0.7;

  } else if (example_idx_ == 1) { // trifinger 180

    for (int i = 0; i < 3; i++) {
      // Position constraints
      A_x(3*i, 3*i) = 1;
      A_x(3*i+1, 3*i+1) = 1;
      A_x(3*i+2, 3*i+2) = 1;

      // Velocity constraints
      A_x(16 + 3*i, 16 + 3*i) = 1;
      A_x(16 + 3*i + 1, 16 + 3*i + 1) = 1;
      A_x(16 + 3*i + 2, 16 + 3*i + 2) = 1;

      // Offset from initial position
      lower_bound_x(3*i) = xd(3*i) - 0.06;
      lower_bound_x(3*i+1) = xd(3*i+1) - 0.06;
      lower_bound_x(3*i+2) = xd(3*i+2) - 0.01;

      lower_bound_x(16 + 3*i) = -0.2;
      lower_bound_x(16 + 3*i+1) = -0.2;
      lower_bound_x(16 + 3*i+2) = -0.05;

      upper_bound_x(3*i) = xd(3*i) + 0.06;
      upper_bound_x(3*i+1) = xd(3*i+1) + 0.06;
      upper_bound_x(3*i+2) = xd(3*i+2) + 0.01;

      upper_bound_x(16 + 3*i) = 0.2;
      upper_bound_x(16 + 3*i+1) = 0.2;
      upper_bound_x(16 + 3*i+2) = 0.05;

      A_u(3*i, 3*i) = 1;
      A_u(3*i+1, 3*i+1) = 1;
      A_u(3*i+2, 3*i+2) = 1;

      lower_bound_u(3*i) = -0.4;
      lower_bound_u(3*i+1) = -0.4;
      lower_bound_u(3*i+2) = 0.15;
      
      upper_bound_u(3*i) = 0.4;
      upper_bound_u(3*i+1) = 0.4;
      upper_bound_u(3*i+2) = 0.25;
    }

  } else if (example_idx_ == 2) { // trifinger pivot

    for (int i = 0; i < 3; i++) {
      // Position constraints
      A_x(3*i, 3*i) = 1;
      A_x(3*i+1, 3*i+1) = 1;
      A_x(3*i+2, 3*i+2) = 1;

      // Velocity constraints
      A_x(16 + 3*i, 16 + 3*i) = 1;
      A_x(16 + 3*i + 1, 16 + 3*i + 1) = 1;
      A_x(16 + 3*i + 2, 16 + 3*i + 2) = 1;

      // Offset from initial position
      lower_bound_x(3*i) = xd(3*i) - 0.07;
      lower_bound_x(3*i+1) = xd(3*i+1) - 0.07;
      lower_bound_x(3*i+2) = xd(3*i+2) - 0.01;

      lower_bound_x(16 + 3*i) = -0.1;
      lower_bound_x(16 + 3*i+1) = -0.1;
      lower_bound_x(16 + 3*i+2) = -0.1;

      upper_bound_x(3*i) = xd(3*i) + 0.07;
      upper_bound_x(3*i+1) = xd(3*i+1) + 0.07;
      upper_bound_x(3*i+2) = xd(3*i+2) + 0.1;

      upper_bound_x(16 + 3*i) = 0.1;
      upper_bound_x(16 + 3*i+1) = 0.1;
      upper_bound_x(16 + 3*i+2) = 0.1;

      A_u(3*i, 3*i) = 1;
      A_u(3*i+1, 3*i+1) = 1;
      A_u(3*i+2, 3*i+2) = 1;

      lower_bound_u(3*i) = -5;
      lower_bound_u(3*i+1) = -5;
      lower_bound_u(3*i+2) = -2;
      
      upper_bound_u(3*i) = 5;
      upper_bound_u(3*i+1) = 5;
      upper_bound_u(3*i+2) = 2;
    }
  }

  ResolveContacts(context, context_rollout);

  LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
      contact_geoms_, controller_options_.lcs_factory_options);

  HybridMPC hybrid_mpc_controller(plant_rollout_, lcs_factory, rollout_diagram_, std::move(rollout_diagram_context_), 
          contact_geoms_rollout_, mpc_options, ms_ic3_options_, example_idx_, controller_options_.lcs_factory_options.mu, 
          A_x, lower_bound_x, upper_bound_x, A_u, lower_bound_u, upper_bound_u);

  // Run hybrid mpc over final trajectory to ensure feasibility
  vector<MatrixXd> x_traj;
  vector<MatrixXd> u_traj;
  vector<MatrixXd> lambda_traj;

  for (size_t i = 0; i < x0s.size(); i++) {
    auto [x_hat_out, u_hat_out, lambda_hat_out] = 
      hybrid_mpc_controller.SimulateHybridMPC(x0s[i], x_hat, u_hat, lambda_hat, context_rollout);
    x_traj.push_back(x_hat_out);
    u_traj.push_back(u_hat_out);
    lambda_traj.push_back(lambda_hat_out);
  }

  return {x_traj, u_traj, lambda_traj};
}

VectorXd MSiC3Parallel::ProjectContactVertical(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx) {
    
    VectorXd x_out = x_init;
    plant_.SetPositionsAndVelocities(&context, x_init);
    multibody::GeomGeomCollider collider(plant_, geom_pair);
    auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);

    // HARDCODED FOR PIVOTING
    double phi_check = phi;
    while (phi_check < -3e-4) {
      // Displace z of object upwards 
      x_out(z_idx) += 0.002;
      
      plant_.SetPositionsAndVelocities(&context, x_out);
      auto [phi_new, J_new] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);
      phi_check = phi_new;
    }
    x_out(z_idx) = x_out(z_idx) + 0.002; // Add a little extra to ensure no penetration

    return x_out;
}

VectorXd MSiC3Parallel::ProjectContactPlate(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx, int pitch_idx, 
                                MatrixXd A_x, VectorXd lb_x, VectorXd ub_x) {
    
    VectorXd x_out = x_init;
    plant_.SetPositionsAndVelocities(&context, x_init);
    multibody::GeomGeomCollider collider(plant_, geom_pair);
    auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);

    // HARDCODED FOR PLATE EXAMPLE
    int counter = 0;
    double phi_check = phi;
    while (phi_check < -1e-4 && counter < 50) {
      // Displace z of object upwards 
      x_out(z_idx) += 0.002;
      
      // Displace pitch of plate towards object
      // HARDCODED SCALAR
      double displacement = 0.2 * (x_init(9) - x_init(0));
      x_out(pitch_idx) += displacement;
  
      // Threshold
      for (int i = 0; i < n_q_; i++) {
        if (A_x(i, i) == 1) { // Assumes diagonal
          x_out(i) = std::min(std::max(x_out(i), lb_x(i)), ub_x(i));
        }
      }

      plant_.SetPositionsAndVelocities(&context, x_out);
      auto [phi_new, J_new] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);
      phi_check = phi_new;
      counter++;
    }
    if (counter > 10) {
      std::cout << "counter " << counter << std::endl;
    }
    x_out(z_idx) = x_out(z_idx) -= 0.002; // Add a little extra to ensure no penetration

    return x_out;
}

VectorXd MSiC3Parallel::ProjectContact(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int start_idx, int q_size, MatrixXd A_x, VectorXd lb_x, VectorXd ub_x) {

    VectorXd x_out = x_init;
    plant_.SetPositionsAndVelocities(&context, x_init);
    multibody::GeomGeomCollider collider(plant_, geom_pair);
    auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);

    double phi_check = phi;
    VectorXd contact_normal = J.row(0).segment(start_idx, q_size);

    // HARDCODED: project slightly more than 1 unit out
    int counter = 0;
    while (phi_check < -3e-4 && counter < 5) {

      x_out.segment(start_idx, q_size) = 
        x_out.segment(start_idx, q_size) - 1.05 * (phi_check / contact_normal.squaredNorm()) * contact_normal;
      // Threshold
      for (int i = 0; i < q_size; i++) {
        int idx = start_idx + i;
        if (A_x(idx, idx) == 1) { // Assumes diagonal
          x_out(idx) = std::min(std::max(x_out(idx), lb_x(idx)), ub_x(idx));
        }
      }
      plant_.SetPositionsAndVelocities(&context, x_out);
      auto [phi_new, J_new] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);
      phi_check = phi_new;
      contact_normal = J_new.row(0).segment(start_idx, q_size);
      counter++;
    }
    if (counter > 1) {
      std::cout << "projection counter: " << counter << std::endl;
      std::cout << "phi end " << phi_check << std::endl;
    }
    
    return x_out;
}

tuple<LCS, MatrixXd, MatrixXd, MatrixXd> MSiC3Parallel::DoLCSRollout(VectorXd x0, MatrixXd u_hat, 
  LCSFactory factory, LCSFactory rollout_factory, MatrixXd A_constraint_x, VectorXd lower_bound_x, 
  VectorXd upper_bound_x, MatrixXd A_constraint_u, VectorXd lower_bound_u, VectorXd upper_bound_u, 
  vector<MatrixXd> K, vector<VectorXd> k_ff, double alpha) {

  int N = u_hat.cols();
  int factor = ms_ic3_options_.rollout_dt_scaling;

  rollout_factory.SetNewDt(dt_ / factor);

  MatrixXd x_hat(x0.size(), N*factor + 1);
  MatrixXd lambda_hat(n_lambda_, N*factor);
  MatrixXd u_hat_fb(n_u_, N*factor);

  x_hat.col(0) = x0;
  VectorXd x_curr = x0;
  VectorXd x_next;

  for (int k = 0; k < N*factor; k++) {

    // Linearize about current point
    VectorXd u_nominal = u_hat.col(k / factor);

    for (int i = 0; i < A_constraint_u.rows(); i++) {
      if (A_constraint_u(i, i) != 0) { // Assumes diagonal
        u_nominal(i) = std::clamp(u_nominal(i), lower_bound_u(i), upper_bound_u(i));
      }
    }

    u_hat_fb.col(k) = u_nominal;

    rollout_factory.UpdateStateAndInput(x_curr, u_nominal);
    LCS lcs = rollout_factory.GenerateLCS();     

    // Do one rollout step
    auto pair = lcs.SimulateAndReturnForce(x_curr, u_nominal, true);
    x_next = pair.first;

    // HARDCODED thresholding x's
    if (example_idx_ == 1 || example_idx_ == 2) {
      for (int i = 0; i < A_constraint_x.rows(); i++) {
        if (A_constraint_x(i, i) == 1) { // Assumes diagonal
          x_next(i) = std::min(std::max(x_next(i), lower_bound_x(i)), upper_bound_x(i));
        }
      }
    }

    lambda_hat.col(k) = pair.second;

    x_hat.col(k+1) = x_next;
    x_curr = x_next;
  }

  MatrixXd x_hat_downsampled(MatrixXd::Zero(n_x_, N + 1));
  MatrixXd lambda_hat_downsampled(MatrixXd::Zero(n_lambda_, N));
  MatrixXd u_hat_fb_downsampled(MatrixXd::Zero(n_u_, N));

  for (int i = 0; i < N; i++) {
    x_hat_downsampled.col(i) = x_hat.col(i * factor);
    lambda_hat_downsampled.col(i) = lambda_hat.col(i * factor);
    u_hat_fb_downsampled.col(i) = u_hat_fb.col(i * factor);
  }
  x_hat_downsampled.col(N) = x_hat.col(N * factor);

  LCS output_lcs = MakeTimeVaryingLCS(x_hat_downsampled, u_hat_fb_downsampled, lambda_hat_downsampled, factory);

  if ((x_hat_downsampled.array().isNaN()).any()) {
    std::cout << "XHAT NOT FINITE" << std::endl;
  }
  return {output_lcs, x_hat_downsampled, u_hat_fb_downsampled, lambda_hat_downsampled};

}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd, vector<vector<MatrixXd>>, vector<vector<VectorXd>>> MSiC3Parallel::DoC3Rollout(
    VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, 
    MatrixXd lambda_hat, VectorXd ud, VectorXd x_anchor_next,
    LCSFactory factory, LCSFactory rollout_factory, vector<MatrixXd> H, 
    vector<VectorXd> g, int start_idx, int segment_idx,                                          
    MatrixXd A_x, VectorXd lb_x, VectorXd ub_x,
    MatrixXd A_u, VectorXd lb_u, VectorXd ub_u,
    drake::systems::Context<double>& context, drake::systems::Context<double>& context_rollout,
    c3::C3Plus& c3_tracking) {

  DRAKE_DEMAND(start_idx < N_);
  DRAKE_DEMAND(static_cast<int>(H.size()) == N_ + 1);
  DRAKE_DEMAND(static_cast<int>(g.size()) == N_ + 1);
  DRAKE_DEMAND(x_hat.cols() == N_ + 1);

  int num_steps = u_hat.cols();
  int factor = ms_ic3_options_.rollout_dt_scaling;

  MatrixXd x_hat_output(n_x_, num_steps * factor + 1);
  MatrixXd lambda_hat_out(n_lambda_, num_steps * factor);
  MatrixXd u_hat_fb(n_u_, num_steps * factor);

  MatrixXd gamma_matrix(MatrixXd::Zero(n_lambda_ / 4, num_steps * factor));
  MatrixXd in_contact_matrix(MatrixXd::Zero(n_lambda_ / 4, num_steps * factor));

  vector<vector<MatrixXd>> seg_delta_projections;
  vector<vector<VectorXd>> seg_z_sols;

  x_hat_output.col(0) = x0;
  VectorXd x_curr = x0;
  VectorXd x_next;

  if (!controller_options_.x_des.has_value()) std::cerr << "Set x des" << std::endl;
  
  std::vector<double> x_des = controller_options_.x_des.value();
  VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());

  int tracking_N = controller_options_.lcs_factory_options.N;

  // Run receding horizon C3 MPC
  for (int t = 0; t < num_steps; t++) {

    vector<MatrixXd> Q;
    vector<MatrixXd> R;
    vector<MatrixXd> G;
    vector<MatrixXd> U;
    vector<VectorXd> x_targets_shortened;
    vector<VectorXd> x_reg_targets;
    vector<VectorXd> u_targets_shortened;
    vector<VectorXd> u_reg_targets;
    MatrixXd x_hat_for_lcs(MatrixXd::Zero(n_x_, tracking_N+1));
    MatrixXd u_hat_for_lcs(MatrixXd::Zero(n_u_, tracking_N));
    MatrixXd lambda_hat_for_lcs(MatrixXd::Zero(n_lambda_, tracking_N));

    double discount_factor = 1;
    for (int i = 0; i < tracking_N + 1; i++) {
      int x_idx = std::min(N_, start_idx + t + i);
      int u_idx = std::min(num_steps-1, t + i);
      int R_idx = std::min(N_-1, start_idx + t + i);

      x_targets_shortened.push_back(xd);
      x_reg_targets.push_back(x_hat.col(x_idx));

      x_hat_for_lcs.col(i) = x_curr;

      Q.push_back(discount_factor * Q_[x_idx]);

      if (i < tracking_N) {
        u_targets_shortened.push_back(ud);
        u_reg_targets.push_back(u_hat.col(u_idx));
        u_hat_for_lcs.col(i) = u_hat.col(u_idx);
        R.push_back(discount_factor * R_[R_idx]);

        lambda_hat_for_lcs.col(i) = lambda_hat.col(R_idx);
        G.push_back(discount_factor * G_[R_idx]);      
        U.push_back(discount_factor * U_[R_idx]);
      }
      discount_factor *= controller_options_.c3_options.gamma;
    }

    vector<MatrixXd> Q_updated = UpdateQuaternionCosts(x_curr, x_targets_shortened, Q);
    C3::CostMatrices costs(Q_updated, R, G, U);

    LCS lcs;
    #pragma omp critical
    {
      lcs = MakeTimeVaryingLCSWithEE(x_hat_for_lcs, u_hat_for_lcs, lambda_hat_for_lcs, factory, x_curr.segment(0, n_u_), 0);
    }

    vector<double> norms;
    norms.push_back(1.0);
    VectorXd x_temp = x_curr;
    for (int i = 1; i < tracking_N+1; i++) {
      VectorXd u_nominal = u_hat_for_lcs.col(i-1);
      x_temp = lcs.SimulateAtTimestep(x_temp, u_nominal, true, i-1);
      for (auto quat_idx : controller_options_.quaternion_indices) {
        double norm = x_temp.segment(quat_idx, 4).norm();
        norms.push_back(norm);
        x_targets_shortened[i].segment(quat_idx, 4) *= norm;
        x_reg_targets[i].segment(quat_idx, 4) *= norm;
      }
    }

    c3_tracking.UpdateCostMatrices(costs);
    c3_tracking.UpdateLCS(lcs);
    c3_tracking.UpdateTarget(x_targets_shortened);
    c3_tracking.UpdateInputTarget(u_targets_shortened);
    c3_tracking.SetPenalizeChange(false);

    if (ms_ic3_options_.add_terminal_constraint) {
      MatrixXd Q_slack_base = ms_ic3_options_.terminal_slack_vector.asDiagonal();
      MatrixXd Q_slack = UpdateQuaternionCostsSlack(x_curr, x_anchor_next, Q_slack_base);
      c3_tracking.UpdateTerminalTarget(Q_slack, x_anchor_next);
    }

    // If tracking c3 horizon goes past iC3 N, just use last H, g
    int lqr_idx = std::min(start_idx + t + tracking_N, N_); 
    double scaling = ms_ic3_options_.value_function_scaling;

    // Note: this gets added on top of the base x regularization weight
    MatrixXd Q_trust = UpdateQuaternionCosts(x_curr, {x_hat.col(lqr_idx)}, {Q_[lqr_idx]})[0];
    Q_trust *= ms_ic3_options_.vf_trust_region_weight;
    VectorXd bias = g[lqr_idx] - (H[lqr_idx] + Q_trust) * x_hat.col(lqr_idx);
    c3_tracking.UpdateFinalCost(scaling * (H[lqr_idx] + Q_trust), scaling * bias);

    auto c3_start = std::chrono::high_resolution_clock::now();
    c3_tracking.Solve(x_curr);
    auto c3_end = std::chrono::high_resolution_clock::now();
    auto c3_elapsed = c3_end - c3_start;
    double c3_solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(c3_elapsed).count() / 1e6;
    vector<MatrixXd> delta_proj = c3_tracking.GetDeltaProjection();
    
    // Rescale lambda, eta
    double AnDn = c3_tracking.GetAnDn();
    for (size_t ii = 0; ii < delta_proj.size(); ii++) {
      for (int jj = 0; jj < delta_proj[ii].cols(); jj++) {
        delta_proj[ii].col(jj).segment(n_x_, n_lambda_) *= AnDn;
        delta_proj[ii].col(jj).segment(n_x_+n_lambda_+n_u_, n_lambda_) *= AnDn;
      }
    }

    seg_delta_projections.push_back(delta_proj); // store projected step

    vector<Eigen::VectorXd> z_sol = c3_tracking.GetFullSolution();

    vector<Eigen::VectorXd> z_sol_scaled = z_sol;
    for (size_t ii = 0; ii < z_sol_scaled.size(); ii++) {
      z_sol_scaled[ii].segment(n_x_, n_lambda_) *= AnDn;
      z_sol_scaled[ii].segment(n_x_+n_lambda_+n_u_, n_lambda_) *= AnDn;
    } 
    seg_z_sols.push_back(z_sol_scaled);

    VectorXd c3_u = z_sol[0].segment(n_x_ + n_lambda_, n_u_);
    VectorXd c3_x = z_sol[0].segment(0, n_x_);
    VectorXd c3_x_next = z_sol[1].segment(0, n_x_);

    if (ms_ic3_options_.print_costs) {
      std::cout << "c3 solve time " << c3_solve_time << std::endl;
    }

    if (use_drake_sim_) {
      int q_idx = 0;
      int v_idx = 0;
      if (n_u_ == 5) {
        q_idx = 0;
        v_idx = 12;
      } else if (n_u_ == 9) {
        q_idx = 0;
        v_idx = 16;
      }
      MatrixXd Kp = ms_ic3_options_.rollout_Kp.asDiagonal();
      MatrixXd Kd = ms_ic3_options_.rollout_Kd.asDiagonal();

      Context<double>& root_context = simulators_[segment_idx]->get_mutable_context();
      Context<double>& sim_plant_context = rollout_diagram_.GetMutableSubsystemContext(
          plant_rollout_, &root_context);

      for (int i = 0; i < factor; i++) {      
        plant_rollout_.SetPositionsAndVelocities(&sim_plant_context, x_curr);

        // Apply PD to C3 plan
        VectorXd c3_x_tracking = (i * c3_x_next + (factor-i) * c3_x) / factor;
        VectorXd u_tracking = c3_u + Kp * (c3_x_tracking.segment(q_idx, Kp.rows()) - x_curr.segment(q_idx, Kp.rows())) 
          + Kd * (c3_x_tracking.segment(v_idx, Kd.rows()) - x_curr.segment(v_idx, Kd.rows()));

        for (int j = 0; j < A_u.rows(); j++) {
          if (A_u(j, j) == 1) {
            u_tracking(j) = std::clamp(u_tracking(j), lb_u(j), ub_u(j));
          }
        }

        plant_rollout_.get_actuation_input_port().FixValue(&sim_plant_context, u_tracking);

        double target_time = root_context.get_time() + dt_ / factor;
        simulators_[segment_idx]->AdvanceTo(target_time);

        x_next = plant_rollout_.GetPositionsAndVelocities(sim_plant_context);

        // Ensure consistent quaternion convention
        for (size_t quat_i = 0; quat_i < controller_options_.quaternion_indices.size(); quat_i++) {
          int idx = controller_options_.quaternion_indices[quat_i];
          if (x_curr.segment(idx, 4).dot(x_next.segment(idx, 4)) < 0) {
            x_next.segment(idx, 4) *= -1;
          }
          x_next.segment(idx, 4) = x_next.segment(idx, 4).normalized();
        }

        // Clamp velocities
        if (example_idx_ == 1 || example_idx_ == 2) {
          for (int j = n_q_; j < A_x.rows(); j++) {
            if (A_x(j, j) == 1) { // Assumes diagonal
              x_next(j) = std::clamp(x_next(j), lb_x(j), ub_x(j));
            }
          }
        }
        auto abstract_contact_results = drake::AbstractValue::Make<drake::multibody::ContactResults<double>>({});
        plant_rollout_.get_contact_results_output_port().Calc(sim_plant_context, abstract_contact_results.get());
        const auto& contact_results = abstract_contact_results->get_value<drake::multibody::ContactResults<double>>();

        x_hat_output.col(factor * t + i + 1) = x_next;
        u_hat_fb.col(factor * t + i) = u_tracking;

        auto [lambda, gamma_out, in_contact_out] = 
            ConstructLambdasFromContactResults(contact_results, controller_options_.lcs_factory_options.contact_model); 
        lambda_hat_out.col(factor * t + i) = lambda;
        gamma_matrix.col(factor * t + i) = gamma_out;
        in_contact_matrix.col(factor * t + i) = in_contact_out;

        x_curr = x_next;                         
      }

    } else {
      rollout_factory.SetNewDt(dt_ / factor);

      // Rollout this u with LCS
      for (int i = 0; i < factor; i++) {
        // Normalize quaternions
        for (int quat_idx : controller_options_.quaternion_indices) {
          x_curr.segment(quat_idx, 4) = x_curr.segment(quat_idx, 4).normalized();
        }

        // Apply PD to C3 plan
        int q_idx = 0;
        int v_idx = 0;
        if (n_u_ == 5) {
          q_idx = 0;
          v_idx = 12;
        } else if (n_u_ == 9) {
          q_idx = 0;
          v_idx = 16;
        }
        MatrixXd Kp = ms_ic3_options_.rollout_Kp.asDiagonal();
        MatrixXd Kd = ms_ic3_options_.rollout_Kd.asDiagonal();

        VectorXd c3_x_tracking = (i * c3_x_next + (factor-i) * c3_x) / factor;
        VectorXd u_tracking = c3_u + Kp * (c3_x_tracking.segment(q_idx, Kp.rows()) - x_curr.segment(q_idx, Kp.rows())) 
          + Kd * (c3_x_tracking.segment(v_idx, Kd.rows()) - x_curr.segment(v_idx, Kd.rows()));

        for (int j = 0; j < A_u.rows(); j++) {
          if (A_u(j, j) == 1) {
            u_tracking(j) = std::clamp(u_tracking(j), lb_u(j), ub_u(j));
          }
        }

        LCS lcs_rollout;
        #pragma omp critical
        {
          rollout_factory.UpdateStateAndInput(x_curr, u_tracking);
          lcs_rollout = rollout_factory.GenerateLCS();  

          // Debugging phi
          for (size_t g_idx = 0; g_idx < std::min<size_t>(3, contact_geoms_.size()); g_idx++) {
            multibody::GeomGeomCollider collider(plant_, contact_geoms_[g_idx]);
            plant_.SetPositionsAndVelocities(&context, x_curr);
            auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
              drake::multibody::JacobianWrtVariable::kQDot);
            if (phi < -1e-3) {
              std::cout << "contact " << g_idx << " phi " << phi << std::endl;
            }
          }
        }

        auto pair = lcs_rollout.SimulateAndReturnForce(x_curr, u_tracking, true);
        x_next = pair.first;

        if (example_idx_ == 1 || example_idx_ == 2) {
          for (int j = 0; j < A_x.rows(); j++) {
            if (A_x(j, j) == 1) { // Assumes diagonal
              x_next(j) = std::clamp(x_next(j), lb_x(j), ub_x(j));
            }
          }
        }
        x_hat_output.col(factor * t + i + 1) = x_next;
        lambda_hat_out.col(factor * t + i) = pair.second;
        u_hat_fb.col(factor * t + i) = u_tracking;

        x_curr = x_next;
      }
    }
  }

  MatrixXd x_hat_downsampled(MatrixXd::Zero(n_x_, num_steps + 1));
  MatrixXd u_hat_downsampled(MatrixXd::Zero(n_u_, num_steps));
  MatrixXd lambda_hat_downsampled(MatrixXd::Zero(n_lambda_, num_steps));
  MatrixXd gamma_downsampled(MatrixXd::Zero(n_lambda_ / 4, num_steps));
  MatrixXd in_contact_downsampled(MatrixXd::Zero(n_lambda_ / 4, num_steps));

  for (int i = 0; i < num_steps; i++) {
    x_hat_downsampled.col(i) = x_hat_output.col(i * factor);
    u_hat_downsampled.col(i) = u_hat_fb.col(i * factor);
    lambda_hat_downsampled.col(i) = lambda_hat_out.col(i * factor);
    gamma_downsampled.col(i) = gamma_matrix.col(i * factor);
    in_contact_downsampled.col(i) = in_contact_matrix.col(i * factor);
  }
  x_hat_downsampled.col(num_steps) = x_hat_output.col(num_steps * factor);

  return {x_hat_downsampled, u_hat_downsampled, lambda_hat_downsampled, gamma_downsampled, in_contact_downsampled, seg_delta_projections, seg_z_sols};                                              
}

std::tuple<vector<MatrixXd>, vector<VectorXd>, vector<MatrixXd>, vector<VectorXd>> 
MSiC3Parallel::ComputeLQRValueFunction(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat,
      LCS lcs, VectorXd xd, VectorXd ud, MatrixXd defects) {
  
  const vector<MatrixXd>& A = lcs.A();
  const vector<MatrixXd>& B = lcs.B();
  const vector<MatrixXd>& D = lcs.D();
  const vector<MatrixXd>& E = lcs.E();
  const vector<MatrixXd>& F = lcs.F();
  const vector<MatrixXd>& H = lcs.H();

  const vector<MatrixXd>& Q = Q_;
  const vector<MatrixXd>& R = R_;

  vector<MatrixXd> H_vf(N_+1, MatrixXd::Zero(n_x_, n_x_));
  vector<VectorXd> g(N_+1, VectorXd::Zero(n_x_));
  vector<MatrixXd> K(N_, MatrixXd::Zero(n_u_, n_x_));
  vector<VectorXd> k_ff(N_, VectorXd::Zero(n_u_));    

  // Terminal value function
  H_vf[N_] = Q[N_];

  for (int t = N_ - 1; t >= 0; t--) {
    int k = t / L_;

    VectorXd x_t = x_hat.col(t);
    VectorXd u_t = u_hat.col(t);

    // Active contact detection
    double active_tol = 1e-6;
    std::vector<int> active;
    active.reserve(n_lambda_);
    for (int j = 0; j < n_lambda_; j++) {
      if (lambda_hat.col(t)(j) > active_tol) {
        active.push_back(j);
      }
    }

    MatrixXd f_x, f_u;
    if (active.empty()) {
      f_x = A[t];
      f_u = B[t];
    } else {
      int na = static_cast<int>(active.size());

      MatrixXd D_a(n_x_, na);
      MatrixXd E_a(na, n_x_);
      MatrixXd H_a(na, n_u_);
      MatrixXd F_aa(na, na);

      for (int a = 0; a < na; a++) {
        int ja = active[a];
        D_a.col(a) = D[t].col(ja);
        E_a.row(a) = E[t].row(ja);
        H_a.row(a) = H[t].row(ja);
        for (int b = 0; b < na; b++) {
          F_aa(a, b) = F[t](ja, active[b]);
        }
      }

      // Solve (F_aa)^{-1}E_a and (F_aa)^{-1}H_a
      Eigen::ColPivHouseholderQR<MatrixXd> qr(F_aa);
      MatrixXd FinvE = qr.solve(E_a);
      MatrixXd FinvH = qr.solve(H_a); 
      
      f_x = A[t] - D_a * FinvE;
      f_u = B[t] - D_a * FinvH;
    }

    // Quadratic expansions
    MatrixXd Q_xx = Q[t] + f_x.transpose() * H_vf[t+1] * f_x;
    MatrixXd Q_uu = R[t] + f_u.transpose() * H_vf[t+1] * f_u;
    MatrixXd Q_ux = f_u.transpose() * H_vf[t+1] * f_x;

    // Linear expansions
    VectorXd Q_x = Q[t] * (x_t - xd) + f_x.transpose() * (g[t+1] + H_vf[t+1] * defects.col(k+1));
    VectorXd Q_u = R[t] * (u_t - ud) + f_u.transpose() * (g[t+1] + H_vf[t+1] * defects.col(k+1));

    // Regularize Q_uu for robust inversion
    Q_uu.diagonal().array() += 1e-6;

    Eigen::LDLT<MatrixXd> solver(Q_uu);
    K[t] = -solver.solve(Q_ux);
    k_ff[t] = -solver.solve(Q_u);

    // Backpropagate Value Function
    H_vf[t] = Q_xx + K[t].transpose() * Q_ux; // Mathematically identical to Q_xx - K^T Q_uu K
    H_vf[t] = 0.5 * (H_vf[t] + H_vf[t].transpose()); // Enforce symmetry

    // Threshold small values for conditioning
    constexpr double EPS_ABS = 1e-4;
    for (int i = 0; i < n_x_; ++i) {
        for (int j = 0; j < n_x_; ++j) {
            if (std::abs(H_vf[t](i, j)) < EPS_ABS) {
                H_vf[t](i, j) = 0.0;
            }
        }
    }

    g[t] = Q_x + K[t].transpose() * Q_u;
  }

  return std::make_tuple(H_vf, g, K, k_ff);
}

std::tuple<vector<MatrixXd>, vector<VectorXd>, vector<MatrixXd>, vector<VectorXd>> 
MSiC3Parallel::ComputeBoxDDPValueFunction(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat,
      LCS lcs, VectorXd xd, VectorXd ud, MatrixXd defects,
      VectorXd u_min, VectorXd u_max) {
  
  const vector<MatrixXd>& A = lcs.A();
  const vector<MatrixXd>& B = lcs.B();
  const vector<MatrixXd>& D = lcs.D();
  const vector<MatrixXd>& E = lcs.E();
  const vector<MatrixXd>& F = lcs.F();
  const vector<MatrixXd>& H = lcs.H();

  const vector<MatrixXd>& Q = Q_;
  const vector<MatrixXd>& R = R_;

  vector<MatrixXd> H_vf(N_+1, MatrixXd::Zero(n_x_, n_x_));
  vector<VectorXd> g(N_+1, VectorXd::Zero(n_x_));
  vector<MatrixXd> K(N_, MatrixXd::Zero(n_u_, n_x_));
  vector<VectorXd> k_ff(N_, VectorXd::Zero(n_u_));    

  // Terminal value function
  H_vf[N_] = Q[N_];

  for (int t = N_ - 1; t >= 0; t--) {
    int k_idx = t / L_;

    VectorXd x_t = x_hat.col(t);
    VectorXd u_t = u_hat.col(t);

    // Active contact detection
    double active_tol = 1e-6;
    std::vector<int> active;
    active.reserve(n_lambda_);
    for (int j = 0; j < n_lambda_; j++) {
      if (lambda_hat.col(t)(j) > active_tol) {
        active.push_back(j);
      }
    }

    MatrixXd f_x, f_u;
    if (active.empty()) {
      f_x = A[t];
      f_u = B[t];
    } else {
      int na = static_cast<int>(active.size());

      MatrixXd D_a(n_x_, na);
      MatrixXd E_a(na, n_x_);
      MatrixXd H_a(na, n_u_);
      MatrixXd F_aa(na, na);

      for (int a = 0; a < na; a++) {
        int ja = active[a];
        D_a.col(a) = D[t].col(ja);
        E_a.row(a) = E[t].row(ja);
        H_a.row(a) = H[t].row(ja);
        for (int b = 0; b < na; b++) {
          F_aa(a, b) = F[t](ja, active[b]);
        }
      }

      Eigen::ColPivHouseholderQR<MatrixXd> qr(F_aa);
      MatrixXd FinvE = qr.solve(E_a);
      MatrixXd FinvH = qr.solve(H_a);
      
      f_x = A[t] - D_a * FinvE;
      f_u = B[t] - D_a * FinvH;
    }

    // Quadratic expansions
    MatrixXd Q_xx = Q[t] + f_x.transpose() * H_vf[t+1] * f_x;
    MatrixXd Q_uu = R[t] + f_u.transpose() * H_vf[t+1] * f_u;
    MatrixXd Q_ux = f_u.transpose() * H_vf[t+1] * f_x;

    // Linear expansions
    VectorXd Q_x = Q[t] * (x_t - xd) + f_x.transpose() * (g[t+1] + H_vf[t+1] * defects.col(k_idx+1));
    VectorXd Q_u = R[t] * (u_t - ud) + f_u.transpose() * (g[t+1] + H_vf[t+1] * defects.col(k_idx+1));

    // Regularize Q_uu for robust inversion
    Q_uu.diagonal().array() += 1e-6;

    // =========================================================================
    // BOX-DDP INCORPORATION (Projected Newton / BoxQP)
    // =========================================================================
    VectorXd lower_bound = u_min - u_t;
    VectorXd upper_bound = u_max - u_t;

    // 1. Solve for bounded k_ff using Projected Newton
    std::vector<int> free_idx;
    VectorXd k_init = VectorXd::Zero(n_u_);
    k_ff[t] = SolveBoxQP(Q_uu, Q_u, lower_bound, upper_bound, k_init, free_idx);

    // 2. Compute Feedback Gain K on the free subspace (K_c = 0)
    K[t].setZero();
    int n_f = static_cast<int>(free_idx.size());
    if (n_f > 0) {
      MatrixXd Q_uu_ff(n_f, n_f);
      MatrixXd Q_ux_f(n_f, n_x_);

      for (int i = 0; i < n_f; ++i) {
        Q_ux_f.row(i) = Q_ux.row(free_idx[i]);
        for (int j = 0; j < n_f; ++j) {
          Q_uu_ff(i, j) = Q_uu(free_idx[i], free_idx[j]);
        }
      }

      Eigen::LDLT<MatrixXd> free_solver(Q_uu_ff);
      MatrixXd K_f = -free_solver.solve(Q_ux_f);

      for (int i = 0; i < n_f; ++i) {
        K[t].row(free_idx[i]) = K_f.row(i);
      }
    }

    // 3. Backpropagate Value Function
    MatrixXd Kt_Quu_K = K[t].transpose() * Q_uu * K[t];
    MatrixXd Kt_Qux = K[t].transpose() * Q_ux;

    H_vf[t] = Q_xx + Kt_Quu_K + Kt_Qux + Kt_Qux.transpose();
    H_vf[t] = 0.5 * (H_vf[t] + H_vf[t].transpose()); // Enforce symmetry

    // Threshold small values for conditioning
    constexpr double EPS_ABS = 1e-4;
    for (int i = 0; i < n_x_; ++i) {
        for (int j = 0; j < n_x_; ++j) {
            if (std::abs(H_vf[t](i, j)) < EPS_ABS) {
                H_vf[t](i, j) = 0.0;
            }
        }
    }

    g[t] = Q_x + K[t].transpose() * (Q_uu * k_ff[t] + Q_u) + Q_ux.transpose() * k_ff[t];
  }

  return std::make_tuple(H_vf, g, K, k_ff);
}

LCS MSiC3Parallel::MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, LCSFactory factory) {
  DRAKE_DEMAND(x_hat.cols() >= u_hat.cols());

  vector<Eigen::MatrixXd> A;
  vector<Eigen::MatrixXd> B;
  vector<Eigen::MatrixXd> D;
  vector<Eigen::VectorXd> d;
  vector<Eigen::MatrixXd> E;
  vector<Eigen::MatrixXd> F;
  vector<Eigen::MatrixXd> H;
  vector<Eigen::VectorXd> c;

  int N = u_hat.cols();

  for (int k = 0; k < N; k++) {
    
    for (auto idx : controller_options_.quaternion_indices) {
      x_hat.col(k).segment(idx, 4) = x_hat.col(k).segment(idx, 4).normalized(); // Normalize quaternions
    }

    // Linearize about kth xhat, uhat
    factory.UpdateStateAndInput(x_hat.col(k), u_hat.col(k));

    VectorXd lambda_nom(VectorXd::Zero(n_lambda_));
    if (ms_ic3_options_.use_lambdas_for_lcs) {
      lambda_nom = lambda_hat.col(k);
    } 

    LCS lcs = factory.GenerateLCS();
    A.push_back(lcs.A()[0]);
    B.push_back(lcs.B()[0]);
    D.push_back(lcs.D()[0]);
    d.push_back(lcs.d()[0]);
    E.push_back(lcs.E()[0]);
    F.push_back(lcs.F()[0]);
    H.push_back(lcs.H()[0]);
    c.push_back(lcs.c()[0]);      
  }

  for (int k = 0; k < 1; k++) {
    if (A[k].array().isNaN().any()) {
      std::cerr << "A " << k << " has NaN" << std::endl;
      std::cerr << "x hat " << x_hat.col(k).transpose() << std::endl;
      std::cerr << "u hat " << u_hat.col(k).transpose() << std::endl << std::endl;
    }
    if (B[k].array().isNaN().any()) {
      std::cerr << "B " << k << " has NaN" << std::endl;
      std::cerr << "x hat " << x_hat.col(k).transpose() << std::endl;
      std::cerr << "u hat " << u_hat.col(k).transpose() << std::endl << std::endl;
    }
    if (D[k].array().isNaN().any()) {
      std::cerr << "D " << k << " has NaN" << std::endl;
      std::cerr << "x hat " << x_hat.col(k).transpose() << std::endl;
      std::cerr << "u hat " << u_hat.col(k).transpose() << std::endl << std::endl;
    }
    if (d[k].array().isNaN().any()) {
      std::cerr << "d " << k << " has NaN" << std::endl;
      std::cerr << "x hat " << x_hat.col(k).transpose() << std::endl;
      std::cerr << "u hat " << u_hat.col(k).transpose() << std::endl << std::endl;
    }
    if (E[k].array().isNaN().any()) {
      std::cerr << "E " << k << " has NaN" << std::endl;
      std::cerr << "x hat " << x_hat.col(k).transpose() << std::endl;
      std::cerr << "u hat " << u_hat.col(k).transpose() << std::endl << std::endl;
    }
    if (F[k].array().isNaN().any()) {
      std::cerr << "F " << k << " has NaN" << std::endl;
      std::cerr << "x hat " << x_hat.col(k).transpose() << std::endl;
      std::cerr << "u hat " << u_hat.col(k).transpose() << std::endl << std::endl;
    }
    if (H[k].array().isNaN().any()) {
      std::cerr << "H " << k << " has NaN" << std::endl;
      std::cerr << "x hat " << x_hat.col(k).transpose() << std::endl;
      std::cerr << "u hat " << u_hat.col(k).transpose() << std::endl << std::endl;
    }
    if (c[k].array().isNaN().any()) {
      std::cerr << "c " << k << " has NaN" << std::endl;
      std::cerr << "x hat " << x_hat.col(k).transpose() << std::endl;
      std::cerr << "u hat " << u_hat.col(k).transpose() << std::endl << std::endl;
    }
  }

  return LCS(A, B, D, d, E, F, H, c, dt_);
}

LCS MSiC3Parallel::MakeTimeVaryingLCSWithEE(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, 
                                  LCSFactory factory, VectorXd ee_pose, int ee_idx) {
  MatrixXd x_hat_copy = x_hat;
  for (int i = 0; i < x_hat_copy.cols(); i++) {
    x_hat_copy.col(i).segment(ee_idx, ee_pose.size()) = ee_pose;
  }
  return MakeTimeVaryingLCS(x_hat_copy, u_hat, lambda_hat, factory);
}

LCS MSiC3Parallel::GetLCSSegment(LCS lcs, int start_idx, int length) {
  DRAKE_DEMAND(start_idx < lcs.N());

  vector<Eigen::MatrixXd> A;
  vector<Eigen::MatrixXd> B;
  vector<Eigen::MatrixXd> D;
  vector<Eigen::VectorXd> d;
  vector<Eigen::MatrixXd> E;
  vector<Eigen::MatrixXd> F;
  vector<Eigen::MatrixXd> H;
  vector<Eigen::VectorXd> c;

  for (int i = 0; i < length; i++) {
    int idx = std::min(start_idx + i, lcs.N()-1);

    if (lcs.A()[idx].array().isNaN().any()) {
      std::cout << "lcs A " << idx << " is NAN" << std::endl;
    }
    if (lcs.B()[idx].array().isNaN().any()) {
      std::cout << "lcs B " << idx << " is NAN" << std::endl;
    }
    if (lcs.D()[idx].array().isNaN().any()) {
      std::cout << "lcs D " << idx << " is NAN" << std::endl;
    }
    if (lcs.d()[idx].array().isNaN().any()) {
      std::cout << "lcs d " << idx << " is NAN" << std::endl;
    }
    if (lcs.E()[idx].array().isNaN().any()) {
      std::cout << "lcs E " << idx << " is NAN" << std::endl;
    }
    if (lcs.F()[idx].array().isNaN().any()) {
      std::cout << "lcs F " << idx << " is NAN" << std::endl;
    }
    if (lcs.H()[idx].array().isNaN().any()) {
      std::cout << "lcs H " << idx << " is NAN" << std::endl;
    }
    if (lcs.c()[idx].array().isNaN().any()) {
      std::cout << "lcs c " << idx << " is NAN" << std::endl;
    }

    A.push_back(lcs.A()[idx]);
    B.push_back(lcs.B()[idx]);
    D.push_back(lcs.D()[idx]);
    d.push_back(lcs.d()[idx]);
    E.push_back(lcs.E()[idx]);
    F.push_back(lcs.F()[idx]);
    H.push_back(lcs.H()[idx]);
    c.push_back(lcs.c()[idx]);      
  }

  return LCS(A, B, D, d, E, F, H, c, dt_);
}

std::tuple<VectorXd, VectorXd, VectorXd> MSiC3Parallel::ConstructLambdasFromContactResults(ContactResults<double> contact_results, std::string contact_model) {
  DRAKE_DEMAND(controller_options_.lcs_factory_options.num_friction_directions == 2);

  // Assumes 2 friction directions
  VectorXd lambda(VectorXd::Zero(n_lambda_));

  VectorXd gamma(VectorXd::Zero(n_lambda_ / 4));
  VectorXd in_contact(VectorXd::Zero(n_lambda_ / 4));

  int n_contacts = contact_geoms_rollout_.size();

  for (int i = 0; i < n_contacts; i++) {
    GeometryId geom_A = contact_geoms_rollout_[i].first();
    GeometryId geom_B = contact_geoms_rollout_[i].second();

    for (int j = 0; j < contact_results.num_point_pair_contacts(); j++) {
      const auto& info = contact_results.point_pair_contact_info(j);
      const auto& pair = info.point_pair();

      GeometryId id_A = pair.id_A;
      GeometryId id_B = pair.id_B;

      // Search for matching contact result
      if ((geom_A == id_A && geom_B == id_B) || (geom_A == id_B && geom_B == id_A)) {
        bool is_swapped = (geom_A == id_B && geom_B == id_A);

        Vector3d n_W;
        Vector3d f_W;

        in_contact(i) = 1;
        gamma(i) = info.slip_speed();

        if (is_swapped) {
           n_W = -pair.nhat_BA_W;
           f_W = info.contact_force(); 
        } else {
           n_W = pair.nhat_BA_W;
           f_W = -info.contact_force(); 
        }

        // Get tangent basis
        auto R_WC = RotationMatrix<double>::MakeFromOneVector(n_W, 0);
        Eigen::Vector3d t1_W = R_WC.col(1);
        Eigen::Vector3d t2_W = R_WC.col(2);

        double f_n = std::max(0.0, f_W.dot(n_W));
        double f_t1 = f_W.dot(t1_W);
        double f_t2 = f_W.dot(t2_W);

        if (contact_model == "anitescu") {
          double mu = controller_options_.lcs_factory_options.mu[i];

          double l1_base = std::max(0.0, f_t1 / mu);
          double l2_base = std::max(0.0, -f_t1 / mu);
          double l3_base = std::max(0.0, f_t2 / mu);
          double l4_base = std::max(0.0, -f_t2 / mu);

          double base_normal_sum = l1_base + l2_base + l3_base + l4_base;
          double deficit = std::max(0.0, f_n - base_normal_sum);
          double offset = deficit / 4.0;

          lambda(4*i) = l1_base + offset;
          lambda(4*i + 1) = l2_base + offset;
          lambda(4*i + 2) = l3_base + offset;
          lambda(4*i + 3) = l4_base + offset;

        } else if (contact_model == "stewart_and_trinkle") {

          lambda(i) = info.slip_speed(); // gamma
          lambda(n_contacts + i) = f_n; // lambda_n
          lambda(2 * n_contacts + 4*i) = std::max(0.0,  f_t1);      
          lambda(2 * n_contacts + 4*i + 1) = std::max(0.0, -f_t1);
          lambda(2 * n_contacts + 4*i + 2) = std::max(0.0,  f_t2); 
          lambda(2 * n_contacts + 4*i + 3) = std::max(0.0, -f_t2); 

        } else {
          std::cerr << "UNKNOWN CONTACT MODEL" << std::endl;
        }

      }
    }
  }
  return {lambda, gamma, in_contact};
}

void MSiC3Parallel::UpdateQuaternionCosts(
  MatrixXd x_hat, VectorXd x_des) {
  
  Q_.clear();
  R_.clear();
  G_.clear();
  U_.clear();

  for (int i = 0; i < N_+1; i++) {
    Q_.push_back(controller_options_.c3_options.Q);
    if (i < N_) {
      R_.push_back(controller_options_.c3_options.R);
      G_.push_back(controller_options_.c3_options.G);
      U_.push_back(controller_options_.c3_options.U);
    }
  }  

  for (int i = 0; i < N_ + 1; i++) {
    int j = 0;
    for (int index : controller_options_.quaternion_indices) {

      // make quaternion costs time-varying based on x_hat
      Eigen::VectorXd quat_curr_i = x_hat.col(i).segment(index, 4).normalized();
      Eigen::VectorXd quat_des_i = x_des.segment(index, 4).normalized();

      Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

      // Regularize hessian so Q is always PSD
      double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();

      Eigen::MatrixXd Q_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd Q_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
      Eigen::MatrixXd Q_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

      Q_[i].block(index, index, 4, 4) = 
        controller_options_.c3_options.w_Q * 
        controller_options_.Q_quaternion_weight * (quat_hessian_i + Q_quat_regularizer_1 + 
        controller_options_.quaternion_regularizer_fraction * Q_quat_regularizer_2 + Q_quat_regularizer_3);

      j++;
    }
  }
}

vector<MatrixXd> MSiC3Parallel::UpdateQuaternionCosts(
    VectorXd x_curr, vector<VectorXd> x_des, vector<MatrixXd> Q_in) {
  
  DRAKE_DEMAND(x_des.size() == Q_in.size());

  vector<MatrixXd> Q = Q_in;

  double discount_factor = 1;
  for (size_t i = 0; i < Q.size(); i++) {
    int j = 0;
    for (int index : controller_options_.quaternion_indices) {

      // make quaternion costs time-varying based on x_hat
      Eigen::VectorXd quat_curr_i = x_curr.segment(index, 4).normalized();
      Eigen::VectorXd quat_des_i = x_des[i].segment(index, 4).normalized();

      Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

      // Regularize hessian so Q is always PSD
      double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();

      Eigen::MatrixXd Q_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd Q_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
      Eigen::MatrixXd Q_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

      Q[i].block(index, index, 4, 4) = 
        discount_factor * 
        controller_options_.c3_options.w_Q * 
        controller_options_.Q_quaternion_weight * (quat_hessian_i + Q_quat_regularizer_1 + 
        controller_options_.quaternion_regularizer_fraction * Q_quat_regularizer_2 + Q_quat_regularizer_3);

      j++;
    }
    discount_factor *= controller_options_.c3_options.gamma;
  }
  return Q;
}

MatrixXd MSiC3Parallel::UpdateQuaternionCostsSlack(
    VectorXd x_curr, VectorXd x_des, MatrixXd Q_in) {
  
  MatrixXd Q = Q_in;

  double discount_factor = 1;
  int j = 0;
  for (int index : controller_options_.quaternion_indices) {

    // make quaternion costs time-varying based on x_hat
    Eigen::VectorXd quat_curr_i = x_curr.segment(index, 4).normalized();
    Eigen::VectorXd quat_des_i = x_des.segment(index, 4).normalized();

    Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

    // Regularize hessian so Q is always PSD
    double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();

    Eigen::MatrixXd Q_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd Q_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
    Eigen::MatrixXd Q_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

    Q.block(index, index, 4, 4) = 
      discount_factor * ms_ic3_options_.terminal_slack_quaternion_weight * 
        (quat_hessian_i + Q_quat_regularizer_1 + Q_quat_regularizer_3);

    j++;
  }
  discount_factor *= controller_options_.c3_options.gamma;
  
  return Q;
}

Eigen::Quaterniond MSiC3Parallel::slerpLong(const Eigen::Quaterniond& q0,
                             const Eigen::Quaterniond& q1, double t) {
    Eigen::Quaterniond a = q0.normalized();
    Eigen::Quaterniond b = q1.normalized();

    double d = a.dot(b);

    // For the LONG way we want the larger quaternion arc,
    // i.e. force the dot product to be <= 0.
    if (d > 0.0) {
        b.coeffs() = -b.coeffs();
        d = -d;
    }

    double theta    = std::acos(std::clamp(d, -1.0, 1.0)); // obtuse now
    double sinTheta = std::sin(theta);

    if (sinTheta < 1e-9)               // degenerate / antipodal
        return a;

    double s0 = std::sin((1.0 - t) * theta) / sinTheta;
    double s1 = std::sin(t * theta)         / sinTheta;

    Eigen::Quaterniond res;
    res.coeffs() = s0 * a.coeffs() + s1 * b.coeffs();
    return res.normalized();
}

VectorXd MSiC3Parallel::SolveBoxQP(const MatrixXd& Q_uu,
                    const VectorXd& Q_u,
                    const VectorXd& lower_bound,
                    const VectorXd& upper_bound,
                    const VectorXd& k_init,
                    std::vector<int>& free_indices) {

  const int n_u = Q_u.size();
  const int max_iters = 30;
  const double tol = 1e-6;
  const double armijo_c1 = 0.1;
  const double step_decay = 0.5;
  const int max_linesearch_iters = 10;

  // Initialize and clamp initial guess
  VectorXd k = k_init;
  for (int i = 0; i < n_u; ++i) {
    k(i) = std::clamp(k(i), lower_bound(i), upper_bound(i));
  }

  auto compute_cost = [&](const VectorXd& x) -> double {
    return 0.5 * x.dot(Q_uu * x) + Q_u.dot(x);
  };

  VectorXd grad(n_u);
  VectorXd delta_k(n_u);

  for (int iter = 0; iter < max_iters; ++iter) {
    grad = Q_u + Q_uu * k;

    // 1. Identify Free (Inactive) and Clamped (Active) sets
    free_indices.clear();
    std::vector<int> clamped_indices;
    free_indices.reserve(n_u);
    clamped_indices.reserve(n_u);

    for (int i = 0; i < n_u; ++i) {
      bool at_lower = (k(i) <= lower_bound(i) + 1e-9);
      bool at_upper = (k(i) >= upper_bound(i) - 1e-9);

      // A variable is clamped if it is at the boundary and the gradient points outside
      if ((at_lower && grad(i) > 0.0) || (at_upper && grad(i) < 0.0)) {
        clamped_indices.push_back(i);
      } else {
        free_indices.push_back(i);
      }
    }

    // 2. Check first-order optimality (Projected Gradient Norm)
    double projected_grad_norm = 0.0;
    for (int i = 0; i < n_u; ++i) {
      double proj_step = std::clamp(k(i) - grad(i), lower_bound(i), upper_bound(i)) - k(i);
      projected_grad_norm = std::max(projected_grad_norm, std::abs(proj_step));
    }
    if (projected_grad_norm < tol) {
      break;
    }

    // 3. Compute Newton Search Direction
    delta_k.setZero();

    // Clamped variables move toward projected bounds
    for (int i : clamped_indices) {
      delta_k(i) = std::clamp(k(i) - grad(i), lower_bound(i), upper_bound(i)) - k(i);
    }

    const int n_f = free_indices.size();
    if (n_f > 0) {
      // Extract free sub-Hessian and sub-gradient
      MatrixXd H_ff(n_f, n_f);
      VectorXd g_f(n_f);

      for (int r = 0; r < n_f; ++r) {
        int i = free_indices[r];
        // g_f = Q_u_f + H_fc * delta_k_c
        double coupling = 0.0;
        for (int j : clamped_indices) {
          coupling += Q_uu(i, j) * delta_k(j);
        }
        g_f(r) = grad(i) + coupling;

        for (int c = 0; c < n_f; ++c) {
          int j = free_indices[c];
          H_ff(r, c) = Q_uu(i, j);
        }
      }

      // Solve H_ff * delta_k_f = -g_f via Cholesky (LLT)
      LLT<MatrixXd> llt(H_ff);
      VectorXd delta_k_free;
      if (llt.info() == Eigen::Success) {
        delta_k_free = llt.solve(-g_f);
      } else {
        // Fallback to scaled gradient descent if free Hessian is singular
        delta_k_free = -g_f;
      }

      for (int r = 0; r < n_f; ++r) {
        delta_k(free_indices[r]) = delta_k_free(r);
      }
    }

    // 4. Projected Armijo Line Search
    double current_cost = compute_cost(k);
    double directional_deriv = grad.dot(delta_k);
    if (directional_deriv > 0.0) {
      // If search direction is not descent due to numerical error, use projected gradient
      delta_k = -grad;
      directional_deriv = -grad.squaredNorm();
    }

    double alpha = 1.0;
    VectorXd k_trial(n_u);
    bool accepted = false;

    for (int ls = 0; ls < max_linesearch_iters; ++ls) {
      // Project candidate point along the arc
      for (int i = 0; i < n_u; ++i) {
        k_trial(i) = std::clamp(k(i) + alpha * delta_k(i), lower_bound(i), upper_bound(i));
      }

      double trial_cost = compute_cost(k_trial);
      // Armijo condition: f(k_trial) <= f(k) + c1 * alpha * grad^T * delta_k
      if (trial_cost <= current_cost + armijo_c1 * alpha * directional_deriv) {
        k = k_trial;
        accepted = true;
        break;
      }
      alpha *= step_decay;
    }

    if (!accepted) {
      // Line search failed to find sufficient decrease; terminate
      break;
    }
  }

  return k;
}

vector<MatrixXd> MSiC3Parallel::GetLowRankApproximation(vector<MatrixXd> H_in) {
  vector<MatrixXd> H_out;

  for (size_t i = 0; i < H_in.size(); i++) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(H_in[i]);
    VectorXd evals = es.eigenvalues();            // ascending
    double lambda_max = evals(evals.size() - 1);

    double tol = 1e-2;  
    int r = 0;
    for (int j = evals.size() - 1; j >= 0; --j) {
      if (evals(j) > tol * lambda_max) ++r;
      else break;  
    }

    MatrixXd V = es.eigenvectors().rightCols(r);
    VectorXd d = evals.tail(r).cwiseMax(0.0);
    MatrixXd H_lr = V * d.asDiagonal() * V.transpose();

    H_out.push_back(H_lr);
  }
  return H_out;
}

void MSiC3Parallel::SetSolverOptions(const drake::solvers::SolverOptions& solver_options) {
  solver_options_ = solver_options;
  for (auto& c3_ptr : c3_trackings_) {
    if (c3_ptr) {
      c3_ptr->SetSolverOptions(solver_options);
    }
  }
  if (c3_tracking_) {
    c3_tracking_->SetSolverOptions(solver_options);
  }
}

void MSiC3Parallel::SetSolverOptions(const std::string& solver_options_file) {
  std::cout << "solver options file " << solver_options_file << std::endl;
  drake::solvers::SolverOptions solver_options =
      drake::yaml::LoadYamlFile<c3::SolverOptionsFromYaml>(solver_options_file)
          .GetAsSolverOptions(drake::solvers::OsqpSolver::id());
  SetSolverOptions(solver_options);
}

} // namespace systems
} // namespace c3
