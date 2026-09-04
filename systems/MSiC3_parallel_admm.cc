#include "MSiC3_parallel_admm.h"

#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>

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

MSiC3ParallelAdmm::MSiC3ParallelAdmm(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<vector<SortedPair<GeometryId>>>& contact_groups,
    const vector<vector<SortedPair<GeometryId>>>& contact_groups_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options,
    HybridMpcOptions mpc_options, int example_idx, bool is_optuna)
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
    } else {
      if (controller_options_.lcs_factory_options.mu.empty()) {
        controller_options_.lcs_factory_options.mu.resize(contact_geoms_.size(), 0.3);
      } else {
        controller_options_.lcs_factory_options.mu.resize(
            contact_geoms_.size(), controller_options_.lcs_factory_options.mu[0]);
      }
    }
  } else {
    controller_options_.lcs_factory_options.num_contacts = contact_geoms_.size();
    if (controller_options_.lcs_factory_options.mu.empty()) {
      controller_options_.lcs_factory_options.mu.resize(contact_geoms_.size(), 0.3);
    } else if (controller_options_.lcs_factory_options.mu.size() == 1) {
      controller_options_.lcs_factory_options.mu.resize(
          contact_geoms_.size(), controller_options_.lcs_factory_options.mu[0]);
    }
  }

  n_q_ = plant_.num_positions();
  n_v_ = plant_.num_velocities();
  n_x_ = n_q_ + n_v_;
  n_u_ = plant_.num_actuators();
  n_lambda_ = multibody::LCSFactory::GetNumContactVariables(
      controller_options_.lcs_factory_options);
  dt_ = controller_options_.lcs_factory_options.dt;

  num_segments_ = ms_ic3_options_.num_segments;
  L_ = N_ / num_segments_;

  for (int i = 0; i < N_ + 1; i++) {
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

MSiC3ParallelAdmm::MSiC3ParallelAdmm(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<vector<SortedPair<GeometryId>>>& contact_groups,
    const vector<vector<SortedPair<GeometryId>>>& contact_groups_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options,
    int example_idx, bool is_optuna)
    : MSiC3ParallelAdmm(plant, plant_ad, plant_rollout, plant_ad_rollout, rollout_diagram,
                        std::move(rollout_diagram_context), contact_groups, contact_groups_rollout,
                        controller_options, ms_ic3_options, HybridMpcOptions{}, example_idx, is_optuna) {}

MSiC3ParallelAdmm::MSiC3ParallelAdmm(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const vector<SortedPair<GeometryId>>& contact_geoms_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options,
    HybridMpcOptions mpc_options, int example_idx)
    : MSiC3ParallelAdmm(plant, plant_ad, plant_rollout, plant_ad_rollout, rollout_diagram,
                        std::move(rollout_diagram_context),
                        vector<vector<SortedPair<GeometryId>>>{contact_geoms},
                        vector<vector<SortedPair<GeometryId>>>{contact_geoms_rollout},
                        controller_options, ms_ic3_options, mpc_options, example_idx) {}

MSiC3ParallelAdmm::MSiC3ParallelAdmm(
    const MultibodyPlant<double>& plant,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    const MultibodyPlant<double>& plant_rollout,
    const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    drake::systems::Diagram<double>& rollout_diagram,
    std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,
    const vector<SortedPair<GeometryId>>& contact_geoms,
    const vector<SortedPair<GeometryId>>& contact_geoms_rollout,
    C3ControllerOptions controller_options, MSiC3Options ms_ic3_options,
    int example_idx)
    : MSiC3ParallelAdmm(plant, plant_ad, plant_rollout, plant_ad_rollout, rollout_diagram,
                        std::move(rollout_diagram_context),
                        vector<vector<SortedPair<GeometryId>>>{contact_geoms},
                        vector<vector<SortedPair<GeometryId>>>{contact_geoms_rollout},
                        controller_options, ms_ic3_options, HybridMpcOptions{}, example_idx) {}

void MSiC3ParallelAdmm::ResolveContacts(
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
        int n_active = (g < res_list.size())
            ? std::min(res_list[g], static_cast<int>(contact_groups_[g].size()))
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
            contact_geoms_.size(), controller_options_.lcs_factory_options.mu[0]);
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
  }
}

void MSiC3ParallelAdmm::SetSolverOptions(const drake::solvers::SolverOptions& solver_options) {
  solver_options_ = solver_options;
  for (auto& c3_ptr : c3_trackings_) {
    if (c3_ptr) {
      c3_ptr->SetSolverOptions(solver_options);
    }
  }
}

void MSiC3ParallelAdmm::SetSolverOptions(const std::string& solver_options_file) {
  std::cout << "solver options file " << solver_options_file << std::endl;
  drake::solvers::SolverOptions solver_options =
      drake::yaml::LoadYamlFile<c3::SolverOptionsFromYaml>(solver_options_file)
          .GetAsSolverOptions(drake::solvers::OsqpSolver::id());
  SetSolverOptions(solver_options);
}

tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>> MSiC3ParallelAdmm::ComputeTrajectory(
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

  DRAKE_DEMAND(N_ % num_segments_ == 0);
  L_ = N_ / num_segments_;

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

  if (example_idx_ == 0) { // plate
    A_x(0, 0) = 1; A_x(1, 1) = 1; A_x(2, 2) = 1; A_x(3, 3) = 1; A_x(4, 4) = 1;
    lower_bound_x(0) = -0.08; lower_bound_x(1) = -0.08; lower_bound_x(2) = x0(2) - 0.1;
    lower_bound_x(3) = -0.8;  lower_bound_x(4) = -0.8;
    upper_bound_x(0) = 0.08;  upper_bound_x(1) = 0.08;  upper_bound_x(2) = x0(2) + 0.1;
    upper_bound_x(3) = 0.8;   upper_bound_x(4) = 0.8;

    A_u(0, 0) = 1; A_u(1, 1) = 1; A_u(2, 2) = 1; A_u(3, 3) = 1; A_u(4, 4) = 1;
    lower_bound_u(0) = -1; lower_bound_u(1) = -1; lower_bound_u(2) = 9.81 * 0.85 - 3.2;
    lower_bound_u(3) = (plate_u_torque_bound == -1) ? -2.4 : -plate_u_torque_bound;
    lower_bound_u(4) = (plate_u_torque_bound == -1) ? -2.4 : -plate_u_torque_bound;
    upper_bound_u(0) = 1; upper_bound_u(1) = 1; upper_bound_u(2) = 9.81 * 0.85 + 3.2;
    upper_bound_u(3) = (plate_u_torque_bound == -1) ? 2.4 : plate_u_torque_bound;
    upper_bound_u(4) = (plate_u_torque_bound == -1) ? 2.4 : plate_u_torque_bound;

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

  }

  VectorXd gravity(VectorXd::Zero(n_u_));
  if (example_idx_ == 0) {
    gravity(2) = 9.81 * 0.85;
  }

  // Trajectory storage across ADMM iterations
  vector<MatrixXd> x_traj;
  vector<MatrixXd> u_traj;
  vector<MatrixXd> lambda_traj;

  MatrixXd x_hat(MatrixXd::Zero(n_x_, N_+1));
  MatrixXd u_hat(MatrixXd::Zero(n_u_, N_));
  MatrixXd lambda_hat(MatrixXd::Zero(n_lambda_, N_));

  ResolveContacts(context, *contexts_rollout[0]);

  LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
      contact_geoms_, controller_options_.lcs_factory_options);
  LCSFactory lcs_factory_rollout(plant_rollout_, *contexts_rollout[0], plant_ad_rollout_,
      context_ad_rollout, contact_geoms_rollout_, controller_options_.lcs_factory_options);

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

  // Setup C3Plus instance for each segment
  c3_trackings_.resize(num_segments_);
  for (int s = 0; s < num_segments_; s++) {
    c3_trackings_[s] = std::make_unique<C3Plus>(
        lcs_temp, costs_temp, x_targets_temp, controller_options_.c3_options);

    if (solver_options_.has_value()) {
      c3_trackings_[s]->SetSolverOptions(*solver_options_);
    }

    if (ms_ic3_options_.add_position_constraints) {
      c3_trackings_[s]->AddLinearConstraint(A_x, lower_bound_x, upper_bound_x,
                                            ConstraintVariable::STATE);
    }

    if (ms_ic3_options_.add_input_constraints) {
      c3_trackings_[s]->AddLinearConstraint(A_u, lower_bound_u, upper_bound_u,
                                            ConstraintVariable::INPUT);
    }

    if (ms_ic3_options_.penalize_acceleration) {
      c3_trackings_[s]->AddAccelerationCost(n_q_, n_v_, ms_ic3_options_.acceleration_cost_weight);
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

  // 1. Set initial guess trajectory (linear interpolation in x, gravity in u, sim rollout in lambda)
  VectorXd x_diff = xd - x0;
  for (int k = 0; k < N_+1; k++) {
    x_hat.col(k) = x0 + k * x_diff / (N_);
    
    // Interpolate quaternions correctly
    for (auto idx : controller_options_.quaternion_indices) {
      double rotation = (double)k / (N_);

      Eigen::Quaterniond q0(x0(idx), x0(idx+1), x0(idx+2), x0(idx+3));
      Eigen::Quaterniond qd_q(xd(idx), xd(idx+1), xd(idx+2), xd(idx+3));
      VectorXd v0 = x0.segment(idx, 4);
      VectorXd vd = xd.segment(idx, 4);

      if (v0.dot(vd) < 0) {
        vd = -vd;
        qd_q = Eigen::Quaterniond(vd(0), vd(1), vd(2), vd(3));
      }

      if (-1e-3 < q0.dot(qd_q) && q0.dot(qd_q) < 1e-3) {
        Eigen::Vector4d mid = v0 + vd;
        Eigen::Vector4d tangent = (mid - mid.dot(v0) * v0).normalized();
        double theta = std::acos(std::clamp(v0.dot(vd), -1.0, 1.0));
        Eigen::Vector4d v_interpolated = v0 * std::cos(rotation * theta) + tangent * std::sin(rotation * theta);
        x_hat.col(k).segment(idx, 4) = v_interpolated;
      } else {
        if (example_idx_ == 0) {
          Eigen::Quaterniond slerp = slerpLong(q0, qd_q, rotation);
          x_hat.col(k).segment(idx, 4) << slerp.w(), slerp.x(), slerp.y(), slerp.z();
        } else {
          Eigen::Quaterniond slerp = q0.slerp(rotation, qd_q);
          x_hat.col(k).segment(idx, 4) << slerp.w(), slerp.x(), slerp.y(), slerp.z();
        }
      }
    }

    x_hat.col(k) = ProjectFeasible(x_hat.col(k), context, A_x, lower_bound_x, upper_bound_x);

    if (k < N_) {
      u_hat.col(k) = gravity;
    }
  }

  // Get lambda_hat initial guess from simulator
  if (use_drake_sim_) {
    Context<double>& root_context = simulators_[0]->get_mutable_context();
    Context<double>& sim_plant_context = rollout_diagram_.GetMutableSubsystemContext(
        plant_rollout_, &root_context);
    VectorXd x_curr = x0;
    for (int i = 0; i < N_; i++) {
      plant_rollout_.SetPositionsAndVelocities(&sim_plant_context, x_curr);
      plant_rollout_.get_actuation_input_port().FixValue(&sim_plant_context, u_hat.col(i));

      double target_time = root_context.get_time() + dt_;
      simulators_[0]->AdvanceTo(target_time);

      x_curr = plant_rollout_.GetPositionsAndVelocities(sim_plant_context);
      const auto& contact_results =
          plant_rollout_.get_contact_results_output_port()
              .Eval<ContactResults<double>>(sim_plant_context);
      auto [sim_lambda, sim_gamma, sim_in_contact] =
          ConstructLambdasFromContactResults(contact_results,
                                             controller_options_.lcs_factory_options.contact_model);
      lambda_hat.col(i) = sim_lambda;
    }
  }

  // Store initial guess at index 0
  x_traj.push_back(x_hat);
  u_traj.push_back(u_hat);
  lambda_traj.push_back(lambda_hat);

  // 2. Define consensus anchors x_bar at segment boundaries
  // x_bar[i] is the consensus state at the end of segment i
  vector<VectorXd> x_bar(num_segments_);
  for (int i = 0; i < num_segments_; i++) {
    if (i == num_segments_ - 1) {
      x_bar[i] = xd;
    } else {
      x_bar[i] = ProjectFeasible(x_hat.col((i + 1) * L_), context, A_x, lower_bound_x, upper_bound_x);
    }
  }

  // 3. Initialize ADMM dual variables y (end coupling) to 0
  vector<VectorXd> y(num_segments_, VectorXd::Zero(n_x_));

  // Initialize x_k_end to the last element of the kth segment in x_hat
  vector<VectorXd> x_k_end(num_segments_);
  for (int i = 0; i < num_segments_; i++) {
    x_k_end[i] = x_hat.col((i + 1) * L_);
  }

  // Note: ADMM penalty parameter rho can be configured via ms_ic3_options.anchor_rho or defaults to 1.0
  double rho = (ms_ic3_options_.anchor_rho.has_value() && ms_ic3_options_.anchor_rho.value() > 0)
      ? ms_ic3_options_.anchor_rho.value() : 1.0;

  int num_iters = ms_ic3_options_.num_iters;
  int num_threads = ms_ic3_options_.num_threads.value_or(1);

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

  // =========================================================================
  // 4. Main Parallel ADMM Loop
  // =========================================================================
  for (int iter = 0; iter < num_iters; iter++) {
    auto iter_start = std::chrono::high_resolution_clock::now();
    std::cout << "iC3 iteration " << (iter + 1) << std::endl;

    // Segment start states a_i and consensus targets b_i:
    // a_i: start state for segment i (x0 for first segment, x_bar[i-1] for subsequent)
    // b_i: consensus target for segment i (x_bar[i] - y[i])
    vector<VectorXd> a(num_segments_);
    vector<VectorXd> b(num_segments_);

    for (int i = 0; i < num_segments_; i++) {
      if (i == 0) {
        a[0] = x0; // First segment start is ALWAYS fixed at x0
      } else {
        a[i] = x_bar[i - 1];
      }

      for (int m = 0; m < n_x_; m++) {
        if (!std::isfinite(a[i](m))) {
          a[i](m) = (i == 0) ? x0(m) : xd(m);
        }
      }

      // Prevent cube from drifting too much (kinda hacky)
      if (example_idx_ == 1) {
        a[i](13) = std::clamp(a[i](13), -0.03, 0.03);
        a[i](14) = std::clamp(a[i](14), -0.03, 0.03);
      }

      a[i] = ProjectFeasible(a[i], context, A_x, lower_bound_x, upper_bound_x);

      b[i] = x_bar[i] - y[i];
      for (int m = 0; m < n_x_; m++) {
        if (!std::isfinite(b[i](m))) {
          b[i](m) = xd(m);
        }
      }
    }

    // Step 1: Local subproblem for segment i (parallel over segments)
    vector<MatrixXd> seg_x_hat_out(num_segments_);
    vector<MatrixXd> seg_u_hat_out(num_segments_);
    vector<MatrixXd> seg_lambda_hat_out(num_segments_);
    vector<vector<MatrixXd>> seg_J_policy_out(num_segments_);
    vector<vector<StepAdmmMetrics>> seg_admm_metrics_out(num_segments_);
    vector<VectorXd> p_k(num_segments_, VectorXd::Zero(n_x_));

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_segments_; i++) {
      VectorXd x0_seg = a[i];
      VectorXd x_end_target = b[i];

      auto [x_out, u_out, lambda_out, J_pol, res_out] = DoC3Rollout(
          x0_seg, x_hat, u_hat.middleCols(i * L_, L_), lambda_hat,
          gravity, x_end_target, lcs_factory, lcs_factory_rollout,
          i * L_, i, A_x, lower_bound_x, upper_bound_x,
          A_u, lower_bound_u, upper_bound_u,
          context, *contexts_rollout[i], *c3_trackings_[i], rho);

      seg_x_hat_out[i] = x_out;
      seg_u_hat_out[i] = u_out;
      seg_lambda_hat_out[i] = lambda_out;
      seg_J_policy_out[i] = J_pol;
      seg_admm_metrics_out[i] = res_out;
      x_k_end[i] = x_out.col(L_);

      // Step 2: Compute p_k for segment i
      // 2b: Make time-varying LCS for segment i about (x_out, u_out, lambda_out)
      LCS lcs_seg;
      #pragma omp critical
      {
        lcs_seg = MakeTimeVaryingLCS(x_out, u_out, lambda_out, lcs_factory);
      }

      // 2c: Compute linearized dynamics derivatives f_x[t], f_u[t]
      const auto& A_lcs = lcs_seg.A();
      const auto& B_lcs = lcs_seg.B();
      const auto& D_lcs = lcs_seg.D();
      const auto& E_lcs = lcs_seg.E();
      const auto& F_lcs = lcs_seg.F();
      const auto& H_lcs = lcs_seg.H();

      vector<MatrixXd> A_cl(L_);
      vector<VectorXd> g_vec(L_);
      double active_tol = 1e-4;

      MatrixXd Q_mat = controller_options_.c3_options.Q;
      MatrixXd R_mat = controller_options_.c3_options.R;

      for (int t = 0; t < L_; t++) {
        std::vector<int> active;
        for (int j = 0; j < n_lambda_; j++) {
          if (lambda_out.col(t)(j) > active_tol) {
            active.push_back(j);
          }
        }

        MatrixXd f_x, f_u;
        if (active.empty()) {
          f_x = A_lcs[t];
          f_u = B_lcs[t];
        } else {
          int na = static_cast<int>(active.size());
          MatrixXd D_a(n_x_, na);
          MatrixXd E_a(na, n_x_);
          MatrixXd H_a(na, n_u_);
          MatrixXd F_aa(na, na);

          for (int a = 0; a < na; a++) {
            int ja = active[a];
            D_a.col(a) = D_lcs[t].col(ja);
            E_a.row(a) = E_lcs[t].row(ja);
            H_a.row(a) = H_lcs[t].row(ja);
            for (int cb = 0; cb < na; cb++) {
              F_aa(a, cb) = F_lcs[t](ja, active[cb]);
            }
          }

          MatrixXd F_reg = F_aa + 1e-6 * MatrixXd::Identity(na, na);
          Eigen::LDLT<MatrixXd> solver(F_reg);
          MatrixXd FinvE = solver.solve(E_a);
          MatrixXd FinvH = solver.solve(H_a);

          f_x = A_lcs[t] - D_a * FinvE;
          f_u = B_lcs[t] - D_a * FinvH;
        }

        // 2d: A[t] = f_x[t] + f_u[t] * J_policy[t]
        A_cl[t] = f_x + f_u * J_pol[t];

        // 2e: g[t] = 2 * Q * (x[t] - xd) + 2 * J_policy[t]^T * R * u[t]
        VectorXd x_t = x_out.col(t);
        VectorXd u_t = u_out.col(t);
        VectorXd x_err = x_t - xd;
        VectorXd g_x = 2.0 * Q_mat * x_err;
        for (int quat_idx : controller_options_.quaternion_indices) {
          VectorXd quat_curr = x_t.segment(quat_idx, 4);
          VectorXd quat_des = xd.segment(quat_idx, 4);
          VectorXd quat_grad = common::gradient_of_squared_quaternion_angle_difference(quat_curr, quat_des);
          g_x.segment(quat_idx, 4) = controller_options_.Q_quaternion_weight * quat_grad;
        }
        g_vec[t] = g_x + 2.0 * J_pol[t].transpose() * R_mat * u_t;
      }

      // 2f: Terminal costate nu[L] including downstream ADMM consensus coupling
      VectorXd x_end = x_out.col(L_);
      VectorXd x_end_err = x_end - xd;
      VectorXd nu = 2.0 * Q_mat * x_end_err;

      // Add terminal ADMM consensus gradient: d/dx_L (rho * ||x_L - b_i||^2)
      VectorXd consensus_err = x_end - b[i];
      for (int quat_idx : controller_options_.quaternion_indices) {
        if (x_end.segment(quat_idx, 4).dot(b[i].segment(quat_idx, 4)) < 0.0) {
          consensus_err.segment(quat_idx, 4) = x_end.segment(quat_idx, 4) + b[i].segment(quat_idx, 4);
        }
      }
      nu += 2.0 * rho * consensus_err;

      for (int quat_idx : controller_options_.quaternion_indices) {
        VectorXd quat_curr = x_end.segment(quat_idx, 4);
        VectorXd quat_des = xd.segment(quat_idx, 4);
        VectorXd quat_grad = common::gradient_of_squared_quaternion_angle_difference(quat_curr, quat_des);
        nu.segment(quat_idx, 4) += controller_options_.Q_quaternion_weight * quat_grad;

        // Project terminal costate onto tangent space of S^3
        Eigen::Vector4d q_e = quat_curr.normalized();
        nu.segment(quat_idx, 4) -= q_e * (q_e.dot(nu.segment(quat_idx, 4)));
      }

      double gamma_dyn = controller_options_.c3_options.gamma;
      for (int t = L_ - 1; t >= 0; t--) {
        nu = g_vec[t] + gamma_dyn * A_cl[t].transpose() * nu;
        // Tangent projection along the trajectory to prevent radial drift
        for (int quat_idx : controller_options_.quaternion_indices) {
          Eigen::Vector4d q_t = x_out.col(t).segment(quat_idx, 4).normalized();
          nu.segment(quat_idx, 4) -= q_t * (q_t.dot(nu.segment(quat_idx, 4)));
        }
        if (nu.norm() > 1e4) {
          nu *= 1e4 / nu.norm();
        }
      }
      p_k[i] = nu;
    }

    // Update overall trajectory with parallel rollout outputs
    for (int i = 0; i < num_segments_; i++) {
      x_hat.middleCols(i * L_, L_) = seg_x_hat_out[i].leftCols(L_);
      u_hat.middleCols(i * L_, L_) = seg_u_hat_out[i].leftCols(L_);
      lambda_hat.middleCols(i * L_, L_) = seg_lambda_hat_out[i].leftCols(L_);

      for (int t = 0; t < L_; t++) {
        int global_t = i * L_ + t;
        const auto& metrics = seg_admm_metrics_out[i][t];
        for (int k = 0; k < static_cast<int>(metrics.lambda_res.size()); ++k) {
            double step_val = (k < static_cast<int>(metrics.step_changes.size())) ? metrics.step_changes[k] : 0.0;
            double comp_val = (k < static_cast<int>(metrics.comp_slacks.size())) ? metrics.comp_slacks[k] : 0.0;
            lambda_residual_records.push_back({iter + 1, i, global_t, k, metrics.lambda_res[k], step_val, comp_val});
        }
        c3_plan_step_records.push_back({iter + 1, i, global_t, metrics.x0, metrics.x1, metrics.lambda, metrics.eta, metrics.lambda_last, metrics.eta_last});
        for (size_t k = 0; k < metrics.full_x_lookahead.size(); ++k) {
          c3_full_lookahead_records.push_back({iter + 1, i, global_t, static_cast<int>(k), metrics.full_x_lookahead[k]});
        }
      }
    }
    x_hat.col(N_) = seg_x_hat_out[num_segments_ - 1].col(L_);

    // Defect Logging between consecutive segments
    for (int i = 0; i < num_segments_ - 1; i++) {
      VectorXd x_i_end = seg_x_hat_out[i].col(L_);
      VectorXd x_next_start = seg_x_hat_out[i + 1].col(0);
      VectorXd defect = x_i_end - x_next_start;

      VectorXd ee_defect;
      VectorXd object_defect;
      if (example_idx_ == 0) {
        ee_defect = defect.segment(0, 5);
        object_defect = defect.segment(9, 3);
      } else if (example_idx_ == 1 || example_idx_ == 2) {
        ee_defect = defect.segment(0, 9);
        object_defect = defect.segment(13, 3);
      }

      std::cout << "Segment " << i << " ee defect: " << ee_defect.transpose() << std::endl;
      double quat_defect_deg = 0.0;
      for (auto quat_idx : controller_options_.quaternion_indices) {
        Eigen::Quaterniond q_end(x_i_end(quat_idx), x_i_end(quat_idx+1), x_i_end(quat_idx+2), x_i_end(quat_idx+3));
        Eigen::Quaterniond q_start(x_next_start(quat_idx), x_next_start(quat_idx+1), x_next_start(quat_idx+2), x_next_start(quat_idx+3));
        quat_defect_deg = q_end.angularDistance(q_start) * 180.0 / M_PI;
        std::cout << "Segment " << i << " quaternion defect (deg): " << quat_defect_deg << std::endl;
      }
      std::cout << "Segment " << i << " object defect: " << object_defect.transpose() << std::endl;

      defect_records.push_back({iter + 1, i, ee_defect.norm(), quat_defect_deg, object_defect.norm()});
    }

    // Step 3: Curvature-Regularized Proximal Consensus Update
    for (int i = 0; i < num_segments_ - 1; i++) {
      if (!p_k[i + 1].allFinite()) {
        p_k[i + 1] = VectorXd::Zero(n_x_);
      }

      // 1. Ensure segment end and consensus share the same quaternion hemisphere
      for (int quat_idx : controller_options_.quaternion_indices) {
        if (x_k_end[i].segment(quat_idx, 4).dot(x_bar[i].segment(quat_idx, 4)) < 0.0) {
          x_k_end[i].segment(quat_idx, 4) *= -1.0;
        }
      }

      VectorXd xbar_old = x_bar[i];
      VectorXd xbar_new = xbar_old;

      // -----------------------------------------------------------------------
      // A. Non-Quaternion States (Positions, Velocities)
      // -----------------------------------------------------------------------
      for (int m = 0; m < n_x_; m++) {
        bool is_quat = false;
        for (int quat_idx : controller_options_.quaternion_indices) {
          if (m >= quat_idx && m < quat_idx + 4) {
            is_quat = true;
            break;
          }
        }
        if (!is_quat) {
          // Estimated Hessian curvature along horizon: H_m ≈ 2 * L * Q_mm
          double q_cost = (m < controller_options_.c3_options.Q.rows()) ? controller_options_.c3_options.Q(m, m) : 1.0;
          double H_m = 2.0 * L_ * q_cost;

          // Proximal update balancing segment i defect against segment i+1 sensitivity
          double delta_m = (rho * (x_k_end[i](m) + y[i](m) - xbar_old(m)) - p_k[i + 1](m)) / (rho + H_m);
          xbar_new(m) = xbar_old(m) + delta_m;
        }
      }

      // -----------------------------------------------------------------------
      // B. Quaternion States (Lie Manifold Retraction)
      // -----------------------------------------------------------------------
      for (int quat_idx : controller_options_.quaternion_indices) {
        Eigen::Vector4d q_old = xbar_old.segment(quat_idx, 4).normalized();
        Eigen::Vector4d q_end = x_k_end[i].segment(quat_idx, 4);

        // Geodesic defect in the tangent space of S^3 at q_old: (I - q q^T) * (q_end - q_old)
        Eigen::Vector4d v_defect = q_end - q_old * (q_old.dot(q_end));

        // Tangent projection of dual variable y
        Eigen::Vector4d y_quat = y[i].segment(quat_idx, 4);
        Eigen::Vector4d v_dual = y_quat - q_old * (q_old.dot(y_quat));

        // Tangent projection of sensitivity p_k[i+1]
        Eigen::Vector4d p_quat = p_k[i + 1].segment(quat_idx, 4);
        Eigen::Vector4d v_sens = p_quat - q_old * (q_old.dot(p_quat));

        // Curvature of quaternion trajectory cost: H_quat ≈ 8 * (L + 1) * Q_quaternion_weight
        double H_quat = 8.0 * (L_ + 1) * controller_options_.Q_quaternion_weight;

        // Intrinsic tangent step vector (Δq ∈ T_q S^3)
        Eigen::Vector4d v_step = (rho * (v_defect + v_dual) - v_sens) / (rho + H_quat);

        // Retraction back to S^3
        Eigen::Vector4d q_new = (q_old + v_step).normalized();

        // Maintain hemisphere consistency with nominal goal
        if (q_new.dot(xd.segment(quat_idx, 4)) < 0.0) {
          q_new *= -1.0;
        }

        xbar_new.segment(quat_idx, 4) = q_new;
      }

      std::cout << "step " << i << " " << (xbar_new - xbar_old).transpose() << std::endl;

      // Prevent cube drift on consensus state
      if (example_idx_ == 1) {
        xbar_new(13) = std::clamp(xbar_new(13), -0.03, 0.03);
        xbar_new(14) = std::clamp(xbar_new(14), -0.03, 0.03);
      }

      // Project consensus state to feasible contact manifold
      x_bar[i] = ProjectFeasible(xbar_new, context, A_x, lower_bound_x, upper_bound_x);
    }
    x_bar[num_segments_ - 1] = xd;

    // Step 4: Dual updates y_k = y_k + x_k[end] - xbar_k
    // Dual updates with antipodal check and tangent space projection
    for (int i = 0; i < num_segments_; i++) {
      for (int quat_idx : controller_options_.quaternion_indices) {
        if (x_k_end[i].segment(quat_idx, 4).dot(x_bar[i].segment(quat_idx, 4)) < 0.0) {
          x_k_end[i].segment(quat_idx, 4) *= -1.0;
        }
      }

      y[i] += (x_k_end[i] - x_bar[i]);

      for (int quat_idx : controller_options_.quaternion_indices) {
        Eigen::Vector4d q_b = x_bar[i].segment(quat_idx, 4).normalized();
        y[i].segment(quat_idx, 4) -= q_b * (q_b.dot(y[i].segment(quat_idx, 4)));
      }

      for (int m = 0; m < n_x_; m++) {
        if (!std::isfinite(y[i](m))) {
          y[i](m) = 0.0;
        }
      }
    }

    // Print updated consensus states for each segment
    std::cout << "--- Updated Consensus States x_bar (first " << n_q_ << " terms) ---" << std::endl;
    for (int i = 0; i < num_segments_; i++) {
      std::cout << "Segment " << i << " x_bar (q): " << x_bar[i].head(n_q_).transpose() << std::endl;
    }

    // Save snapshot of current iteration
    x_traj.push_back(x_hat);
    u_traj.push_back(u_hat);
    lambda_traj.push_back(lambda_hat);

    auto iter_end = std::chrono::high_resolution_clock::now();
    double iter_runtime = std::chrono::duration<double>(iter_end - iter_start).count();
    std::cout << "Iteration runtime: " << iter_runtime << " seconds\n" << std::endl;
  }

  // Export defects to CSV
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
          for (const auto& rec : defect_records) {
            file << rec.iter << "," << rec.segment << ","
                 << rec.ee_defect_norm << "," << rec.quat_defect_deg << ","
                 << rec.object_defect_norm << "\n";
          }
          file.close();
        }
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to write defects to " << path << ": " << e.what() << std::endl;
      }
    }
    std::cout << "Saved defect data to defects.csv (" << defect_records.size() << " records)\n";

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
  double total_runtime = std::chrono::duration<double>(end_total - start_total).count();
  std::cout << "Total runtime: " << total_runtime << " seconds\n" << std::endl;

  return {x_traj, u_traj, lambda_traj};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, vector<MatrixXd>, vector<MSiC3ParallelAdmm::StepAdmmMetrics>> MSiC3ParallelAdmm::DoC3Rollout(
    VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, 
    MatrixXd lambda_hat, VectorXd ud, VectorXd x_boundary_target,
    LCSFactory factory, LCSFactory rollout_factory,
    int start_idx, int segment_idx,                                          
    MatrixXd A_x, VectorXd lb_x, VectorXd ub_x,
    MatrixXd A_u, VectorXd lb_u, VectorXd ub_u,
    drake::systems::Context<double>& context, drake::systems::Context<double>& context_rollout,
    c3::C3Plus& c3_tracking, double rho) {

  int num_steps = u_hat.cols();
  int factor = ms_ic3_options_.rollout_dt_scaling;

  MatrixXd x_hat_output(n_x_, num_steps * factor + 1);
  MatrixXd lambda_hat_out(n_lambda_, num_steps * factor);
  MatrixXd u_hat_fb(n_u_, num_steps * factor);
  vector<MatrixXd> J_policy(num_steps);
  vector<StepAdmmMetrics> admm_metrics_per_step(num_steps);

  x_hat_output.col(0) = x0;
  VectorXd x_curr = x0;
  VectorXd x_next;

  std::vector<double> x_des = controller_options_.x_des.value();
  VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());

  int tracking_N = controller_options_.lcs_factory_options.N;

  // Run receding horizon C3 MPC for local subproblem
  for (int t = 0; t < num_steps; t++) {

    vector<MatrixXd> Q;
    vector<MatrixXd> R;
    vector<MatrixXd> G;
    vector<MatrixXd> U;
    vector<VectorXd> x_targets_shortened;
    vector<VectorXd> x_des_shortened;
    vector<VectorXd> x_admm_shortened;
    vector<VectorXd> u_targets_shortened;
    MatrixXd x_hat_for_lcs(MatrixXd::Zero(n_x_, tracking_N + 1));
    MatrixXd u_hat_for_lcs(MatrixXd::Zero(n_u_, tracking_N));
    MatrixXd lambda_hat_for_lcs(MatrixXd::Zero(n_lambda_, tracking_N));

    double discount_factor = 1.0;
    for (int i = 0; i < tracking_N + 1; i++) {
      int x_idx = std::min(N_, start_idx + t + i);
      int u_idx = std::min(num_steps - 1, t + i);
      int R_idx = std::min(N_ - 1, start_idx + t + i);

      // Base C3 task cost matrix
      MatrixXd Q_base = discount_factor * Q_[x_idx];
      MatrixXd Q_total = Q_base;
      VectorXd x_target = xd;

      // ADMM consensus penalty only on terminal stage i == tracking_N:
      // J(x, u) + (rho/2) ||x_N - xbar_k + y_k||^2
      if (i == tracking_N) {
        VectorXd x_admm = x_boundary_target;
        MatrixXd Q_admm = rho * MatrixXd::Identity(n_x_, n_x_);
        Q_total = Q_base + Q_admm;

        for (int m = 0; m < n_x_; m++) {
          double w_tot = Q_total(m, m);
          if (w_tot > 1e-12) {
            x_target(m) = (Q_base(m, m) * xd(m) + Q_admm(m, m) * x_admm(m)) / w_tot;
          }
        }

        for (int quat_idx : controller_options_.quaternion_indices) {
          double w_base_q = discount_factor * controller_options_.Q_quaternion_weight;
          double w_admm_q = rho;
          double w_tot_q = w_base_q + w_admm_q;
          if (w_tot_q > 1e-12) {
            double alpha_q = w_admm_q / w_tot_q;
            Eigen::Quaterniond q_d(xd(quat_idx), xd(quat_idx+1), xd(quat_idx+2), xd(quat_idx+3));
            Eigen::Quaterniond q_a(x_admm(quat_idx), x_admm(quat_idx+1), x_admm(quat_idx+2), x_admm(quat_idx+3));
            if (q_d.dot(q_a) < 0) {
              q_a = Eigen::Quaterniond(-q_a.w(), -q_a.x(), -q_a.y(), -q_a.z());
            }
            Eigen::Quaterniond q_t = q_d.slerp(alpha_q, q_a).normalized();
            x_target.segment(quat_idx, 4) << q_t.w(), q_t.x(), q_t.y(), q_t.z();
          }
        }
        x_admm_shortened.push_back(x_admm);
      } else {
        x_admm_shortened.push_back(xd);
      }

      x_targets_shortened.push_back(x_target);
      x_des_shortened.push_back(xd);
      Q.push_back(Q_total);

      x_hat_for_lcs.col(i) = x_curr;

      if (i < tracking_N) {
        u_targets_shortened.push_back(ud);
        u_hat_for_lcs.col(i) = u_hat.col(u_idx);
        R.push_back(discount_factor * R_[R_idx]);
        lambda_hat_for_lcs.col(i) = lambda_hat.col(R_idx);
        G.push_back(discount_factor * G_[R_idx]);
        U.push_back(discount_factor * U_[R_idx]);
      }
      discount_factor *= controller_options_.c3_options.gamma;
    }

    vector<MatrixXd> Q_updated = UpdateQuaternionCosts(x_curr, x_des_shortened, x_admm_shortened, Q, rho);
    C3::CostMatrices costs(Q_updated, R, G, U);

    LCS lcs;
    #pragma omp critical
    {
      lcs = MakeTimeVaryingLCSWithEE(x_hat_for_lcs, u_hat_for_lcs, lambda_hat_for_lcs, factory, x_curr.segment(0, n_u_), 0);
    }

    VectorXd x_temp = x_curr;
    for (int i = 1; i < tracking_N + 1; i++) {
      VectorXd u_nominal = u_hat_for_lcs.col(i - 1);
      x_temp = lcs.SimulateAtTimestep(x_temp, u_nominal, true, i - 1);
      for (auto quat_idx : controller_options_.quaternion_indices) {
        double norm = x_temp.segment(quat_idx, 4).norm();
        x_targets_shortened[i].segment(quat_idx, 4) *= norm;
      }
    }

    c3_tracking.UpdateCostMatrices(costs);
    c3_tracking.UpdateLCS(lcs);
    c3_tracking.UpdateTarget(x_targets_shortened);
    c3_tracking.UpdateInputTarget(u_targets_shortened);
    c3_tracking.SetPenalizeChange(false);

    c3_tracking.Solve(x_curr);
    admm_metrics_per_step[t].lambda_res = c3_tracking.GetLambdaMinusDeltaLambdaAvgHorizonNorms();
    admm_metrics_per_step[t].step_changes = c3_tracking.GetIterateStepChangeNorms();
    admm_metrics_per_step[t].comp_slacks = c3_tracking.GetComplementaritySlackness();

    vector<Eigen::VectorXd> z_sol = c3_tracking.GetFullSolution();
    admm_metrics_per_step[t].x0 = z_sol[0].segment(0, n_x_);
    admm_metrics_per_step[t].x1 = (z_sol.size() > 1) ? VectorXd(z_sol[1].segment(0, n_x_)) : VectorXd(z_sol[0].segment(0, n_x_));
    admm_metrics_per_step[t].lambda = (z_sol[0].size() >= n_x_ + n_lambda_) ? VectorXd(z_sol[0].segment(n_x_, n_lambda_)) : VectorXd::Zero(n_lambda_);
    admm_metrics_per_step[t].eta = (z_sol[0].size() >= n_x_ + 2 * n_lambda_ + n_u_)
                                   ? VectorXd(z_sol[0].segment(n_x_ + n_lambda_ + n_u_, n_lambda_))
                                   : VectorXd::Zero(n_lambda_);
    size_t last_k = z_sol.size() > 0 ? z_sol.size() - 1 : 0;
    admm_metrics_per_step[t].lambda_last = (z_sol[last_k].size() >= n_x_ + n_lambda_) ? VectorXd(z_sol[last_k].segment(n_x_, n_lambda_)) : VectorXd::Zero(n_lambda_);
    admm_metrics_per_step[t].eta_last = (z_sol[last_k].size() >= n_x_ + 2 * n_lambda_ + n_u_)
                                        ? VectorXd(z_sol[last_k].segment(n_x_ + n_lambda_ + n_u_, n_lambda_))
                                        : VectorXd::Zero(n_lambda_);
    admm_metrics_per_step[t].full_x_lookahead.resize(z_sol.size());
    for (size_t k = 0; k < z_sol.size(); ++k) {
      admm_metrics_per_step[t].full_x_lookahead[k] = z_sol[k].segment(0, n_x_);
    }

    // Compute policy Jacobian J_policy = du0/dx0 from KKT sensitivity
    auto start_policy_jacobian = std::chrono::high_resolution_clock::now();
    J_policy[t] = c3_tracking.ComputePolicyJacobian();
    auto end_policy_jacobian = std::chrono::high_resolution_clock::now();
    double policy_jacobian_runtime = std::chrono::duration<double>(end_policy_jacobian - start_policy_jacobian).count();
    // std::cout << "Policy Jacobian runtime: " << policy_jacobian_runtime << " seconds\n" << std::endl;

    VectorXd c3_u = z_sol[0].segment(n_x_ + n_lambda_, n_u_);
    VectorXd c3_x = z_sol[0].segment(0, n_x_);
    VectorXd c3_x_next = (z_sol.size() > 1) ? z_sol[1].segment(0, n_x_) : c3_x;

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
        VectorXd c3_x_tracking = (i * c3_x_next + (factor - i) * c3_x) / factor;
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

        VectorXd c3_x_tracking = (i * c3_x_next + (factor - i) * c3_x) / factor;
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
        u_hat_fb.col(factor * t + i) = u_tracking;
        lambda_hat_out.col(factor * t + i) = pair.second;

        x_curr = x_next;
      }
    }
  }

  MatrixXd x_hat_downsampled(MatrixXd::Zero(n_x_, num_steps + 1));
  MatrixXd lambda_hat_downsampled(MatrixXd::Zero(n_lambda_, num_steps));
  MatrixXd u_hat_fb_downsampled(MatrixXd::Zero(n_u_, num_steps));

  for (int i = 0; i < num_steps; i++) {
    x_hat_downsampled.col(i) = x_hat_output.col(i * factor);
    u_hat_fb_downsampled.col(i) = u_hat_fb.col(i * factor);
    lambda_hat_downsampled.col(i) = lambda_hat_out.col(i * factor);
  }
  x_hat_downsampled.col(num_steps) = x_hat_output.col(num_steps * factor);

  return std::make_tuple(x_hat_downsampled, u_hat_fb_downsampled, lambda_hat_downsampled, J_policy, admm_metrics_per_step);
}

VectorXd MSiC3ParallelAdmm::ProjectFeasible(
    const VectorXd& x_in, drake::systems::Context<double>& context,
    const MatrixXd& A_x, const VectorXd& lb_x, const VectorXd& ub_x) {
  VectorXd x_projected = x_in;

  // Normalize quaternions first
  for (auto quat_idx : controller_options_.quaternion_indices) {
    x_projected.segment(quat_idx, 4) = x_projected.segment(quat_idx, 4).normalized();
  }

  if (example_idx_ == 0) {
    drake::geometry::GeometryId plate_geom =
        plant_.GetCollisionGeometriesForBody(plant_.GetBodyByName("plate"))[0];
    drake::geometry::GeometryId cube_geom =
        plant_.GetCollisionGeometriesForBody(plant_.GetBodyByName("cube"))[0];
    SortedPair<GeometryId> cube_plate(plate_geom, cube_geom);
    x_projected = ProjectContactPlate(context, cube_plate, x_projected, 2, 11, A_x, lb_x, ub_x);
  } else if (example_idx_ == 1) {
    for (int j = 0; j < 3; j++) {
      x_projected = ProjectContact(context, contact_geoms_[j], x_projected, 3*j, 3, A_x, lb_x, ub_x);
    }
  } else if (example_idx_ == 2) {
    drake::geometry::GeometryId cube_geom =
        plant_.GetCollisionGeometriesForBody(plant_.GetBodyByName("cube"))[0];
    drake::geometry::GeometryId ground_geom =
        plant_.GetCollisionGeometriesForBody(plant_.GetBodyByName("ground"))[0];
    SortedPair<GeometryId> cube_ground(cube_geom, ground_geom);
    x_projected = ProjectContactVertical(context, cube_ground, x_projected, 15, A_x, lb_x, ub_x);
    for (int j = 0; j < 3; j++) {
      x_projected = ProjectContact(context, contact_geoms_[j], x_projected, 3*j, 3, A_x, lb_x, ub_x);
    }
  }

  // Ensure C3 linear state constraints (A_x, lb_x, ub_x) are respected
  for (int i = 0; i < A_x.rows(); i++) {
    if (A_x(i, i) == 1) {
      x_projected(i) = std::clamp(x_projected(i), lb_x(i), ub_x(i));
    }
  }

  // Normalize quaternions again
  for (auto quat_idx : controller_options_.quaternion_indices) {
    x_projected.segment(quat_idx, 4) = x_projected.segment(quat_idx, 4).normalized();
  }

  return x_projected;
}

VectorXd MSiC3ParallelAdmm::ProjectContactVertical(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx,
                                MatrixXd A_x, VectorXd lb_x, VectorXd ub_x) {
    VectorXd x_out = x_init;
    for (auto quat_idx : controller_options_.quaternion_indices) {
      x_out.segment(quat_idx, 4) = x_out.segment(quat_idx, 4).normalized();
    }
    plant_.SetPositionsAndVelocities(&context, x_out);
    multibody::GeomGeomCollider collider(plant_, geom_pair);
    auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);

    double phi_check = phi;
    int counter = 0;
    while (phi_check < -3e-4 && counter < 50) {
      x_out(z_idx) += 0.002;
      if (A_x.rows() > z_idx && A_x(z_idx, z_idx) == 1) {
        x_out(z_idx) = std::clamp(x_out(z_idx), lb_x(z_idx), ub_x(z_idx));
      }
      for (auto quat_idx : controller_options_.quaternion_indices) {
        x_out.segment(quat_idx, 4) = x_out.segment(quat_idx, 4).normalized();
      }
      plant_.SetPositionsAndVelocities(&context, x_out);
      auto [phi_new, J_new] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);
      phi_check = phi_new;
      counter++;
    }
    x_out(z_idx) = x_out(z_idx) + 0.002;
    if (A_x.rows() > z_idx && A_x(z_idx, z_idx) == 1) {
      x_out(z_idx) = std::clamp(x_out(z_idx), lb_x(z_idx), ub_x(z_idx));
    }
    return x_out;
}

VectorXd MSiC3ParallelAdmm::ProjectContactPlate(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx, int pitch_idx, 
                                MatrixXd A_x, VectorXd lb_x, VectorXd ub_x) {
    VectorXd x_out = x_init;
    for (auto quat_idx : controller_options_.quaternion_indices) {
      x_out.segment(quat_idx, 4) = x_out.segment(quat_idx, 4).normalized();
    }
    plant_.SetPositionsAndVelocities(&context, x_out);
    multibody::GeomGeomCollider collider(plant_, geom_pair);
    auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);

    int counter = 0;
    double phi_check = phi;
    while (phi_check < -1e-4 && counter < 50) {
      x_out(z_idx) += 0.002;
      double displacement = 0.2 * (x_init(9) - x_init(0));
      x_out(pitch_idx) += displacement;
  
      for (int i = 0; i < n_q_; i++) {
        if (A_x(i, i) == 1) {
          x_out(i) = std::min(std::max(x_out(i), lb_x(i)), ub_x(i));
        }
      }

      for (auto quat_idx : controller_options_.quaternion_indices) {
        x_out.segment(quat_idx, 4) = x_out.segment(quat_idx, 4).normalized();
      }
      plant_.SetPositionsAndVelocities(&context, x_out);
      auto [phi_new, J_new] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);
      phi_check = phi_new;
      counter++;
    }
    x_out(z_idx) -= 0.002;
    return x_out;
}

VectorXd MSiC3ParallelAdmm::ProjectContact(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int start_idx, int q_size, MatrixXd A_x, VectorXd lb_x, VectorXd ub_x) {
    VectorXd x_out = x_init;
    plant_.SetPositionsAndVelocities(&context, x_init);
    multibody::GeomGeomCollider collider(plant_, geom_pair);
    auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);

    double phi_check = phi;
    VectorXd contact_normal = J.row(0).segment(start_idx, q_size);

    int counter = 0;
    while (phi_check < -3e-4 && counter < 5) {
      double norm_sq = contact_normal.squaredNorm();
      if (norm_sq < 1e-8) break;
      x_out.segment(start_idx, q_size) = 
        x_out.segment(start_idx, q_size) - 1.05 * (phi_check / norm_sq) * contact_normal;
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
    return x_out;
}

LCS MSiC3ParallelAdmm::MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, LCSFactory factory) {
  int N = u_hat.cols();
  vector<Eigen::MatrixXd> A;
  vector<Eigen::MatrixXd> B;
  vector<Eigen::MatrixXd> D;
  vector<Eigen::VectorXd> d;
  vector<Eigen::MatrixXd> E;
  vector<Eigen::MatrixXd> F;
  vector<Eigen::MatrixXd> H;
  vector<Eigen::VectorXd> c;

  for (int k = 0; k < N; k++) {
    for (auto idx : controller_options_.quaternion_indices) {
      x_hat.col(k).segment(idx, 4) = x_hat.col(k).segment(idx, 4).normalized();
    }
    factory.UpdateStateAndInput(x_hat.col(k), u_hat.col(k));
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
  return LCS(A, B, D, d, E, F, H, c, dt_);
}

LCS MSiC3ParallelAdmm::MakeTimeVaryingLCSWithEE(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, LCSFactory factory, VectorXd ee_pose, int ee_idx) {
  MatrixXd x_hat_copy = x_hat;
  for (int i = 0; i < x_hat_copy.cols(); i++) {
    x_hat_copy.col(i).segment(ee_idx, ee_pose.size()) = ee_pose;
  }
  return MakeTimeVaryingLCS(x_hat_copy, u_hat, lambda_hat, factory);
}

std::tuple<VectorXd, VectorXd, VectorXd> MSiC3ParallelAdmm::ConstructLambdasFromContactResults(
    drake::multibody::ContactResults<double> contact_results, std::string contact_model) {
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

      if ((geom_A == id_A && geom_B == id_B) || (geom_A == id_B && geom_B == id_A)) {
        bool is_swapped = (geom_A == id_B && geom_B == id_A);

        Vector3d n_W = is_swapped ? -pair.nhat_BA_W : pair.nhat_BA_W;
        Vector3d f_W = is_swapped ? info.contact_force() : -info.contact_force();

        in_contact(i) = 1;
        gamma(i) = info.slip_speed();

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
          lambda(i) = info.slip_speed();
          lambda(n_contacts + i) = f_n;
          lambda(2 * n_contacts + 4*i) = std::max(0.0, f_t1);
          lambda(2 * n_contacts + 4*i + 1) = std::max(0.0, -f_t1);
          lambda(2 * n_contacts + 4*i + 2) = std::max(0.0, f_t2);
          lambda(2 * n_contacts + 4*i + 3) = std::max(0.0, -f_t2);
        }
      }
    }
  }
  return {lambda, gamma, in_contact};
}

void MSiC3ParallelAdmm::UpdateQuaternionCosts(MatrixXd x_hat, VectorXd x_des) {
  Q_.clear();
  R_.clear();
  G_.clear();
  U_.clear();

  for (int i = 0; i < N_ + 1; i++) {
    Q_.push_back(controller_options_.c3_options.Q);
    if (i < N_) {
      R_.push_back(controller_options_.c3_options.R);
      G_.push_back(controller_options_.c3_options.G);
      U_.push_back(controller_options_.c3_options.U);
    }
  }  

  for (int i = 0; i < N_ + 1; i++) {
    for (int index : controller_options_.quaternion_indices) {
      Eigen::VectorXd quat_curr_i = x_hat.col(i).segment(index, 4).normalized();
      Eigen::VectorXd quat_des_i = x_des.segment(index, 4).normalized();

      Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);
      double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();

      Eigen::MatrixXd Q_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd Q_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
      Eigen::MatrixXd Q_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

      Q_[i].block(index, index, 4, 4) = 
        controller_options_.Q_quaternion_weight * (quat_hessian_i + Q_quat_regularizer_1 + Q_quat_regularizer_3);
    }
  }
}

vector<MatrixXd> MSiC3ParallelAdmm::UpdateQuaternionCosts(
    VectorXd x_curr,
    const vector<VectorXd>& x_des,
    const vector<VectorXd>& x_admm,
    const vector<MatrixXd>& Q_in,
    double rho) {
  vector<MatrixXd> Q = Q_in;
  double discount_factor = 1.0;
  size_t N_stages = Q.size();

  for (size_t i = 0; i < N_stages; i++) {
    for (int index : controller_options_.quaternion_indices) {
      Eigen::VectorXd quat_curr = x_curr.segment(index, 4).normalized();
      Eigen::VectorXd quat_des = x_des[i].segment(index, 4).normalized();

      // 1. Base task quaternion Hessian
      Eigen::MatrixXd quat_hessian_base =
          common::hessian_of_squared_quaternion_angle_difference(quat_curr, quat_des);
      double min_eigenval_base = quat_hessian_base.eigenvalues().real().minCoeff();
      Eigen::MatrixXd reg_base_1 = std::max(0.0, -min_eigenval_base) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd reg_base_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);
      double w_base = discount_factor * controller_options_.Q_quaternion_weight;

      // 2. ADMM consensus quaternion Hessian (only on terminal stage i == N_stages - 1)
      if (i == N_stages - 1) {
        Eigen::VectorXd quat_admm = x_admm[i].segment(index, 4).normalized();
        Eigen::MatrixXd quat_hessian_admm =
            common::hessian_of_squared_quaternion_angle_difference(quat_curr, quat_admm);
        double min_eigenval_admm = quat_hessian_admm.eigenvalues().real().minCoeff();
        Eigen::MatrixXd reg_admm_1 = std::max(0.0, -min_eigenval_admm) * Eigen::MatrixXd::Identity(4, 4);
        Eigen::MatrixXd reg_admm_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);
        double w_admm = rho;

        Q[i].block(index, index, 4, 4) = 
            w_base * (quat_hessian_base + reg_base_1 + reg_base_3) +
            w_admm * (quat_hessian_admm + reg_admm_1 + reg_admm_3);
      } else {
        Q[i].block(index, index, 4, 4) = 
            w_base * (quat_hessian_base + reg_base_1 + reg_base_3);
      }
    }
    discount_factor *= controller_options_.c3_options.gamma;
  }
  return Q;
}

Eigen::Quaterniond MSiC3ParallelAdmm::slerpLong(const Eigen::Quaterniond& q0, const Eigen::Quaterniond& q1, double t) {
  Eigen::Quaterniond a = q0.normalized();
  Eigen::Quaterniond b = q1.normalized();
  double dot = a.dot(b);

  if (dot > 0.0) {
    b = Eigen::Quaterniond(-b.w(), -b.x(), -b.y(), -b.z());
    dot = -dot;
  }

  dot = std::clamp(dot, -1.0, 1.0);
  double theta = std::acos(dot);

  if (std::abs(theta) < 1e-6) {
    return a;
  }

  double sin_theta = std::sin(theta);
  double w1 = std::sin((1.0 - t) * theta) / sin_theta;
  double w2 = std::sin(t * theta) / sin_theta;

  return Eigen::Quaterniond(
      w1 * a.w() + w2 * b.w(),
      w1 * a.x() + w2 * b.x(),
      w1 * a.y() + w2 * b.y(),
      w1 * a.z() + w2 * b.z()).normalized();
}

} // namespace systems
} // namespace c3
