#include "MSiC3_parallel.h"

#include <cmath>
#include <chrono>

#include <Eigen/Dense>

#include "core/c3_miqp.h"
#include "core/c3_plus.h"
#include "core/c3_qp.h"
#include "multibody/lcs_factory.h"
#include "multibody/geom_geom_collider.h"
#include "common/quaternion_error_hessian.h"
#include "systems/hybrid_mpc.h"

#include "drake/common/text_logging.h"
#include <drake/multibody/parsing/parser.h>
#include <chrono>
#include <omp.h>

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::BasicVector;
using drake::multibody::MultibodyPlant;
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

namespace c3 {
namespace systems {

MSiC3Parallel::MSiC3Parallel(const MultibodyPlant<double>& plant, const MultibodyPlant<drake::AutoDiffXd>& plant_ad, 
  const MultibodyPlant<double>& plant_rollout, const MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout, 
  drake::systems::Diagram<double>& rollout_diagram, std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,    
  const vector<SortedPair<GeometryId>>& contact_geoms, const vector<SortedPair<GeometryId>>& contact_geoms_rollout,
  C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, HybridMpcOptions mpc_options, int example_idx)
    : plant_(plant),
      plant_ad_(plant_ad),
      plant_rollout_(plant_rollout),
      plant_ad_rollout_(plant_ad_rollout),
      rollout_diagram_(rollout_diagram),
      rollout_diagram_context_(std::move(rollout_diagram_context)),
      contact_geoms_(contact_geoms),
      contact_geoms_rollout_(contact_geoms_rollout),
      use_drake_sim_(ms_ic3_options.use_drake_sim),
      controller_options_(controller_options),
      ms_ic3_options_(ms_ic3_options),
      mpc_options_(mpc_options),
      N_(ms_ic3_options.N),
      example_idx_(example_idx) {

  if (rollout_diagram_context_ == nullptr) {
    std::cout << "NULLPTR SPJOIGNSUIPD" << std::endl;
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

  simulators_.resize(num_segments_);
  if (use_drake_sim_) {
    for (int i = 0; i < num_segments_; i++) {
      auto context_clone = rollout_diagram_.CreateDefaultContext();   
      simulators_[i] = std::make_unique<drake::systems::Simulator<double>>(
          rollout_diagram_, std::move(context_clone));
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
  drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout) {

  auto start_total = std::chrono::high_resolution_clock::now();
  
  // num segements must divide the entire time horizon (might be unnecessary but for ease of implementation)
  DRAKE_DEMAND((double)(N_ / num_segments_) == (double)N_ / num_segments_);

  std::vector<double> x_init = *controller_options_.x_init;
  VectorXd x0 = Eigen::Map<VectorXd>(x_init.data(), x_init.size());    

  std::vector<double> x_des = *controller_options_.x_des;
  VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());  
  MatrixXd xd_mat = xd.replicate(1, N_/L_ + 1);

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

    lower_bound_x(0) = -0.1;
    lower_bound_x(1) = -0.1;
    lower_bound_x(2) = -0.15; 
    lower_bound_x(3) = -0.6;
    lower_bound_x(4) = -0.6;

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

    A_x(13, 13) = 1;
    A_x(14, 14) = 1;

    lower_bound_x(13) = -0.03;
    lower_bound_x(14) = -0.03;

    upper_bound_x(13) = 0.03;
    upper_bound_x(14) = 0.03;

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
      double z_bound = (i == 0) ? 0.07 : 0.04;

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

      double u_bound_xy = (i == 0) ? 3 : 2;

      lower_bound_u(3*i) = -u_bound_xy;
      lower_bound_u(3*i+1) = -u_bound_xy;
      lower_bound_u(3*i+2) = -0.8;
      
      upper_bound_u(3*i) = u_bound_xy;
      upper_bound_u(3*i+1) = u_bound_xy;
      upper_bound_u(3*i+2) = 1.2;
    }
  }

  std::cout << "lb x " << lower_bound_x.transpose() << std::endl;
  std::cout << "ub x " << upper_bound_x.transpose() << std::endl;


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

  VectorXd gravity;
  if (example_idx_ == 0) {
    gravity = VectorXd::Zero(5);
    gravity[2] = 8.33;
  } else if (example_idx_ == 1 || example_idx_ == 2) {
    gravity = VectorXd::Zero(9);
    gravity[2] = 0.196;
    gravity[5] = 0.196;
    gravity[8] = 0.196;
  }

  LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
      contact_geoms_, controller_options_.lcs_factory_options);

  LCSFactory lcs_factory_rollout(plant_rollout_, *contexts_rollout[0], plant_ad_rollout_,
      context_ad_rollout, contact_geoms_rollout_, controller_options_.lcs_factory_options);

  std::cout << "plant lcs dt " << plant_.time_step() << std::endl;
  std::cout << "plant rollout dt " << plant_rollout_.time_step() << std::endl;


  HybridMPC hybrid_mpc_controller(plant_rollout_, lcs_factory, rollout_diagram_, std::move(rollout_diagram_context_), 
            contact_geoms_rollout_, mpc_options_, ms_ic3_options_, example_idx_, controller_options_.lcs_factory_options.mu, 
            A_x, lower_bound_x, upper_bound_x, A_u, lower_bound_u, upper_bound_u);


  // Set initial guess to something kinda reasonable
  // Set initial guess for x - linear interpolation (including in quaternion space)
  // But project contacts so that each x is feasible (i.e. no penetration)
  VectorXd x_diff = xd - x0;
  for (int k = 0; k < N_+1; k++) {
    x_hat.col(k) = x0 + k * x_diff / (N_);
    
    // Linearly interpolate quaternions correctly
    for (auto idx : controller_options_.quaternion_indices) {
      double rotation = (double)k / (N_);

      Quaterniond q0(x0(idx), x0(idx+1), x0(idx+2), x0(idx+3));
      Quaterniond qd(xd(idx), xd(idx+1), xd(idx+2), xd(idx+3));
      VectorXd v0 = x0.segment(idx, 4);
      VectorXd vd = xd.segment(idx, 4);

      // Ensure quaternions are in the same hemisphere 
      if (v0.dot(vd) < 0) {
          vd = -vd;
          qd = Quaterniond(vd(0), vd(1), vd(2), vd(3));
      }

      if (-1e-3 < q0.dot(qd) && q0.dot(qd) < 1e-3) { 
        // Fallback for antipodal points, use linear interpolation in R3 to get default axis
        Eigen::Vector4d mid = v0 + vd;
        Eigen::Vector4d tangent = (mid - mid.dot(v0) * v0).normalized();

        double theta = std::acos(std::clamp(v0.dot(vd), -1.0, 1.0)); 
        Eigen::Vector4d v_interpolated = v0 * std::cos(rotation * theta) + tangent * std::sin(rotation * theta);

        x_hat.col(k).segment(idx, 4) = v_interpolated;

      } else {
        Quaterniond slerp = q0.slerp(rotation, qd);
        x_hat.col(k).segment(idx, 4) << slerp.w(), slerp.x(), slerp.y(), slerp.z(); 
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

      x_projected = ProjectContactVertical(context, cube_plate_contact, x_projected, 11);
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
      SortedPair<GeometryId>cube_plate_contact(cube_collision_geom, ground_collision_geom);

      x_projected = ProjectContactVertical(context, cube_plate_contact, x_projected, 15);


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

  for (int i = 0; i < num_segments_+1; i++) {
    x_anchors.col(i) = x_hat.col(i*L_);
  }


  // Get lambda_hat initial guess and defects
  MatrixXd x_hat_init(MatrixXd::Zero(n_x_, N_+1));
  if (use_drake_sim_) {
    VectorXd x_curr = x0;

    Context<double>& root_context = simulators_[0]->get_mutable_context();
    Context<double>& plant_context =
        rollout_diagram_.GetMutableSubsystemContext(plant_rollout_, &root_context);
    root_context.SetTime(0.0);
    simulators_[0]->Initialize();  

    for (int i = 0; i < N_; i++) {
      x_hat_init.col(i) = x_curr;
      plant_rollout_.SetPositionsAndVelocities(&plant_context, x_curr);

      plant_rollout_.get_actuation_input_port().FixValue(&plant_context, u_hat.col(i));
      double target_time = root_context.get_time() + dt_;
      simulators_[0]->AdvanceTo(target_time);

      auto abstract_contact_results = drake::AbstractValue::Make<drake::multibody::ContactResults<double>>({});
      plant_rollout_.get_contact_results_output_port().Calc(plant_context, abstract_contact_results.get());
      const auto& contact_results = abstract_contact_results->get_value<drake::multibody::ContactResults<double>>();
        
      x_curr = plant_rollout_.GetPositionsAndVelocities(plant_context);
      if (ms_ic3_options_.use_rollout_lambdas) {
        auto [lambda, gamma, in_contact] = 
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

      x_projected = ProjectContactVertical(context, cube_plate_contact, x_projected, 11);
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
      SortedPair<GeometryId>cube_plate_contact(cube_collision_geom, ground_collision_geom);

      x_projected = ProjectContactVertical(context, cube_plate_contact, x_projected, 15);


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
  vector<MatrixXd> all_defects; // Currently unused
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

  LCS lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);

  int num_iters = ms_ic3_options_.num_iters;

  // AuLa multiplier/penalty
  MatrixXd xis(MatrixXd::Zero(n_x_, num_segments_));
  VectorXd rhos(VectorXd::Ones(num_segments_));

  // TODO: update these to be params
  double rho_multiplier = 2;
  double rho_max = 32;

  for (int iter = 1; iter <= num_iters; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "iC3 iteration " << iter << std::endl;

    delta_projection_iter_.clear();
    z_sol_iter_.clear();

    UpdateQuaternionCosts(x_hat, xd); // Note: this overrides R, G, U as well
    std::cout << "before compute lqr value fun" << std::endl;
    // // Backwards Pass - Compute Value Function
    auto [H, g, K, k_ff] = ComputeLQRValueFunction(x_hat, u_hat, lambda_hat, lcs, xd_mat, u_nominal[0], defects);
    std::cout << "after compute lqr value fun" << std::endl;

    Hs.push_back(H);
    gs.push_back(g);
    Ks.push_back(K);
    k_ffs.push_back(k_ff);
    

    MatrixXd new_x_anchors(MatrixXd::Zero(n_x_, num_segments_+1));
    new_x_anchors.col(0) = x0;

    // Set alpha
    // double alpha_ee = std::min(1.0, ms_ic3_options_.alpha_ee + (iter-1) * ms_ic3_options_.alpha_ee_step);
    // double alpha_object = std::min(1.0, ms_ic3_options_.alpha_object + (iter-1) * ms_ic3_options_.alpha_object_step);

    // std::cout << "alpha ee " << alpha_ee << " alpha object " << alpha_object << std::endl;
    std::cout << "num segments " << num_segments_ << std::endl;
    std::cout << "L " << L_ << std::endl;

    // delta_projection_iter_.resize(num_segments_);
    // z_sol_iter_.resize(num_segments_);

    // Solve C3 over each segment
    #pragma omp parallel for num_threads(ms_ic3_options_.num_threads.value_or(1))
    for (int i = 0; i < num_segments_; i++) {

      auto [x_hat_out, u_hat_out, lambda_hat_out, gamma_out, in_contact_out] = 
         DoC3Rollout(x_anchors.col(i), x_hat, u_hat.middleCols(i*L_, L_), xd, gravity,
                      lcs_factory, lcs_factory_rollout, H, g, i*L_, i,
                      A_x, lower_bound_x, upper_bound_x, A_u, lower_bound_u, upper_bound_u,
                      context, *contexts_rollout[i]);

      gamma.block(0, i*L_, n_lambda_ / 4, L_) = gamma_out;
      in_contact.block(0, i*L_, n_lambda_ / 4, L_) = in_contact_out;

      // std::cout << x_hat_out.topRows(12).transpose() << std::endl;
      VectorXd x_L = x_hat_out.col(L_);
      VectorXd x_anchor_next = x_anchors.col(i+1);

      // Update anchors based on value function (encodes sensitivity)
      new_x_anchors.col(i+1) = UpdateAnchor(x_L, x_anchor_next, H[(i+1) * L_]);

      // Don't update final anchor
      if (i == num_segments_) {
        new_x_anchors.col(i+1) = xd;
      }

      VectorXd x_projected = new_x_anchors.col(i+1);

      #pragma omp critical
      {
        // Ensure anchors don't have penetration
        if (example_idx_ == 0) {
          // HARDCODED CONTACT GEOM
          drake::geometry::GeometryId plate_collision_geom =
              plant_.GetCollisionGeometriesForBody(
                  plant_.GetBodyByName("plate"))[0];
          drake::geometry::GeometryId cube_collision_geom =
            plant_.GetCollisionGeometriesForBody(
                plant_.GetBodyByName("cube"))[0];
          SortedPair<GeometryId>cube_plate_contact(plate_collision_geom, cube_collision_geom);

          x_projected = ProjectContactVertical(context, cube_plate_contact, x_projected, 11);
          
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
      } // end pragma omp critical

      new_x_anchors.col(i+1) = x_projected;

      defects.col(i+1) = x_hat_out.col(L_) - new_x_anchors.col(i+1);

      x_hat.middleCols(i*L_, L_) = x_hat_out.leftCols(L_);
      u_hat.middleCols(i*L_, L_) = u_hat_out.leftCols(L_);
      lambda_hat.middleCols(i*L_, L_) = lambda_hat_out.leftCols(L_);

      if (i == num_segments_-1) { // Tack on final state if last segment
        x_hat.col(N_) = x_hat_out.col(L_);
      }
    } 
    // END OF PARALLELIZATION

    x_anchors = new_x_anchors;



    MatrixXd x_anchors_prev = all_x_anchors[iter-1];
    for (int j = 0; j < num_segments_; j++) {
      std::cout << "Segment " << j << std::endl;
      if (example_idx_ == 0) {
        std::cout << "x_anchor pancake " << x_anchors_prev.col(j).segment(5, 7).transpose() << std::endl;
        std::cout << "x_hat[L] pancake: " << x_hat.col((j+1) * L_ - 1).segment(5, 7).transpose() << std::endl;
        std::cout << "new anchor pancake: " << x_anchors.col(j+1).segment(5, 7).transpose() << std::endl << std::endl;
      } else if (example_idx_ == 1 || example_idx_ == 2) {
        std::cout << "x_anchor cube " << x_anchors_prev.col(j).segment(9, 7).transpose() << std::endl;
        std::cout << "x_hat[L] cube: " << x_hat.col((j+1) * L_ - 1).segment(9, 7).transpose() << std::endl;
        std::cout << "new anchor cube: " << x_anchors.col(j+1).segment(9, 7).transpose() << std::endl << std::endl;
      }
    }

    // Linearize about new nominal trajectory
    lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);

    // Don't store if warmup
    all_x_hats.push_back(x_hat);
    all_u_hats.push_back(u_hat);
    all_lambda_hats.push_back(lambda_hat);
    all_defects.push_back(defects);
    all_x_anchors.push_back(x_anchors);
    all_delta_projections.push_back(delta_projection_iter_);
    all_z_sols.push_back(z_sol_iter_);
    all_gammas.push_back(gamma);
    all_in_contacts.push_back(in_contact);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Iteration runtime: " << duration.count() << " seconds\n\n " << std::endl;

  }


  UpdateQuaternionCosts(x_hat, xd);
  std::cout << "Before compute lqr value function" << std::endl;

  auto [H, g, K, k_ff] = ComputeLQRValueFunction(x_hat, u_hat, lambda_hat, lcs, xd_mat, u_nominal[0], defects);
  Hs.push_back(H);
  gs.push_back(g);
  Ks.push_back(K);
  k_ffs.push_back(k_ff);

  std::cout << std::endl;

  auto start_mpc = std::chrono::high_resolution_clock::now();
  // Run hybrid mpc over final trajectory to ensure feasibility
  // auto [x_hat_out, u_hat_out, lambda_hat_out] = 
  //   hybrid_mpc_controller.SimulateHybridMPC(x0, x_hat, u_hat, lambda_hat, *contexts_rollout[0]);

  // Final, fully feasible pass
  auto [x_hat_out, u_hat_out, lambda_hat_out, gamma_out, in_contact_out] = 
      DoC3Rollout(x0, x_hat, u_hat, xd, gravity,
                  lcs_factory, lcs_factory_rollout, H, g, 0, 0,
                  A_x, lower_bound_x, upper_bound_x, A_u, lower_bound_u, upper_bound_u,
                  context, *contexts_rollout[0]);

  auto end_mpc = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_mpc = end_mpc - start_mpc;
  std::cout << "Final rollout runtime: " << duration_mpc.count() << " seconds " << std::endl;

  if (example_idx_ == 0) {
    std::cout << "x pancake final " << x_hat_out.col(x_hat_out.cols()-1).segment(5, 7).transpose() << std::endl;
  } else if (example_idx_ == 1 || example_idx_ == 2) {
    std::cout << "x cube final " << x_hat_out.col(x_hat_out.cols()-1).segment(9, 7).transpose() << std::endl;
  }
  std::cout << std::endl;

  all_x_hats.push_back(x_hat_out);
  all_u_hats.push_back(u_hat_out);
  all_lambda_hats.push_back(lambda_hat_out);

  // Get value function over hybrid mpc trajectory
  UpdateQuaternionCosts(x_hat_out, xd);
  lcs = MakeTimeVaryingLCS(x_hat_out, u_hat_out, lcs_factory);

  MatrixXd x_anchors_final(n_x_, N_ / L_);
  MatrixXd defects_final(MatrixXd::Zero(n_x_, N_ / L_)); // fully feasible so defects are 0
  for (int i = 0; i < N_; i+= L_) {
    x_anchors_final.col(i / L_) = x_hat_out.col(i);
  }

  
  auto [H_final, g_final, K_final, k_ff_final] = ComputeLQRValueFunction(x_hat_out, u_hat_out, lambda_hat_out, lcs, xd_mat, u_nominal[0], defects_final);
  Hs.push_back(H_final);
  gs.push_back(g_final);
  Ks.push_back(K_final);
  k_ffs.push_back(k_ff_final);
  std::cout << std::endl;

  auto end_total = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end_total - start_total;
  std::cout << "Total runtime: " << duration.count() << " seconds\n\n " << std::endl;

  return std::make_tuple(all_x_hats, all_u_hats, all_lambda_hats, Hs, gs, Ks, k_ffs, 
                        all_delta_projections, all_z_sols, all_gammas, all_in_contacts);
}

VectorXd MSiC3Parallel::ProjectContactVertical(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int z_idx) {
    
    VectorXd x_out = x_init;
    plant_.SetPositionsAndVelocities(&context, x_init);
    multibody::GeomGeomCollider collider(plant_, geom_pair);
    auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);

    // HARDCODED FOR PLATE EXAMPLE and pivoting
    double phi_check = phi;
    while (phi_check < -3e-4) {
      // Displace z of object upwards
      x_out(z_idx) = x_out(z_idx) + 0.002;
      
      plant_.SetPositionsAndVelocities(&context, x_out);
      auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);
      phi_check = phi;
    }
    x_out(z_idx) = x_out(z_idx) + 0.002; // Add a little extra to ensure no penetration

    return x_out;
}

// TODO: clean up this function
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
      auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);
      phi_check = phi;
      contact_normal = J.row(0).segment(start_idx, q_size);
      counter++;
    }
    if (counter > 1) {
      std::cout << "projection counter: " << counter << std::endl;
      std::cout << "phi end " << phi_check << std::endl;
    }
    
    return x_out;
}

// TODO: clean this function up
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

  // HARDCODED
  int q_idx;
  int v_idx;
  if (x0.size() == 23) {
    q_idx = 0;
    v_idx = 12;
  } else if (x0.size() == 31) {
    q_idx = 0;
    v_idx = 16;
  }

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

  LCS output_lcs = MakeTimeVaryingLCS(x_hat_downsampled, u_hat_fb_downsampled, factory);

  if ((x_hat_downsampled.array().isNaN()).any()) {
    std::cout << "XHAT NOT FINITE" << std::endl;
  }
  return {output_lcs, x_hat_downsampled, u_hat_fb_downsampled, lambda_hat_downsampled};

}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd> MSiC3Parallel::DoC3Rollout(VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, 
                                              VectorXd xd, VectorXd ud, LCSFactory factory, 
                                              LCSFactory rollout_factory, vector<MatrixXd> H, 
                                              vector<VectorXd> g, int start_idx, int segment_idx, 
                                              MatrixXd A_x, VectorXd lb_x, VectorXd ub_x,
                                              MatrixXd A_u, VectorXd lb_u, VectorXd ub_u,
                                              Context<double>& context, Context<double>& context_rollout) {
  // Assume that x_hat, H, g, x_targets correspond to the entire iC3 horizon

  DRAKE_DEMAND(start_idx < N_);
  DRAKE_DEMAND(H.size() == N_ + 1);
  DRAKE_DEMAND(g.size() == N_ + 1);
  DRAKE_DEMAND(x_hat.cols() == N_ + 1);

  auto c3_rollout_start = std::chrono::high_resolution_clock::now();

  int num_steps = u_hat.cols();

  int factor = ms_ic3_options_.rollout_dt_scaling;

  MatrixXd x_hat_output(n_x_, num_steps * factor + 1);
  MatrixXd lambda_hat(n_lambda_, num_steps * factor);
  MatrixXd u_hat_fb(n_u_, num_steps * factor);

  MatrixXd gamma_matrix(MatrixXd::Zero(n_lambda_ / 4, num_steps * factor));
  MatrixXd in_contact_matrix(MatrixXd::Zero(n_lambda_ / 4, num_steps * factor));

  x_hat_output.col(0) = x0;
  VectorXd x_curr = x0;
  VectorXd x_next;

  // std::cout << "x anchor fingers " << x0.segment(0,9).transpose() << std::endl;
  LCS lcs_test;
  #pragma omp critical
  {
    factory.UpdateStateAndInput(x0, u_hat.col(0));
    lcs_test = factory.GenerateLCS();
  }

  // std::cout << "x anchor fingers " << x0.segment(0, 9).transpose() << std::endl;
  // if (example_idx_ == 0) {
  //   std::cout << "x anchor ee " << x0.segment(0, 5).transpose() << std::endl;
  //   std::cout << "x anchor pancake " << x0.segment(5, 7).transpose() << std::endl;
  // } else if (example_idx_ == 1 || example_idx_ == 2) {
  //   std::cout << "x anchor ee " << x0.segment(0, 9).transpose() << std::endl;
  //   std::cout << "x anchor cube " << x0.segment(9, 7).transpose() << std::endl;
  // }

  int tracking_N = controller_options_.lcs_factory_options.N;

  rollout_factory.SetNewDt(dt_ / factor);

  // Run receding horizon C3 MPC
  for (int t = 0; t < num_steps; t++) {

    VectorXd u_nominal = u_hat.col(t);
    
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


    double discount_factor = 1;
    for (int i = 0; i < tracking_N + 1; i++) {
      int x_idx = std::min(N_, start_idx + t + i);
      int u_idx = std::min(num_steps-1, t + i);
      int R_idx = std::min(N_-1, start_idx + t + i);

      x_targets_shortened.push_back(xd);
      x_reg_targets.push_back(x_hat.col(x_idx));
      x_hat_for_lcs.col(i) = x_hat.col(x_idx);
      Q.push_back(discount_factor * Q_[x_idx]);

      if (i < tracking_N) {
        u_targets_shortened.push_back(ud);
        u_reg_targets.push_back(u_hat.col(u_idx));
        u_hat_for_lcs.col(i) = u_hat.col(u_idx);
        R.push_back(discount_factor * R_[R_idx]);

        G.push_back(discount_factor * G_[R_idx]);      
        U.push_back(discount_factor * U_[R_idx]);
      }
      discount_factor *= controller_options_.c3_options.gamma;
    }

    vector<MatrixXd> Q_updated = UpdateQuaternionCosts(x_curr, x_targets_shortened, Q);
    C3::CostMatrices costs(Q_updated, R, G, U);

    vector<MatrixXd> Q_reg = UpdateQuaternionCosts(x_curr, x_reg_targets, Q);
    vector<MatrixXd> R_reg = R;

    // HARDCODED ee start idx
    LCS lcs;
    #pragma omp critical
    {
      lcs = MakeTimeVaryingLCSWithEE(x_hat_for_lcs, u_hat_for_lcs, factory, x_curr.segment(0, n_u_), 0);
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
        // std::cout << "norm " << norm << std::endl;
      }
    }

    std::unique_ptr<C3Plus> c3_tracking = std::make_unique<C3Plus>(lcs, costs, x_targets_shortened,
                                  controller_options_.c3_options);
    c3_tracking->UpdateInputTarget(u_targets_shortened);
    c3_tracking->SetPenalizeChange(false); // TODO: change the interface of regularization costs so this isn't needed

    c3_tracking->SetXHat(x_reg_targets);
    c3_tracking->SetUHat(u_reg_targets);
    c3_tracking->AddRegularizationCostsState(Q_reg);
    c3_tracking->AddRegularizationCostsInput(R_reg);

    if (ms_ic3_options_.add_position_constraints) {
      c3_tracking->AddLinearConstraint(A_x, lb_x, ub_x,
                                ConstraintVariable::STATE);
    }  
    if (ms_ic3_options_.add_input_constraints) {
      c3_tracking->AddLinearConstraint(A_u, lb_u, ub_u,
                                ConstraintVariable::INPUT);
    }
    
    if (n_u_ == 9 && ms_ic3_options_.penalize_acceleration) {
      c3_tracking->AddAccelerationCost(n_q_, n_v_, ms_ic3_options_.acceleration_cost_weight);
    }

    // If tracking c3 horizon goes past iC3 N, just use last H, g
    int lqr_idx = std::min(start_idx + t + tracking_N, N_); 
    double scaling = ms_ic3_options_.value_function_scaling;

    // Note: this gets added on top of the base x regularization weight
    MatrixXd Q_trust = UpdateQuaternionCosts(x_curr, {x_hat.col(lqr_idx)}, {Q_[lqr_idx]})[0];
    Q_trust *= ms_ic3_options_.vf_trust_region_weight;
    VectorXd bias = g[lqr_idx] - (H[lqr_idx] + Q_trust) * x_hat.col(lqr_idx);
    c3_tracking->UpdateFinalCost(scaling * (H[lqr_idx] + Q_trust), scaling * bias);

    auto c3_start = std::chrono::high_resolution_clock::now();
    c3_tracking->Solve(x_curr);
    auto c3_end = std::chrono::high_resolution_clock::now();
    auto c3_elapsed = c3_end - c3_start;
    double c3_solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(c3_elapsed).count() / 1e6;
    vector<MatrixXd> delta_proj = c3_tracking->GetDeltaProjection();

    // Rescale lambda, eta
    double AnDn = c3_tracking->GetAnDn();
    for (int ii = 0; ii < delta_proj.size(); ii++) {
      for (int jj = 0; jj < delta_proj[ii].cols(); jj++) {
        delta_proj[ii].col(jj).segment(n_x_, n_lambda_) *= AnDn;
        delta_proj[ii].col(jj).segment(n_x_+n_lambda_+n_u_, n_lambda_) *= AnDn;
      }
    }

    #pragma omp critical
    {
      delta_projection_iter_.push_back(delta_proj);
    }

    vector<Eigen::VectorXd> z_sol = c3_tracking->GetFullSolution();

    vector<Eigen::VectorXd> z_sol_scaled = z_sol;
    for (int ii = 0; ii < z_sol_scaled.size(); ii++) {
      z_sol_scaled[ii].segment(n_x_, n_lambda_) *= AnDn;
      z_sol_scaled[ii].segment(n_x_+n_lambda_+n_u_, n_lambda_) *= AnDn;
    } 
    #pragma omp critical
    {
      z_sol_iter_.push_back(z_sol_scaled);
    }

    VectorXd c3_u = z_sol[0].segment(n_x_ + n_lambda_, n_u_);
    VectorXd c3_x = z_sol[0].segment(0, n_x_);
    VectorXd c3_x_next = z_sol[1].segment(0, n_x_);

    if (ms_ic3_options_.print_costs) {
      std::cout << "c3 solve time " << c3_solve_time << std::endl;
    }


    auto rollout_start = std::chrono::high_resolution_clock::now();
    if (use_drake_sim_) {
      int q_idx;
      int v_idx;
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
      Context<double>& plant_context =
          rollout_diagram_.GetMutableSubsystemContext(plant_rollout_, &root_context);
      root_context.SetTime(0.0);
      simulators_[segment_idx]->Initialize();  

      for (int i = 0; i < factor; i++) {
        plant_rollout_.SetPositionsAndVelocities(&plant_context, x_curr);

        // Apply PD to C3 plan
        VectorXd c3_x_tracking = (i * c3_x_next + (factor-i) * c3_x) / factor;
        VectorXd u_tracking = c3_u + Kp * (c3_x_tracking.segment(q_idx, Kp.rows()) - x_curr.segment(q_idx, Kp.rows())) 
          + Kd * (c3_x_tracking.segment(v_idx, Kd.rows()) - x_curr.segment(v_idx, Kd.rows()));

        for (int j = 0; j < A_u.rows(); j++) {
          if (A_u(j, j) == 1) {
            u_tracking(j) = std::clamp(u_tracking(j), lb_u(j), ub_u(j));
          }
        }
        // std::cout << "u: " << u_tracking.transpose() << std::endl;

        plant_rollout_.get_actuation_input_port().FixValue(&plant_context, u_tracking);

        double target_time = root_context.get_time() + dt_ / factor;

        auto simulator_start = std::chrono::high_resolution_clock::now();
        simulators_[segment_idx]->AdvanceTo(target_time);
        auto simulator_end = std::chrono::high_resolution_clock::now();
        auto simulator_elapsed = simulator_end - simulator_start;
        double simulator_solve_time =
            std::chrono::duration_cast<std::chrono::microseconds>(simulator_elapsed).count() / 1e6;

        x_next = plant_rollout_.GetPositionsAndVelocities(plant_context);

        // Ensure consistent quaternion convention
        for (int i = 0; i < controller_options_.quaternion_indices.size(); i++) {
          int idx = controller_options_.quaternion_indices[i];
          if (x_curr.segment(idx, 4).dot(x_next.segment(idx, 4)) < 0) {
            x_next.segment(idx, 4) *= -1;
          }
        }

        // std::cout << "x next " << x_next.transpose() << std::endl;  
        // std::cout << "u " << u_tracking.transpose() << std::endl;

        // Clamp velocities
        if (example_idx_ == 1 || example_idx_ == 2) {
          for (int j = n_q_; j < A_x.rows(); j++) {
            if (A_x(j, j) == 1) { // Assumes diagonal
              x_next(j) = std::clamp(x_next(j), lb_x(j), ub_x(j));
            }
          }
        }
        auto abstract_contact_results = drake::AbstractValue::Make<drake::multibody::ContactResults<double>>({});
        plant_rollout_.get_contact_results_output_port().Calc(plant_context, abstract_contact_results.get());
        const auto& contact_results = abstract_contact_results->get_value<drake::multibody::ContactResults<double>>();

        x_hat_output.col(factor * t + i + 1) = x_next;
        u_hat_fb.col(factor * t + i) = u_tracking;

        auto [lambda, gamma, in_contact] = 
            ConstructLambdasFromContactResults(contact_results, controller_options_.lcs_factory_options.contact_model); 
        lambda_hat.col(factor * t + i) = lambda;
        gamma_matrix.col(factor * t + i) = gamma;
        in_contact_matrix.col(factor * t + i) = in_contact;


        x_curr = x_next;                         
      }
      

    } else {
      // Rollout this u with LCS
      for (int i = 0; i < factor; i++) {
        // Normalize quaternions
        for (int quat_idx : controller_options_.quaternion_indices) {
          x_curr.segment(quat_idx, 4) = x_curr.segment(quat_idx, 4).normalized();
        }

        // Apply PD to C3 plan
        int q_idx;
        int v_idx;
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
        // std::cout << "tracking u " << u_tracking.transpose() << std::endl;

        for (int j = 0; j < A_u.rows(); j++) {
          if (A_u(j, j) == 1) {
            u_tracking(j) = std::clamp(u_tracking(j), lb_u(j), ub_u(j));
          }
        }
        // std::cout << "u: " << u_tracking.transpose() << std::endl;
        LCS lcs_rollout;
        #pragma omp critical
        {
          rollout_factory.UpdateStateAndInput(x_curr, u_tracking);
          lcs_rollout = rollout_factory.GenerateLCS();  

          // Debugging phi
          for (int g = 0; g < 3; g++) {
            multibody::GeomGeomCollider collider(plant_, contact_geoms_[g]);
            plant_.SetPositionsAndVelocities(&context, x_curr);
            auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
              drake::multibody::JacobianWrtVariable::kQDot);
            if (phi < -1e-3) {
              std::cout << "contact " << g << " phi " << phi << std::endl;
            }
          }
        } // end pragma omp critical

        // std::cout << "u tracking " << u_tracking.transpose() << std::endl;
        auto pair = lcs_rollout.SimulateAndReturnForce(x_curr, u_tracking, true);
        x_next = pair.first;
        // if (i == 0) {
        //   std::cout << "lambda rollout " << pair.second.transpose() << std::endl;
        // }

        if (example_idx_ == 1 || example_idx_ == 2) {
          for (int j = 0; j < A_x.rows(); j++) {
            if (A_x(j, j) == 1) { // Assumes diagonal
              x_next(j) = std::clamp(x_next(j), lb_x(j), ub_x(j));
            }
          }
          // Force end effector z to be 0.05
          // x_next(2) = 0.05; 
          // x_next(5) = 0.05; 
          // x_next(8) = 0.05; 
        }
        x_hat_output.col(factor * t + i + 1) = x_next;
        lambda_hat.col(factor * t + i) = pair.second;
        u_hat_fb.col(factor * t + i) = u_tracking;

        x_curr = x_next;
      }
    }
    auto rollout_end = std::chrono::high_resolution_clock::now();
    auto rollout_elapsed = rollout_end - rollout_start;
    double rollout_solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(rollout_elapsed).count() / 1e6;
  

    if (ms_ic3_options_.print_costs) {
      // std::cout << "rollout time " << rollout_solve_time << std::endl;
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
    lambda_hat_downsampled.col(i) = lambda_hat.col(i * factor);
    gamma_downsampled.col(i) = gamma_matrix.col(i * factor);
    in_contact_downsampled.col(i) = in_contact_matrix.col(i * factor);

  }
  x_hat_downsampled.col(num_steps) = x_hat_output.col(num_steps * factor);

  auto c3_rollout_end = std::chrono::high_resolution_clock::now();
  auto c3_rollout_elapsed = c3_rollout_end - c3_rollout_start;
  double c3_rollout_solve_time =
      std::chrono::duration_cast<std::chrono::microseconds>(c3_rollout_elapsed).count() / 1e6;
  std::cout << "C3 rollout time segment " << start_idx / L_ << " " << c3_rollout_solve_time << std::endl;

  return {x_hat_downsampled, u_hat_downsampled, lambda_hat_downsampled, gamma_downsampled, in_contact_downsampled};                                              
}


std::tuple<vector<MatrixXd>, vector<VectorXd>, vector<MatrixXd>, vector<VectorXd>> 
MSiC3Parallel::ComputeLQRValueFunction(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat,
      LCS lcs, MatrixXd x_anchors, VectorXd ud, MatrixXd defects) {
  
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
  // If terminal state has an anchor offset:
  // g[N_] = Q[N_] * (x_hat.col(N_) - x_anchors.col(N_ / L_));

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
    VectorXd Q_x = Q[t] * (x_t - x_anchors.col(k+1)) + f_x.transpose() * (g[t+1] + H_vf[t+1] * defects.col(k+1));
    VectorXd Q_u = R[t] * (u_t - ud) + f_u.transpose() * (g[t+1] + H_vf[t+1] * defects.col(k+1));

    // Regularize Q_uu for robust inversion
    Q_uu.diagonal().array() += 1e-6;

    Eigen::LDLT<MatrixXd> solver(Q_uu);
    K[t] = -solver.solve(Q_ux);
    k_ff[t] = -solver.solve(Q_u);

    // Backpropagate Value Function
    H_vf[t] = Q_xx + K[t].transpose() * Q_ux; // Mathematically identical to Q_xx - K^T Q_uu K
    H_vf[t] = 0.5 * (H_vf[t] + H_vf[t].transpose()); // Enforce symmetry
    
    g[t] = Q_x + K[t].transpose() * Q_u;
  }

  return std::make_tuple(H_vf, g, K, k_ff);
}

LCS MSiC3Parallel::MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory) {
  DRAKE_DEMAND(x_hat.cols() >= u_hat.cols());

  vector<MatrixXd> A;
  vector<MatrixXd> B;
  vector<MatrixXd> D;
  vector<VectorXd> d;
  vector<MatrixXd> E;
  vector<MatrixXd> F;
  vector<MatrixXd> H;
  vector<VectorXd> c;

  int N = u_hat.cols();

  for (int k = 0; k < N; k++) {
    
    for (auto idx : controller_options_.quaternion_indices) {
      x_hat.col(k).segment(idx, 4) = x_hat.col(k).segment(idx, 4).normalized(); // Normalize quaternions
    }

    // Linearize about kth xhat, uhat
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

LCS MSiC3Parallel::MakeTimeVaryingLCSWithEE(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory, VectorXd ee_pose, int ee_idx) {
  MatrixXd x_hat_copy = x_hat;
  for (int i = 0; i < x_hat_copy.cols(); i++) {
    x_hat_copy.col(i).segment(ee_idx, ee_pose.size()) = ee_pose;
  }
  return MakeTimeVaryingLCS(x_hat_copy, u_hat, factory);
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

        // TODO: CHECK THIS
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
  // std::cout << "lambda " << lambda.transpose() << std::endl;
  return {lambda, gamma, in_contact};
}

VectorXd MSiC3Parallel::UpdateAnchor(VectorXd x_L, VectorXd x_anchor_next, MatrixXd H) {
  double rho = ms_ic3_options_.anchor_rho.value_or(1);

  VectorXd x_updated = x_anchor_next;

  // Track which indices belong to quaternion blocks
  std::unordered_set<int> quat_indices_set;
  for (int q_idx : controller_options_.quaternion_indices) {
    quat_indices_set.insert(q_idx);
    quat_indices_set.insert(q_idx + 1);
    quat_indices_set.insert(q_idx + 2);
    quat_indices_set.insert(q_idx + 3);
  }

  // ----------------------------------------------------------------------
  // 1. Calculate Average Curvature across 3D Tangent Space
  // ----------------------------------------------------------------------
  double trace_tangent = 0.0;
  int minimal_dim = 0;

  for (int i = 0; i < n_x_; ) {
    // Check if i is the start of a quaternion block
    if (std::find(controller_options_.quaternion_indices.begin(), 
                  controller_options_.quaternion_indices.end(), i) 
                  != controller_options_.quaternion_indices.end()) {
      // Extract Quaternion q_curr = [w, x, y, z]
      double w = x_anchor_next(i), x = x_anchor_next(i + 1), y = x_anchor_next(i + 2), z = x_anchor_next(i + 3);

      // Compute 4x3 G(q) matrix
      Eigen::Matrix<double, 4, 3> G;
      G << -x, -y, -z,
            w, -z,  y,
            z,  w, -x,
            -y,  x,  w;

      Eigen::Matrix<double, 4, 3> E_q = 0.5 * G;

      // Project 4x4 Hessian block to 3x3 Tangent block
      Eigen::Matrix3d V_theta_theta = E_q.transpose() * H.block<4, 4>(i, i) * E_q;
      trace_tangent += V_theta_theta.trace();
      minimal_dim += 3;

      i += 4; // Skip full quaternion block
    } else {
      // Standard Euclidean component
      trace_tangent += std::max(0.0, H(i, i));
      minimal_dim += 1;
      i += 1;
    }
  }

  double mean_curvature = (minimal_dim > 0) ? (trace_tangent / minimal_dim) : 1.0;
  mean_curvature = std::max(mean_curvature, 1e-6); // Prevent division by zero

  // ----------------------------------------------------------------------
  // 2. Update Euclidean Components
  // ----------------------------------------------------------------------
  for (int i = 0; i < n_x_; ++i) {
    if (quat_indices_set.find(i) == quat_indices_set.end()) {
      double h_ii = std::max(0.0, H(i, i));
      
      // Relative stiffness ratio compared to average state stiffness at this time step
      double r_i = h_ii / mean_curvature;
      
      // Component-wise step size
      double alpha_i = rho / (r_i + rho);
      
      // Convex combination
      x_updated(i) = (1.0 - alpha_i) * x_anchor_next(i) + alpha_i * x_L(i);
    }
  }

  // ----------------------------------------------------------------------
  // 3. Update Quaternion Components
  // ----------------------------------------------------------------------
  for (int q_idx : controller_options_.quaternion_indices) {
    // Extract quaternions [w, x, y, z]
    Quaterniond q_curr(x_anchor_next(q_idx), x_anchor_next(q_idx + 1), x_anchor_next(q_idx + 2), x_anchor_next(q_idx + 3));
    Quaterniond q_L(x_L(q_idx), x_L(q_idx + 1), x_L(q_idx + 2), x_L(q_idx + 3));

    // A. Project 4x4 Hessian to 3x3 Tangent Space
    double w = q_curr.w(), x = q_curr.x(), y = q_curr.y(), z = q_curr.z();
    Eigen::Matrix<double, 4, 3> G;
    G << -x, -y, -z,
          w, -z,  y,
          z,  w, -x,
          -y,  x,  w;

    Eigen::Matrix<double, 4, 3> E_q = 0.5 * G;
    Matrix3d V_theta_theta = E_q.transpose() * H.block<4, 4>(q_idx, q_idx) * E_q;

    // B. Compute relative rotational stiffness ratio r_theta & alpha_theta
    double mean_v_theta = V_theta_theta.trace() / 3.0;
    double r_theta = std::max(0.0, mean_v_theta / mean_curvature);
    double alpha_theta = rho / (r_theta + rho);

    // C. Inline Quaternion Log Map: d_theta = 2 * Log(q_curr^-1 * q_L)
    Quaterniond q_err = (q_curr.conjugate() * q_L).normalized();
    if (q_err.w() < 0.0) {
      q_err.coeffs() = -q_err.coeffs(); // Ensure shortest path
    }

    Vector3d v = q_err.vec();
    double v_norm = v.norm();
    double w_err = q_err.w();

    Vector3d d_theta;
    if (v_norm < 1e-8) {
      d_theta = 2.0 * v;
    } else {
      double angle = 2.0 * std::atan2(v_norm, w_err);
      d_theta = angle * (v / v_norm);
    }

    // D. Inline Quaternion Exp Map & Apply Convex Combination on S^3
    Vector3d scaled_d_theta = alpha_theta * d_theta;
    double step_angle = scaled_d_theta.norm();

    Quaterniond exp_d_theta;
    if (step_angle < 1e-8) {
      exp_d_theta = Quaterniond(1.0, 0.5 * scaled_d_theta.x(), 0.5 * scaled_d_theta.y(), 0.5 * scaled_d_theta.z()).normalized();
    } else {
      Vector3d axis = scaled_d_theta / step_angle;
      exp_d_theta = Quaterniond(Eigen::AngleAxisd(step_angle, axis));
    }

    Quaterniond q_updated = (q_curr * exp_d_theta).normalized();

    // Write back updated quaternion [w, x, y, z]
    x_updated(q_idx) = q_updated.w();
    x_updated(q_idx + 1) = q_updated.x();
    x_updated(q_idx + 2) = q_updated.y();
    x_updated(q_idx + 3) = q_updated.z();
  }

  return x_updated;
}


void MSiC3Parallel::UpdateQuaternionCosts(
  MatrixXd x_hat, VectorXd x_des) {
  
  // std::cout << x_hat.rows() << ", " << x_hat.cols() << std::endl;
  // std::cout << "xd: " << x_des.transpose() << std::endl;
  // std::cout << c3_quat_norms.size() << std::endl;

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

      //std::cout << "xhat q: " << quat_curr_i.transpose() << std::endl;

      Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

      // Regularize hessian so Q is always PSD
      double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
      //std::cout << min_eigenval << std::endl;

      Eigen::MatrixXd Q_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd Q_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
      Eigen::MatrixXd Q_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

      Q_[i].block(index, index, 4, 4) = 
        controller_options_.c3_options.w_Q * 
        controller_options_.Q_quaternion_weight * (quat_hessian_i + Q_quat_regularizer_1 + 
        controller_options_.quaternion_regularizer_fraction * Q_quat_regularizer_2 + Q_quat_regularizer_3);

      // double q_min_eigenval = Q_[i].eigenvalues().real().minCoeff();
      // std::cout << "Q_" << i << " min eigenvalue " <<  q_min_eigenval << std::endl;
      j++;
    }
  }
  //std::cout << std::endl;
  //Q_[N_] = Q_[N_-1];
}


vector<MatrixXd> MSiC3Parallel::UpdateQuaternionCosts(
    VectorXd x_curr, vector<VectorXd> x_des, vector<MatrixXd> Q_in) {
  
  // std::cout << x_hat.rows() << ", " << x_hat.cols() << std::endl;
  // std::cout << "xd: " << x_des.transpose() << std::endl;
  // std::cout << c3_quat_norms.size() << std::endl;

  DRAKE_DEMAND(x_des.size() == Q_in.size());

  vector<MatrixXd> Q = Q_in;

  double discount_factor = 1;
  for (int i = 0; i < Q.size(); i++) {
    int j = 0;
    for (int index : controller_options_.quaternion_indices) {

      // make quaternion costs time-varying based on x_hat
      Eigen::VectorXd quat_curr_i = x_curr.segment(index, 4).normalized();
      Eigen::VectorXd quat_des_i = x_des[i].segment(index, 4).normalized();

      //std::cout << "xhat q: " << quat_curr_i.transpose() << std::endl;

      Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

      // Regularize hessian so Q is always PSD
      double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
      //std::cout << min_eigenval << std::endl;

      Eigen::MatrixXd Q_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd Q_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
      Eigen::MatrixXd Q_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

      Q[i].block(index, index, 4, 4) = 
        discount_factor * 
        controller_options_.c3_options.w_Q * 
        controller_options_.Q_quaternion_weight * (quat_hessian_i + Q_quat_regularizer_1 + 
        controller_options_.quaternion_regularizer_fraction * Q_quat_regularizer_2 + Q_quat_regularizer_3);

      // double q_min_eigenval = Q_[i].eigenvalues().real().minCoeff();
      // std::cout << "Q_" << i << " min eigenvalue " <<  q_min_eigenval << std::endl;
      j++;
    }
    discount_factor *= controller_options_.c3_options.gamma;
  }
  return Q;
}


} // namespace systems
} // namespace c3
