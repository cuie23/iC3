#include "MSiC3.h"

#include <cmath>
#include <chrono>

#include <Eigen/Dense>

#include "core/c3_miqp.h"
#include "core/c3_plus.h"
#include "core/c3_qp.h"
#include "multibody/lcs_factory.h"
#include "multibody/geom_geom_collider.h"
#include "common/quaternion_error_hessian.h"

#include "drake/common/text_logging.h"
#include <drake/multibody/parsing/parser.h>
#include <chrono>

using drake::multibody::ModelInstanceIndex;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::BasicVector;
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
using drake::math::RotationMatrix;
using drake::math::RollPitchYaw;
using drake::multibody::ContactResults;
using drake::geometry::GeometryId;
using drake::SortedPair;

namespace c3 {
namespace systems {

MSiC3::MSiC3(MultibodyPlant<double>& plant, MultibodyPlant<drake::AutoDiffXd>& plant_ad, 
  MultibodyPlant<double>& plant_rollout, MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout, 
  drake::systems::Diagram<double>& rollout_diagram, std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context,    
  const vector<SortedPair<GeometryId>>& contact_geoms, const vector<SortedPair<GeometryId>>& contact_geoms_rollout,
  C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, int example_idx)
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

  if (use_drake_sim_) {
    simulator_ = std::make_unique<drake::systems::Simulator<double>>(
        rollout_diagram_, std::move(rollout_diagram_context_)
    );
  }
}

tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>, vector<vector<MatrixXd>>, vector<vector<VectorXd>>, 
  vector<vector<MatrixXd>>, vector<vector<VectorXd>>, vector<vector<vector<MatrixXd>>>> MSiC3::ComputeTrajectory(
  drake::systems::Context<double>& context,
  drake::systems::Context<drake::AutoDiffXd>& context_ad, 
  drake::systems::Context<double>& context_rollout,
  drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout) {

  // num segements must divide the entire time horizon (might be unnecessary but for ease of implementation)
  DRAKE_DEMAND((double)(N_ / num_segments_) == (double)N_ / num_segments_);

  std::vector<double> x_init = *controller_options_.x_init;
  VectorXd x0 = Eigen::Map<VectorXd>(x_init.data(), x_init.size());    

  std::vector<double> x_des = *controller_options_.x_des;
  VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());  
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

  std::cout << "rollout factory before " << std::endl;
  LCSFactory lcs_factory_rollout(plant_rollout_, context_rollout, plant_ad_rollout_,
      context_ad_rollout, contact_geoms_rollout_, controller_options_.lcs_factory_options);
  std::cout << "rollout factory after " << std::endl;

  // Set initial guess to something kinda reasonable
  // Set initial guess for x - linear interpolation (including in quaternion space)
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

      if (-1e-3 < q0.dot(qd) && q0.dot(qd) < 1e-3) { 
        // Fallback for antipodal points, use linear interpolation in R3 to get default axis
        Eigen::Vector4d mid = v0 + vd;
        Eigen::Vector4d tangent = (mid - mid.dot(v0) * v0).normalized();

        double theta = std::acos(std::clamp(v0.dot(vd), -1.0, 1.0)); 
        Eigen::Vector4d v_interpolated = v0 * std::cos(rotation * theta) + tangent * std::sin(rotation * theta);

        x_hat.col(k).segment(idx, 4) = v_interpolated;

      } else {
        Eigen::Quaterniond slerp = q0.slerp(rotation, qd);
        x_hat.col(k).segment(idx, 4) << slerp.w(), slerp.x(), slerp.y(), slerp.z(); 
      }
    }

    if (k < N_) {
      u_hat.col(k) = gravity;
    }
  }

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

    lower_bound_x(0) = -0.2;
    lower_bound_x(1) = -0.2;
    lower_bound_x(2) = -0.2; 
    lower_bound_x(3) = -0.6;
    lower_bound_x(4) = -0.6;

    upper_bound_x(0) = 0.2;
    upper_bound_x(1) = 0.2;
    upper_bound_x(2) = 0.2;
    upper_bound_x(3) = 0.6;
    upper_bound_x(4) = 0.6;

    // Actuation limits
    A_u(2, 2) = 1;
    A_u(3, 3) = 1;
    A_u(4, 4) = 1;

    lower_bound_u(2) = 0;
    lower_bound_u(3) = -2;
    lower_bound_u(4) = -2;

    upper_bound_u(2) = 15;
    upper_bound_u(3) = 2;
    upper_bound_u(4) = 2;

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

      lower_bound_x(16 + 3*i) = -0.075;
      lower_bound_x(16 + 3*i+1) = -0.075;
      lower_bound_x(16 + 3*i+2) = -0.05;

      upper_bound_x(3*i) = xd(3*i) + 0.06;
      upper_bound_x(3*i+1) = xd(3*i+1) + 0.06;
      upper_bound_x(3*i+2) = xd(3*i+2) + 0.01;

      upper_bound_x(16 + 3*i) = 0.075;
      upper_bound_x(16 + 3*i+1) = 0.075;
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

    A_x(13, 13) = 1;
    A_x(14, 14) = 1;

    lower_bound_x(13) = -0.05;
    lower_bound_x(14) = -0.05;

    upper_bound_x(13) = 0.05;
    upper_bound_x(14) = 0.05;

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
      lower_bound_x(3*i) = xd(3*i) - 0.08;
      lower_bound_x(3*i+1) = xd(3*i+1) - 0.08;
      lower_bound_x(3*i+2) = xd(3*i+2) - 0.03;

      lower_bound_x(16 + 3*i) = -0.1;
      lower_bound_x(16 + 3*i+1) = -0.1;
      lower_bound_x(16 + 3*i+2) = -0.1;

      upper_bound_x(3*i) = xd(3*i) + 0.08;
      upper_bound_x(3*i+1) = xd(3*i+1) + 0.08;
      upper_bound_x(3*i+2) = xd(3*i+2) + 0.08;

      upper_bound_x(16 + 3*i) = 0.1;
      upper_bound_x(16 + 3*i+1) = 0.1;
      upper_bound_x(16 + 3*i+2) = 0.1;

      A_u(3*i, 3*i) = 1;
      A_u(3*i+1, 3*i+1) = 1;
      A_u(3*i+2, 3*i+2) = 1;

      lower_bound_u(3*i) = -0.5;
      lower_bound_u(3*i+1) = -0.5;
      lower_bound_u(3*i+2) = 0;
      
      upper_bound_u(3*i) = 0.5;
      upper_bound_u(3*i+1) = 0.5;
      upper_bound_u(3*i+2) = 0.4;
    }
  }


  // Get lambda_hat initial guess   
  if (use_drake_sim_) {
    for (int i = 0; i < N_; i++) {
      plant_rollout_.SetPositionsAndVelocities(&context_rollout, x_hat.col(i));

      plant_rollout_.get_actuation_input_port().FixValue(&context_rollout, u_hat.col(i));

      Context<double>& root_context = simulator_->get_mutable_context();
      double target_time = root_context.get_time() + dt_;
      simulator_->AdvanceTo(target_time);

      auto abstract_contact_results = drake::AbstractValue::Make<drake::multibody::ContactResults<double>>({});
      plant_rollout_.get_contact_results_output_port().Calc(context_rollout, abstract_contact_results.get());
      const auto& contact_results = abstract_contact_results->get_value<drake::multibody::ContactResults<double>>();
        
      // std::cout << plant_rollout_.GetPositionsAndVelocities(context_rollout).transpose() << std::endl;

      lambda_hat.col(i) = ConstructLambdasFromContactResults(contact_results, controller_options_.lcs_factory_options.contact_model);
      // std::cout << "lambda " << i << " " << lambda_hat.col(i).transpose() << std::endl;
    }  
  } else {
    std::vector<MatrixXd> K_init(N_, Eigen::MatrixXd::Identity(n_u_, n_u_));
    std::vector<VectorXd> K_ff_init(N_, VectorXd::Zero(n_u_));
    auto [lcs_init_out, x_hat_init_out, u_hat_init_out, lambda_hat_init_out] = DoLCSRollout(x0, u_hat, 
      lcs_factory, lcs_factory_rollout, MatrixXd::Zero(n_x_, n_x_), VectorXd::Zero(n_x_), VectorXd::Zero(n_x_), 
      MatrixXd::Zero(n_u_, n_u_), VectorXd::Zero(n_u_), VectorXd::Zero(n_u_), 
      K_init, K_ff_init, 0);
    lambda_hat = lambda_hat_init_out;
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
    defects.col(i) = x_hat.col(i * L_) - x_anchors.col(i);
  }


  vector<VectorXd> u_nominal(N_, gravity);

  vector<MatrixXd> all_x_hats;
  vector<MatrixXd> all_u_hats;
  vector<MatrixXd> all_lambda_hats;
  vector<MatrixXd> all_defects;
  vector<MatrixXd> all_x_anchors;

  vector<vector<MatrixXd>> Hs;
  vector<vector<VectorXd>> gs;
  vector<vector<MatrixXd>> Ks;
  vector<vector<VectorXd>> k_ffs;

  // ic3 iteration, ic3 timestep, admm iter, c3 timestep
  vector<vector<vector<MatrixXd>>> all_delta_projections;

  all_x_hats.push_back(x_hat);
  all_u_hats.push_back(u_hat);
  all_lambda_hats.push_back(lambda_hat);
  all_defects.push_back(defects);
  all_x_anchors.push_back(x_anchors);

  LCS lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);

  int num_iters = ms_ic3_options_.num_iters;
  int num_warmup_iters = ms_ic3_options_.num_warmup_iters;

  for (int iter = 1 - num_warmup_iters; iter <= num_iters; iter++) {
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << "iC3 iteration " << iter << std::endl;

    delta_projection_iter_.clear();

    bool is_warmup = (iter < 1);

    UpdateQuaternionCosts(x_hat, xd); // Note: this overrides R, G, U as well

    // Backwards Pass - Compute Value Function
    auto [H, g, K, k_ff] = ComputeLQRValueFunction(x_hat, u_hat, lambda_hat, lcs, xd, u_nominal[0], defects);

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

    // Forwards Pass - Do C3 MPC with value function terminal cost
    for (int i = 0; i < num_segments_; i++) {

      std::cout << "segment " << i << std::endl;

      auto [x_hat_out, u_hat_out, lambda_hat_out] = 
         DoC3Rollout(new_x_anchors.col(i), x_hat, u_hat.middleCols(i*L_, L_), gravity,
                      lcs_factory, lcs_factory_rollout, H, g, i*L_,
                      A_x, lower_bound_x, upper_bound_x, A_u, lower_bound_u, upper_bound_u,
                      context, context_rollout);


      VectorXd x_L = x_hat_out.col(L_);
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
      new_x_anchors.col(i+1) = x_projected;

      if (example_idx_ == 0) {
        std::cout << "x_hat[L] pancake: " << x_hat_out.col(L_).segment(5, 7).transpose() << std::endl << std::endl;
      } else if (example_idx_ == 1 || example_idx_ == 2) {
        std::cout << "x_hat[L] cube: " << x_hat_out.col(L_).segment(9, 7).transpose() << std::endl << std::endl;
      }

      defects.col(i+1) = x_hat_out.col(L_) - new_x_anchors.col(i+1);

      // Don't update initial x guess during warm start
      if (!is_warmup) {
        x_hat.middleCols(i*L_, L_) = x_hat_out.leftCols(L_);
      }
      u_hat.middleCols(i*L_, L_) = u_hat_out.leftCols(L_);
      lambda_hat.middleCols(i*L_, L_) = lambda_hat_out.leftCols(L_);

      if (i == num_segments_-1) { // Tack on final state if last segment
        x_hat.col(N_) = x_hat_out.col(L_);
      }
    } 

    x_anchors = new_x_anchors;

    // Linearize about new nominal trajectory
    lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);

    // Don't store if warmup
    if (!is_warmup) {
      all_x_hats.push_back(x_hat);
      all_u_hats.push_back(u_hat);
      all_lambda_hats.push_back(lambda_hat);
      all_defects.push_back(defects);
      all_x_anchors.push_back(x_anchors);
      all_delta_projections.push_back(delta_projection_iter_);
    }


    // Think up some logic for this, probably looking at some norm of the defects
    if (ms_ic3_options_.early_termination) {

    }

    // Print costs
    if (ms_ic3_options_.print_costs) {

      // std::cout << "finger defect costs: ";
      // for (int j = 0; j < num_segments_+1; j++) {
      //   std::cout << defects.col(j).segment(0, 9).transpose() * P_[j*L_].block(0, 0, 9, 9) * defects.col(j).segment(0, 9) << ", ";
      // }
      // std::cout << std::endl;

      // std::cout << "cube rotation defect costs: ";
      // for (int j = 0; j < num_segments_+1; j++) {
      //   std::cout <<   defects.col(j).segment(9, 4).transpose() * P_[j*L_].block(9, 9, 4, 4) * defects.col(j).segment(9, 4) << ", ";
      // }
      // std::cout << std::endl;

      // std::cout << "cube position defect costs: ";
      // for (int j = 0; j < num_segments_+1; j++) {
      //   std::cout << defects.col(j).segment(13, 3).transpose() * P_[j*L_].block(13, 13, 3, 3) * defects.col(j).segment(13, 3) << ", ";
      // }
      // std::cout << "\n " << std::endl;

      // double total_finger_defect_cost = 0;
      // double total_cube_rot_defect_cost = 0;
      // double total_cube_pos_defect_cost = 0;
      // double total_defect_cost = 0;

      //  for (int j = 0; j < num_segments_+1; j++) {
      //   total_finger_defect_cost += defects.col(j).segment(0, 9).transpose() *  P_[j*L_].block(0, 0, 9, 9) * defects.col(j).segment(0, 9);
      //   total_cube_rot_defect_cost += defects.col(j).segment(9, 4).transpose() *  P_[j*L_].block(9, 9, 4, 4) * defects.col(j).segment(9, 4);
      //   total_cube_pos_defect_cost += defects.col(j).segment(13, 3).transpose() *  P_[j*L_].block(13, 13, 3, 3) * defects.col(j).segment(13, 3);

      //   total_defect_cost += defects.col(j).transpose() * P_[j*L_] * defects.col(j);
      // }
      // std::cout << "FINGER DEFECT COST: " << total_finger_defect_cost << std::endl;
      // std::cout << "CUBE ROT DEFECT COST: " << total_cube_rot_defect_cost << std::endl;
      // std::cout << "CUBE POS DEFECT COST: " << total_cube_pos_defect_cost << std::endl;
      // std::cout << "TOTAL DEFECT COST: " << total_defect_cost << std::endl;
      // std::cout << std::endl;

    
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Iteration runtime: " << duration.count() << " seconds\n\n " << std::endl;

  }
  UpdateQuaternionCosts(x_hat, xd);
  std::cout << "Before compute lqr value function" << std::endl;
  auto [H, g, K, k_ff] = ComputeLQRValueFunction(x_hat, u_hat, lambda_hat, lcs, xd, u_nominal[0], defects);
  Hs.push_back(H);
  gs.push_back(g);
  Ks.push_back(K);
  k_ffs.push_back(k_ff);

  return std::make_tuple(all_x_hats, all_u_hats, all_lambda_hats, Hs, gs, Ks, k_ffs, all_delta_projections);
}

VectorXd MSiC3::ProjectContactVertical(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
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

    return x_out;
}

// TODO: clean up this function
VectorXd MSiC3::ProjectContact(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
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
tuple<LCS, MatrixXd, MatrixXd, MatrixXd> MSiC3::DoLCSRollout(VectorXd x0, MatrixXd u_hat, 
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

tuple<MatrixXd, MatrixXd, MatrixXd> MSiC3::DoC3Rollout(VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, VectorXd ud, 
                                              LCSFactory factory, LCSFactory rollout_factory, vector<MatrixXd> H, 
                                              vector<VectorXd> g, int start_idx,
                                              MatrixXd A_x, VectorXd lb_x, VectorXd ub_x,
                                              MatrixXd A_u, VectorXd lb_u, VectorXd ub_u,
                                              Context<double>& context, Context<double>& context_rollout) {
  // Assume that x_hat, H, g, x_targets correspond to the entire iC3 horizon

  DRAKE_DEMAND(start_idx < N_);
  DRAKE_DEMAND(H.size() == N_ + 1);
  DRAKE_DEMAND(g.size() == N_ + 1);
  DRAKE_DEMAND(x_hat.cols() == N_ + 1);

  int num_steps = u_hat.cols();

  int factor = ms_ic3_options_.rollout_dt_scaling;

  MatrixXd x_hat_output(n_x_, num_steps * factor + 1);
  MatrixXd lambda_hat(n_lambda_, num_steps * factor);
  MatrixXd u_hat_fb(n_u_, num_steps * factor);

  x_hat_output.col(0) = x0;
  VectorXd x_curr = x0;
  VectorXd x_next;

  // std::cout << "x anchor fingers " << x0.segment(0,9).transpose() << std::endl;
  factory.UpdateStateAndInput(x0, u_hat.col(0));
  LCS lcs_test = factory.GenerateLCS();
  // std::cout << "x anchor fingers " << x0.segment(0, 9).transpose() << std::endl;
  if (example_idx_ == 0) {
    std::cout << "x anchor pancake " << x0.segment(5, 7).transpose() << std::endl;
  } else if (example_idx_ == 1 || example_idx_ == 2) {
    std::cout << "x anchor ee " << x0.segment(0, 9).transpose() << std::endl;
    std::cout << "x anchor cube " << x0.segment(9, 7).transpose() << std::endl;
  }

  if (!controller_options_.x_des.has_value()) std::cerr << "Set x des" << std::endl;
  
  std::vector<double> x_des = controller_options_.x_des.value();
  VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());

  int tracking_N = controller_options_.lcs_factory_options.N;

  rollout_factory.SetNewDt(dt_ / factor);

  // Run receding horizon C3 MPC
  for (int t = 0; t < num_steps; t++) {

    VectorXd u_nominal = u_hat.col(t);
    
    vector<MatrixXd> Q;
    vector<MatrixXd> R;
    vector<MatrixXd> G;
    vector<MatrixXd> U;
    vector<VectorXd> u_targets_shortened;
    MatrixXd x_hat_for_lcs(MatrixXd::Zero(n_x_, tracking_N+1));
    MatrixXd u_hat_for_lcs(MatrixXd::Zero(n_u_, tracking_N));

    double discount_factor = 1;
    for (int i = 0; i < tracking_N + 1; i++) {
      int x_idx = std::min(N_, start_idx + t + i);
      int u_idx = std::min(num_steps-1, t + i);
      int R_idx = std::min(N_-1, start_idx + t + i);

      Q.push_back(discount_factor * Q_[x_idx]);
      x_hat_for_lcs.col(i) = x_hat.col(x_idx);

      if (i < tracking_N) {
        // u_targets_shortened.push_back(u_hat.col(u_idx));
        u_targets_shortened.push_back(ud);
        u_hat_for_lcs.col(i) = u_hat.col(u_idx);

        R.push_back(discount_factor * R_[R_idx]);
        G.push_back(discount_factor * G_[R_idx]);      
        U.push_back(discount_factor * U_[R_idx]);
      }
      discount_factor *= controller_options_.c3_options.gamma;
    }

    C3::CostMatrices costs(Q, R, G, U);

    costs = UpdateQuaternionCosts(x_curr, xd, costs);

    // HARDCODED ee start idx
    LCS lcs = MakeTimeVaryingLCSWithEE(x_hat_for_lcs, u_hat_for_lcs, factory, x_curr.segment(0, n_u_), 0);

    vector<VectorXd> x_targets;
    vector<double> norms;
    // x_targets.push_back(x_hat_for_lcs.col(0));
    x_targets.push_back(xd);
    norms.push_back(1.0);
    VectorXd x_temp = x_curr;
    for (int i = 1; i < tracking_N+1; i++) {
      VectorXd u_nominal = u_hat_for_lcs.col(i-1);
      x_temp = lcs.SimulateAtTimestep(x_temp, u_nominal, true, i-1);
      // VectorXd xd_copy = x_hat_for_lcs.col(i);
      VectorXd xd_copy = xd;
      for (auto quat_idx : controller_options_.quaternion_indices) {
        double norm = x_temp.segment(quat_idx, 4).norm();
        norms.push_back(norm);
        xd_copy.segment(quat_idx, 4) *= norm;
        // std::cout << "norm " << norm << std::endl;
      }
      x_targets.push_back(xd_copy);
    }

    std::unique_ptr<C3Plus> c3_tracking = std::make_unique<C3Plus>(lcs, costs, x_targets,
                                  controller_options_.c3_options);
    c3_tracking->UpdateInputTarget(u_targets_shortened);

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

    VectorXd bias = g[lqr_idx] - H[lqr_idx] * x_hat.col(lqr_idx);
    c3_tracking->UpdateFinalCost(H[lqr_idx], bias);

    auto c3_start = std::chrono::high_resolution_clock::now();
    c3_tracking->Solve(x_curr);
    auto c3_end = std::chrono::high_resolution_clock::now();
    auto c3_elapsed = c3_end - c3_start;
    double c3_solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(c3_elapsed).count() / 1e6;
    if (ms_ic3_options_.print_costs) {
      std::cout << "c3 solve time " << c3_solve_time << std::endl;
    }

    vector<MatrixXd> delta_proj = c3_tracking->GetDeltaProjection();
    delta_projection_iter_.push_back(delta_proj);

    vector<Eigen::VectorXd> z_sol = c3_tracking->GetFullSolution();

    // for (int ii = 0; ii < 1; ii++) {
    //   std::cout << "lambda " << ii << ": " << z_sol[ii].segment(n_x_, n_lambda_).transpose() << std::endl;
    //   std::cout << "eta " << ii << ": " << z_sol[ii].segment(n_x_+n_lambda_+n_u_, n_lambda_).transpose() << std::endl;
    //   std::cout << "Ex " << ii << ": " << (lcs.E()[ii] * z_sol[ii].segment(0, n_x_)).transpose() << std::endl;
    //   std::cout << "F\\lambda " << ii << ": " << (lcs.F()[ii] * z_sol[ii].segment(n_x_, n_lambda_)).transpose() << std::endl;
    //   std::cout << "Hu " << ii << ": " << (lcs.H()[ii] * z_sol[ii].segment(n_x_+n_lambda_, n_u_)).transpose() << std::endl;
    //   std::cout << "c " << ii << ": " << (lcs.c()[ii]).transpose() << std::endl;
    //   std::cout << std::endl;
    // }

    
    // int delta_size = delta_proj.size();
    // for (int ii = 0; ii < 4; ii++) {
    //   double AnDn = c3_tracking->GetAnDn();
    //   for (int jj = 0; jj < delta_size; jj++) {
    //     std::cout << "delta_proj lambda " << ii << " admm " << jj << ": " << AnDn * delta_proj[jj].col(ii).segment(n_x_, n_lambda_).transpose() << std::endl;
    //     std::cout << "delta_proj eta " << ii << " admm " << jj << ": " << AnDn * delta_proj[jj].col(ii).segment(n_x_+n_lambda_+n_u_, n_lambda_).transpose() << std::endl;
    //   }
    //   std::cout << "c3 lambda " << ii << ": " << AnDn * z_sol[ii].segment(n_x_, n_lambda_).transpose() << std::endl;
    //   std::cout << "c3 eta " << ii << ": " << AnDn * z_sol[ii].segment(n_x_+n_lambda_+n_u_, n_lambda_).transpose() << std::endl;

    //   // std::cout << "c3 pos " << ii << ": " << z_sol[ii].segment(0, n_q_).transpose() << std::endl;
    //   // std::cout << "c3 velo " << ii << ": " << z_sol[ii].segment(n_q_, n_v_).transpose() << std::endl;
      
    //   // std::cout << "delta_proj Ex " << ii << ": " << (lcs.E()[ii] * delta_proj[delta_size-1].col(ii).segment(0, n_x_)).transpose() << std::endl;
    //   // std::cout << "delta_proj F\\lambda " << ii << ": " << (lcs.F()[ii] * delta_proj[delta_size-1].col(ii).segment(n_x_, n_lambda_)).transpose() << std::endl;
    //   // std::cout << "delta_proj Hu " << ii << ": " << (lcs.H()[ii] * delta_proj[delta_size-1].col(ii).segment(n_x_+n_lambda_, n_u_)).transpose() << std::endl;
    //   // std::cout << "delta_proj c " << ii << ": " << (lcs.c()[ii]).transpose() << std::endl;
      

    //   std::cout << "c3 proj phi pred " << ii << ": ";
    //   for (int g = 0; g < 7; g++) {
    //     double phi_g_pred = dt_ * AnDn * delta_proj[delta_size-1].col(ii).segment(n_x_+n_lambda_+n_u_+4*g, 4).sum();
    //     std::cout << (phi_g_pred / 4) << ", ";       
    //   }
    //   std::cout << std::endl;

    //   std::cout << "c3 zsol phi pred " << ii << ": ";
    //   for (int g = 0; g < 7; g++) {
    //     double phi_g_pred = dt_ * AnDn * z_sol[ii].segment(n_x_+n_lambda_+n_u_+4*g, 4).sum();
    //     std::cout << (phi_g_pred / 4) << ", ";       
    //   }
    //   std::cout << std::endl;

    //   std::cout << "phis (zsol): ";
    //   for (int g = 0; g < 7; g++) {
    //     multibody::GeomGeomCollider collider(plant_, contact_geoms_[g]);
    //     plant_.SetPositionsAndVelocities(&context, z_sol[ii].segment(0, n_x_));
    //     auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
    //       drake::multibody::JacobianWrtVariable::kQDot);
    //     std::cout << phi << ", ";
    //   }
    //   std::cout << std::endl;

    //   std::cout << std::endl;
    // }

    // for (int ii = 0; ii < z_sol.size(); ii++) {
    //   std::cout << "z_sol velo " << z_sol.at(ii).segment(n_q_, n_v_).transpose() << std::endl;
    // }
    // for (int ii = 0; ii < z_sol.size()-1; ii++) {
    //   std::cout << "finite diff velo " << ((z_sol.at(ii+1) - z_sol.at(ii)) / dt_)
    //                                         .segment(n_q_, n_v_).transpose() << std::endl;
    // }

    // std::cout << "lambda proj " << delta_proj.at(delta_proj.size()-1).col(0).segment(n_x_, n_lambda_).transpose() << std::endl;
    // std::cout << "eta proj " << delta_proj.at(delta_proj.size()-1).col(0).segment(n_x_+n_lambda_+n_u_, n_lambda_).transpose() << std::endl;

    // std::cout << "lambda qp " << z_sol[0].segment(n_x_, n_lambda_).transpose() << std::endl;
    // std::cout << "eta qp " << z_sol[0].segment(n_x_+n_lambda_+n_u_, n_lambda_).transpose() << std::endl;

    VectorXd c3_u = z_sol[0].segment(n_x_ + n_lambda_, n_u_);
    VectorXd c3_x = z_sol[0].segment(0, n_x_);
    VectorXd c3_x_next = z_sol[1].segment(0, n_x_);

    if (ms_ic3_options_.print_costs) {
      if (example_idx_ == 1 || example_idx_ == 2) {
        for (int i = 0; i < z_sol.size(); i++) {
          double finger_pos_cost = (z_sol[i].segment(0, 9) - x_targets[i].segment(0, 9)).transpose() 
                                            * Q_[i].block(0, 0, 9, 9) * (z_sol[i].segment(0, 9) - x_targets[i].segment(0, 9));
          double cube_rot_cost = (z_sol[i].segment(9, 4) - x_targets[i].segment(9, 4)).transpose() 
                                            * Q_[i].block(9, 9, 4, 4) * (z_sol[i].segment(9, 4) - x_targets[i].segment(9, 4));
          double cube_pos_cost = (z_sol[i].segment(13, 3) - x_targets[i].segment(13, 3)).transpose() 
                                            * Q_[i].block(13, 13, 3, 3) * (z_sol[i].segment(13, 3) - x_targets[i].segment(13, 3));
          std::cout << "c3 cost " << i << " finger cost " << finger_pos_cost << ", cube rot " 
                << cube_rot_cost << ", cube pos " << cube_pos_cost << std::endl;
        }
        VectorXd x_final = c3_tracking->GetFinalStateSolution();
        double final_finger_pos_cost = (x_final.segment(0, 9) - x_hat.col(lqr_idx).segment(0, 9)).transpose() * 
                                          H[lqr_idx].block(0, 0, 9, 9) * (x_final.segment(0, 9) - x_hat.col(lqr_idx).segment(0, 9)); 
        double final_cube_rot_cost = (x_final.segment(9, 4) - x_hat.col(lqr_idx).segment(9, 4)).transpose() * 
                                          H[lqr_idx].block(9, 9, 4, 4) * (x_final.segment(9, 4) - x_hat.col(lqr_idx).segment(9, 4)); 
        double final_cube_pos_cost = (x_final.segment(13, 3) - x_hat.col(lqr_idx).segment(13, 3)).transpose() * 
                                          H[lqr_idx].block(13, 13, 3, 3) * (x_final.segment(13, 3) - x_hat.col(lqr_idx).segment(13, 3)); 

        std::cout << "c3 cost final "<< " finger cost " << final_finger_pos_cost << ", cube rot " 
                << final_cube_rot_cost << ", cube pos " << final_cube_pos_cost << std::endl;

        final_finger_pos_cost += g[lqr_idx].segment(0, 9).dot(x_final.segment(0, 9) - x_hat.col(lqr_idx).segment(0, 9));
        final_cube_rot_cost += g[lqr_idx].segment(9, 4).dot(x_final.segment(9, 4) - x_hat.col(lqr_idx).segment(9, 4));
        final_cube_pos_cost += g[lqr_idx].segment(13, 3).dot(x_final.segment(13, 3) - x_hat.col(lqr_idx).segment(13, 3));

        std::cout << "c3 cost final with affine " << " finger cost " << final_finger_pos_cost << ", cube rot " 
                << final_cube_rot_cost << ", cube pos " << final_cube_pos_cost << std::endl;
      }
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

      Context<double>& root_context = simulator_->get_mutable_context();

      for (int i = 0; i < factor; i++) {      
        plant_rollout_.SetPositionsAndVelocities(&context_rollout, x_curr);

        // Apply PD to C3 plan
        VectorXd c3_x_tracking = (i * c3_x_next + (factor-i) * c3_x) / factor;
        VectorXd u_tracking = c3_u + Kp * (c3_x_tracking.segment(q_idx, Kp.rows()) - x_curr.segment(q_idx, Kp.rows())) 
          + Kd * (c3_x_tracking.segment(v_idx, Kd.rows()) - x_curr.segment(v_idx, Kd.rows()));

        // std::cout << "u: " << u_tracking.transpose() << std::endl;

        // std::cout << u_tracking.transpose() << std::endl;
        plant_rollout_.get_actuation_input_port().FixValue(&context_rollout, u_tracking);

        double target_time = root_context.get_time() + dt_ / factor;

        auto simulator_start = std::chrono::high_resolution_clock::now();
        simulator_->AdvanceTo(target_time);
        auto simulator_end = std::chrono::high_resolution_clock::now();
        auto simulator_elapsed = simulator_end - simulator_start;
        double simulator_solve_time =
            std::chrono::duration_cast<std::chrono::microseconds>(simulator_elapsed).count() / 1e6;

        if (ms_ic3_options_.print_costs) {
          std::cout << "simulator time " << simulator_solve_time << std::endl;
        }

        x_next = plant_rollout_.GetPositionsAndVelocities(context_rollout);

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
        plant_rollout_.get_contact_results_output_port().Calc(context_rollout, abstract_contact_results.get());
        const auto& contact_results = abstract_contact_results->get_value<drake::multibody::ContactResults<double>>();

        // for (int i = 0; i < factor; i++) {
        x_hat_output.col(factor * t + i + 1) = x_next;
        u_hat_fb.col(factor * t + i) = u_tracking;
        lambda_hat.col(factor * t + i) = ConstructLambdasFromContactResults(contact_results, 
          controller_options_.lcs_factory_options.contact_model);
        // }

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


        rollout_factory.UpdateStateAndInput(x_curr, u_tracking);
        LCS lcs_rollout = rollout_factory.GenerateLCS();  

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
      std::cout << "rollout time " << rollout_solve_time << std::endl;
    }
  }

  MatrixXd x_hat_downsampled(MatrixXd::Zero(n_x_, num_steps + 1));
  MatrixXd u_hat_downsampled(MatrixXd::Zero(n_u_, num_steps));
  MatrixXd lambda_hat_downsampled(MatrixXd::Zero(n_lambda_, num_steps));

  for (int i = 0; i < num_steps; i++) {
    x_hat_downsampled.col(i) = x_hat_output.col(i * factor);
    u_hat_downsampled.col(i) = u_hat_fb.col(i * factor);
    lambda_hat_downsampled.col(i) = lambda_hat.col(i * factor);
  }
  x_hat_downsampled.col(num_steps) = x_hat_output.col(num_steps * factor);

  return {x_hat_downsampled, u_hat_downsampled, lambda_hat_downsampled};                                              
}

std::tuple<vector<MatrixXd>, vector<VectorXd>, vector<MatrixXd>, vector<VectorXd>> 
  MSiC3::ComputeLQRValueFunction(MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, 
        LCS lcs, VectorXd xd, VectorXd ud, MatrixXd defects) {
  
  vector<MatrixXd> A = lcs.A();
  vector<MatrixXd> B = lcs.B();
  vector<MatrixXd> D = lcs.D();
  vector<VectorXd> d = lcs.d();
  vector<MatrixXd> Q = Q_;
  vector<MatrixXd> R = R_;
  vector<MatrixXd> P = P_;    

  vector<VectorXd> c; // Bias term from contact forces
  for (int t = 0; t < N_; t++) {
    c.push_back(D[t] * lambda_hat.col(t) + d[t]);
    // if (t % 10 == 0) {
      // std::cout << "D lambda " << t << " " << (D[t] * lambda_hat.col(t)).transpose() << std::endl;
      // std::cout << "lambda " << lambda_hat.col(t).transpose() << std::endl;
    // }

  }

  // Solve time-varying affine LQR about nominal trajectory
  vector<MatrixXd> H(N_+1, MatrixXd::Zero(n_x_, n_x_));
  vector<VectorXd> g(N_+1, VectorXd::Zero(n_x_));
  vector<MatrixXd> K(N_, MatrixXd::Zero(n_u_, n_x_));
  vector<VectorXd> k_ff(N_, VectorXd::Zero(n_u_));    

  H[N_] = Q[N_] + P[N_]; // terminal condition

  for (int t = N_-1; t >= 0; t--) {
    int k = t / L_;

    VectorXd x_t = x_hat.col(t);
    VectorXd u_t = u_hat.col(t);
          
    MatrixXd Q_xx = Q[t] + A[t].transpose()*(H[t+1]+P[t])*A[t];
    MatrixXd Q_uu = R[t] + B[t].transpose()*(H[t+1]+P[t])*B[t];
    MatrixXd Q_ux = B[t].transpose()*(H[t+1]+P[t])*A[t];

    VectorXd Q_x = Q[t]*(x_t - xd) + A[t].transpose()*g[t+1] + 
                    A[t].transpose()*(H[t+1]+P[t])*(c[t]+defects.col(k+1));

    VectorXd Q_u = R[t]*(u_t - ud) + B[t].transpose()*g[t+1] + 
                    B[t].transpose()*(H[t+1]+P[t])*(c[t]+defects.col(k+1));

    Eigen::LDLT<MatrixXd> solver(Q_uu);
    K[t] = -solver.solve(Q_ux);
    k_ff[t] = -solver.solve(Q_u);

    double reg = 1e-5;

    H[t] = Q_xx - Q_ux.transpose() * solver.solve(Q_ux) + reg * MatrixXd::Identity(n_x_, n_x_);
    g[t] = Q_x  - Q_ux.transpose() * solver.solve(Q_u);     


    // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver_H(H[k]);
    // std::cout << "min eigenvalue: " << solver_H.eigenvalues().minCoeff() << std::endl;

  }

  return std::make_tuple(H, g, K, k_ff);
}


LCS MSiC3::MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory) {
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

LCS MSiC3::MakeTimeVaryingLCSWithEE(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory, VectorXd ee_pose, int ee_idx) {
  MatrixXd x_hat_copy = x_hat;
  for (int i = 0; i < x_hat_copy.cols(); i++) {
    x_hat_copy.col(i).segment(ee_idx, ee_pose.size()) = ee_pose;
  }
  return MakeTimeVaryingLCS(x_hat_copy, u_hat, factory);
}

LCS MSiC3::GetLCSSegment(LCS lcs, int start_idx, int length) {
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


VectorXd MSiC3::ConstructLambdasFromContactResults(ContactResults<double> contact_results, std::string contact_model) {

  // Assumes 2 friction directions
  VectorXd lambda(VectorXd::Zero(n_lambda_));

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

        // std::cout << "contact " << i << std::endl;
        Vector3d n_W;
        Vector3d f_W;

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
  return lambda;
}

void MSiC3::UpdateQuaternionCosts(
  MatrixXd x_hat, VectorXd x_des) {
  
  // std::cout << x_hat.rows() << ", " << x_hat.cols() << std::endl;
  // std::cout << "xd: " << x_des.transpose() << std::endl;
  // std::cout << c3_quat_norms.size() << std::endl;

  Q_.clear();
  R_.clear();
  G_.clear();
  U_.clear();

  P_.clear();

  std::cout << ms_ic3_options_.P.rows() << ", " << ms_ic3_options_.P.cols() << std::endl;

  for (int i = 0; i < N_+1; i++) {
    Q_.push_back(controller_options_.c3_options.Q);
    P_.push_back(ms_ic3_options_.P);
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

      Eigen::MatrixXd P_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd P_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
      Eigen::MatrixXd P_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

      P_[i].block(index, index, 4, 4) = 
        ms_ic3_options_.w_P * 
        ms_ic3_options_.defect_quaternion_weight * (quat_hessian_i + P_quat_regularizer_1 + 
        ms_ic3_options_.defect_quaternion_regularizer_fraction * P_quat_regularizer_2 + P_quat_regularizer_3);

      // double q_min_eigenval = Q_[i].eigenvalues().real().minCoeff();
      // std::cout << "Q_" << i << " min eigenvalue " <<  q_min_eigenval << std::endl;
      j++;
    }
  }
  //std::cout << std::endl;
  //Q_[N_] = Q_[N_-1];
}


void MSiC3::UpdateQuaternionCostAtIdx(
  VectorXd x_curr, VectorXd x_des, int idx) {

  for (int index : controller_options_.quaternion_indices) {

    // make quaternion costs time-varying based on x_hat
    Eigen::VectorXd quat_curr_i = x_curr.segment(index, 4).normalized();
    Eigen::VectorXd quat_des_i = x_des.segment(index, 4).normalized();

    Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

    // Regularize hessian so Q is always PSD
    double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
    //std::cout << min_eigenval << std::endl;

    Eigen::MatrixXd Q_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd Q_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
    Eigen::MatrixXd Q_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

    Q_[idx].block(index, index, 4, 4) = 
      controller_options_.c3_options.w_Q * 
      controller_options_.Q_quaternion_weight * (quat_hessian_i + Q_quat_regularizer_1 + 
      controller_options_.quaternion_regularizer_fraction * Q_quat_regularizer_2 + Q_quat_regularizer_3);

    Eigen::MatrixXd P_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd P_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
    Eigen::MatrixXd P_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

    P_[idx].block(index, index, 4, 4) = 
      ms_ic3_options_.w_P * 
      ms_ic3_options_.defect_quaternion_weight * (quat_hessian_i + P_quat_regularizer_1 + 
      ms_ic3_options_.defect_quaternion_regularizer_fraction * P_quat_regularizer_2 + P_quat_regularizer_3);

    // double q_min_eigenval = Q_[i].eigenvalues().real().minCoeff();
    // std::cout << "Q_" << i << " min eigenvalue " <<  q_min_eigenval << std::endl;
  }

}


C3::CostMatrices MSiC3::UpdateQuaternionCosts(
    VectorXd x_curr, VectorXd x_des, C3::CostMatrices costs) {
  
  // std::cout << x_hat.rows() << ", " << x_hat.cols() << std::endl;
  // std::cout << "xd: " << x_des.transpose() << std::endl;
  // std::cout << c3_quat_norms.size() << std::endl;

  vector<MatrixXd> Q = costs.Q;
  vector<MatrixXd> R = costs.R;
  vector<MatrixXd> G = costs.G;
  vector<MatrixXd> U = costs.U;

  for (int i = 0; i < Q.size(); i++) {
    int j = 0;
    for (int index : controller_options_.quaternion_indices) {

      // make quaternion costs time-varying based on x_hat
      Eigen::VectorXd quat_curr_i = x_curr.segment(index, 4).normalized();
      Eigen::VectorXd quat_des_i = x_des.segment(index, 4).normalized();

      //std::cout << "xhat q: " << quat_curr_i.transpose() << std::endl;

      Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

      // Regularize hessian so Q is always PSD
      double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
      //std::cout << min_eigenval << std::endl;

      Eigen::MatrixXd Q_quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
      Eigen::MatrixXd Q_quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
      Eigen::MatrixXd Q_quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

      Q[i].block(index, index, 4, 4) = 
        controller_options_.c3_options.w_Q * 
        controller_options_.Q_quaternion_weight * (quat_hessian_i + Q_quat_regularizer_1 + 
        controller_options_.quaternion_regularizer_fraction * Q_quat_regularizer_2 + Q_quat_regularizer_3);

      // double q_min_eigenval = Q_[i].eigenvalues().real().minCoeff();
      // std::cout << "Q_" << i << " min eigenvalue " <<  q_min_eigenval << std::endl;
      j++;
    }
  }
  return C3::CostMatrices(Q, R, G, U);
}

} // namespace systems
} // namespace c3