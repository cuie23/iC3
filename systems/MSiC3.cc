#include "MSiC3.h"

#include <cmath>

#include <Eigen/Dense>

#include "core/c3_miqp.h"
#include "core/c3_plus.h"
#include "core/c3_qp.h"
#include "multibody/lcs_factory.h"
#include "multibody/geom_geom_collider.h"
#include "common/quaternion_error_hessian.h"

#include "drake/common/text_logging.h"
#include <drake/multibody/parsing/parser.h>

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

namespace c3 {
namespace systems {

MSiC3::MSiC3(MultibodyPlant<double>& plant, MultibodyPlant<drake::AutoDiffXd>& plant_ad, 
  MultibodyPlant<double>& plant_rollout, MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout, 
  C3ControllerOptions controller_options, MSiC3Options ms_ic3_options, int example_idx)
    : plant_(plant),
      plant_ad_(plant_ad),
      plant_rollout_(plant_rollout),
      plant_ad_rollout_(plant_ad_rollout),
      controller_options_(controller_options),
      ms_ic3_options_(ms_ic3_options),
      N_(ms_ic3_options.N),
      example_idx_(example_idx) {

  // Initialize dimensions
  n_q_ = plant_.num_positions();
  n_v_ = plant_.num_velocities();
  n_u_ = plant_.num_actuators();
  n_x_ = n_q_ + n_v_;
  dt_ = controller_options_.lcs_factory_options.dt;

  // Determine the size of lambda based on the contact model
  n_lambda_ = multibody::LCSFactory::GetNumContactVariables(
      controller_options_.lcs_factory_options);

  std::cout << "n lambda: " << n_lambda_ << std::endl;

  num_segments_ = ms_ic3_options_.num_segments;
  L_ = N_ / num_segments_;
}

tuple<vector<MatrixXd>, vector<MatrixXd>, vector<vector<MatrixXd>>, vector<vector<VectorXd>>, 
  vector<vector<MatrixXd>>, vector<vector<VectorXd>>> MSiC3::ComputeTrajectory(
  drake::systems::Context<double>& context,
  drake::systems::Context<drake::AutoDiffXd>& context_ad, 
  drake::systems::Context<double>& context_rollout,
  drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout, 
  const std::vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms,
  const std::vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms_rollout) {

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
    gravity[2] = 5;
  } else if (example_idx_ == 1) {
    gravity = VectorXd::Zero(9);
    gravity[2] = 0.196;
    gravity[5] = 0.196;
    gravity[8] = 0.196;
  }

  LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
      contact_geoms, controller_options_.lcs_factory_options);

  LCSFactory lcs_factory_rollout(plant_rollout_, context_rollout, plant_ad_rollout_,
      context_ad_rollout, contact_geoms_rollout, controller_options_.lcs_factory_options);

  // Set initial guess to something kinda reasonable
  // Set initial guess for x - linear interpolation (including in quaternion space)
  VectorXd x_diff = xd - x0;
  for (int k = 0; k < N_+1; k++) {
    x_hat.col(k) = x0 + k * x_diff / (N_+1);
    
    // Linearly interpolate quaternions correctly
    for (auto idx : controller_options_.quaternion_indices) {
      double rotation = (double)k / (N_+1);

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
        x_hat.col(k).segment(idx, 4) = q0.slerp(rotation, qd).coeffs(); 
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
  if (n_u_ == 5) { // plate
  } else if (n_u_ == 9) { // trifinger

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
      lower_bound_x(3*i) = xd(3*i) - 0.1;
      lower_bound_x(3*i+1) = xd(3*i+1) - 0.1;
      lower_bound_x(3*i+2) = xd(3*i+2) - 0.01;

      lower_bound_x(16 + 3*i) = -0.25;
      lower_bound_x(16 + 3*i+1) = -0.25;
      lower_bound_x(16 + 3*i+2) = -0.05;

      upper_bound_x(3*i) = xd(3*i) + 0.1;
      upper_bound_x(3*i+1) = xd(3*i+1) + 0.1;
      upper_bound_x(3*i+2) = xd(3*i+2) + 0.01;

      upper_bound_x(16 + 3*i) = 0.25;
      upper_bound_x(16 + 3*i+1) = 0.25;
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
    A_x(15, 15) = 1;

    lower_bound_x(13) = -0.06;
    lower_bound_x(14) = -0.06;
    lower_bound_x(15) = -0.06;

    upper_bound_x(13) = 0.06;
    upper_bound_x(14) = 0.06;
    upper_bound_x(15) = 0.06;
  }


  // Get lambda_hat initial guess   
  std::vector<MatrixXd> K_init(N_, Eigen::MatrixXd::Identity(n_u_, n_u_));
  std::vector<VectorXd> K_ff_init(N_, VectorXd::Zero(n_u_));
  auto [lcs_init_out, x_hat_init_out, u_hat_init_out, lambda_hat_init_out] = DoLCSRollout(x0, x_hat, x_hat.leftCols(N_), u_hat, 
    lcs_factory, lcs_factory_rollout, MatrixXd::Zero(n_x_, n_x_), VectorXd::Zero(n_x_), VectorXd::Zero(n_x_), 
    MatrixXd::Zero(n_u_, n_u_), VectorXd::Zero(n_u_), VectorXd::Zero(n_u_), 
    K_init, K_ff_init, 0);
  lambda_hat = lambda_hat_init_out;

  for (int i = 0; i < num_segments_+1; i++) {
    // Ensure anchors don't have penetration
    VectorXd x_projected = x_hat.col(i * L_);
    if (example_idx_ == 1) {
      for (int i = 0; i < 3; i++) {
        x_projected = ProjectContact(context, contact_geoms[i], x_projected, 3*i, 3);
      }
    }
    x_anchors.col(i) = x_projected;

    // Threshold so fingers are within joint limits
    for (int i = 0; i < A_x.rows(); i++) {
      if (A_x(i, i) != 0) { // Assumes diagonal
        x_projected(i) = std::min(std::max(x_projected(i), lower_bound_x(i)), upper_bound_x(i));
      }
    }
  
    defects.col(i) = x_hat_init_out.col(i * L_) - x_anchors.col(i);
  }



  vector<VectorXd> u_nominal(N_, gravity);

  vector<MatrixXd> all_x_hats;
  vector<MatrixXd> all_u_hats;
  vector<MatrixXd> all_defects;
  vector<MatrixXd> all_x_anchors;

  vector<vector<MatrixXd>> Hs;
  vector<vector<VectorXd>> gs;
  vector<vector<MatrixXd>> Ks;
  vector<vector<VectorXd>> k_ffs;

  all_x_hats.push_back(x_hat);
  all_u_hats.push_back(u_hat);
  all_defects.push_back(defects);
  all_x_anchors.push_back(x_anchors);

  LCS lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);

  int num_iters = ms_ic3_options_.num_iters;
  for (int iter = 1; iter <= num_iters; iter++) {

    std::cout << "iC3 iteration " << iter << std::endl;
    UpdateQuaternionCosts(x_hat, xd); // Note: this overrides R, G, U as well

    // Backwards Pass - Compute Value Function
    auto [H, g, K, k_ff] = ComputeLQRValueFunction(x_hat, u_hat, lambda_hat, lcs, xd, u_nominal[0], defects);
    Hs.push_back(H);
    gs.push_back(g);
    Ks.push_back(K);
    k_ffs.push_back(k_ff);

    MatrixXd new_x_anchors(MatrixXd::Zero(n_x_, num_segments_+1));
    new_x_anchors.col(0) = x0;

    // Forwards Pass - Do C3 MPC with value function terminal cost
    for (int i = 0; i < num_segments_; i++) {

      std::cout << "segment " << i << std::endl;

      // Potentially change u_hat to just be u_nominal for every iteration
      auto [x_hat_out, u_hat_out, lambda_hat_out] = 
         DoC3Rollout(new_x_anchors.col(i), x_hat, u_hat.middleCols(i*L_, L_), 
                      lcs_factory, lcs_factory_rollout, H, g, x_targets, i*L_,
                      A_x, lower_bound_x, upper_bound_x, A_u, lower_bound_u, upper_bound_u);

      // Update anchor for next segment
      double alpha_ee = ms_ic3_options_.alpha_ee;
      double alpha_object = ms_ic3_options_.alpha_object;

      // HARDCODED INDICES
      if (example_idx_ == 1) {
        new_x_anchors.col(i+1).segment(0, 9) = x_hat_out.col(L_).segment(0, 9) - (1-alpha_ee) * (x_hat_out.col(L_) - x_anchors.col(i+1)).segment(0, 9);
        new_x_anchors.col(i+1).segment(16, 9) = x_hat_out.col(L_).segment(16, 9) - (1-alpha_ee) * (x_hat_out.col(L_) - x_anchors.col(i+1)).segment(16, 9);
        new_x_anchors.col(i+1).segment(9, 7) = x_hat_out.col(L_).segment(9, 7) - (1-alpha_ee) * (x_hat_out.col(L_) - x_anchors.col(i+1)).segment(9, 7);
        new_x_anchors.col(i+1).segment(25, 6) = x_hat_out.col(L_).segment(25, 6) - (1-alpha_ee) * (x_hat_out.col(L_) - x_anchors.col(i+1)).segment(25, 6);
      }
      
      // Ensure anchors don't have penetration
      VectorXd x_projected = new_x_anchors.col(i+1);
      if (example_idx_ == 1) {
        for (int i = 0; i < 3; i++) {
          x_projected = ProjectContact(context, contact_geoms[i], x_projected, 3*i, 3);
        }
        
        // Threshold so fingers are within joint limits
        for (int i = 0; i < A_x.rows(); i++) {
          if (A_x(i, i) != 0) { // Assumes diagonal
            x_projected(i) = std::min(std::max(x_projected(i), lower_bound_x(i)), upper_bound_x(i));
          }
        }
      }
      new_x_anchors.col(i+1) = x_projected;

      std::cout << "x_hat[L] cube: " << x_hat_out.col(L_).segment(9, 7).transpose() << std::endl << std::endl;

      defects.col(i+1) = x_hat_out.col(L_) - new_x_anchors.col(i+1);


      x_hat.middleCols(i*L_, L_) = x_hat_out.leftCols(L_);
      u_hat.middleCols(i*L_, L_) = u_hat_out.leftCols(L_);
      lambda_hat.middleCols(i*L_, L_) = lambda_hat_out.leftCols(L_);

      if (i == num_segments_-1) { // Tack on final state if last segment
        x_hat.col(N_) = x_hat_out.col(L_);
      }
    } 

    x_anchors = new_x_anchors;

    // Linearize about new nominal trajectory
    lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);
    
    all_x_hats.push_back(x_hat);
    all_u_hats.push_back(u_hat);
    all_defects.push_back(defects);
    all_x_anchors.push_back(x_anchors);

    // Think up some logic for this, probably looking at some norm of the defects
    if (ms_ic3_options_.early_termination) {

    }

    // Print costs
    if (ms_ic3_options_.print_costs) {

      std::cout << "finger defect costs: ";
      for (int j = 0; j < num_segments_+1; j++) {
        std::cout << defects.col(j).segment(0, 9).transpose() * P_[j*L_].block(0, 0, 9, 9) * defects.col(j).segment(0, 9) << ", ";
      }
      std::cout << std::endl;

      std::cout << "cube rotation defect costs: ";
      for (int j = 0; j < num_segments_+1; j++) {
        std::cout <<   defects.col(j).segment(9, 4).transpose() * P_[j*L_].block(9, 9, 4, 4) * defects.col(j).segment(9, 4) << ", ";
      }
      std::cout << std::endl;

      std::cout << "cube position defect costs: ";
      for (int j = 0; j < num_segments_+1; j++) {
        std::cout << defects.col(j).segment(13, 3).transpose() * P_[j*L_].block(13, 13, 3, 3) * defects.col(j).segment(13, 3) << ", ";
      }
      std::cout << "\n " << std::endl;

      double total_finger_defect_cost = 0;
      double total_cube_rot_defect_cost = 0;
      double total_cube_pos_defect_cost = 0;
      double total_defect_cost = 0;

       for (int j = 0; j < num_segments_+1; j++) {
        total_finger_defect_cost += defects.col(j).segment(0, 9).transpose() *  P_[j*L_].block(0, 0, 9, 9) * defects.col(j).segment(0, 9);
        total_cube_rot_defect_cost += defects.col(j).segment(9, 4).transpose() *  P_[j*L_].block(9, 9, 4, 4) * defects.col(j).segment(9, 4);
        total_cube_pos_defect_cost += defects.col(j).segment(13, 3).transpose() *  P_[j*L_].block(13, 13, 3, 3) * defects.col(j).segment(13, 3);

        total_defect_cost += defects.col(j).transpose() * P_[j*L_] * defects.col(j);
      }
      std::cout << "FINGER DEFECT COST: " << total_finger_defect_cost << std::endl;
      std::cout << "CUBE ROT DEFECT COST: " << total_cube_rot_defect_cost << std::endl;
      std::cout << "CUBE POS DEFECT COST: " << total_cube_pos_defect_cost << std::endl;
      std::cout << "TOTAL DEFECT COST: " << total_defect_cost << std::endl;
      std::cout << "\n\n" << std::endl;

    
    }
    

  }
  UpdateQuaternionCosts(x_hat, xd);
  std::cout << "Before compute lqr value function" << std::endl;
  auto [H, g, K, k_ff] = ComputeLQRValueFunction(x_hat, u_hat, lambda_hat, lcs, xd, u_nominal[0], defects);
  Hs.push_back(H);
  gs.push_back(g);
  Ks.push_back(K);
  k_ffs.push_back(k_ff);

  return std::make_tuple(all_x_hats, all_u_hats, Hs, gs, Ks, k_ffs);
}


VectorXd MSiC3::ProjectContact(drake::systems::Context<double>& context, SortedPair<GeometryId> geom_pair, 
                                VectorXd x_init, int start_idx, int q_size) {

    VectorXd x_out = x_init;
    plant_.SetPositionsAndVelocities(&context, x_init);
    multibody::GeomGeomCollider collider(plant_, geom_pair);
    auto [phi, J] = collider.EvalPolytope(context, controller_options_.lcs_factory_options.num_friction_directions, 
        drake::multibody::JacobianWrtVariable::kQDot);

    if (phi < 0) {
      VectorXd contact_normal = J.row(0).segment(start_idx, q_size);
      x_out.segment(start_idx, q_size) = x_init.segment(start_idx, q_size) - (phi / (contact_normal.transpose() * contact_normal)) * contact_normal;
    }
    
    return x_out;
}

// TODO: clean this function up
tuple<LCS, MatrixXd, MatrixXd, MatrixXd> MSiC3::DoLCSRollout(VectorXd x0, MatrixXd x_hat_prev, MatrixXd c3_x_hat, 
  MatrixXd u_hat, LCSFactory factory, LCSFactory rollout_factory, MatrixXd A_constraint_x, VectorXd lower_bound_x, 
  VectorXd upper_bound_x, MatrixXd A_constraint_u, VectorXd lower_bound_u, VectorXd upper_bound_u, 
  vector<MatrixXd> K, vector<VectorXd> k_ff, double alpha) {

  DRAKE_DEMAND(c3_x_hat.cols() == u_hat.cols());
  DRAKE_DEMAND(K.size() >= u_hat.cols());
  DRAKE_DEMAND(k_ff.size() >= u_hat.cols());

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
    VectorXd x_nominal = x_hat_prev.col(k / factor);
    VectorXd c3_x = c3_x_hat.col(k / factor);

    MatrixXd Kp = ms_ic3_options_.rollout_Kp.asDiagonal();
    MatrixXd Kd = ms_ic3_options_.rollout_Kd.asDiagonal();

    for (int i = 0; i < A_constraint_u.rows(); i++) {
      if (A_constraint_u(i, i) != 0) { // Assumes diagonal
        u_nominal(i) = std::min(std::max(u_nominal(i), lower_bound_u(i)), upper_bound_u(i));
      }
    }

    VectorXd u_k = u_nominal + Kp * (c3_x.segment(q_idx, Kp.rows()) - x_curr.segment(q_idx, Kp.rows())) 
      + Kd * (c3_x.segment(v_idx, Kd.rows()) - x_curr.segment(v_idx, Kd.rows()));

    // VectorXd u_k = u_nominal + K[k / factor] * (x_curr - x_nominal) + alpha * k_ff[k / factor];
    // VectorXd u_k = u_nominal;
    u_hat_fb.col(k) = u_k;

    rollout_factory.UpdateStateAndInput(x_curr, u_k);
    LCS lcs = rollout_factory.GenerateLCS();     

    // Do one rollout step
    auto pair = lcs.SimulateAndReturnForce(x_curr, u_k, true);
    x_next = pair.first;

    // HARDCODED thresholding x's
    if (example_idx_ == 1) {
      for (int i = 0; i < A_constraint_x.rows(); i++) {
        if (A_constraint_x(i, i) != 0) { // Assumes diagonal
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

tuple<MatrixXd, MatrixXd, MatrixXd> MSiC3::DoC3Rollout(VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, 
                                              LCSFactory factory, LCSFactory rollout_factory, vector<MatrixXd> H, 
                                              vector<VectorXd> g, vector<VectorXd> x_targets, int start_idx,
                                              MatrixXd A_x, VectorXd lb_x, VectorXd ub_x,
                                              MatrixXd A_u, VectorXd lb_u, VectorXd ub_u) {
  // Assume that x_hat, H, g, x_targets correspond to the entire iC3 horizon

  DRAKE_DEMAND(start_idx < N_);
  DRAKE_DEMAND(H.size() == N_ + 1);
  DRAKE_DEMAND(g.size() == N_ + 1);
  DRAKE_DEMAND(x_hat.cols() == N_ + 1);
  DRAKE_DEMAND(x_targets.size() == N_ + 1);

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
  std::cout << "x anchor cube " << x0.segment(9, 7).transpose() << std::endl;

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
    vector<VectorXd> x_targets_shortened;
    vector<VectorXd> u_targets_shortened;
    MatrixXd x_hat_for_lcs(MatrixXd::Zero(n_x_, tracking_N+1));
    MatrixXd u_hat_for_lcs(MatrixXd::Zero(n_u_, tracking_N));

    double discount_factor = 1;
    for (int i = 0; i < tracking_N + 1; i++) {
      int x_idx = std::min(N_, start_idx + t + i);
      int u_idx = std::min(num_steps-1, start_idx + t + i);
      int R_idx = std::min(N_-1, start_idx + t + i);

      x_targets_shortened.push_back(x_targets.at(x_idx));
      Q.push_back(discount_factor * Q_[x_idx]);
      x_hat_for_lcs.col(i) = x_hat.col(x_idx);

      if (i < tracking_N) {
        u_targets_shortened.push_back(u_hat.col(u_idx));
        u_hat_for_lcs.col(i) = u_hat.col(u_idx);

        R.push_back(discount_factor * R_[R_idx]);
        G.push_back(discount_factor * G_[R_idx]);      
        U.push_back(discount_factor * U_[R_idx]);
      }
      discount_factor *= controller_options_.c3_options.gamma;
    }
    C3::CostMatrices costs(Q, R, G, U);

    // HARDCODED ee start idx
    LCS lcs = MakeTimeVaryingLCSWithEE(x_hat_for_lcs, u_hat_for_lcs, factory, x_curr.segment(0, n_u_), 0);

    std::unique_ptr<C3Plus> c3_tracking = std::make_unique<C3Plus>(lcs, costs, x_targets_shortened,
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
    int lqr_idx = std::min(start_idx + t, N_); 
    c3_tracking->UpdateFinalCost(H[lqr_idx], g[lqr_idx]);

    // std::cout << "x curr " << x_curr.transpose() << std::endl;
    c3_tracking->Solve(x_curr);

    vector<Eigen::VectorXd> z_sol = c3_tracking->GetFullSolution();
    VectorXd c3_u = z_sol[0].segment(n_x_ + n_lambda_, n_u_);
    VectorXd c3_x = z_sol[0].segment(0, n_x_);

    // std::cout << "c3 u: " << c3_u.transpose() << std::endl << std::endl;

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
      VectorXd u_tracking = c3_u + Kp * (c3_x.segment(q_idx, Kp.rows()) - x_curr.segment(q_idx, Kp.rows())) 
        + Kd * (c3_x.segment(v_idx, Kd.rows()) - x_curr.segment(v_idx, Kd.rows()));
      // std::cout << "tracking u " << u_tracking.transpose() << std::endl;

      rollout_factory.UpdateStateAndInput(x_curr, u_tracking);
      LCS lcs_rollout = rollout_factory.GenerateLCS();  

      // std::cout << "u tracking " << u_tracking.transpose() << std::endl;
      auto pair = lcs_rollout.SimulateAndReturnForce(x_curr, u_tracking, true);
      x_next = pair.first;

      if (example_idx_ == 1) {
        for (int j = 0; j < A_x.rows(); j++) {
          if (A_x(j, j) != 0) { // Assumes diagonal
            x_next(j) = std::min(std::max(x_next(j), lb_x(j)), ub_x(j));
          }
        }
      }
      x_hat_output.col(factor * t + i + 1) = x_next;
      lambda_hat.col(factor * t + i) = pair.second;
      u_hat_fb.col(factor * t + i) = u_tracking;

      x_curr = x_next;
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

    H[k] = Q_xx - Q_ux.transpose() * solver.solve(Q_ux) + reg * MatrixXd::Identity(n_x_, n_x_);
    g[k] = Q_x  - Q_ux.transpose() * solver.solve(Q_u);     


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

  for (int k = 0; k < N; k++) {
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

void MSiC3::UpdateQuaternionCosts(
  MatrixXd x_hat, const Eigen::VectorXd& x_des) {
  
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
      Eigen::VectorXd quat_des_i = x_des.segment(index, 4);

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

} // namespace systems
} // namespace c3