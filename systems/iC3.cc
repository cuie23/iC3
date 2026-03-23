#include "iC3.h"

#include <cmath>

#include <Eigen/Dense>

#include "core/c3_miqp.h"
#include "core/c3_plus.h"
#include "core/c3_qp.h"
#include "multibody/lcs_factory.h"
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

iC3::iC3(MultibodyPlant<double>& plant, MultibodyPlant<drake::AutoDiffXd>& plant_ad, 
  MultibodyPlant<double>& plant_rollout, MultibodyPlant<drake::AutoDiffXd>& plant_ad_rollout,
    C3::CostMatrices& costs, C3ControllerOptions controller_options, iC3Options ic3_options, int example_idx)
    : plant_(plant),
      plant_ad_(plant_ad),
      plant_rollout_(plant_rollout),
      plant_ad_rollout_(plant_ad_rollout),
      controller_options_(controller_options),
      ic3_options_(ic3_options),
      N_(controller_options_.lcs_factory_options.N),
      example_idx_(example_idx) {
  this->set_name("iC3");

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

  // Placeholder vector for initialization
  VectorXd zeros = VectorXd::Zero(n_x_ + n_lambda_ + n_u_);

  // Create placeholder LCS and desired state for the base C3 problem
  auto lcs_placeholder =
      LCS::CreatePlaceholderLCS(n_x_, n_u_, n_lambda_, N_, dt_);
  auto x_desired_placeholder =
      std::vector<VectorXd>(N_ + 1, VectorXd::Zero(n_x_));

  // Initialize the C3 problem based on the projection type
  if (controller_options_.projection_type == "MIQP") {
    c3_ =
        std::make_unique<C3MIQP>(lcs_placeholder, costs, x_desired_placeholder,
                                 controller_options_.c3_options);
  } else if (controller_options_.projection_type == "QP") {
    c3_ = std::make_unique<C3QP>(lcs_placeholder, costs, x_desired_placeholder,
                                 controller_options_.c3_options);
  } else if (controller_options_.projection_type == "C3+") {
    c3_ =
        std::make_unique<C3Plus>(lcs_placeholder, costs, x_desired_placeholder,
                                 controller_options_.c3_options);
  } else {
    drake::log()->error("Unknown projection type : {}",
                        controller_options_.projection_type);
  }
  DRAKE_THROW_UNLESS(c3_ != nullptr);

}

  tuple<vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>, vector<MatrixXd>,
    vector<MatrixXd>, vector<VectorXd>, vector<MatrixXd>, vector<VectorXd>> iC3::ComputeTrajectory(
    drake::systems::Context<double>& context,
    drake::systems::Context<drake::AutoDiffXd>& context_ad, 
    drake::systems::Context<double>& context_rollout,
    drake::systems::Context<drake::AutoDiffXd>& context_ad_rollout, 
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>& contact_geoms_rollout) {

    std::vector<double> x_init = *controller_options_.x_init;
    VectorXd x0 = Eigen::Map<VectorXd>(x_init.data(), x_init.size());    

    std::vector<double> x_des = *controller_options_.x_des;
    VectorXd xd = Eigen::Map<VectorXd>(x_des.data(), x_des.size());  

    // ith column = ith timestep
    MatrixXd x_hat = x0.replicate(1, N_+1);
    MatrixXd u_hat(Eigen::MatrixXd::Zero(n_u_, N_));
    MatrixXd c3_xs = MatrixXd::Zero(n_x_, N_);
    MatrixXd x_real = MatrixXd::Zero(n_x_, N_);
    MatrixXd lambda_hat = MatrixXd::Zero(n_lambda_, N_);

    // Set initial guess to something kinda reasonable
    // TODO: make this a yaml option or use drake slerp
    VectorXd gravity;
    if (example_idx_ == 0) {
      gravity = VectorXd::Zero(5);
      gravity[2] = 5;
    } else if (example_idx_ == 1) {
      gravity = VectorXd::Zero(16);
    } else if (example_idx_ == 2) {
      gravity = VectorXd::Zero(9);
      gravity[2] = 0.196;
      gravity[5] = 0.196;
      gravity[8] = 0.196;
      // gravity[11] = 1.96;

    }
  
    vector<VectorXd> u_nominal(N_, gravity);
    vector<VectorXd> u_sol_for_penalization(N_, gravity);

    VectorXd x_diff = xd - x0;
    for (int k = 0; k < N_+1; k++) {
      x_hat.col(k) = x0 + k * x_diff / (N_+1);
      if (k < N_) u_hat.col(k) = gravity;
    }

    vector<MatrixXd> all_x_hats;
    vector<MatrixXd> all_u_hats;
    vector<MatrixXd> all_c3_x;
    vector<MatrixXd> all_x_real;

    all_x_hats.push_back(x_hat);
    all_u_hats.push_back(u_hat);
    all_x_real.push_back(x_real);
    all_c3_x.push_back(x_hat);

    LCSFactory lcs_factory(plant_, context, plant_ad_, context_ad, 
        contact_geoms, controller_options_.lcs_factory_options);

    LCSFactory lcs_factory_rollout(plant_rollout_, context_rollout, plant_ad_rollout_,
       context_ad_rollout, contact_geoms_rollout, controller_options_.lcs_factory_options);

    LCS lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);

    vector<VectorXd> c3_quat_norms;
    for (int index : controller_options_.quaternion_indices) {
      c3_quat_norms.push_back(VectorXd::Ones(N_ + 1));
    }

    int num_iters = ic3_options_.num_iters;
    for (int iter = 1; iter <= num_iters; iter++) {

      std::cout << "iC3 iteration " << iter << std::endl;
      UpdateQuaternionCosts(x_hat, xd, c3_quat_norms);

      
      // Add actuation/position limits
      std::cout << "n_x_: " << n_x_ << std::endl;
      std::cout << "n_u_: " << n_u_ << std::endl;
      MatrixXd A(MatrixXd::Zero(n_x_, n_x_));
      MatrixXd A_u(MatrixXd::Zero(n_u_, n_u_));

      VectorXd lower_bound(VectorXd::Zero(n_x_));
      VectorXd upper_bound(VectorXd::Zero(n_x_));
      VectorXd lower_bound_u(VectorXd::Zero(n_u_));
      VectorXd upper_bound_u(VectorXd::Zero(n_u_));

      if (example_idx_ == 0) {
        // Plate position constraints
        A(0, 0) = 1;
        A(1, 1) = 1;
        A(2, 2) = 1;
        A(3, 3) = 1;
        A(4, 4) = 1;

        lower_bound(0) = -0.2;
        lower_bound(1) = -0.2;
        lower_bound(2) = -0.2; 
        lower_bound(3) = -0.6;
        lower_bound(4) = -0.6;

        upper_bound(0) = 0.2;
        upper_bound(1) = 0.2;
        upper_bound(2) = 0.2;
        upper_bound(3) = 0.6;
        upper_bound(4) = 0.6;

        // Actuation limits
        A_u(2, 2) = 1;
        A_u(3, 3) = 1;
        A_u(4, 4) = 1;

        lower_bound_u(2) = 0;
        lower_bound_u(3) = -1;
        lower_bound_u(4) = -1;

        upper_bound_u(2) = 15;
        upper_bound_u(3) = 1;
        upper_bound_u(4) = 1;

      } else if (example_idx_ == 1) {
        for (int i = 0; i < 16; i++) {
          A(i, i) = 1;
        }
        // non-thumb joint limits
        for (int i = 0; i < 3; i++) {
          lower_bound(4*i) = -0.47;
          lower_bound(4*i + 1) = -0.2;
          lower_bound(4*i + 2) = -0.17;
          lower_bound(4*i + 3) = -0.23;

          upper_bound(4*i) = 0.47;
          upper_bound(4*i + 1) = 1.61;
          upper_bound(4*i + 2) = 1.71;
          upper_bound(4*i + 3) = 1.62;
        }

        lower_bound(12) = 0.36;
        lower_bound(13) = -0.2;
        lower_bound(14) = -0.17;
        lower_bound(15) = -0.23;

        upper_bound(12) = 1.50;
        upper_bound(13) = 1.26;
        upper_bound(14) = 1.71;
        upper_bound(15) = 1.62;      
    } else if (example_idx_ == 2) {
      for (int i = 0; i < 3; i++) {
        // Position constraints
        A(3*i, 3*i) = 1;
        A(3*i+1, 3*i+1) = 1;
        A(3*i+2, 3*i+2) = 1;

        // Velocity constraints
        A(16 + 3*i, 16 + 3*i) = 1;
        A(16 + 3*i + 1, 16 + 3*i + 1) = 1;
        A(16 + 3*i + 2, 16 + 3*i + 2) = 1;

        // Offset from initial position
        lower_bound(3*i) = x0(3*i) - 0.1;
        lower_bound(3*i+1) = x0(3*i+1) - 0.1;
        lower_bound(3*i+2) = x0(3*i+2) - 0.01;

        lower_bound(16 + 3*i) = -0.5;
        lower_bound(16 + 3*i+1) = -0.5;
        lower_bound(16 + 3*i+2) = -0.05;

        upper_bound(3*i) = x0(3*i) + 0.1;
        upper_bound(3*i+1) = x0(3*i+1) + 0.1;
        upper_bound(3*i+2) = x0(3*i+2) + 0.01;

        upper_bound(16 + 3*i) = 0.5;
        upper_bound(16 + 3*i+1) = 0.5;
        upper_bound(16 + 3*i+2) = 0.05;


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

    }
    
      vector<Eigen::VectorXd> x_sol;
      vector<Eigen::VectorXd> u_sol;
      vector<Eigen::VectorXd> z_sol;
        
      std::vector<VectorXd> target =
        std::vector<VectorXd>(N_ + 1, xd);

      std::cout << ic3_options_.num_segments << std::endl;
      int segment_length = N_ / ic3_options_.num_segments;
      VectorXd x_start = x_hat.col(0);
      int indexer = 0;

      vector<VectorXd> u_sol_prev_iter = u_sol_for_penalization;

      for (int i = 0; i < ic3_options_.num_segments; i++) {
        LCS shortened_lcs = ShortenLCSFront(lcs, i * segment_length);
        C3::CostMatrices shortened_costs = ShortenCostsFront(i * segment_length);
        std::vector<VectorXd> shortened_targets;

        // Scale quaternions in target based on previous iteration
        for (int k = i * segment_length; k < N_+1; k++) {
          VectorXd target_k = target[k];
          for (int j = 0; j < controller_options_.quaternion_indices.size(); j++) {
              int index = controller_options_.quaternion_indices[j];
              double norm = c3_quat_norms[j](k);
              target_k.segment(index, 4) *= norm;
           }
           shortened_targets.push_back(target_k);
        }

        // Update c3_ to match new length
        c3_ = std::make_unique<C3Plus>(shortened_lcs, shortened_costs, shortened_targets,
                                 controller_options_.c3_options);

        if (controller_options_.c3_options.penalize_input_change) {
          int N_penalize = std::max(0, ic3_options_.N_penalize_input_change - i * segment_length);
          c3_->SetNPenalizeInputChange(N_penalize);
        }

        vector<VectorXd> u_nominal_short(u_nominal.begin() + i * segment_length, u_nominal.end());
        c3_->UpdateInputTarget(u_nominal_short);

        if ((controller_options_.c3_options.penalize_input_change)) {
          vector<VectorXd> u_sol_keep;
          if (i == 0) { 
            // Take from previous iC3 iteration
            std::cout << "u for penal size i=0: " << u_sol_for_penalization.size() << std::endl;
            u_sol_keep = u_sol_for_penalization;
            u_sol_for_penalization.clear();
          } else {
            //std::cout << "u sol size " << u_sol.size() << std::endl;
            //u_sol_keep = std::vector<VectorXd>(u_sol.begin() + segment_length, u_sol.end()); 
            u_sol_keep = std::vector<VectorXd>(u_sol_prev_iter.begin() + i * segment_length, 
                                              u_sol_prev_iter.end());            
          } 
          c3_->SetUSol(u_sol_keep);
        }
        
        if ((controller_options_.c3_options.warm_start)) {
          vector<VectorXd> warm_start_x;
          for (int i = 0; i < N_ + 1; i++) {
            warm_start_x.push_back(x_hat.col(i));
          }
          c3_->SetXSol(warm_start_x);
          c3_->SetUSol(u_sol_prev_iter);
        }

        if (ic3_options_.add_position_constraints) {
          c3_->AddLinearConstraint(A, lower_bound, upper_bound,
                                            ConstraintVariable::STATE);
        }
        if (ic3_options_.add_input_constraints) {
          c3_->AddLinearConstraint(A_u, lower_bound_u, upper_bound_u,
                                    ConstraintVariable::INPUT);
        }
        //std::cout << "Before c3 solve segment " << i << std::endl;
        c3_->Solve(x_start);
        if (i % 5 == 0) {
          std::cout << "after c3 solve segment " << i << std::endl;
        }

        z_sol = c3_->GetFullSolution();

        x_sol.clear();
        u_sol.clear();

        for (int r = 0; r < z_sol.size(); r++) {
          x_sol.push_back(z_sol[r].segment(0, n_x_));
          u_sol.push_back(z_sol[r].segment(n_x_ + n_lambda_, n_u_));

          // std::cout << "x sol " << r << " " << x_sol[r].segment(0, 16).transpose() << std::endl;
        }


        // Only keep segment_length x's and u's
        for (int j = 0; j < segment_length; j++) {
          c3_xs.col(indexer) = x_sol[j];
          u_hat.col(indexer) = u_sol[j];
          indexer++;
          u_sol_for_penalization.push_back(u_sol[j]);
        }
        if (i < ic3_options_.num_segments) { 
          if (example_idx_ == 2) {
            MatrixXd segment_u_hat(MatrixXd::Zero(n_u_, segment_length));
            for (int w = 0; w < segment_length; w++) {
              segment_u_hat.col(w) = u_sol[w];
            }
            
            // auto [lcs_out, x_hat_out, lambda_hat_out] = DoLCSRollout(x_start, segment_u_hat, lcs_factory_rollout, A, lower_bound, upper_bound);
            // x_start = x_hat_out.col(x_hat_out.cols()-1);
            x_start = x_sol[segment_length];
          } else {
            x_start = x_sol[segment_length];
          }
        }
        //std::cout << "After keeping stuff" << std::endl;
      }
      for (int i = indexer; i < N_; i++) {
        c3_xs.col(i) = x_sol[i - indexer];
        u_hat.col(i) = u_sol[i - indexer]; 
        u_sol_for_penalization.push_back(u_sol[i - indexer]);
      } 

      // Update norms of c3 quaternion outputs
      c3_quat_norms.clear();
      for (int index : controller_options_.quaternion_indices) {
        VectorXd c3_norm(VectorXd::Ones(N_+1));
        for (int i = 0; i < N_; i++) {
          c3_norm(i) = c3_xs.col(i).segment(index, 4).norm();
        } 
        c3_norm(N_) = c3_norm(N_-1); // c3 xsol only has N_ points
        c3_quat_norms.push_back(c3_norm);
        
      }

      auto lcs_start = std::chrono::high_resolution_clock::now();
      auto [lcs_out, x_hat_out, lambda_hat_out] = DoLCSRollout(x0, u_hat, lcs_factory_rollout, A, lower_bound, upper_bound);
      auto lcs_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> lcs_elapsed = lcs_end - lcs_start;
      std::cout << "LCS rollout time: " << lcs_elapsed.count() << " seconds\n";

      lcs = lcs_out;
      x_hat = x_hat_out;
      lambda_hat = lambda_hat_out;

      // for (int i = 0; i < x_hat.cols(); i++) {
      //   std::cout << x_hat.col(i).transpose() << std::endl;
      // }

      // if (example_idx_ == 0) {
      //   auto [lcs_out, x_hat_out, lambda_hat_out] = DoLCSRollout(x0, u_hat, lcs_factory);
      //   x_hat = x_hat_out;
      //   lambda_hat = lambda_hat_out;
      //   lcs = lcs_out;
      // } else if (example_idx_ == 1) {
      //   auto [x_hat_out, lambda_hat_out] = RolloutUHatHand(x0, c3_xs, u_hat, contact_geoms);
      //   x_hat = x_hat_out;
      //   lambda_hat = lambda_hat_out;
      //   lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory);
      // } else if (example_idx_ == 2) {
      //   auto rollout_start = std::chrono::high_resolution_clock::now();
      //   auto [x_hat_out, lambda_hat_out] = RolloutUHatPointHand(x0, c3_xs, u_hat, contact_geoms);
      //   auto rollout_end = std::chrono::high_resolution_clock::now();
      //   std::chrono::duration<double> rollout_elapsed = rollout_end - rollout_start;
      //   std::cout << "Rollout time: " << rollout_elapsed.count() << " seconds\n";


      //   x_hat = x_hat_out;
      //   lambda_hat = lambda_hat_out;       
      //   lcs = MakeTimeVaryingLCS(x_hat, u_hat, lcs_factory); 
      // }

      // if (example_idx_ == 0) {
      //   x_real = RolloutUHatPlate(x0, c3_xs, u_hat);
      // } else if (example_idx_ == 1) {
      //   // x_real = RolloutUHatHand(x0, u_hat);
      //   x_real = x_hat;
      // } else if (example_idx_ == 2) {
      //   x_real = x_hat;
      // }

      // normalize xhat quaternions
      for (int i = 0; i < x_hat.cols(); i++) {
        for (int index : controller_options_.quaternion_indices) {
          x_hat.col(i).segment(index, 4).normalize();
        }
      }

      all_x_hats.push_back(x_hat);
      all_u_hats.push_back(u_hat);
      all_c3_x.push_back(c3_xs);
      all_x_real.push_back(x_real);


      // Print costs
      // Assumes Q is diagonal
      if (ic3_options_.print_costs) {
        double x_cost = 0;
        double cube_pos_cost = 0;
        double plate_pos_cost = 0;
        double thumb_pos_cost = 0;
        double index_pos_cost = 0;
        double middle_pos_cost = 0;
        double ring_pos_cost = 0;

        double rot_cost = 0;
        double rot_cost_rollout = 0;
        double v_cost = 0;
        double u_cost = 0;

        double pos_cost_rollout = 0;
        double rot_angle_diff_rollout = 0;

        vector<VectorXd> xds;
        for (int k = 0; k < N_+1; k++) {
          VectorXd target_k = target[k];
          for (int j = 0; j < controller_options_.quaternion_indices.size(); j++) {
              int index = controller_options_.quaternion_indices[j];
              double norm = c3_quat_norms[j](k);
              target_k.segment(index, 4) *= norm;
           }
           xds.push_back(target_k);
        }

        // Since for all examples there's only one object
        int quat_idx = controller_options_.quaternion_indices[0];
        for (int i = 0; i < N_; i++) {
          VectorXd x_curr = c3_xs.col(i);
          VectorXd x_rollout = x_hat.col(i);
          VectorXd xd = xds[i];

          x_cost += (x_curr - xd).transpose() * Q_[i] * (x_curr - xd);

          rot_cost += (x_curr.segment(quat_idx, 4) - xd.segment(quat_idx, 4)).transpose() * 
              Q_[i].block(quat_idx, quat_idx, 4, 4) * (x_curr.segment(quat_idx, 4) - xd.segment(quat_idx, 4));
          rot_cost_rollout += (x_rollout.segment(quat_idx, 4) - xd.segment(quat_idx, 4)).transpose() * 
              Q_[i].block(quat_idx, quat_idx, 4, 4) * (x_rollout.segment(quat_idx, 4) - xd.segment(quat_idx, 4));

          VectorXd v_curr = x_curr.tail(n_v_);    
          VectorXd vd = xd.tail(n_v_);    

          v_cost += (v_curr - vd).transpose() * Q_[i].bottomRightCorner(n_v_, n_v_) * (v_curr - vd);

          Eigen::Vector4d v_des = xd.segment<4>(quat_idx).normalized();
          Eigen::Vector4d v_rollout = x_rollout.segment<4>(quat_idx).normalized();
          Eigen::Quaterniond quat_des(v_des[0], v_des[1], v_des[2], v_des[3]);
          Eigen::Quaterniond quat_rollout(v_rollout[0], v_rollout[1], v_rollout[2], v_rollout[3]);

          double dot = quat_des.dot(quat_rollout);
          double angle_diff = 2.0 * std::acos(std::min(1.0, std::abs(dot)));

          double angle_diff_deg = angle_diff * 180.0 / M_PI;
          rot_angle_diff_rollout += angle_diff_deg;

          Eigen::Quaterniond q_curr(x_curr(quat_idx), x_curr(quat_idx+1), x_curr(quat_idx+2), x_curr(quat_idx+3));
          Eigen::Quaterniond q_des(xd(quat_idx), xd(quat_idx+1), xd(quat_idx+2), xd(quat_idx+3));
          Eigen::AngleAxisd angle_axis(q_des * q_curr.inverse());
          double angle = angle_axis.angle();


          plate_pos_cost += (x_curr.segment(0, 3) - xd.segment(0, 3)).transpose() * 
              Q_[i].block(0, 0, 3, 3) * (x_curr.segment(0, 3) - xd.segment(0, 3));

          if (example_idx_ == 1) {
            index_pos_cost += (x_curr.segment(0, 4) - xd.segment(0, 4)).transpose() * 
                Q_[i].block(0, 0, 4, 4) * (x_curr.segment(0, 4) - xd.segment(0, 4));
            middle_pos_cost += (x_curr.segment(4, 4) - xd.segment(4, 4)).transpose() * 
                Q_[i].block(4, 4, 4, 4) * (x_curr.segment(4, 4) - xd.segment(4, 4));
            ring_pos_cost += (x_curr.segment(8, 4) - xd.segment(8, 4)).transpose() * 
                Q_[i].block(8, 8, 4, 4) * (x_curr.segment(8, 4) - xd.segment(8, 4));
            thumb_pos_cost += (x_curr.segment(12, 4) - xd.segment(12, 4)).transpose() * 
                Q_[i].block(12, 12, 4, 4) * (x_curr.segment(12, 4) - xd.segment(12, 4));
          } else if (example_idx_ == 2) {
            index_pos_cost += (x_curr.segment(0, 3) - xd.segment(0, 3)).transpose() * 
                Q_[i].block(0, 0, 3, 3) * (x_curr.segment(0, 3) - xd.segment(0, 3));
            middle_pos_cost += (x_curr.segment(3, 3) - xd.segment(3, 3)).transpose() * 
                Q_[i].block(3, 3, 3, 3) * (x_curr.segment(3, 3) - xd.segment(3, 3));
            ring_pos_cost += (x_curr.segment(6, 3) - xd.segment(6, 3)).transpose() * 
                Q_[i].block(6, 6, 3, 3) * (x_curr.segment(6, 3) - xd.segment(6, 3));
          }


          int cube_pos_idx;
          if (example_idx_ == 0) {
            cube_pos_idx = 9;
          } else if (example_idx_ == 1) {
            cube_pos_idx = 20;
          } else if (example_idx_ == 2) {
            cube_pos_idx = 13;
          }

          cube_pos_cost += (x_curr.segment(cube_pos_idx, 3) - xd.segment(cube_pos_idx, 3)).transpose() * 
              Q_[i].block(cube_pos_idx, cube_pos_idx, 3, 3) * (x_curr.segment(cube_pos_idx, 3) - xd.segment(cube_pos_idx, 3));

          pos_cost_rollout += (x_rollout.segment(cube_pos_idx, 3) - xd.segment(cube_pos_idx, 3)).transpose() * 
              Q_[i].block(cube_pos_idx, cube_pos_idx, 3, 3) * (x_rollout.segment(cube_pos_idx, 3) - xd.segment(cube_pos_idx, 3));
          
          // pos_cost += (x_curr.segment(0, 3) - xd.segment(0, 3)).transpose() * 
          //     Q_[i].block(0, 0, 3, 3) * (x_curr.segment(0, 3) - xd.segment(0, 3));

          // std::cout << "i: " << i << ", cube pos cost: " << (x_curr.segment(9, 3) - xd.segment(9, 3)).transpose() * 
          //     Q_[i].block(9, 9, 3, 3) * (x_curr.segment(9, 3) - xd.segment(9, 3)) << std::endl;
          // std::cout << "i: " << i << ", plate pos cost: " << (x_curr.segment(0, 3) - xd.segment(0, 3)).transpose() * 
          //     Q_[i].block(0, 0, 3, 3) * (x_curr.segment(0, 3) - xd.segment(0, 3)) << std::endl << std::endl;
                 

          VectorXd u_curr = u_hat.col(i);
          // if (i < 5 && !is_franka_) {
          //   std::cout << "u_" << i << ": " << u_curr.transpose() << std::endl;
          // }
          if (controller_options_.c3_options.penalize_input_change){
            VectorXd u_prev = u_sol_for_penalization[i];
            if (!u_prev.allFinite()) {
              std::cout << u_sol_for_penalization[i].transpose() << std::endl;
              std::cout << "u prev not all finite " << i << std::endl;
            }
            if (!u_curr.allFinite()) {
              std::cout << "u curr not all finite " << i << std::endl;
            }
            if (!R_[i].allFinite()) {
              std::cout << "R not all finite" << i << std::endl;
            }
            u_cost += (u_curr - u_prev).transpose() * R_[i] * (u_curr - u_prev);
            u_cost += (u_curr - gravity).transpose() * R_[i] * (u_curr - gravity);
          } else {
            u_cost += (u_curr - gravity).transpose() * R_[i] * (u_curr-gravity);
          }
          
          //std::cout << "u cost " << i << ": " << (u_curr - u_prev).transpose() * R_[i] * (u_curr - u_prev) << std::endl;;
        } 

        std::cout << "x cost: " << x_cost << std::endl;
        std::cout << "cube position cost: " << cube_pos_cost << std::endl;
        if (example_idx_ == 0) {
          std::cout << "plate position cost: " << plate_pos_cost << std::endl;
        } else if(example_idx_ == 1 || example_idx_ == 2) {
          std::cout << "finger 1 position cost: " << index_pos_cost << std::endl;
          std::cout << "finger 2 position cost: " << middle_pos_cost << std::endl;
          std::cout << "finger 3 position cost: " << ring_pos_cost << std::endl;
          if (example_idx_ == 1) {
            std::cout << "finger 4 position cost: " << thumb_pos_cost << std::endl;
          }

        }
        std::cout << "rotation cost: " << rot_cost << std::endl;
        std::cout << "rotation rollout cost: " << rot_cost_rollout << std::endl;
        std::cout << "position rollout cost: " << pos_cost_rollout << std::endl;
        std::cout << "avg rotation angle diff: " << rot_angle_diff_rollout / N_ << std::endl;
        std::cout << "v cost " << v_cost << std::endl;
        std::cout << "u cost " << u_cost << std::endl;

        // terminate early if rotation goal met
        if (ic3_options_.early_termination) {
          int matched_count = 0;
          vector<int> quat_idxs = controller_options_.quaternion_indices;

          for (int s = (int)(0.8 * N_); s < x_hat.cols(); s++) {

            for (int r = 0; r < quat_idxs.size(); r++) {
              VectorXd v_curr = x_hat.col(s).segment(quat_idxs[r], 4).normalized(); 
              VectorXd v_des = xd.segment(quat_idxs[r], 4).normalized();
              
              Eigen::Map<Eigen::Quaterniond> q_curr(v_curr.data());
              Eigen::Map<Eigen::Quaterniond> q_des(v_des.data());

              double angle = q_curr.angularDistance(q_des);
              std::cout << "s: " << s << ", angle: " << angle << std::endl;
              if (angle < 0.3) {
                matched_count++;
              }
            }
          }
          // Random heuristic for stopping condition
          std::cout << "matched count: " << matched_count << std::endl;
          if (matched_count > 0.5 * (0.2 * N_)) {
            iter = 99999;
          }
        }
      }
        
      

    }

    std::cout << std::endl;
    for (int i = 0; i < N_; i+= 5) {
      std::cout << "u_hat " << i << ": " << u_hat.col(i).transpose() << std::endl;
    }
    std::cout << std::endl;


    // for (int i = 0; i < N_; i++) {
    //   std::cout << "velo: " << (x_hat.col(i).segment(12, 5)).transpose() << std::endl;
    // }
  
    UpdateQuaternionCosts(x_hat, xd, c3_quat_norms);
    std::cout << "Before compute lqr value function" << std::endl;
    auto [H, g, K, k_ff] = ComputeLQRValueFunction(x_hat, u_hat, lambda_hat, xd, u_nominal[0], lcs);

    return std::make_tuple(all_x_hats, all_u_hats, all_c3_x, all_x_real, H, g, K, k_ff);
  }


  tuple<LCS, MatrixXd, MatrixXd> iC3::DoLCSRollout(VectorXd x0, MatrixXd u_hat, LCSFactory input_factory, 
      MatrixXd A_constraint, VectorXd lower_bound_x, VectorXd upper_bound_x) {

    int factor = ic3_options_.rollout_dt_scaling;
    int N = u_hat.cols();

    LCSFactory factory = input_factory;
    factory.SetNewDt(dt_ / factor);

    // Set up time varying LCS
    vector<Eigen::MatrixXd> A;
    vector<Eigen::MatrixXd> B;
    vector<Eigen::MatrixXd> D;
    vector<Eigen::VectorXd> d;
    vector<Eigen::MatrixXd> E;
    vector<Eigen::MatrixXd> F;
    vector<Eigen::MatrixXd> H;
    vector<Eigen::VectorXd> c;
    A.clear();
    B.clear();
    D.clear();
    d.clear();
    E.clear();
    F.clear();
    H.clear();
    c.clear();

    MatrixXd x_hat(x0.size(), N*factor + 1);
    MatrixXd lambda_hat(n_lambda_, N*factor);
    x_hat.col(0) = x0;
    VectorXd x_curr = x0;
    VectorXd x_next;

    for (int k = 0; k < N*factor; k++) {

      // Linearize about current point
      factory.UpdateStateAndInput(x_curr, u_hat.col(k / factor));
      LCS lcs = factory.GenerateLCS();
      A.push_back(lcs.A()[0]);
      B.push_back(lcs.B()[0]);
      D.push_back(lcs.D()[0]);
      d.push_back(lcs.d()[0]);
      E.push_back(lcs.E()[0]);
      F.push_back(lcs.F()[0]);
      H.push_back(lcs.H()[0]);
      c.push_back(lcs.c()[0]);      

      // Do one rollout step
      VectorXd u_k = u_hat.col(k / factor);

      // std::cout << "lcs simulate timestep " << k << std::endl;
      // std::cout << "x curr " << x_curr.transpose() << std::endl;
      // std::cout << "u_k " << u_k.transpose() << std::endl;
      // std::cout << u_k.size() << std::endl;
      auto pair = lcs.SimulateAndReturnForce(x_curr, u_k, true);
      x_next = pair.first;

      // HARDCODED
      if (example_idx_ == 2) {
        for (int i = 0; i < A_constraint.rows(); i++) {
          if (A_constraint(i, i) != 0) { // Assumes diagonal
            x_next(i) = std::min(std::max(x_next(i), lower_bound_x(i)), upper_bound_x(i));
          }
        }
      }

      // Normalize quaternions
      for (auto idx : controller_options_.quaternion_indices) {
        x_next.segment(idx, 4) = x_next.segment(idx, 4).normalized();
      }

      lambda_hat.col(k) = pair.second;

      x_hat.col(k+1) = x_next;
      x_curr = x_next;
    }

    MatrixXd x_hat_downsampled(MatrixXd::Zero(n_x_, N + 1));
    MatrixXd lambda_hat_downsampled(MatrixXd::Zero(n_lambda_, N));

    for (int i = 0; i < N; i++) {
      x_hat_downsampled.col(i) = x_hat.col(i * factor);
      lambda_hat_downsampled.col(i) = lambda_hat.col(i * factor);
    }
    x_hat_downsampled.col(N) = x_hat.col(N * factor);

    LCS output_lcs = MakeTimeVaryingLCS(x_hat_downsampled, u_hat, input_factory);

    if ((x_hat_downsampled.array().isNaN()).any()) {
      std::cout << "XHAT NOT FINITE" << std::endl;
    }
    return {output_lcs, x_hat_downsampled, lambda_hat_downsampled};

  }


  MatrixXd iC3::RolloutUHatPlate(VectorXd x0, MatrixXd c3_x, MatrixXd c3_u) {
    DiagramBuilder<double> builder;
    auto [plant_sim, scene_graph] =
        drake::multibody::AddMultibodyPlantSceneGraph(&builder, 0.00001);
    Parser parser(&plant_sim, &scene_graph);
    
    const std::string plate_file_lcs = "examples/resources/plate/plate.sdf";	
    const std::string cube_file_lcs = "examples/resources/plate/cube.sdf";

    parser.AddModels(plate_file_lcs);
    parser.AddModels(cube_file_lcs);

    plant_sim.Finalize();

    auto* broadcaster = builder.AddSystem<PdInputSource>(c3_x, c3_u, 0, 
          plant_sim.num_positions(), dt_, N_, ic3_options_.rollout_Kp, ic3_options_.rollout_Kd);
    builder.Connect(plant_sim.get_state_output_port(), broadcaster->get_input_port_state());
    builder.Connect(broadcaster->get_output_port_u(), plant_sim.get_actuation_input_port());

    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();

    auto& plant_context =
      diagram->GetMutableSubsystemContext(plant_sim, context.get());
    plant_sim.SetPositionsAndVelocities(&plant_context, x0);

    drake::systems::Simulator<double> simulator(*diagram, std::move(context));
    simulator.Initialize();

    MatrixXd x_hat(n_x_, N_+1);
    x_hat.col(0) = x0;

    int idx = 1;
    for (double t = 0.0; t < N_ * dt_ - 0.0001; t += dt_) {
      simulator.AdvanceTo(t);
      x_hat.col(idx) = plant_sim.GetPositionsAndVelocities(plant_context);
      idx++;
    }

    return x_hat;

  }

  tuple<MatrixXd, MatrixXd> iC3::RolloutUHatHand(VectorXd x0, MatrixXd c3_x, MatrixXd c3_u, 
            const vector<SortedPair<GeometryId>>& contact_geoms) {
    DiagramBuilder<double> builder;
    auto [plant_sim, scene_graph] =
        drake::multibody::AddMultibodyPlantSceneGraph(&builder, 0.0001);
    Parser parser(&plant_sim, &scene_graph);

    const std::string hand_file = "examples/resources/multifinger_hand/allegro_hand_description_right.urdf";
    const std::string cube_file = "examples/resources/multifinger_hand/cube.sdf";
    const std::string ground_file = "examples/resources/multifinger_hand/ground.urdf";

    parser.AddModels(hand_file);
    parser.AddModels(cube_file);
    parser.AddModels(ground_file);

    RigidTransform<double> X_G = RigidTransform<double>(
      drake::math::RotationMatrix<double>(), {0, 0, 0});

    RotationMatrix<double> R =
        RollPitchYaw<double>(0.0, (5.0 / 8) * M_PI, 0.0).ToRotationMatrix();
    RigidTransform<double> X_H = RigidTransform<double>(R, {-0.04, 0, 0.17});

    plant_sim.WeldFrames(plant_sim.world_frame(),
                          plant_sim.GetFrameByName("hand_root"), X_H);
    plant_sim.WeldFrames(plant_sim.world_frame(),
                          plant_sim.GetFrameByName("ground"), X_G);

    plant_sim.Finalize();

    auto* broadcaster = builder.AddSystem<PdInputSource>(c3_x, c3_u, 0, 
          plant_sim.num_positions(), dt_, N_, ic3_options_.rollout_Kp, ic3_options_.rollout_Kd);
    builder.Connect(plant_sim.get_state_output_port(), broadcaster->get_input_port_state());
    builder.Connect(broadcaster->get_output_port_u(), plant_sim.get_actuation_input_port());

    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();

    auto& plant_context =
      diagram->GetMutableSubsystemContext(plant_sim, context.get());
    plant_sim.SetPositionsAndVelocities(&plant_context, x0);

    drake::systems::Simulator<double> simulator(*diagram, std::move(context));
    simulator.set_target_realtime_rate(1);
    simulator.Initialize();

    MatrixXd x_hat(n_x_, N_+1);
    MatrixXd lambda_hat(n_lambda_, N_);
    x_hat.col(0) = x0;

    int idx = 1;
    for (double t = 0.0; t < N_ * dt_ - 0.0001; t += dt_) {
      simulator.AdvanceTo(t);
      x_hat.col(idx) = plant_sim.GetPositionsAndVelocities(plant_context);
      

      const auto& contact_results =
          plant_sim.get_contact_results_output_port()
              .Eval<ContactResults<double>>(plant_context);
      lambda_hat.col(idx - 1) = GetLambdaFromContacts(contact_results, contact_geoms);
      idx++;
    }

    return {x_hat, lambda_hat};

  }

  tuple<MatrixXd, MatrixXd> iC3::RolloutUHatPointHand(VectorXd x0, MatrixXd c3_x, MatrixXd c3_u, 
            const vector<SortedPair<GeometryId>>& contact_geoms) {
    DiagramBuilder<double> builder;

    // for (int i = 0; i < c3_x.cols(); i+=5) {
    //   std::cout << "c3 u " << i << ": " << c3_u.col(i).transpose() << std::endl;
    // }

    auto [plant_sim, scene_graph] =
        drake::multibody::AddMultibodyPlantSceneGraph(&builder, 0.0001);
    Parser parser(&plant_sim, &scene_graph);

    const std::string hand_file = "examples/resources/multifinger_hand/simplified_hand.sdf";
    const std::string cube_file = "examples/resources/multifinger_hand/cube.sdf";
    const std::string ground_file = "examples/resources/multifinger_hand/ground.urdf";

    parser.AddModels(hand_file);
    parser.AddModels(cube_file);
    parser.AddModels(ground_file);

    RigidTransform<double> X_G = RigidTransform<double>(
      drake::math::RotationMatrix<double>(), {0, 0, 0.0});

    // RigidTransform<double> X_1 = RigidTransform<double>(
    //   drake::math::RotationMatrix<double>(), {0.08, 0, 0.05});
    // RigidTransform<double> X_2 = RigidTransform<double>(
    //   drake::math::RotationMatrix<double>(), {-0.08, 0, 0.05});
    // RigidTransform<double> X_3 = RigidTransform<double>(
    //   drake::math::RotationMatrix<double>(), {0, 0.08, 0.05});
    // RigidTransform<double> X_4 = RigidTransform<double>(
    //   drake::math::RotationMatrix<double>(), {0, -0.08, 0.05});
      
    RigidTransform<double> X_1 = RigidTransform<double>(
      drake::math::RotationMatrix<double>(), {0.08, 0, 0.05});
    RigidTransform<double> X_2 = RigidTransform<double>(
      drake::math::RotationMatrix<double>(), {-0.08, -0.04, 0.05});
    RigidTransform<double> X_3 = RigidTransform<double>(
      drake::math::RotationMatrix<double>(), {-0.08, 0.04, 0.05});

    plant_sim.WeldFrames(plant_sim.world_frame(),
                        plant_sim.GetFrameByName("base_link_1"), X_1);
    plant_sim.WeldFrames(plant_sim.world_frame(),
                        plant_sim.GetFrameByName("base_link_2"), X_2);
    plant_sim.WeldFrames(plant_sim.world_frame(),
                        plant_sim.GetFrameByName("base_link_3"), X_3);
    // plant_sim.WeldFrames(plant_sim.world_frame(),
    //                     plant_sim.GetFrameByName("base_link_4"), X_4);                                                  
    plant_sim.WeldFrames(plant_sim.world_frame(),
                        plant_sim.GetFrameByName("ground"), X_G);

    plant_sim.Finalize();

    int N = c3_u.cols();

    auto* broadcaster = builder.AddSystem<PdInputSource>(c3_x, c3_u, 0, 
          plant_sim.num_positions(), dt_, N_, ic3_options_.rollout_Kp, ic3_options_.rollout_Kd);
    builder.Connect(plant_sim.get_state_output_port(), broadcaster->get_input_port_state());
    builder.Connect(broadcaster->get_output_port_u(), plant_sim.get_actuation_input_port());

    auto diagram = builder.Build();
    auto context = diagram->CreateDefaultContext();

    auto& plant_context =
      diagram->GetMutableSubsystemContext(plant_sim, context.get());
    plant_sim.SetPositionsAndVelocities(&plant_context, x0);

    drake::systems::Simulator<double> simulator(*diagram, std::move(context));
    simulator.set_target_realtime_rate(0);
    simulator.Initialize();

    MatrixXd x_hat(n_x_, N+1);
    MatrixXd lambda_hat(n_lambda_, N);
    x_hat.col(0) = x0;

    int idx = 1;
    for (double t = 0.0; t < N * dt_ - 0.0001; t += dt_) {
      simulator.AdvanceTo(t);
      x_hat.col(idx) = plant_sim.GetPositionsAndVelocities(plant_context);
      

      const auto& contact_results =
          plant_sim.get_contact_results_output_port()
              .Eval<ContactResults<double>>(plant_context);
      lambda_hat.col(idx - 1) = GetLambdaFromContacts(contact_results, contact_geoms);
      idx++;
    }

    return {x_hat, lambda_hat};

  }

  VectorXd iC3::GetLambdaFromContacts(ContactResults<double> contact_results,
      const vector<SortedPair<GeometryId>>& contact_geoms) {

    // HARDCODED for 2 friction directions
    VectorXd lambda_hat(VectorXd::Zero(4 * contact_geoms.size()));

    for (int i = 0; i < contact_geoms.size(); i++) {
      auto contact_A = contact_geoms[i].first();
      auto contact_B = contact_geoms[i].second();
      Vector3d force(Vector3d::Zero());
      Vector3d normal;
      for (int j = 0; j < contact_results.num_point_pair_contacts(); j++){
        const auto& result = contact_results.point_pair_contact_info(j);
        auto pp = result.point_pair();
        auto idA = pp.id_A;
        auto idB = pp.id_B;
        if ((idA == contact_A && idB == contact_B)) {
          force = result.contact_force();  
          normal = -1 * pp.nhat_BA_W;
          std::cout << "found" << std::endl;
          break;
        } 
        if ((idB == contact_A && idA == contact_B)) {
          std::cout << "found opposite" << std::endl;
        }
      }
      if (force == Vector3d::Zero()) {
        lambda_hat.segment(4*i, 4) = VectorXd::Zero(4);
        break;
      }

      double f_n = force.dot(normal);
      Vector3d t1 = normal.unitOrthogonal().normalized();
      Vector3d t2 = normal.cross(t1).normalized();

      Vector3d f_t = force - f_n * normal;
      double f_t1 = f_t.dot(t1);
      double f_t2 = f_t.dot(t2);

      double mu = controller_options_.lcs_factory_options.mu[i];

      double l1 = std::max(f_t1 / mu, 0.0);
      double l2 = std::max(-f_t1 / mu, 0.0);
      double l3 = std::max(f_t2 / mu, 0.0);
      double l4 = std::max(-f_t2 / mu, 0.0);
      double sum = l1 + l2 + l3 + l4;

      if (sum <= f_n && f_n > 0) {
        double add = (f_n - sum) / 4.0;
        l1 += add;
        l2 += add;
        l3 += add;
        l4 += add;
      } else if (sum > 1e-12) {
        double scale = f_n / sum;
        l1 *= scale;
        l2 *= scale;
        l3 *= scale;
        l4 *= scale;
      }

      lambda_hat(4*i) = l1;
      lambda_hat(4*i + 1) = l2;
      lambda_hat(4*i + 2) = l3;
      lambda_hat(4*i + 3) = l4;
    }
    return lambda_hat;
  }

  std::tuple<vector<MatrixXd>, vector<VectorXd>, vector<MatrixXd>, vector<VectorXd>> 
    iC3::ComputeLQRValueFunction(MatrixXd x_hat, MatrixXd u_hat, 
        MatrixXd lambda_hat, VectorXd xd, VectorXd ud, LCS lcs) {
    
    vector<MatrixXd> A = lcs.A();
    vector<MatrixXd> B = lcs.B();
    vector<MatrixXd> D = lcs.D();
    vector<VectorXd> d = lcs.d();
    vector<MatrixXd> Q = Q_;
    vector<MatrixXd> R = R_;

    vector<VectorXd> c; // Bias term from contact forces
    for (int i = 0; i < N_; i++) {
      c.push_back(D[i] * lambda_hat.col(i) + d[i]);
    }

    // Solve time-varying affine LQR about nominal trajectory
    vector<MatrixXd> H(N_+1, MatrixXd::Zero(n_x_, n_x_));
    vector<VectorXd> g(N_+1, VectorXd::Zero(n_x_));
    vector<MatrixXd> K(N_, MatrixXd::Zero(n_u_, n_x_));
    vector<VectorXd> k_ff(N_, VectorXd::Zero(n_u_));    

    H[N_] = Q[N_]; // terminal condition

    for (int k = N_-1; k >= 0; k--) {
      VectorXd x_k = x_hat.col(k);
      VectorXd u_k = u_hat.col(k);
            
      MatrixXd Q_xx = Q[k] + A[k].transpose()*H[k+1]*A[k];
      MatrixXd Q_uu = R[k] + B[k].transpose()*H[k+1]*B[k];
      MatrixXd Q_ux = B[k].transpose()*H[k+1]*A[k];

      VectorXd Q_x = Q[k]*(x_k - xd) + A[k].transpose()*g[k+1] + A[k].transpose()*H[k+1]*c[k]; 
      VectorXd Q_u = R[k]*(u_k - ud) + B[k].transpose()*g[k+1] + B[k].transpose()*H[k+1]*c[k];



      Eigen::LDLT<MatrixXd> solver(Q_uu);
      K[k] = -solver.solve(Q_ux);
      k_ff[k] = -solver.solve(Q_u);

      double reg = 1e-5;

      H[k] = Q_xx - Q_ux.transpose() * solver.solve(Q_ux) + reg * MatrixXd::Identity(n_x_, n_x_);
      g[k] = Q_x  - Q_ux.transpose() * solver.solve(Q_u);     

    }

    return std::make_tuple(H, g, K, k_ff);
  }


  LCS iC3::MakeTimeVaryingLCS(MatrixXd x_hat, MatrixXd u_hat, LCSFactory factory) {
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

    return LCS(A, B, D, d, E, F, H, c, dt_);
  }

  LCS iC3::ShortenLCSFront(LCS lcs, int num_timesteps_to_remove) {
    vector<Eigen::MatrixXd> A(lcs.A().begin() + num_timesteps_to_remove, lcs.A().end());
    vector<Eigen::MatrixXd> B(lcs.B().begin() + num_timesteps_to_remove, lcs.B().end());
    vector<Eigen::MatrixXd> D(lcs.D().begin() + num_timesteps_to_remove, lcs.D().end());
    vector<Eigen::VectorXd> d(lcs.d().begin() + num_timesteps_to_remove, lcs.d().end());
    vector<Eigen::MatrixXd> E(lcs.E().begin() + num_timesteps_to_remove, lcs.E().end());
    vector<Eigen::MatrixXd> F(lcs.F().begin() + num_timesteps_to_remove, lcs.F().end());
    vector<Eigen::MatrixXd> H(lcs.H().begin() + num_timesteps_to_remove, lcs.H().end());
    vector<Eigen::VectorXd> c(lcs.c().begin() + num_timesteps_to_remove, lcs.c().end());

    return LCS(A, B, D, d, E, F, H, c, dt_);
  }

  C3::CostMatrices iC3::ShortenCostsFront(int num_timesteps_to_remove) {
    vector<Eigen::MatrixXd> Q(Q_.begin() + num_timesteps_to_remove, Q_.end());
    vector<Eigen::MatrixXd> R(R_.begin() + num_timesteps_to_remove, R_.end());
    vector<Eigen::MatrixXd> G(G_.begin() + num_timesteps_to_remove, G_.end());
    vector<Eigen::MatrixXd> U(U_.begin() + num_timesteps_to_remove, U_.end());

    return C3::CostMatrices(Q, R, G, U);
  }


  void iC3::UpdateQuaternionCosts(
    MatrixXd x_hat, const Eigen::VectorXd& x_des, vector<VectorXd> c3_quat_norms) {
    
    // std::cout << x_hat.rows() << ", " << x_hat.cols() << std::endl;
    // std::cout << "xd: " << x_des.transpose() << std::endl;
    // std::cout << c3_quat_norms.size() << std::endl;

    Q_.clear();
    R_.clear();
    G_.clear();
    U_.clear();


    double discount_factor = 1;
    for (int i = 0; i < N_; i++) {
      Q_.push_back(discount_factor * controller_options_.c3_options.Q);
      discount_factor *=  controller_options_.c3_options.gamma;
      if (i < N_) {
        R_.push_back(discount_factor * controller_options_.c3_options.R);
        G_.push_back(discount_factor * controller_options_.c3_options.G);
        U_.push_back(discount_factor * controller_options_.c3_options.U);
      }
    }  
    Q_.push_back(discount_factor * controller_options_.c3_options.Q); 

    for (int i = 0; i < N_ + 1; i++) {
      int j = 0;
      for (int index : controller_options_.quaternion_indices) {
        double norm = c3_quat_norms[j](i);

        // make quaternion costs time-varying based on x_hat
        Eigen::VectorXd quat_curr_i = x_hat.col(i).segment(index, 4).normalized();
        Eigen::VectorXd quat_des_i = x_des.segment(index, 4);

        //std::cout << "xhat q: " << quat_curr_i.transpose() << std::endl;

        Eigen::MatrixXd quat_hessian_i = common::hessian_of_squared_quaternion_angle_difference(quat_curr_i, quat_des_i);

        // Regularize hessian so Q is always PSD
        double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
        //std::cout << min_eigenval << std::endl;

        Eigen::MatrixXd quat_regularizer_1 = std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
        Eigen::MatrixXd quat_regularizer_2 = quat_des_i * quat_des_i.transpose();
        Eigen::MatrixXd quat_regularizer_3 = 1e-4 * Eigen::MatrixXd::Identity(4, 4);

        double discount_factor = 1;
        Q_[i].block(index, index, 4, 4) = 
          discount_factor * controller_options_.Q_quaternion_weight * 
          (quat_hessian_i + quat_regularizer_1 + 
          controller_options_.quaternion_regularizer_fraction * quat_regularizer_2 + quat_regularizer_3);
        discount_factor *= controller_options_.c3_options.gamma;

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