#include "systems/hybrid_mpc.h"

namespace c3 {
namespace systems {

using drake::systems::Context;
using drake::multibody::ContactResults;

HybridMPC::HybridMPC(const MultibodyPlant<double>& plant_rollout, drake::systems::Diagram<double>& rollout_diagram, 
  std::unique_ptr<drake::systems::Context<double>> rollout_diagram_context, 
  const vector<SortedPair<GeometryId>>& contact_geoms_rollout, HybridMPCOptions mpc_options,
  MSiC3Options ms_ic3_options, int example_idx,
  MatrixXd A_x, VectorXd lb_x, VectorXd ub_x, MatrixXd A_u, VectorXd lb_u, VectorXd ub_u) 
  : plant_rollout_(plant_rollout),
    lcs_factory_(lcs_factory),
    rollout_diagram_(rollout_diagram),
    rollout_diagram_context_(std::move(rollout_diagram_context)),
    contact_geoms_rollout_(contact_geoms_rollout),
    mpc_options_(mpc_options),
    ms_ic3_options_(ms_ic3_options),
    example_idx_(example_idx),
    N_(mpc_options.N),
    dt_(mpc_options.dt),
    Q_(mpc_options.Q),
    R_(mpc_options.R),
    S_(mpc_options.S),
    G_(mpc_options.G),
    A_x_(A_x),
    lb_x_(lb_x),
    ub_x_(ub_x),
    A_u_(A_u),
    lb_u_(lb_u),
    ub_u_(ub_u),
    prog_(drake::solvers::MathematicalProgram()),
    osqp_(drake::solvers::OsqpSolver()) { 

    n_q_ = plant_rollout_.num_positions();
    n_v_ = plant_rollout_.num_velocities();
    n_u_ = plant_rollout_.num_actuators();
    n_x_ = n_q_ + n_v_;

    // TODO: don't hardcode this
    if (example_idx_ == 0) {
      n_lambda_ = 8 * 4;
    } else if (example_idx_ == 1) {
      n_lambda_ = 7 * 4;
    } else if (example_idx_ == 2) {
      n_lambda_ = 11 * 4;
    }
    
    // Add decision variables
    for (int i = 0; i < N_+1; i++) {
      x_.push_back(prog_.NewContinuousVariables(n_x_, "x_" + std::to_string(i)));
      if (i == N_) break;
      u_.push_back(prog_.NewContinuousVariables(n_u_, "u_" + std::to_string(i)));
      lambda_.push_back(prog_.NewContinuousVariables(n_lambda_, "lambda_" + std::to_string(i)));
      epsilon_.push_back(prog_.NewContinuousVariables(n_lambda_, "epsilon_" + std::to_string(i)));
    }

    // Add placeholder costs
    target_costs_.resize(N_ + 1);
    input_costs_.resize(N_);
    force_costs_.resize(N_);
    slack_costs_.resize(N_);
    for (int i = 0; i < N_+1; i++) {
      target_costs_[i] = 
          prog_.AddQuadraticCost(2*Q_, VectorXd::Zero(n_x_), x_[i]).evaluator().get();
      if (i == N_) break;
      input_costs_[i] = 
          prog_.AddQuadraticCost(2*R_, VectorXd::Zero(n_u_), u_[i]).evaluator().get();
      force_costs_[i] = 
          prog_.AddQuadraticCost(2*S_, VectorXd::Zero(n_lambda_), lambda_[i]).evaluator().get();
      slack_costs_[i] = 
          prog_.AddQuadraticCost(2*G_, VectorXd::Zero(n_lambda_), epsilon_[i]).evaluator().get();
    }

    // End effector acceleration cost
    // Assumes first n_u_ terms of velocity correspond to end effector
    double accel_cost = mpc_options_.accel_cost;
    for (int i = 0; i < N_; i++) {
      MatrixXd Q_accel = accel_cost * MatrixXd::Identity(2*n_u_, 2*n_u_);
      Q_accel.block(0, n_u_, n_u_, n_u_) = -accel_cost * MatrixXd::Identity(n_u_, n_u_);
      Q_accel.block(n_u_, 0, n_u_, n_u_) = -accel_cost * MatrixXd::Identity(n_u_, n_u_);
      prog_.AddQuadraticCost(2*Q_accel, VectorXd::Zero(2*n_u_), {x_[i].segment(n_q_, n_u_), x_[i+1].segment(n_q_, n_u_)});
    }


    // Placeholder initial state constraint
    initial_state_constraint_ = 
        prog_.AddLinearEqualityConstraint(
            MatrixXd::Identity(n_x_, n_x_), VectorXd::Zero(n_x_), x_.at(0)
        ).evaluator().get();

    // Placeholder dynamics constraint
    dynamics_constraints_.resize(N_);
    MatrixXd A_dyn(MatrixXd::Zero(n_x_, n_x_ + n_u_ + n_lambda_ + n_x_));
    for (int i = 0; i < N_; i++) {
      dynamics_constraints_[i] = 
          prog_.AddLinearEqualityConstraint(
              A_dyn, VectorXd::Zero(n_x_), {x_.at(i), u_.at(i), lambda_.at(i), x_.at(i+1)}
          ).evaluator().get();
    }

    // Placeholder mode constraint
    lambda_constraints_.resize(N_);
    // Need 2 constraints for each [λ ε]
    MatrixXd A_lambda(MatrixXd::Zero(2 * n_lambda_, 2 * n_lambda_));
    for (int i = 0; i < N_; i++) {
      lambda_constraints_[i] = 
          prog_.AddLinearConstraint(
              A_lambda, VectorXd::Zero(2 * n_lambda_), VectorXd::Zero(2 * n_lambda_), {lambda_.at(i), epsilon_.at(i)}
          ).evaluator().get();
    }

    eta_constraints_.resize(N_);
    // Need 2 constraints for each [η ε]
    MatrixXd A_eta(MatrixXd::Zero(2 * n_lambda_, n_x_ + n_lambda_ + n_u_ + n_lambda_));
    for (int i = 0; i < N_; i++) {
      eta_constraints_[i] = 
          prog_.AddLinearConstraint(
              A_eta, VectorXd::Zero(2 * n_lambda_), VectorXd::Zero(2 * n_lambda_), {x_.at(i), lambda_.at(i), u_.at(i), epsilon_.at(i)}
          ).evaluator().get();
    }

    // Ensure epsilon is greater than 0
    for (int i = 0; i < N_; i++) {
      prog_.AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), epsilon_[i]);
    }


    // Add linear constraints
    if (ic3_options_.add_position_constraints) {
      for (int i = 1; i < N_ + 1; i++) { // Don't put constraint on x0
        prog_.AddLinearConstraint(A_x_, lb_x_, ub_x_, x_.at(i)); 
      }
      std::cout << "A_x " << A_x_.diagonal().transpose() << std::endl;
      std::cout << "lb_x " << lb_x_.transpose() << std::endl;
      std::cout << "ub_x " << ub_x_.transpose() << std::endl;
    }
    if (ic3_options_.add_input_constraints) {
      for (int i = 0; i < N_; i++) {
        prog_.AddLinearConstraint(A_u_, lb_u_, ub_u_, u_.at(i));
      }
      std::cout << "A_u " << A_u_.diagonal().transpose() << std::endl;
      std::cout << "lb_u " << lb_u_.transpose() << std::endl;
      std::cout << "ub_u " << ub_u_.transpose() << std::endl;
    }

    X_delta_ = drake::math::RigidTransform<double>::Identity();

    simulator_ = std::make_unique<drake::systems::Simulator<double>>(
        rollout_diagram_, std::move(rollout_diagram_context_));
}


std::tuple<MatrixXd, MatrixXd, MatrixXd> HybridMPC::SimulateHybridMPC(
  VectorXd x0, MatrixXd x_hat, MatrixXd u_hat, MatrixXd lambda_hat, drake::systems::Context<double>& context_rollout) {
  
  MatrixXd x_out(n_x_, ms_ic3_options_.N+1);
  MatrixXd u_out(n_u_, ms_ic3_options_.N);
  MatrixXd lambda_out(n_lambda_, ms_ic3_options_.N);

  x_out.col(0) = x0;

  VectorXd x_curr = x0;
  for (int i = 0; i < ms_ic3_options_.N; i++) {

    // Get targets
    vector<VectorXd> x_noms;
    vector<VectorXd> u_noms;
    vector<VectorXd> lambda_noms;
    for (int t = 0; t < N_; t++) {
      int x_idx = std::min(i + t, ms_ic3_options_.N);
      int u_idx = std::min(i + t, ms_ic3_options_.N-1);
      x_noms.push_back(x_hat.col(x_idx));
      u_noms.push_back(u_hat.col(u_idx));
      lambda_noms.push_back(lambda_hat.col(u_idx));
    }

    if (example_idx_ == 0) {
      // Don't get transform for plate example
    } else if (example_idx_ == 1 || example_idx_ == 2) {
      // HARDCODED INDICES

      // Get end effector transform
      UpdateXDelta(x_curr, x_hat.col(i));

      for (int t = 0; t < N_; t++) {
        VectorXd ee_nom = x_noms[t].segment(0, 9);
        VectorXd u_nom = u_noms[t];

        for (int f = 0; f < 3; f++) {
          ee_nom.segment(3*f, 3) = X_delta_ * ee_nom.segment(3*f, 3);
          u_nom.segment(3*f, 3) = X_delta_.rotation() * u_nom.segment(3*f, 3);
        }
        x_noms[t].segment(0, 9) = ee_nom;
        u_noms[t] = u_nom;
      }
    }

    LCS lcs = MakeLCS();
    UpdateQP(x_curr, lcs, x_noms, u_noms, lambda_noms);

    drake::solvers::MathematicalProgramResult result = osqp_.Solve(prog_, std::nullopt, solver_options_);
    if (!result.is_success()) {
      const auto& details = result.get_solver_details<drake::solvers::OsqpSolver>();
      std::cout << "Hybrid MPC QP failed" << std::endl;
      std::cout << "OSQP Status: " << details.status_val << std::endl;
      std::cout << "Iterations: " << details.iter << std::endl;
      std::cout << "Primal Res: " << details.primal_res << std::endl;
      std::cout << "Dual Res: " << details.dual_res << std::endl;
    }

    VectorXd u_mpc = result.GetSolution(u_[0]);

    // Simulate
    Context<double>& root_context = simulator_->get_mutable_context();
    Context<double>& plant_context =
        rollout_diagram_.GetMutableSubsystemContext(plant_rollout_, &root_context);
      
    for (int j = 0; j < ms_ic3_options_.rollout_dt_scaling; j++) {
      
    }
    


  }


}

void UpdateXDelta(VectorXd x_curr, VectorXd x_nom) {
  if (example_idx_ == 0) {
    // Don't do anything for plate example
  } else if (example_idx_ == 1 || example_idx_ == 2) {

    // HARDCODED indices
    Eigen::Quaterniond cube_rot_plan(x_nom(9), x_nom(10), x_nom(11), x_nom(12));
    Eigen::Vector3d cube_pos_plan(x_nom.segment(13, 3));
    drake::math::RigidTransform<double> X_W_Nom(cube_rot_plan, cube_pos_plan);

    Eigen::Quaterniond cube_rot_curr(x_curr(9), x_curr(10), x_curr(11), x_curr(12));
    Eigen::Vector3d cube_pos_curr(x_curr.segment(13, 3));
    drake::math::RigidTransform<double> X_W_Curr(cube_rot_curr, cube_pos_curr);

    X_delta_ = X_W_Curr * X_W_Nom.inverse();
  }
}

void UpdateQP(VectorXd x_curr, LCS lcs, vector<VectorXd> x_noms, vector<VectorXd> u_noms, vector<VectorXd> lambda_noms) {

  DRAKE_DEMAND(lcs.N() == N_);
  DRAKE_DEMAND(x_noms.size() == N_+1);
  DRAKE_DEMAND(u_noms.size() == N_);
  DRAKE_DEMAND(lambda_noms.size() == N_);

  // Update initial state constraint
  initial_state_constraint_->UpdateCoefficients(MatrixXd::Identity(n_x_, n_x_), x_curr);

  // Update dynamics constraint
  // x_next = Ax + Bu + Dλ + d
  // x_k, u_k, lambda_k, x_{k+1}
  for (int i = 0; i < N_; i++) {
    MatrixXd A_dyn(n_x_, n_x_ + n_u_ + n_lambda_ + n_x_);
    A_dyn.block(0, 0, n_x_, n_x_) = lcs.A()[i];
    A_dyn.block(0, n_x_, n_x_, n_u_) = lcs.B()[i];
    A_dyn.block(0, n_x_+n_u_, n_x_, n_lambda_) = lcs.D()[i];
    A_dyn.block(0, n_x_+n_u_+n_lambda_, n_x_, n_x_) = -1 * MatrixXd::Identity(n_x_, n_x_);

    if (A_dyn.array().isNaN().any()) {
      std::cout << "A_dyn has NAN " << std::endl;
    }
    
    VectorXd affine_term = -lcs.d()[i];
    dynamics_constraints_[i]->UpdateCoefficients(A_dyn, affine_term);
  }


  /* Update lambda/eta constraints
    -ε <= λ <= ε if λ_hat = 0 
    equivalent to [0 ] <= [1  1] [λ] <= [∞]
                  [-∞]    [1 -1] [ε]    [0]
    
    -ε <= λ <= ∞ if λ_hat > 0 
    equivalent to [0] <= [1 1] [λ] <= [∞]
                  [0]    [0 0] [ε]    [0]
  */
  for (int i = 0; i < N_; i++) {
    VectorXd lambda_nom = lambda_noms[i];

    MatrixXd E = lcs.E()[i];
    MatrixXd F = lcs.F()[i];
    MatrixXd H = lcs.H()[i];
    VectorXd c = lcs.c()[i];

    // row j and j+n_lambda correspond to the same component of lambda/eta
    VectorXd lambda_lb(2 * n_lambda_);
    VectorXd eta_lb(2 * n_lambda_);
    VectorXd lambda_ub(2 * n_lambda_);
    VectorXd eta_ub(2 * n_lambda_);
    MatrixXd A_lambda(MatrixXd::Zero(2 * n_lambda_, 2 * n_lambda_));
    MatrixXd A_eta(MatrixXd::Zero(2 * n_lambda_, n_x_ + n_lambda_ + n_u_ + n_lambda_));

    A_lambda.block(0, 0, n_lambda_, n_lambda_) = MatrixXd::Identity(n_lambda_, n_lambda_);
    A_lambda.block(0, n_lambda_, n_lambda_, n_lambda_) = MatrixXd::Identity(n_lambda_, n_lambda_);

    A_eta.block(0, 0, n_lambda_, n_x_) = E;
    A_eta.block(0, n_x_, n_lambda_, n_lambda_) = F;
    A_eta.block(0, n_x_+n_lambda_, n_lambda_, n_u_) = H;
    A_eta.block(0, n_x_+n_lambda_+n_u_, n_lambda_, n_lambda_) = MatrixXd::Identity(n_lambda_, n_lambda_);

    lambda_ub.segment(0, n_lambda_) = VectorXd::Constant(n_lambda_, std::numeric_limits<double>::infinity());
    lambda_ub.segment(n_lambda_, n_lambda_) = VectorXd::Zero(n_lambda_);

    eta_ub.segment(0, n_lambda_) = VectorXd::Constant(n_lambda_, std::numeric_limits<double>::infinity());

    double tolerance = 1e-5;
    for (int j = 0; j < n_lambda_; j++) {
      lambda_lb(j) = (lambda_nom(j) <= tolerance) ? 0 : lambda_threshold_(j);
      lambda_lb(n_lambda_+j) = (lambda_nom(j) <= tolerance) ? -std::numeric_limits<double>::infinity() : 0;

      eta_lb(j) = -c(j) + ((lambda_nom(j) > tolerance) ? 0 : eta_threshold_(j));
      eta_lb(n_lambda_+j) = (lambda_nom(j) > tolerance) ? -std::numeric_limits<double>::infinity() : 0;
      eta_ub(n_lambda_+j) = (lambda_nom(j) > tolerance) ? -c(j) : 0;

      A_lambda(n_lambda_+j, j) = (lambda_nom(j) <= tolerance) ? 1 : 0;
      A_lambda(n_lambda_+j, n_lambda_+j) = (lambda_nom(j) <= tolerance) ? -1 : 0;    
      
      A_eta.block(n_lambda_+j, 0, 1, n_x_) = 
          (lambda_nom(j) > tolerance) ? E.row(j) : RowVectorXd(RowVectorXd::Zero(n_x_));
      A_eta.block(n_lambda_+j, n_x_, 1, n_lambda_) = 
          (lambda_nom(j) > tolerance) ? F.row(j) : RowVectorXd(RowVectorXd::Zero(n_lambda_));
      A_eta.block(n_lambda_+j, n_x_+n_lambda_, 1, n_u_) = 
          (lambda_nom(j) > tolerance) ? H.row(j) : RowVectorXd(RowVectorXd::Zero(n_u_));

      RowVectorXd eta_row = RowVectorXd::Zero(n_lambda_);
      eta_row(j) = (lambda_nom(j) > tolerance) ? -1 : 0;    
      A_eta.block(n_lambda_+j, n_x_+n_lambda_+n_u_, 1, n_lambda_) = 
          (lambda_nom(j) > tolerance) ? eta_row : RowVectorXd(RowVectorXd::Zero(n_lambda_));
    }


    lambda_constraints_[i]->UpdateCoefficients(A_lambda, lambda_lb, lambda_ub);
    eta_constraints_[i]->UpdateCoefficients(A_eta, eta_lb, eta_ub);
  }

  // Update tracking costs
  for (int i = 0; i < N_+1; i++) {
    x_nom.segment(mpc_options_.quaternion_indices[0], 4) = 
        quat_norms[i] * x_nom.segment(mpc_options_.quaternion_indices[0], 4);
    target_costs_[i]->UpdateCoefficients(2 * Q_, -2 * Q_ * x_noms[i]);

    if (i == N_) break;

    input_costs_[i]->UpdateCoefficients(2 * R_, -2 * R_ * u_noms[i]);

    VectorXd lambda_des = force_data_.col(idx);
    force_costs_[i]->UpdateCoefficients(2 * S_, -2 * S_ * lambda_noms[i]);
  }
}

LCS MakeLCS(VectorXd x_curr, VectorXd u_curr) {
  lcs_factory_.SetNewDt(dt_);
  lcs_factory_.UpdateStateAndInput(x_curr, u_curr);
  LCS lcs = lcs_factory_.GenerateLCS();

  vector<Eigen::MatrixXd> A(lcs.A()[0], N_);
  vector<Eigen::MatrixXd> B(lcs.B()[0], N_);
  vector<Eigen::MatrixXd> D(lcs.D()[0], N_);
  vector<Eigen::VectorXd> d(lcs.d()[0], N_);
  vector<Eigen::MatrixXd> E(lcs.E()[0], N_);
  vector<Eigen::MatrixXd> F(lcs.F()[0], N_);
  vector<Eigen::MatrixXd> H(lcs.H()[0], N_);
  vector<Eigen::VectorXd> c(lcs.c()[0], N_);

  return LCS(A, B, D, d, E, F, H, c);
}

void HybridMPC::UpdateQuaternionCosts(
    const Eigen::VectorXd& x_curr, const Eigen::VectorXd& x_des) const {
    
  // Early return if no quaternions or cost parameters not set
  if (mpc_options_.quaternion_indices.size() == 0) {
    return;
  }

  for (int index : mpc_options_.quaternion_indices) {
    Eigen::VectorXd quat_curr_i = x_curr.segment(index, 4);
    Eigen::VectorXd quat_des_i = x_des.segment(index, 4);

    Eigen::MatrixXd quat_hessian_i =
        c3::systems::common::hessian_of_squared_quaternion_angle_difference(
                  quat_curr_i, quat_des_i);

    // Regularize hessian so Q is always PSD
    double min_eigenval = quat_hessian_i.eigenvalues().real().minCoeff();
    Eigen::MatrixXd quat_regularizer_1 =
        std::max(0.0, -min_eigenval) * Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd quat_regularizer_2 = quat_des_i * quat_des_i.transpose();

    // Additional regularization term to help with numerical issues
    Eigen::MatrixXd quat_regularizer_3 = 1e-8 * Eigen::MatrixXd::Identity(4, 4);

    double quaternion_weight = mpc_options_.quaternion_weight;
    double quaternion_regularizer_fraction = 0.0;

    // Replace quaternion blocks in Q
    double discount_factor = 1;
    for (int i = 0; i < N_ + 1; i++) {
      Q_.block(index, index, 4, 4) =
          mpc_options_.w_Q * quaternion_weight *
          (quat_hessian_i + quat_regularizer_1 +
           quaternion_regularizer_fraction * quat_regularizer_2 +
           quat_regularizer_3);
    }

  }
}



} // namespace systems
} // namespace c3