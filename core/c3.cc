#include "c3.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>

#include <Eigen/Core>
#include <drake/common/find_runfiles.h>
#include <omp.h>

#include "lcs.h"
#include "solver_options_io.h"

#include "drake/common/text_logging.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/moby_lcp_solver.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/solve.h"
#include <drake/common/symbolic/expression.h>

namespace c3 {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::SolutionResult;
using drake::solvers::SolverOptions;

using drake::solvers::OsqpSolver;
using drake::solvers::OsqpSolverDetails;
using drake::solvers::Solve;

C3::CostMatrices::CostMatrices(const std::vector<Eigen::MatrixXd>& Q,
                               const std::vector<Eigen::MatrixXd>& R,
                               const std::vector<Eigen::MatrixXd>& G,
                               const std::vector<Eigen::MatrixXd>& U) {
  this->Q = Q;
  this->R = R;
  this->G = G;
  this->U = U;
}

C3::C3(const LCS& lcs, const CostMatrices& costs,
       const vector<VectorXd>& x_desired, const C3Options& options,
       const int z_size)
    : warm_start_(options.warm_start),
      N_(lcs.N()),
      n_x_(lcs.num_states()),
      n_lambda_(lcs.num_lambdas()),
      n_u_(lcs.num_inputs()),
      n_z_(z_size),
      lcs_(lcs),
      cost_matrices_(costs),
      x_desired_(x_desired),
      options_(options),
      h_is_zero_(lcs.H()[0].isZero(0)),
      prog_(MathematicalProgram()),
      osqp_(OsqpSolver()) {

  DRAKE_DEMAND(cost_matrices_.Q.size() == N_+1);
  DRAKE_DEMAND(cost_matrices_.R.size() == N_);
  DRAKE_DEMAND(cost_matrices_.G.size() == N_);
  DRAKE_DEMAND(cost_matrices_.U.size() == N_);
  DRAKE_DEMAND(lcs_.N() == N_);

  if (warm_start_) {
    warm_start_x_.resize(options_.admm_iter + 1);
    warm_start_lambda_.resize(options_.admm_iter + 1);
    warm_start_u_.resize(options_.admm_iter + 1);
    for (int iter = 0; iter < options_.admm_iter + 1; ++iter) {
      warm_start_x_[iter].resize(N_ + 1);
      for (int i = 0; i < N_ + 1; ++i) {
        warm_start_x_[iter][i] = VectorXd::Zero(n_x_);
      }
      warm_start_lambda_[iter].resize(N_);
      for (int i = 0; i < N_; ++i) {
        warm_start_lambda_[iter][i] = VectorXd::Zero(n_lambda_);
      }
      warm_start_u_[iter].resize(N_);
      for (int i = 0; i < N_; ++i) {
        warm_start_u_[iter][i] = VectorXd::Zero(n_u_);
      }
    }
  }

  ScaleLCS();
  x_ = vector<drake::solvers::VectorXDecisionVariable>();
  u_ = vector<drake::solvers::VectorXDecisionVariable>();
  lambda_ = vector<drake::solvers::VectorXDecisionVariable>();

  z_fin_ = std::make_unique<std::vector<VectorXd>>();
  z_sol_ = std::make_unique<std::vector<VectorXd>>();
  x_sol_ = std::make_unique<std::vector<VectorXd>>();
  lambda_sol_ = std::make_unique<std::vector<VectorXd>>();
  u_sol_ = std::make_unique<std::vector<VectorXd>>();
  w_sol_ = std::make_unique<std::vector<VectorXd>>();
  delta_sol_ = std::make_unique<std::vector<VectorXd>>();

  for (int i = 0; i < N_; ++i) {
    z_sol_->push_back(Eigen::VectorXd::Zero(n_z_));
    x_sol_->push_back(Eigen::VectorXd::Zero(n_x_));
    lambda_sol_->push_back(Eigen::VectorXd::Zero(n_lambda_));
    u_sol_->push_back(Eigen::VectorXd::Zero(n_u_));
    z_fin_->push_back(Eigen::VectorXd::Zero(n_z_));
    w_sol_->push_back(Eigen::VectorXd::Zero(n_z_));
    delta_sol_->push_back(Eigen::VectorXd::Zero(n_z_));
  }

  z_.resize(N_);
  for (int i = 0; i < N_ + 1; ++i) {
    x_.push_back(prog_.NewContinuousVariables(n_x_, "x" + std::to_string(i)));
    if (i < N_) {
      lambda_.push_back(prog_.NewContinuousVariables(
          n_lambda_, "lambda" + std::to_string(i)));
      u_.push_back(prog_.NewContinuousVariables(n_u_, "k" + std::to_string(i)));
      z_.at(i).push_back(x_.back());
      z_.at(i).push_back(lambda_.back());
      z_.at(i).push_back(u_.back());
    }
  }

  // initialize the constraint bindings
  initial_state_constraint_ = nullptr;
  initial_force_constraint_ = nullptr;

  // Add dynamics constraints
  dynamics_constraints_.resize(N_);
  MatrixXd LinEq(n_x_, 2 * n_x_ + n_lambda_ + n_u_);
  LinEq.block(0, n_x_ + n_lambda_ + n_u_, n_x_, n_x_) =
      -1 * MatrixXd::Identity(n_x_, n_x_);
  for (int i = 0; i < N_; ++i) {
    LinEq.block(0, 0, n_x_, n_x_) = lcs_.A().at(i);
    LinEq.block(0, n_x_, n_x_, n_lambda_) = lcs_.D().at(i);
    LinEq.block(0, n_x_ + n_lambda_, n_x_, n_u_) = lcs_.B().at(i);

    dynamics_constraints_[i] =
        prog_
            .AddLinearEqualityConstraint(
                LinEq, -lcs_.d().at(i),
                {x_.at(i), lambda_.at(i), u_.at(i), x_.at(i + 1)})
            .evaluator()
            .get();
    dynamics_constraints_[i]->set_description("dynamics_constraint_" + std::to_string(i));
  }

  // Setup QP costs
  target_costs_.resize(N_ + 1);
  input_costs_.resize(N_);
  augmented_costs_.clear();
  for (int i = 0; i < N_ + 1; ++i) {
    target_costs_[i] =
        prog_
            .AddQuadraticCost(2 * cost_matrices_.Q.at(i),
                              -2 * cost_matrices_.Q.at(i) * x_desired_.at(i),
                              x_.at(i), 1)
            .evaluator()
            .get();
    // Skip input cost at the (N + 1)th time step
    if (i == N_) break;
    input_costs_[i] = prog_
                          .AddQuadraticCost(2 * cost_matrices_.R.at(i),
                                            VectorXd::Zero(n_u_), u_.at(i), 1)
                          .evaluator();
  }

  // Set default solver options
  SetDefaultSolverOptions();

  // for (const auto& binding : prog_.GetAllConstraints()) {
  //   std::cout << "Constraint: " << binding.evaluator()->get_description() << "\n";
  //   std::cout << "Lower bound:\n" << binding.evaluator()->lower_bound().transpose() << "\n";
  //   std::cout << "Upper bound:\n" << binding.evaluator()->upper_bound().transpose() << "\n";
  // }
  // std::cout << std::endl << std::endl; 
}

void C3::SetDefaultSolverOptions() {
  // Set default solver options
  auto main_runfile =
      drake::FindRunfile("_main/core/configs/solver_options_default.yaml");
  auto external_runfile =
      drake::FindRunfile("c3/core/configs/solver_options_default.yaml");
  if (main_runfile.abspath.empty() && external_runfile.abspath.empty()) {
    throw std::runtime_error(fmt::format(
        "Could not find the default solver options YAML file. {}, {}",
        main_runfile.error, external_runfile.error));
  }
  drake::solvers::SolverOptions solver_options =
      drake::yaml::LoadYamlFile<c3::SolverOptionsFromYaml>(
          main_runfile.abspath.empty() ? external_runfile.abspath
                                       : main_runfile.abspath)
          .GetAsSolverOptions(drake::solvers::OsqpSolver::id());
  SetSolverOptions(solver_options);
}

C3::C3(const LCS& lcs, const CostMatrices& costs,
       const vector<VectorXd>& x_desired, const C3Options& options)
    : C3(lcs, costs, x_desired, options,
         lcs.num_states() + lcs.num_lambdas() + lcs.num_inputs()) {}

C3::CostMatrices C3::CreateCostMatricesFromC3Options(const C3Options& options,
                                                     int N) {
  std::vector<Eigen::MatrixXd> Q;  // State cost matrices.
  std::vector<Eigen::MatrixXd> R;  // Input cost matrices.

  std::vector<MatrixXd> G(N,
                          options.G);  // State-input cross-term matrices.
  std::vector<MatrixXd> U(N,
                          options.U);  // Constraint matrices.

  double discount_factor = 1.0;
  for (int i = 0; i < N; ++i) {
    Q.push_back(discount_factor * options.Q);
    R.push_back(discount_factor * options.R);
    discount_factor *= options.gamma;
  }
  Q.push_back(discount_factor * options.Q);

  return CostMatrices(Q, R, G, U);  // Initialize the cost matrices.
}

void C3::ScaleLCS() {
  if (!options_.scale_lcs) {
    // If the LCS is a placeholder or scaling is disabled, we do not scale
    // the complementarity dynamics.
    AnDn_ = 1.0;
    return;
  }

  AnDn_ = lcs_.ScaleComplementarityDynamics();
}

void C3::UpdateLCS(const LCS& lcs) {
  DRAKE_DEMAND(lcs_.HasSameDimensionsAs(lcs));

  lcs_ = lcs;
  h_is_zero_ = lcs_.H()[0].isZero(0);
  ScaleLCS();

  MatrixXd LinEq = MatrixXd::Zero(n_x_, 2 * n_x_ + n_lambda_ + n_u_);
  LinEq.block(0, n_x_ + n_u_ + n_lambda_, n_x_, n_x_) =
      -1 * MatrixXd::Identity(n_x_, n_x_);
  for (int i = 0; i < N_; ++i) {
    LinEq.block(0, 0, n_x_, n_x_) = lcs_.A().at(i);
    LinEq.block(0, n_x_, n_x_, n_lambda_) = lcs_.D().at(i);
    LinEq.block(0, n_x_ + n_lambda_, n_x_, n_u_) = lcs_.B().at(i);

    dynamics_constraints_[i]->UpdateCoefficients(LinEq, -lcs_.d().at(i));
  }
}

const std::vector<drake::solvers::LinearEqualityConstraint*>&
C3::GetDynamicConstraints() {
  return dynamics_constraints_;
}

void C3::UpdateTarget(const std::vector<Eigen::VectorXd>& x_des) {
  DRAKE_DEMAND(x_des.size() == N_ + 1);
  x_desired_ = x_des;
  for (int i = 0; i < N_ + 1; ++i) {
    target_costs_[i]->UpdateCoefficients(
        2 * cost_matrices_.Q.at(i),
        -2 * cost_matrices_.Q.at(i) * x_desired_.at(i));
  }
}

void C3::UpdateInputTarget(const std::vector<Eigen::VectorXd>& u_des) {
  u_desired_ = u_des;
  for (int i = 0; i < N_; ++i) {
    input_costs_[i]->UpdateCoefficients(
        2 * cost_matrices_.R.at(i),
        -2 * cost_matrices_.R.at(i) * u_desired_.at(i));
  }
}

void C3::UpdateCostMatrices(const CostMatrices& costs) {
  DRAKE_DEMAND(cost_matrices_.HasSameDimensionsAs(costs));
  cost_matrices_ = costs;

  for (int i = 0; i < N_ + 1; ++i) {
    target_costs_[i]->UpdateCoefficients(
        2 * cost_matrices_.Q.at(i),
        -2 * cost_matrices_.Q.at(i) * x_desired_.at(i));
    if (i < N_) {
      input_costs_[i]->UpdateCoefficients(2 * cost_matrices_.R.at(i),
                                          VectorXd::Zero(n_u_));
    }
  }
}

void C3::UpdateFinalCost(const Eigen::MatrixXd Q_final, const Eigen::VectorXd bias) {  

  std::vector<Eigen::MatrixXd> Q = cost_matrices_.Q;
  // std::cout << "Q_final " << Q_final.norm() << std::endl;
  // std::cout << "bias " << bias.transpose() << std::endl;
  
  // Convert to symmetric, want 2 * Qf to match rest of C3's cost convention
  Q[N_] = Q_final + Q_final.transpose(); 
  
  auto* qf_evaluator = target_costs_[N_];
  qf_evaluator->UpdateCoefficients(Q[N_], 2 * bias);

}

const std::vector<drake::solvers::QuadraticCost*>& C3::GetTargetCost() {
  return target_costs_;
}

void C3::Solve(const VectorXd& x0) {
  auto start = std::chrono::high_resolution_clock::now();

  delta_projection_.clear();
  lambda_minus_delta_lambda_.clear();
  lambda_minus_delta_lambda_norms_.clear();
  lambda_minus_delta_lambda_avg_horizon_norms_.clear();
  iterate_step_change_norms_.clear();
  complementarity_slackness_.clear();
  lambda_minus_delta_lambda_max_abs_.clear();
  lambda_minus_delta_lambda_all_stages_.clear();

  if (!x0.allFinite()) {
    std::cout << "x0 NOT ALL FINITE" << std::endl;
  }

  // Set the initial state constraint
  if (initial_state_constraint_binding_.has_value() && initial_state_constraint_) {
    initial_state_constraint_->UpdateCoefficients(
        MatrixXd::Identity(n_x_, n_x_), x0);
  } else {
    initial_state_constraint_binding_ =
        prog_.AddLinearEqualityConstraint(MatrixXd::Identity(n_x_, n_x_), x0,
                                         x_[0]);
    initial_state_constraint_ = initial_state_constraint_binding_->evaluator();
  }

  // Set the initial force constraint
  if (h_is_zero_ == 1) {  // No dependence on u, so just simulate passive system
    drake::solvers::MobyLCPSolver<double> LCPSolver;
    VectorXd lambda0;
    LCPSolver.SolveLcpLemke(lcs_.F()[0], lcs_.E()[0] * x0 + lcs_.c()[0],
                            &lambda0);
    // Force constraints to be updated before every solve if no dependence on u
    if (initial_force_constraint_) {
      initial_force_constraint_->UpdateCoefficients(
          MatrixXd::Identity(n_lambda_, n_lambda_), lambda0);
    } else {
      initial_force_constraint_ =
          prog_
              .AddLinearEqualityConstraint(
                  MatrixXd::Identity(n_lambda_, n_lambda_), lambda0, lambda_[0])
              .evaluator();
    }
  }

  if (u_desired_.size() == N_) {
    for (int i = 0; i < N_; ++i) {
      input_costs_[i]->UpdateCoefficients(
          2 * cost_matrices_.R.at(i),
          -2 * cost_matrices_.R.at(i) * u_desired_.at(i));
    }
  }


  if (penalize_change_ && options_.penalize_input_change) {
    std::cout << "penalizing change u " << std::endl;
    if (u_sol_->size() < N_) {
      std::cerr << "u sol not set, penalize input change" << std::endl;
    }

    for (int i = 0; i < N_; ++i) {
      // Penalize deviation from previous input solution:  input cost is
      // (u-u_prev)' * R * (u-u_prev).
      // input_costs_[i]->UpdateCoefficients(
      //     2 * cost_matrices_.R.at(i),
      //     -2 * cost_matrices_.R.at(i) * u_sol_->at(i));

      const double w_diff = options_.input_change_weight;
      const double w_des = 1.0;

      input_costs_[i]->UpdateCoefficients(
          2 * (w_diff + w_des) * cost_matrices_.R.at(i),
          -2 * cost_matrices_.R.at(i) *
              (w_diff * u_sol_->at(i) + w_des * u_desired_.at(i)));;
    }
  }

  if (penalize_change_ && options_.penalize_x_change) {
    std::cout << "penalizing change x" << std::endl;
    if (x_sol_->size() < N_) {
      std::cerr << "x sol not set, penalize x change" << std::endl;
    }

    for (int i = 0; i < N_; ++i) {
      const double w_diff = options_.x_change_weight;
      const double w_des = 1.0;

      target_costs_[i]->UpdateCoefficients(
          2 * (w_diff + w_des) * cost_matrices_.Q.at(i),
          -2 * cost_matrices_.Q.at(i) *
              (w_diff * x_sol_->at(i) + w_des * x_desired_.at(i)));
    }
  }

  

  VectorXd delta_init = VectorXd::Zero(n_z_);
  if (options_.delta_option == 1) {
    delta_init.head(n_x_) = x0;
  }
  std::vector<VectorXd> delta(N_, delta_init);
  std::vector<VectorXd> w(N_, VectorXd::Zero(n_z_));
  vector<MatrixXd> G = cost_matrices_.G;

  for (size_t i = 0; i < delta.size(); ++i) {
      if (!(delta[i].allFinite())) {
          drake::log()->error("delta[{}] contains NaN or Inf", i);
      }
  }

  for (size_t i = 0; i < w.size(); ++i) {
      if (!(w[i].allFinite())) {
          drake::log()->error("w[{}] contains NaN or Inf", i);
      }
  }
  

  prev_delta_ = delta;

  for (int iter = 0; iter < options_.admm_iter; iter++) {
    ADMMStep(x0, &delta, &w, &G, iter);

    MatrixXd delta_projection_iter(n_z_, delta.size());
    for (int i = 0; i < delta.size(); i++) {
      delta_projection_iter.col(i) = delta.at(i);
    }
    delta_projection_.push_back(delta_projection_iter);

    // std::cout << "lambda after admm step " << iter << " " << delta.at(0).segment(n_x_, n_lambda_).transpose() << std::endl;
    // std::cout << "eta after admm step " << iter << " " << delta.at(0).segment(n_x_ + n_u_ + n_lambda_, n_lambda_).transpose() << std::endl;
  }

  vector<VectorXd> WD(N_, VectorXd::Zero(n_z_));
  for (int i = 0; i < N_; ++i) {
    WD.at(i) = delta.at(i) - w.at(i);
  }
  *z_fin_ = SolveQP(x0, G, WD, options_.admm_iter, true);

  *w_sol_ = w;
  *delta_sol_ = delta;

  if (!options_.end_on_qp_step) {
    *z_sol_ = delta;
    z_sol_->at(0).segment(0, n_x_) = x0;
    x_sol_->at(0) = x0;
    for (int i = 1; i < N_; ++i) {
      z_sol_->at(i).segment(0, n_x_) =
          lcs_.A().at(i - 1) * x_sol_->at(i - 1) +
          lcs_.B().at(i - 1) * u_sol_->at(i - 1) +
          lcs_.D().at(i - 1) * lambda_sol_->at(i - 1) + lcs_.d().at(i - 1);
    }
  }

  // Undoing to scaling to put variables back into correct units
  // This only scales lambda
  // for (int i = 0; i < N_; ++i) {
  //   lambda_sol_->at(i) *= AnDn_;
  //   z_sol_->at(i).segment(n_x_, n_lambda_) *= AnDn_;
  // }

  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed = finish - start;
  solve_time_ =
      std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() /
      1e6;
}

void C3::ADMMStep(const VectorXd& x0, vector<VectorXd>* delta,
                  vector<VectorXd>* w, vector<MatrixXd>* G,
                  int admm_iteration) {

  vector<VectorXd> WD(N_, VectorXd::Zero(n_z_));

  for (int i = 0; i < N_; ++i) {
    WD.at(i) = delta->at(i) - w->at(i);
  }

  vector<VectorXd> z = SolveQP(x0, *G, WD, admm_iteration, false);

  vector<VectorXd> ZW(N_, VectorXd::Zero(n_z_));
  for (int i = 0; i < N_; ++i) {
    ZW[i] = w->at(i) + z[i];
  }


  if (cost_matrices_.U[0].isZero(0)) {
    *delta = SolveProjection(*G, ZW, admm_iteration);
  } else {
    *delta = SolveProjection(cost_matrices_.U, ZW, admm_iteration);
  }

  // Record lambda - delta_lambda consensus error for this ADMM iteration
  VectorXd lambda_diff_0 = z[0].segment(n_x_, n_lambda_) - delta->at(0).segment(n_x_, n_lambda_);
  lambda_minus_delta_lambda_.push_back(lambda_diff_0);
  lambda_minus_delta_lambda_norms_.push_back(lambda_diff_0.norm());
  lambda_minus_delta_lambda_max_abs_.push_back(lambda_diff_0.cwiseAbs().maxCoeff());

  MatrixXd lambda_diff_all(n_lambda_, N_);
  double sum_norm = 0.0;
  for (int i = 0; i < N_; ++i) {
    VectorXd diff_i = z[i].segment(n_x_, n_lambda_) - delta->at(i).segment(n_x_, n_lambda_);
    lambda_diff_all.col(i) = diff_i;
    sum_norm += diff_i.norm();
  }
  double avg_horizon_norm = (N_ > 0) ? (sum_norm / N_) : lambda_diff_0.norm();
  lambda_minus_delta_lambda_avg_horizon_norms_.push_back(avg_horizon_norm);
  lambda_minus_delta_lambda_all_stages_.push_back(lambda_diff_all);

  // Compute iterate step change ||delta^k - delta^{k-1}|| across horizon N
  double delta_step_sum = 0.0;
  for (int i = 0; i < N_; ++i) {
    delta_step_sum += (delta->at(i) - prev_delta_[i]).norm();
  }
  iterate_step_change_norms_.push_back(N_ > 0 ? (delta_step_sum / N_) : 0.0);
  prev_delta_ = *delta;

  // Compute complementarity slackness sum(lambda_i * eta_i) across horizon N
  double comp_slack_sum = 0.0;
  for (int i = 0; i < N_; ++i) {
    VectorXd lam = z[i].segment(n_x_, n_lambda_);
    VectorXd eta = z[i].segment(n_x_ + n_u_ + n_lambda_, n_lambda_);
    comp_slack_sum += std::abs(lam.dot(eta));
  }
  complementarity_slackness_.push_back(N_ > 0 ? (comp_slack_sum / N_) : 0.0);

  for (int i = 0; i < N_; ++i) {
    w->at(i) = w->at(i) + z[i] - delta->at(i);
    w->at(i) = w->at(i) / options_.rho_scale;
    G->at(i) = G->at(i) * options_.rho_scale;
  }
}

void C3::SetInitialGuessQP(const Eigen::VectorXd& x0, int admm_iteration) {
  prog_.SetInitialGuess(x_[0], x0);
  if (!warm_start_)
    return;  // No warm start

  if (admm_iteration == 0) {
    if (x_sol_->size() < N_) {
      std::cerr << "x sol not set, warm-start" << std::endl;
    }
    if (u_sol_->size() < N_) {
      std::cerr << "u sol not set, warm-start" << std::endl;
    }
    for (int i = 0; i < N_; ++i) {
      prog_.SetInitialGuess(x_[i], x_sol_->at(i));
      prog_.SetInitialGuess(u_[i], u_sol_->at(i));
    }
    prog_.SetInitialGuess(x_[N_], x_sol_->at(N_));
    return;
  } 
 
  int index = solve_time_ / lcs_.dt();
  double weight = (solve_time_ - index * lcs_.dt()) / lcs_.dt();
  for (int i = 0; i < N_ - 1; ++i) {
    prog_.SetInitialGuess(
        x_[i], (1 - weight) * warm_start_x_[admm_iteration - 1][i] +
                   weight * warm_start_x_[admm_iteration - 1][i + 1]);
    prog_.SetInitialGuess(
        lambda_[i], (1 - weight) * warm_start_lambda_[admm_iteration - 1][i] +
                        weight * warm_start_lambda_[admm_iteration - 1][i + 1]);
    prog_.SetInitialGuess(
        u_[i], (1 - weight) * warm_start_u_[admm_iteration - 1][i] +
                   weight * warm_start_u_[admm_iteration - 1][i + 1]);
  }
  prog_.SetInitialGuess(x_[N_], warm_start_x_[admm_iteration - 1][N_]);
}

void C3::StoreQPResults(const MathematicalProgramResult& result,
                        int admm_iteration, bool is_final_solve) {
  for (int i = 0; i < N_; ++i) {
    if (is_final_solve) {
      x_sol_->at(i) = result.GetSolution(x_[i]);
      lambda_sol_->at(i) = result.GetSolution(lambda_[i]);
      u_sol_->at(i) = result.GetSolution(u_[i]);
    }
    z_sol_->at(i).segment(0, n_x_) = result.GetSolution(x_[i]);
    z_sol_->at(i).segment(n_x_, n_lambda_) = result.GetSolution(lambda_[i]);
    z_sol_->at(i).segment(n_x_ + n_lambda_, n_u_) = result.GetSolution(u_[i]);

    x_sol_final_ = result.GetSolution(x_[N_]);
    //std::cout << result.GetSolution(x_[i]).transpose() << std::endl;

  }

  if (!warm_start_)
    return;  // No warm start, so no need to update warm start parameters
  for (int i = 0; i < N_ + 1; ++i) {
    if (i < N_) {
      warm_start_x_[admm_iteration][i] = result.GetSolution(x_[i]);
      warm_start_lambda_[admm_iteration][i] = result.GetSolution(lambda_[i]);
      warm_start_u_[admm_iteration][i] = result.GetSolution(u_[i]);
    }
    warm_start_x_[admm_iteration][N_] = result.GetSolution(x_[N_]);
  }
}

vector<VectorXd> C3::SolveQP(const VectorXd& x0, const vector<MatrixXd>& G,
                             const vector<VectorXd>& WD, int admm_iteration,
                             bool is_final_solve) {         
  
  double w_final = (is_final_solve && options_.end_on_qp_step) ? (options_.w_G_final.value_or(1)) : 1;     

  // Add or update augmented costs
  if (augmented_costs_.size() == 0) {
    for (int i = 0; i < N_; ++i)
      augmented_costs_.push_back(prog_
                                     .AddQuadraticCost(2 * w_final * G.at(i),
                                                       -2 * w_final * G.at(i) * WD.at(i),
                                                       z_.at(i), 1)
                                     .evaluator());
  } else {
    for (int i = 0; i < N_; ++i)
      augmented_costs_[i]->UpdateCoefficients(2 * w_final * G.at(i),
                                              -2 * w_final * G.at(i) * WD.at(i));
  }
  SetInitialGuessQP(x0, admm_iteration);

  for (int i = 0; i < N_; i++) {
    if (!(lcs_.A()[i].allFinite())) {
      drake::log()->error("A[{}] contains NaN or Inf", i);
    }
    if (!(lcs_.B()[i].allFinite())) {
      drake::log()->error("B[{}] contains NaN or Inf", i);
    }
    if (!(lcs_.D()[i].allFinite())) {
      drake::log()->error("D[{}] contains NaN or Inf", i);
    }
    if (!(lcs_.d()[i].allFinite())) {
      drake::log()->error("d[{}] contains NaN or Inf", i);
    }
    if (!(lcs_.E()[i].allFinite())) {
      drake::log()->error("E[{}] contains NaN or Inf", i);
    }
    if (!(lcs_.F()[i].allFinite())) {
      drake::log()->error("F[{}] contains NaN or Inf", i);
    }
    if (!(lcs_.H()[i].allFinite())) {
      drake::log()->error("H[{}] contains NaN or Inf", i);
    }
    if (!(lcs_.c()[i].allFinite())) {
      drake::log()->error("c[{}] contains NaN or Inf", i);
    }
    if (!(cost_matrices_.Q[i].allFinite())) {
      drake::log()->error("Q[{}] contains NaN or Inf", i);
    }
    if (!(cost_matrices_.R[i].allFinite())) {
      drake::log()->error("R[{}] contains NaN or Inf", i);
    }
    if (!(cost_matrices_.G[i].allFinite())) {
      drake::log()->error("G[{}] contains NaN or Inf", i);
    }
    if (!(cost_matrices_.U[i].allFinite())) {
      drake::log()->error("U[{}] contains NaN or Inf", i);
    }
  }
  if (!(cost_matrices_.Q[N_].allFinite())) {
    drake::log()->error("Q[{}] contains NaN or Inf", N_);
  }
  MathematicalProgramResult result = osqp_.Solve(prog_);

  if (!result.is_success()) {
      const auto& details = result.get_solver_details<drake::solvers::OsqpSolver>();
      std::cout << "OSQP Status: " << details.status_val << std::endl;
      std::cout << "Iterations: " << details.iter << std::endl;
      std::cout << "is_final_solve: " << is_final_solve << std::endl;
      std::cout << "x0: " << x0.transpose() << std::endl;
      std::cout << "Primal Res: " << details.primal_res << std::endl;
      std::cout << "Dual Res: " << details.dual_res << std::endl;
      // while(true){}
  } else {
    // const auto& details = result.get_solver_details<drake::solvers::OsqpSolver>();
    // std::cout << "Iterations: " << details.iter << std::endl;
  }
  StoreQPResults(result, admm_iteration, is_final_solve);

  return *z_sol_;
}

vector<VectorXd> C3::SolveProjection(const vector<MatrixXd>& U,
                                     vector<VectorXd>& WZ, int admm_iteration) {
  vector<VectorXd> deltaProj(N_, VectorXd::Zero(n_z_));

  if (options_.num_threads > 0) {
    omp_set_dynamic(0);  // Explicitly disable dynamic teams
    omp_set_num_threads(options_.num_threads);  // Set number of threads
    omp_set_nested(0);
    omp_set_schedule(omp_sched_static, 0);
  }

  // clang-format off
#pragma omp parallel for num_threads( \
    options_.num_threads) if (use_parallelization_in_projection_)
  // clang-format on

  for (int i = 0; i < N_; ++i) {
    if (warm_start_) {
      if (i == N_ - 1) {
        deltaProj[i] =
            SolveSingleProjection(U[i], WZ[i], lcs_.E()[i], lcs_.F()[i],
                                  lcs_.H()[i], lcs_.c()[i], admm_iteration, -1, i);
      } else {
        deltaProj[i] = SolveSingleProjection(
            U[i], WZ[i], lcs_.E()[i], lcs_.F()[i], lcs_.H()[i], lcs_.c()[i],
            admm_iteration, i + 1, i);
      }
    } else {
      deltaProj[i] =
          SolveSingleProjection(U[i], WZ[i], lcs_.E()[i], lcs_.F()[i],
                                lcs_.H()[i], lcs_.c()[i], admm_iteration, -1, i);
    }
  }

  return deltaProj;
}

void C3::AddLinearConstraint(const Eigen::MatrixXd& A,
                             const VectorXd& lower_bound,
                             const VectorXd& upper_bound,
                             ConstraintVariable constraint) {
  DRAKE_DEMAND(A.allFinite());
  DRAKE_DEMAND(lower_bound.allFinite());
  DRAKE_DEMAND(upper_bound.allFinite());
  for (int i = 0; i < lower_bound.size(); i++) {
    DRAKE_DEMAND(lower_bound(i) <= upper_bound(i));
  }


  if (constraint == 1) {
    DRAKE_DEMAND(A.cols() == n_x_);
    DRAKE_DEMAND(lower_bound.rows() == A.rows());
    DRAKE_DEMAND(upper_bound.rows() == A.rows());

    for (int i = 1; i < N_; ++i) {
      user_constraints_.push_back(
          prog_.AddLinearConstraint(A, lower_bound, upper_bound, x_.at(i)));
    }
  }

  if (constraint == 2) {
    DRAKE_DEMAND(A.cols() == n_u_);
    DRAKE_DEMAND(lower_bound.rows() == A.rows());
    DRAKE_DEMAND(upper_bound.rows() == A.rows());
    for (int i = 0; i < N_; ++i) {
      user_constraints_.push_back(
          prog_.AddLinearConstraint(A, lower_bound, upper_bound, u_.at(i)));
    }
  }

  if (constraint == 3) {
    DRAKE_DEMAND(A.cols() == n_lambda_);
    DRAKE_DEMAND(lower_bound.rows() == A.rows());
    DRAKE_DEMAND(upper_bound.rows() == A.rows());
    for (int i = 0; i < N_; ++i) {
      user_constraints_.push_back(prog_.AddLinearConstraint(
          A, lower_bound, upper_bound, lambda_.at(i)));
    }
  }
}

void C3::AddLinearConstraint(const Eigen::RowVectorXd& A, double lower_bound,
                             double upper_bound,
                             ConstraintVariable constraint) {
  Eigen::VectorXd lb(1);
  lb << lower_bound;
  Eigen::VectorXd ub(1);
  ub << upper_bound;
  AddLinearConstraint(A, lb, ub, constraint);
}

void C3::RemoveConstraints() {
  for (auto& userconstraint : user_constraints_) {
    prog_.RemoveConstraint(userconstraint);
  }
  user_constraints_.clear();
}

void C3::AddTerminalConstraint() {
  epsilon_ = prog_.NewContinuousVariables(n_x_, "epsilon");

  terminal_slack_cost_ = 
      prog_.AddQuadraticCost(MatrixXd::Identity(n_x_, n_x_), VectorXd::Zero(n_x_), epsilon_)            
            .evaluator()
            .get();

  prog_.AddLinearConstraint(MatrixXd::Identity(n_x_, n_x_), VectorXd::Zero(n_x_), 
      VectorXd::Constant(n_x_, std::numeric_limits<double>::infinity()), epsilon_);

  MatrixXd A(2* n_x_, 2 * n_x_);
  A.block(0, 0, n_x_, n_x_) = MatrixXd::Identity(n_x_, n_x_);
  A.block(0, n_x_, n_x_, n_x_) = -MatrixXd::Identity(n_x_, n_x_);
  A.block(n_x_, 0, n_x_, n_x_) = MatrixXd::Identity(n_x_, n_x_);
  A.block(n_x_, n_x_, n_x_, n_x_) = MatrixXd::Identity(n_x_, n_x_);

  terminal_constraint_ = 
      prog_.AddLinearConstraint(A, VectorXd::Zero(2 * n_x_), VectorXd::Zero(2 * n_x_), {x_[N_], epsilon_})
          .evaluator()
          .get();
    
}

void C3::UpdateTerminalTarget(MatrixXd Q_slack, VectorXd x_target_terminal) {
  // std::cout << "update terminal target " << std::endl;
  terminal_slack_cost_->UpdateCoefficients(2 * Q_slack, VectorXd::Zero(n_x_));
  
  MatrixXd A(2* n_x_, 2 * n_x_);
  A.block(0, 0, n_x_, n_x_) = MatrixXd::Identity(n_x_, n_x_);
  A.block(0, n_x_, n_x_, n_x_) = -MatrixXd::Identity(n_x_, n_x_);
  A.block(n_x_, 0, n_x_, n_x_) = MatrixXd::Identity(n_x_, n_x_);
  A.block(n_x_, n_x_, n_x_, n_x_) = MatrixXd::Identity(n_x_, n_x_);

  VectorXd lb(2*n_x_);
  VectorXd ub(2*n_x_);

  lb.segment(0, n_x_) = VectorXd::Constant(n_x_, -std::numeric_limits<double>::infinity());
  lb.segment(n_x_, n_x_) = x_target_terminal;
  ub.segment(0, n_x_) = x_target_terminal;
  ub.segment(n_x_, n_x_) = VectorXd::Constant(n_x_, std::numeric_limits<double>::infinity());

  terminal_constraint_->UpdateCoefficients(A, lb, ub);
}

void C3::AddLambdaBound(double bound, std::vector<int> indices) {
  DRAKE_DEMAND(bound >= 0);

  // const double lb = -std::numeric_limits<double>::infinity();
  const double lb = 0;
  const double ub = bound / AnDn_;

  for (int i = 0; i < N_; i++) {
    for (int idx : indices) {
      prog_.AddBoundingBoxConstraint(lb, ub, lambda_[i](idx));
    }
  }
}

// Manual implementation of L1 norm
void C3::AddL1Cost(MatrixXd A, CostVariable variable, int start_idx) {

  DRAKE_DEMAND(A.rows() == A.cols());

  int var_size = A.rows();

  // position l1 cost
  if (variable == 1 || variable == 2) {
    int discount_factor = 1;
    drake::solvers::VectorXDecisionVariable y;
    std::vector<drake::solvers::VectorXDecisionVariable> s;

    y = prog_.NewContinuousVariables(N_+1);
    for (int i = 0; i < N_+1; i++) {
      s.push_back(prog_.NewContinuousVariables(var_size));
    }

    prog_.AddQuadraticCost(2 * MatrixXd::Identity(N_+1, N_+1), VectorXd::Zero(N_+1), y);
    for (int i = 0; i < N_+1; i++) {
      MatrixXd A_discounted = discount_factor * A;

      MatrixXd M_1(A_discounted.rows(), A_discounted.cols() + var_size);
      MatrixXd M_2(A_discounted.rows(), A_discounted.cols() + var_size);
      M_1 << A_discounted, -MatrixXd::Identity(var_size, var_size);
      M_2 << -A_discounted, -MatrixXd::Identity(var_size, var_size);
      VectorXd b = A_discounted * x_desired_.at(i).segment(start_idx, var_size);

      drake::solvers::VariableRefList vars;
      vars.push_back(x_.at(i).segment(start_idx, var_size));
      vars.push_back(s.at(i));
      drake::solvers::VectorXDecisionVariable z = drake::solvers::ConcatenateVariableRefList(vars);
      Eigen::VectorXd lower_bound = Eigen::VectorXd::Constant(var_size, -std::numeric_limits<double>::infinity());
    
      // [A, -I] [x, s]ᵀ <= A(xd), equiv to A(x-xd) <= s
      // [-A, -I] [x, s]ᵀ <= -A(xd), equiv to -A(x-xd) <= s
      prog_.AddLinearConstraint(M_1, lower_bound, b, z);
      prog_.AddLinearConstraint(M_2, lower_bound, -b, z);

      prog_.AddLinearConstraint(VectorXd::Ones(var_size).transpose() * s.at(i) <= y(i)); // y is a 1d vector
      prog_.AddLinearConstraint(VectorXd::Zero(var_size) <= s.at(i));

      discount_factor *= options_.gamma;
    }
    prog_.AddLinearConstraint(VectorXd::Zero(N_+1) <= y);

  }

  // u l1 cost
  if (variable == 3) {
    int discount_factor = 1;
    drake::solvers::VectorXDecisionVariable y;
    std::vector<drake::solvers::VectorXDecisionVariable> s;

    y = prog_.NewContinuousVariables(N_);
    for (int i = 0; i < N_; i++) {
      s.push_back(prog_.NewContinuousVariables(var_size));
    }

    prog_.AddQuadraticCost(2 * MatrixXd::Identity(N_, N_), VectorXd::Zero(N_), y);
    for (int i = 0; i < N_; i++) {
      MatrixXd A_discounted = discount_factor * A;

      MatrixXd M_1(A_discounted.rows(), A_discounted.cols() + var_size);
      MatrixXd M_2(A_discounted.rows(), A_discounted.cols() + var_size);
      M_1 << A_discounted, -MatrixXd::Identity(var_size, var_size);
      M_2 << -A_discounted, -MatrixXd::Identity(var_size, var_size);
      VectorXd b = A_discounted * u_desired_.at(i).segment(start_idx, var_size);

      drake::solvers::VariableRefList vars;
      vars.push_back(u_.at(i).segment(start_idx, var_size));
      vars.push_back(s.at(i));
      drake::solvers::VectorXDecisionVariable z = drake::solvers::ConcatenateVariableRefList(vars);
      Eigen::VectorXd lower_bound = Eigen::VectorXd::Constant(var_size, -std::numeric_limits<double>::infinity());
    
      // [A, -I] [x, s]ᵀ <= A(xd), equiv to A(x-xd) <= s
      // [-A, -I] [x, s]ᵀ <= -A(xd), equiv to -A(x-xd) <= s
      prog_.AddLinearConstraint(M_1, lower_bound, b, z);
      prog_.AddLinearConstraint(M_2, lower_bound, -b, z);

      prog_.AddLinearConstraint(VectorXd::Ones(var_size).transpose() * s.at(i) <= y(i)); // y is a 1d vector
      prog_.AddLinearConstraint(VectorXd::Zero(var_size) <= s.at(i));

      discount_factor *= options_.gamma;
    }
    prog_.AddLinearConstraint(VectorXd::Zero(N_) <= y);
  }


}

void C3::AddAccelerationCost(int n_q, int n_v, double weight) {
  DRAKE_DEMAND(n_q + n_v == n_x_);
  // (v[i] - v[i+1])' Q_v (v[i] - v[i+1]) = v[i]' Q_v v[i] - 2v[i]' Q_v v[i+1] + v[i+1]' Q_v v[i+1]
  for (int i = 0; i < N_; ++i) {

    /* [Q, -Q
        -Q, Q] */
    MatrixXd cost_matrix(MatrixXd::Zero(2*n_v, 2*n_v));
    // cost_matrix.block(0, 0, n_v, n_v) = cost_matrices_.Q.at(i).block(n_q, n_q, n_v, n_v);
    // cost_matrix.block(n_v, n_v, n_v, n_v) = cost_matrices_.Q.at(i).block(n_q, n_q, n_v, n_v);
    // cost_matrix.block(n_v, 0, n_v, n_v) = -1 * cost_matrices_.Q.at(i).block(n_q, n_q, n_v, n_v);
    // cost_matrix.block(0, n_v, n_v, n_v) = -1 * cost_matrices_.Q.at(i).block(n_q, n_q, n_v, n_v);

    cost_matrix.block(0, 0, n_v, n_v) = MatrixXd::Identity(n_v, n_v);
    cost_matrix.block(n_v, n_v, n_v, n_v) = MatrixXd::Identity(n_v, n_v);
    cost_matrix.block(n_v, 0, n_v, n_v) = -1 * MatrixXd::Identity(n_v, n_v);
    cost_matrix.block(0, n_v, n_v, n_v) = -1 * MatrixXd::Identity(n_v, n_v);

    drake::solvers::VariableRefList vars;
    vars.push_back(x_.at(i).segment(n_q, n_v));
    vars.push_back(x_.at(i+1).segment(n_q, n_v));
    drake::solvers::VectorXDecisionVariable z = drake::solvers::ConcatenateVariableRefList(vars);

    prog_.AddQuadraticCost(
        2 * weight * cost_matrix, VectorXd::Zero(2*n_v), z
    );
  }
  
} 


// y - quadratic slack variables
// v,w - positive/negative outlier slack variables
// NOTE: L matrix is hardcoded to identity currently
void C3::AddHuberCost(MatrixXd L, double weight, double delta, CostVariable variable, int start_idx, int frequency) {
  int var_size = L.rows();

  if (variable == 1 || variable == 2) {
    vector<drake::solvers::VectorXDecisionVariable> y;
    vector<drake::solvers::VectorXDecisionVariable> v;
    vector<drake::solvers::VectorXDecisionVariable> w;

    for (int i = N_; i >= 0 ; i -= frequency) {
      y.push_back(prog_.NewContinuousVariables(var_size));
      v.push_back(prog_.NewContinuousVariables(var_size));
      w.push_back(prog_.NewContinuousVariables(var_size));
    }
        
    
    for (int i = N_; i >= 0 ; i -= frequency) {
      prog_.AddQuadraticCost(2 * weight * MatrixXd::Identity(var_size, var_size), VectorXd::Zero(var_size), y.at(i / frequency));
      prog_.AddLinearCost(weight * delta * Eigen::RowVectorXd::Ones(var_size), 0, v.at(i / frequency));
      prog_.AddLinearCost(weight * delta * Eigen::RowVectorXd::Ones(var_size), 0, w.at(i / frequency));

      MatrixXd M(var_size, 4 * var_size);
      M << L.transpose(), -MatrixXd::Identity(var_size, var_size), 
        -MatrixXd::Identity(var_size, var_size), MatrixXd::Identity(var_size, var_size);

      drake::solvers::VariableRefList vars;
      vars.push_back(x_.at(i).segment(start_idx, var_size));
      vars.push_back(y.at(i / frequency));
      vars.push_back(v.at(i / frequency));
      vars.push_back(w.at(i / frequency));
      drake::solvers::VectorXDecisionVariable z = drake::solvers::ConcatenateVariableRefList(vars);
    
      prog_.AddLinearEqualityConstraint(M, L.transpose() * x_desired_.at(i).segment(start_idx, var_size), z);
      prog_.AddBoundingBoxConstraint(-delta, delta, y.at(i / frequency));
      prog_.AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), v.at(i / frequency));
      prog_.AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), w.at(i / frequency));
    }
  }

  if (variable == 3) {
    vector<drake::solvers::VectorXDecisionVariable> y;
    vector<drake::solvers::VectorXDecisionVariable> v;
    vector<drake::solvers::VectorXDecisionVariable> w;

    for (int i = N_-1; i >= 0 ; i -= frequency) {
      y.push_back(prog_.NewContinuousVariables(var_size));
      v.push_back(prog_.NewContinuousVariables(var_size));
      w.push_back(prog_.NewContinuousVariables(var_size));
    }
        
    
    for (int i = N_-1; i >= 0 ; i -= frequency) {
      prog_.AddQuadraticCost(2 * weight * MatrixXd::Identity(var_size, var_size), VectorXd::Zero(var_size), y.at(i / frequency));
      prog_.AddLinearCost(weight * delta * Eigen::RowVectorXd::Ones(var_size), 0, v.at(i / frequency));
      prog_.AddLinearCost(weight * delta * Eigen::RowVectorXd::Ones(var_size), 0, w.at(i / frequency));

      MatrixXd M(var_size, 4 * var_size);
      M << L.transpose(), -MatrixXd::Identity(var_size, var_size), 
        -MatrixXd::Identity(var_size, var_size), MatrixXd::Identity(var_size, var_size);

      drake::solvers::VariableRefList vars;
      vars.push_back(u_.at(i).segment(start_idx, var_size));
      vars.push_back(y.at(i / frequency));
      vars.push_back(v.at(i / frequency));
      vars.push_back(w.at(i / frequency));
      drake::solvers::VectorXDecisionVariable z = drake::solvers::ConcatenateVariableRefList(vars);
    

      prog_.AddLinearEqualityConstraint(M, L.transpose() * u_desired_.at(i), z);
      prog_.AddBoundingBoxConstraint(-delta, delta, y.at(i / frequency));
      prog_.AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), v.at(i / frequency));
      prog_.AddBoundingBoxConstraint(0, std::numeric_limits<double>::infinity(), w.at(i / frequency));
    }
  }
}

void C3::AddRegularizationCostsState(std::vector<Eigen::MatrixXd> Q_reg) {
  if (!options_.penalize_x_change) return;

  DRAKE_DEMAND(x_hat_.size() == N_+1);
  DRAKE_DEMAND(Q_reg.size() == N_+1);

  for (int i = 0; i < N_+1; i++) {
    prog_.AddQuadraticCost(2 * options_.x_change_weight * Q_reg.at(i), 
        -2 * options_.x_change_weight * Q_reg.at(i) * x_hat_.at(i), x_.at(i));
  }
}
  
void C3::AddRegularizationCostsInput(std::vector<Eigen::MatrixXd> R_reg) {
  if (!options_.penalize_input_change) return;

  DRAKE_DEMAND(u_hat_.size() == N_);
  DRAKE_DEMAND(R_reg.size() == N_);

  for (int i = 0; i < N_; i++) {
    prog_.AddQuadraticCost(2 * options_.input_change_weight * R_reg.at(i), 
        -2 * options_.input_change_weight * R_reg.at(i) * u_hat_.at(i), u_.at(i));
  }
}

const std::vector<LinearConstraintBinding>& C3::GetLinearConstraints() {
  return user_constraints_;
}


Eigen::MatrixXd C3::ComputePolicyJacobian() {
  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  const auto& A_dyn = lcs_.A();
  const auto& B_dyn = lcs_.B();
  const auto& D_dyn = lcs_.D();
  const auto& E_dyn = lcs_.E();
  const auto& F_dyn = lcs_.F();
  const auto& H_dyn = lcs_.H();

  // 1. Terminal cost Hessian P_N = 2 * Q_N
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
  if (N_ < static_cast<int>(cost_matrices_.Q.size())) {
    P = 2.0 * cost_matrices_.Q[N_];
  }
  P.diagonal().array() += 1e-6;

  const double active_tol = 1e-4;
  MatrixXd Q_uu_0 = MatrixXd::Identity(n_u_, n_u_);
  MatrixXd Q_ux_0 = MatrixXd::Zero(n_u_, n_x_);

  // 2. Backward Riccati recursion from k = N-1 down to 0
  for (int k = N_ - 1; k >= 0; --k) {
    MatrixXd A_k = (k < static_cast<int>(A_dyn.size())) ? A_dyn[k] : MatrixXd::Identity(n_x_, n_x_).eval();
    MatrixXd B_k = (k < static_cast<int>(B_dyn.size())) ? B_dyn[k] : MatrixXd::Zero(n_x_, n_u_).eval();
    MatrixXd D_k = (k < static_cast<int>(D_dyn.size())) ? D_dyn[k] : MatrixXd::Zero(n_x_, n_lambda_).eval();
    MatrixXd E_k = (k < static_cast<int>(E_dyn.size())) ? E_dyn[k] : MatrixXd::Zero(n_lambda_, n_x_).eval();
    MatrixXd F_k = (k < static_cast<int>(F_dyn.size())) ? F_dyn[k] : MatrixXd::Zero(n_lambda_, n_lambda_).eval();
    MatrixXd H_k = (k < static_cast<int>(H_dyn.size())) ? H_dyn[k] : MatrixXd::Zero(n_lambda_, n_u_).eval();

    // Identify active contact set at stage k
    std::vector<int> active;
    if (lambda_sol_ && k < static_cast<int>(lambda_sol_->size())) {
      const VectorXd& lam = lambda_sol_->at(k);
      for (int j = 0; j < n_lambda_; ++j) {
        if (lam(j) > active_tol) {
          active.push_back(j);
        }
      }
    }

    // Condense lambda: f_x = A - D_a * F_aa^{-1} * E_a, f_u = B - D_a * F_aa^{-1} * H_a
    MatrixXd f_x, f_u;
    if (active.empty()) {
      f_x = A_k;
      f_u = B_k;
    } else {
      int na = static_cast<int>(active.size());
      MatrixXd D_a(n_x_, na);
      MatrixXd E_a(na, n_x_);
      MatrixXd H_a(na, n_u_);
      MatrixXd F_aa(na, na);

      for (int a = 0; a < na; ++a) {
        int ja = active[a];
        D_a.col(a) = D_k.col(ja);
        E_a.row(a) = E_k.row(ja);
        H_a.row(a) = H_k.row(ja);
        for (int cb = 0; cb < na; ++cb) {
          F_aa(a, cb) = F_k(ja, active[cb]);
        }
      }

      MatrixXd F_reg = F_aa + 1e-6 * MatrixXd::Identity(na, na);
      Eigen::LDLT<MatrixXd> solver(F_reg);
      MatrixXd FinvE = solver.solve(E_a);
      MatrixXd FinvH = solver.solve(H_a);

      f_x = A_k - D_a * FinvE;
      f_u = B_k - D_a * FinvH;
    }

    // Stage control cost R_k and state cost Q_k
    MatrixXd R_k = MatrixXd::Zero(n_u_, n_u_);
    if (k < static_cast<int>(cost_matrices_.R.size())) {
      R_k = 2.0 * cost_matrices_.R[k];
    }
    R_k.diagonal().array() += 1e-6;

    MatrixXd Q_k = MatrixXd::Zero(n_x_, n_x_);
    if (k < static_cast<int>(cost_matrices_.Q.size())) {
      Q_k = 2.0 * cost_matrices_.Q[k];
    }

    // Quadratic expansions
    MatrixXd Q_uu = R_k + f_u.transpose() * P * f_u;
    Q_uu.diagonal().array() += 1e-6;
    MatrixXd Q_ux = f_u.transpose() * P * f_x;
    MatrixXd Q_xx = Q_k + f_x.transpose() * P * f_x;

    if (k == 0) {
      Q_uu_0 = Q_uu;
      Q_ux_0 = Q_ux;
    }

    Eigen::LDLT<MatrixXd> solver_u(Q_uu);
    MatrixXd K_k = -solver_u.solve(Q_ux);

    // Riccati value function update
    P = Q_xx + Q_ux.transpose() * K_k;
    P = 0.5 * (P + P.transpose());
  }

  // 3. Identify active box bounds on u_0
  std::vector<bool> is_clamped(n_u_, false);
  if (u_sol_ && !u_sol_->empty()) {
    const VectorXd& u0_val = u_sol_->at(0);
    for (const auto& binding : prog_.bounding_box_constraints()) {
      const auto& lb = binding.evaluator()->lower_bound();
      const auto& ub = binding.evaluator()->upper_bound();
      const auto& vars = binding.variables();
      for (int i = 0; i < vars.size(); ++i) {
        for (int j = 0; j < n_u_; ++j) {
          if (vars(i).get_id() == u_[0](j).get_id()) {
            double v = u0_val(j);
            if ((std::isfinite(lb(i)) && std::abs(v - lb(i)) < active_tol) ||
                (std::isfinite(ub(i)) && std::abs(v - ub(i)) < active_tol)) {
              is_clamped[j] = true;
            }
          }
        }
      }
    }
  }

  // Partition u_0 into free and clamped indices
  std::vector<int> free_idx;
  for (int j = 0; j < n_u_; ++j) {
    if (!is_clamped[j]) {
      free_idx.push_back(j);
    }
  }

  MatrixXd K_0 = MatrixXd::Zero(n_u_, n_x_);
  if (free_idx.empty()) {
    return K_0;  // All inputs are saturated at box bounds
  }

  int n_free = static_cast<int>(free_idx.size());
  MatrixXd Q_uu_free(n_free, n_free);
  MatrixXd Q_ux_free(n_free, n_x_);
  for (int r = 0; r < n_free; ++r) {
    int jr = free_idx[r];
    Q_ux_free.row(r) = Q_ux_0.row(jr);
    for (int c = 0; c < n_free; ++c) {
      Q_uu_free(r, c) = Q_uu_0(jr, free_idx[c]);
    }
  }
  // 4. Identify active linear inequality constraints involving u_0
  std::vector<VectorXd> active_linear_normals;
  if (u_sol_ && !u_sol_->empty()) {
    for (const auto& binding : prog_.linear_constraints()) {
      const auto& lb = binding.evaluator()->lower_bound();
      const auto& ub = binding.evaluator()->upper_bound();
      const auto& vars = binding.variables();

      // Check if this constraint touches any free components of u_0
      bool has_free_u0 = false;
      for (int i = 0; i < vars.size(); ++i) {
        for (int idx : free_idx) {
          if (vars(i).get_id() == u_[0](idx).get_id()) {
            has_free_u0 = true;
            break;
          }
        }
        if (has_free_u0) break;
      }
      if (!has_free_u0) continue;

      // Extract solution values for variables in this constraint from cached solutions
      VectorXd var_vals = VectorXd::Zero(vars.size());
      for (int i = 0; i < vars.size(); ++i) {
        const auto& v = vars(i);
        const auto vid = v.get_id();
        bool found = false;

        // Check u
        if (u_sol_) {
          for (int k = 0; k < static_cast<int>(u_sol_->size()) && !found; ++k) {
            for (int j = 0; j < n_u_ && !found; ++j) {
              if (vid == u_[k](j).get_id()) {
                var_vals(i) = (*u_sol_)[k](j);
                found = true;
              }
            }
          }
        }

        // Check x (stages 0 to N-1)
        if (!found && x_sol_) {
          for (int k = 0; k < static_cast<int>(x_sol_->size()) && !found; ++k) {
            for (int j = 0; j < n_x_ && !found; ++j) {
              if (vid == x_[k](j).get_id()) {
                var_vals(i) = (*x_sol_)[k](j);
                found = true;
              }
            }
          }
        }

        // Check terminal state x_N
        if (!found && x_.size() > static_cast<size_t>(N_) &&
            x_sol_final_.size() == n_x_) {
          for (int j = 0; j < n_x_ && !found; ++j) {
            if (vid == x_[N_](j).get_id()) {
              var_vals(i) = x_sol_final_(j);
              found = true;
            }
          }
        }

        // Check lambda
        if (!found && lambda_sol_) {
          for (int k = 0; k < static_cast<int>(lambda_sol_->size()) && !found; ++k) {
            for (int j = 0; j < n_lambda_ && !found; ++j) {
              if (vid == lambda_[k](j).get_id()) {
                var_vals(i) = (*lambda_sol_)[k](j);
                found = true;
              }
            }
          }
        }
      }

      VectorXd c_val;
      binding.evaluator()->Eval(var_vals, &c_val);
      const MatrixXd& A_mat = binding.evaluator()->GetDenseA();

      for (int r = 0; r < c_val.size(); ++r) {
        bool active_lb = std::isfinite(lb(r)) && std::abs(c_val(r) - lb(r)) < active_tol;
        bool active_ub = std::isfinite(ub(r)) && std::abs(c_val(r) - ub(r)) < active_tol;
        if (active_lb || active_ub) {
          // Extract gradient row with respect to free components of u_0
          VectorXd a_free = VectorXd::Zero(n_free);
          for (int i = 0; i < vars.size(); ++i) {
            for (int f = 0; f < n_free; ++f) {
              if (vars(i).get_id() == u_[0](free_idx[f]).get_id()) {
                a_free(f) += A_mat(r, i);
              }
            }
          }
          if (a_free.squaredNorm() > 1e-8) {
            active_linear_normals.push_back(a_free);
          }
        }
      }
    }
  }

  // 5. Solve for feedback gain on free subspace:
  //    delta u_free = argmin 1/2 delta u_free^T Q_uu_free delta u_free + delta u_free^T Q_ux_free delta x_0
  //                   s.t.   A_act delta u_free = 0
  int n_act = static_cast<int>(active_linear_normals.size());
  MatrixXd K_free;

  if (n_act == 0) {
    Eigen::LDLT<MatrixXd> solver_free(Q_uu_free);
    K_free = -solver_free.solve(Q_ux_free);
  } else {
    MatrixXd A_act(n_act, n_free);
    for (int i = 0; i < n_act; ++i) {
      A_act.row(i) = active_linear_normals[i];
    }

    // Solve regularized KKT system directly (avoids rank deficiency if LICQ fails)
    MatrixXd KKT(n_free + n_act, n_free + n_act);
    KKT.setZero();
    KKT.topLeftCorner(n_free, n_free) = Q_uu_free;
    KKT.topRightCorner(n_free, n_act) = A_act.transpose();
    KKT.bottomLeftCorner(n_act, n_free) = A_act;
    KKT.bottomRightCorner(n_act, n_act) = -1e-6 * MatrixXd::Identity(n_act, n_act);

    MatrixXd RHS = MatrixXd::Zero(n_free + n_act, n_x_);
    RHS.topRows(n_free) = -Q_ux_free;

    Eigen::ColPivHouseholderQR<MatrixXd> qr_kkt(KKT);
    MatrixXd sol = qr_kkt.solve(RHS);
    K_free = sol.topRows(n_free);
  }

  // 6. Embed free feedback gains back into full K_0
  for (int r = 0; r < n_free; ++r) {
    K_0.row(free_idx[r]) = K_free.row(r);
  }

  if (!K_0.allFinite()) {
    K_0 = MatrixXd::Zero(n_u_, n_x_);
  }

  return K_0;
}

}  // namespace c3
