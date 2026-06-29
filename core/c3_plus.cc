#include "c3_plus.h"

#include <vector>

#include <Eigen/Dense>
#include "drake/common/text_logging.h"

#include "c3_options.h"
#include "lcs.h"

namespace c3 {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

using drake::solvers::MathematicalProgramResult;

C3Plus::C3Plus(const LCS& lcs, const CostMatrices& costs,
               const vector<VectorXd>& xdesired, const C3Options& options)
    : C3(lcs, costs, xdesired, options,
         lcs.num_states() + 2 * lcs.num_lambdas() + lcs.num_inputs()) {

  if (warm_start_) {
    warm_start_eta_.resize(options_.admm_iter + 1);
    for (int iter = 0; iter < options_.admm_iter + 1; ++iter) {
      warm_start_eta_[iter].resize(N_);
      for (int i = 0; i < N_; ++i) {
        warm_start_eta_[iter][i] = VectorXd::Zero(n_lambda_);
      }
    }
  }

  // Initialize eta as optimization variables
  eta_ = vector<drake::solvers::VectorXDecisionVariable>();
  eta_sol_ = std::make_unique<std::vector<VectorXd>>();
  for (int i = 0; i < N_; ++i) {
    eta_sol_->push_back(Eigen::VectorXd::Zero(n_lambda_));
    eta_.push_back(
        prog_.NewContinuousVariables(n_lambda_, "eta" + std::to_string(i)));
    z_.at(i).push_back(eta_.back());
  }

  // Add eta equality constraints η = E * x + F * λ + H * u + c
  MatrixXd EtaLinEq(n_lambda_, n_x_ + 2 * n_lambda_ + n_u_);
  EtaLinEq.block(0, n_x_ + n_lambda_ + n_u_, n_lambda_, n_lambda_) =
      -1 * MatrixXd::Identity(n_lambda_, n_lambda_);
  eta_constraints_.resize(N_);
  for (int i = 0; i < N_; ++i) {

    EtaLinEq.block(0, 0, n_lambda_, n_x_) = lcs_.E().at(i);
    EtaLinEq.block(0, n_x_, n_lambda_, n_lambda_) = lcs_.F().at(i);
    EtaLinEq.block(0, n_x_ + n_lambda_, n_lambda_, n_u_) = lcs_.H().at(i);

    if (EtaLinEq.array().isNaN().any()) {
      std::cout << EtaLinEq.rows() << " x " << EtaLinEq.cols() << std::endl;
      Eigen::Index row, col;
      EtaLinEq.array().isNaN().maxCoeff(&row, &col);
    }
    
    eta_constraints_[i] =
        prog_
            .AddLinearEqualityConstraint(
                EtaLinEq, -lcs_.c().at(i),
                {x_.at(i), lambda_.at(i), u_.at(i), eta_.at(i)})
            .evaluator()
            .get();
  }

  // Disable parallelization for C3+ because of the overhead cost
  use_parallelization_in_projection_ = false;
}

void C3Plus::UpdateLCS(const LCS& lcs) {
  C3::UpdateLCS(lcs);
  MatrixXd EtaLinEq(n_lambda_, n_x_ + 2 * n_lambda_ + n_u_);
  EtaLinEq.block(0, n_x_ + n_lambda_ + n_u_, n_lambda_, n_lambda_) =
      -1 * MatrixXd::Identity(n_lambda_, n_lambda_);
  for (int i = 0; i < N_; ++i) {
    EtaLinEq.block(0, 0, n_lambda_, n_x_) = lcs_.E().at(i);
    EtaLinEq.block(0, n_x_, n_lambda_, n_lambda_) = lcs_.F().at(i);
    EtaLinEq.block(0, n_x_ + n_lambda_, n_lambda_, n_u_) = lcs_.H().at(i);
    eta_constraints_[i]->UpdateCoefficients(EtaLinEq, -lcs_.c().at(i));
  }
}

void C3Plus::SetInitialGuessQP(const Eigen::VectorXd& x0, int admm_iteration) {
  C3::SetInitialGuessQP(x0, admm_iteration);
  if (!warm_start_ || admm_iteration == 0)
    return;  // No warm start for the first iteration
  int index = solve_time_ / lcs_.dt();
  double weight = (solve_time_ - index * lcs_.dt()) / lcs_.dt();
  for (int i = 0; i < N_ - 1; ++i) {
    prog_.SetInitialGuess(
        eta_[i], (1 - weight) * warm_start_eta_[admm_iteration - 1][i] +
                     weight * warm_start_eta_[admm_iteration - 1][i + 1]);
  }
}

void C3Plus::StoreQPResults(const MathematicalProgramResult& result,
                            int admm_iteration, bool is_final_solve) {
  C3::StoreQPResults(result, admm_iteration, is_final_solve);
  for (int i = 0; i < N_; i++) {
    if (is_final_solve) {
      eta_sol_->at(i) = result.GetSolution(eta_[i]);
    }
    z_sol_->at(i).segment(n_x_ + n_lambda_ + n_u_, n_lambda_) =
        result.GetSolution(eta_[i]);
  }

  if (!warm_start_)
    return;  // No warm start, so no need to update warm start parameters
  for (int i = 0; i < N_; ++i) {
    warm_start_eta_[admm_iteration][i] = result.GetSolution(eta_[i]);
  }
}

VectorXd C3Plus::SolveSingleProjection(const MatrixXd& U,
                                       const VectorXd& delta_c,
                                       const MatrixXd& E, const MatrixXd& F,
                                       const MatrixXd& H, const VectorXd& c,
                                       const int admm_iteration,
                                       const int& warm_start_index) {
  return SolveSingleProjection(U, delta_c, E, F, H, c, admm_iteration, warm_start_index, -1);                                 
}


VectorXd C3Plus::SolveSingleProjection(const MatrixXd& U,
                                       const VectorXd& delta_c,
                                       const MatrixXd& E, const MatrixXd& F,
                                       const MatrixXd& H, const VectorXd& c,
                                       const int admm_iteration,
                                       const int& warm_start_index, int timestep) {
  VectorXd delta_proj = delta_c;

  // Extract the weight vectors for lambda and eta from the diagonal of the cost
  // matrix U.
  VectorXd w_eta_vec = U.block(n_x_ + n_lambda_ + n_u_, n_x_ + n_lambda_ + n_u_,
                               n_lambda_, n_lambda_)
                           .diagonal();
  VectorXd w_lambda_vec = U.block(n_x_, n_x_, n_lambda_, n_lambda_).diagonal();

  // Throw an error if any weights are negative.
  if (w_eta_vec.size() > 0 && (w_eta_vec.minCoeff() < 0 || w_lambda_vec.minCoeff() < 0)) {
    throw std::runtime_error(
        "Negative weights in the cost matrix U are not allowed.");
  }

  VectorXd lambda_c = delta_c.segment(n_x_, n_lambda_);
  VectorXd eta_c = delta_c.segment(n_x_ + n_lambda_ + n_u_, n_lambda_);

  // std::cout << "admm " << admm_iteration << std::endl;
  // std::cout << "QP lambda " << admm_iteration << " " << AnDn_ * lambda_c.transpose() << std::endl;
  // std::cout << "QP eta " << admm_iteration << " " << AnDn_ * eta_c.transpose() << std::endl;

  // Set thresholds to 0/inf if not set
  VectorXd lambda_min(VectorXd::Zero(n_lambda_));
  VectorXd lambda_max(VectorXd::Constant(n_lambda_, std::numeric_limits<double>::infinity()));

  VectorXd eta_min(VectorXd::Zero(n_lambda_)); 
  VectorXd eta_max(VectorXd::Constant(n_lambda_, std::numeric_limits<double>::infinity()));


  if (options_.lambda_threshold.has_value() && options_.lambda_threshold.value().size() != 0) {
    lambda_min = Eigen::Map<const Eigen::VectorXd>(
          options_.lambda_threshold.value().data(), options_.lambda_threshold.value().size());
    lambda_min /= AnDn_;
  }
  if (options_.eta_threshold.has_value() && options_.eta_threshold.value().size() != 0) {
    eta_min = Eigen::Map<const Eigen::VectorXd>(
          options_.eta_threshold.value().data(), options_.eta_threshold.value().size());
    eta_min /= AnDn_;
  }

  // Assumes stewart and trinkle
  if (options_.gamma_threshold.has_value() && options_.gamma_threshold.value().size() != 0) {
    int n_contacts = options_.gamma_threshold.value().size();
    VectorXd gamma_threshold(n_contacts);
    gamma_threshold = Eigen::Map<const Eigen::VectorXd>(
      options_.gamma_threshold.value().data(), n_contacts);
    lambda_min.segment(0, n_contacts) = gamma_threshold / AnDn_;
  }

  // Assumes stewart and trinkle
  if (options_.phi_threshold.has_value() && options_.phi_threshold.value().size() != 0) {
    int n_contacts = options_.phi_threshold.value().size();
    VectorXd phi_threshold(n_contacts);
    phi_threshold = Eigen::Map<const Eigen::VectorXd>(
      options_.phi_threshold.value().data(), n_contacts);
    phi_threshold /= AnDn_; // Adjust for scaling
  
    eta_min.segment(n_contacts, n_contacts) = phi_threshold;
  }

  if (options_.add_phi_buffer.value_or(false) && options_.epsilon.has_value() && options_.epsilon.value().size() != 0 &&
      timestep >= 0 && delta_projection_.size() > 0) {
    int n_contacts = options_.epsilon.value().size();
  
    VectorXd eta_prev = delta_projection_[delta_projection_.size()-1]
                          .col(std::max(0, timestep-1)).segment(n_x_+n_lambda_+n_u_, n_lambda_);
    VectorXd eta_next = delta_projection_[delta_projection_.size()-1]
                          .col(std::min(N_, timestep+1)).segment(n_x_+n_lambda_+n_u_, n_lambda_);

    VectorXd epsilon(n_contacts);
    epsilon = Eigen::Map<const Eigen::VectorXd>(options_.epsilon.value().data(), n_contacts);

    VectorXd phi_buffer = epsilon 
        + eta_prev.segment(n_contacts, n_contacts).cwiseMin(eta_next.segment(n_contacts, n_contacts));
    eta_min.segment(n_contacts, n_contacts) = eta_min.segment(n_contacts, n_contacts).cwiseMax(phi_buffer);
  }

  // std::cout << "AnDn " << AnDn_ << std::endl;
  // std::cout << "lambda min " << lambda_min.transpose() << std::endl;
  // std::cout << "eta min " << eta_min.transpose() << std::endl;

  // Compare costs, threshold
  VectorXd eta_star = eta_c.cwiseMax(eta_min).cwiseMin(eta_max);
  VectorXd lambda_star = lambda_c.cwiseMax(lambda_min).cwiseMin(lambda_max);

  // Check if projecting to eta or lambda incurs less cost
  Eigen::Array<bool, Eigen::Dynamic, 1> eta_cost_smaller =
      w_eta_vec.array() * (eta_c - eta_star).array().square() +  w_lambda_vec.array() * lambda_c.array().square() <
      w_lambda_vec.array() * (lambda_c - lambda_star).array().square() +  w_eta_vec.array() * eta_c.array().square();

  // Change thresholds pointwise to obey complimentarity
  lambda_min = eta_cost_smaller.select(VectorXd::Zero(n_lambda_), lambda_min); // 
  eta_min = eta_cost_smaller.select(eta_min, VectorXd::Zero(n_lambda_));

  lambda_c = lambda_c.cwiseMax(lambda_min).cwiseMin(lambda_max);
  eta_c = eta_c.cwiseMax(eta_min).cwiseMin(eta_max);

  // Select variable with smaller cost
  delta_proj.segment(n_x_, n_lambda_) =
      eta_cost_smaller.select(VectorXd::Zero(n_lambda_), lambda_c);
  delta_proj.segment(n_x_ + n_lambda_ + n_u_, n_lambda_) =
      eta_cost_smaller.select(eta_c, VectorXd::Zero(n_lambda_));

  delta_proj.segment(n_x_, n_lambda_) =
      delta_proj.segment(n_x_, n_lambda_).cwiseMax(0);
  delta_proj.segment(n_x_ + n_lambda_ + n_u_, n_lambda_) =
      delta_proj.segment(n_x_ + n_lambda_ + n_u_, n_lambda_).cwiseMax(0);

  // std::cout << "lambda projected " << AnDn_ * delta_proj.segment(n_x_, n_lambda_).transpose() << std::endl;
  // std::cout << "eta projected " << AnDn_ * delta_proj.segment(n_x_ + n_lambda_ + n_u_, n_lambda_).transpose() << std::endl;
  // std::cout << std::endl;

  // if (admm_iteration == 0) {
  //   if (U.array().isNaN().any()) drake::log()->error("NaN found in U");
  //   if (delta_c.array().isNaN().any()) drake::log()->error("NaN found in delta_c");
  //   if (E.array().isNaN().any()) drake::log()->error("NaN found in E");
  //   if (F.array().isNaN().any()) drake::log()->error("NaN found in F");
  //   if (H.array().isNaN().any()) drake::log()->error("NaN found in H");
  //   if (c.array().isNaN().any()) drake::log()->error("NaN found in c");    
  //   std::cout << "delta proj: " << delta_proj.transpose() << std::endl;
  // }

  return delta_proj;
}

}  // namespace c3
