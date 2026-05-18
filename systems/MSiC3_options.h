#pragma once

using Eigen::VectorXd;

namespace c3 {

namespace systems {

struct MSiC3Options {

  int num_iters;

  int N;
  int num_segments;

  bool add_position_constraints;
  bool add_input_constraints;

  bool early_termination;

  bool print_costs;
  int rollout_dt_scaling;

  bool penalize_acceleration;
  double acceleration_cost_weight;

  double defect_quaternion_weight;
  double defect_quaternion_regularizer_fraction;

  double w_P;
  std::vector<double> p_vector; // cost weights for defect

  MatrixXd P;

  VectorXd rollout_Kp;
  VectorXd rollout_Kd;
  
  double alpha_ee;
  double alpha_object;

  double alpha_ee_step;
  double alpha_object_step;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(num_iters));
    a->Visit(DRAKE_NVP(N));
    a->Visit(DRAKE_NVP(num_segments));
    a->Visit(DRAKE_NVP(add_position_constraints));
    a->Visit(DRAKE_NVP(add_input_constraints));
    a->Visit(DRAKE_NVP(early_termination));
    a->Visit(DRAKE_NVP(print_costs));
    a->Visit(DRAKE_NVP(penalize_acceleration));
    a->Visit(DRAKE_NVP(acceleration_cost_weight));
    a->Visit(DRAKE_NVP(rollout_dt_scaling));
    a->Visit(DRAKE_NVP(defect_quaternion_weight));
    a->Visit(DRAKE_NVP(defect_quaternion_regularizer_fraction));
    a->Visit(DRAKE_NVP(w_P));
    a->Visit(DRAKE_NVP(p_vector));
    a->Visit(DRAKE_NVP(rollout_Kp));
    a->Visit(DRAKE_NVP(rollout_Kd));
    a->Visit(DRAKE_NVP(alpha_ee));
    a->Visit(DRAKE_NVP(alpha_object));
    a->Visit(DRAKE_NVP(alpha_ee_step));
    a->Visit(DRAKE_NVP(alpha_object_step));

    Eigen::VectorXd p_diag = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      this->p_vector.data(), this->p_vector.size());
      
    P = w_P * p_diag.asDiagonal();
  }
};


} // Namespace systems
} // Namespace c3