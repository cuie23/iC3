#pragma once

using Eigen::VectorXd;

namespace c3 {

namespace systems {

struct MSiC3Options {

  int num_iters;
  int num_warmup_iters;

  int N;
  int num_segments;

  bool add_position_constraints;
  bool add_input_constraints;

  bool use_rollout_lambdas;

  bool early_termination;

  bool print_costs;

  int rollout_dt_scaling;
  bool use_drake_sim;
  double drake_sim_dt;

  bool penalize_acceleration;
  double acceleration_cost_weight;

  double defect_quaternion_weight;
  double defect_quaternion_regularizer_fraction;

  int num_threads;

  VectorXd rollout_Kp;
  VectorXd rollout_Kd;
  
  double alpha_ee;
  double alpha_object;

  double alpha_ee_step;
  double alpha_object_step;

  double warm_start_alpha;

  double value_function_scaling;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(num_iters));
    a->Visit(DRAKE_NVP(num_warmup_iters));
    a->Visit(DRAKE_NVP(N));
    a->Visit(DRAKE_NVP(num_segments));
    a->Visit(DRAKE_NVP(add_position_constraints));
    a->Visit(DRAKE_NVP(add_input_constraints));
    a->Visit(DRAKE_NVP(early_termination));
    a->Visit(DRAKE_NVP(print_costs));
    a->Visit(DRAKE_NVP(penalize_acceleration));
    a->Visit(DRAKE_NVP(acceleration_cost_weight));
    a->Visit(DRAKE_NVP(rollout_dt_scaling));
    a->Visit(DRAKE_NVP(use_drake_sim));
    a->Visit(DRAKE_NVP(drake_sim_dt));
    a->Visit(DRAKE_NVP(defect_quaternion_weight));
    a->Visit(DRAKE_NVP(defect_quaternion_regularizer_fraction));
    a->Visit(DRAKE_NVP(num_threads));
    a->Visit(DRAKE_NVP(rollout_Kp));
    a->Visit(DRAKE_NVP(rollout_Kd));
    a->Visit(DRAKE_NVP(use_rollout_lambdas));
    a->Visit(DRAKE_NVP(value_function_scaling));
    a->Visit(DRAKE_NVP(alpha_ee));
    a->Visit(DRAKE_NVP(alpha_object));
    a->Visit(DRAKE_NVP(alpha_ee_step));
    a->Visit(DRAKE_NVP(alpha_object_step));
    a->Visit(DRAKE_NVP(warm_start_alpha));

  }
};


} // Namespace systems
} // Namespace c3