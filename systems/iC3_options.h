#pragma once

using Eigen::VectorXd;

namespace c3 {

namespace systems {

struct iC3Options {

  int num_iters;
  bool add_position_constraints;
  bool add_input_constraints;
  int num_segments;

  // does lcs rollout after this many segments
  int segment_rollout_frequency;

  int N_penalize_input_change;

  bool early_termination;

  bool print_costs;

  int rollout_dt_scaling;

  VectorXd rollout_Kp;
  VectorXd rollout_Kd;

  double ff_alpha;

  double position_l2_decoupling_weight;
  double velocity_l2_decoupling_weight;
  double input_l2_decoupling_weight;
  std::vector<std::vector<int>> l2_decoupling_indices;

  double position_l1_weight;
  double velocity_l1_weight;
  double input_l1_weight;  

  double position_huber_weight;
  double velocity_huber_weight;
  double input_huber_weight;  
  double huber_delta;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(num_iters));
    a->Visit(DRAKE_NVP(N_penalize_input_change));
    a->Visit(DRAKE_NVP(add_position_constraints));
    a->Visit(DRAKE_NVP(add_input_constraints));
    a->Visit(DRAKE_NVP(early_termination));
    a->Visit(DRAKE_NVP(num_segments));
    a->Visit(DRAKE_NVP(segment_rollout_frequency));
    a->Visit(DRAKE_NVP(print_costs));
    a->Visit(DRAKE_NVP(rollout_dt_scaling));
    a->Visit(DRAKE_NVP(rollout_Kp));
    a->Visit(DRAKE_NVP(rollout_Kd));
    a->Visit(DRAKE_NVP(ff_alpha));
    a->Visit(DRAKE_NVP(position_l2_decoupling_weight));
    a->Visit(DRAKE_NVP(velocity_l2_decoupling_weight));
    a->Visit(DRAKE_NVP(input_l2_decoupling_weight));
    a->Visit(DRAKE_NVP(l2_decoupling_indices));
    a->Visit(DRAKE_NVP(position_l1_weight));
    a->Visit(DRAKE_NVP(velocity_l1_weight));
    a->Visit(DRAKE_NVP(input_l1_weight));
    a->Visit(DRAKE_NVP(position_huber_weight));
    a->Visit(DRAKE_NVP(velocity_huber_weight));
    a->Visit(DRAKE_NVP(input_huber_weight));
    a->Visit(DRAKE_NVP(huber_delta));

  }
};


} // Namespace systems
} // Namespace c3