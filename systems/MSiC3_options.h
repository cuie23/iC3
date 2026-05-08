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

  double defect_quaternion_weight;
  double defect_quaternion_regularizer_fraction;
  std::vector<double> p_vector; // cost weights for defect

  MatrixXd P;

  VectorXd rollout_Kp;
  VectorXd rollout_Kd;
  
  double ff_alpha;


  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(num_iters));
    a->Visit(DRAKE_NVP(N));
    a->Visit(DRAKE_NVP(num_segments));
    a->Visit(DRAKE_NVP(add_position_constraints));
    a->Visit(DRAKE_NVP(add_input_constraints));
    a->Visit(DRAKE_NVP(print_costs));
    a->Visit(DRAKE_NVP(rollout_dt_scaling));
    a->Visit(DRAKE_NVP(defect_quaternion_weight));
    a->Visit(DRAKE_NVP(defect_quaternion_regularizer_fraction));
    a->Visit(DRAKE_NVP(p_vector));
    a->Visit(DRAKE_NVP(rollout_Kp));
    a->Visit(DRAKE_NVP(rollout_Kd));
    a->Visit(DRAKE_NVP(ff_alpha));

    Eigen::VectorXd P = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      this->p_vector.data(), this->p_vector.size());
  }
};


} // Namespace systems
} // Namespace c3