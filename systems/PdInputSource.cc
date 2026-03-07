#include <iostream>
#include "systems/PdInputSource.h"

namespace c3 {
namespace systems {

PdInputSource::PdInputSource(MatrixXd x_hat, MatrixXd u_hat, int q_idx, int v_idx, 
                            double dt, int N, VectorXd Kp, VectorXd Kd): 
  x_hat_(x_hat),
  u_hat_(u_hat),
  q_idx_(q_idx),
  v_idx_(v_idx),
  dt_(dt),
  N_(N),
  Kp_(Kp),
  Kd_(Kd) {
  
  state_input_port_ = this->DeclareVectorInputPort("x_curr", x_hat_.rows()).get_index();
  u_output_port_ = this->DeclareVectorOutputPort("u_out", u_hat_.rows(),
                                &PdInputSource::CalcOutput).get_index();
}

void PdInputSource::CalcOutput(const Context<double>& context,
                  BasicVector<double>* output) const {
  double t = context.get_time();
  
  const BasicVector<double>* x_curr_in =
    (BasicVector<double>*)this->EvalVectorInput(context, state_input_port_);
  VectorXd x_curr = x_curr_in->get_value();

  VectorXd u_out(VectorXd::Zero(u_hat_.rows()));
  if (t < dt_ * N_) {
    int segment = (int)(t / dt_);
    VectorXd x_nominal = x_hat_.col(segment);
    VectorXd u_nominal = u_hat_.col(segment);

    u_out = u_nominal + Kp_.asDiagonal() * (x_nominal.segment(q_idx_, Kp_.size()) - x_curr.segment(q_idx_, Kp_.size())) 
            + Kd_.asDiagonal() * (x_nominal.segment(v_idx_, Kd_.size()) - x_curr.segment(v_idx_, Kd_.size()));

  } 
  output->get_mutable_value() = u_out;
  
}





} // namespace systems
} // namespace c3