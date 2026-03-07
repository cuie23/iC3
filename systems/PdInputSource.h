#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/leaf_system.h"

using drake::systems::InputPort;
using drake::systems::OutputPort;
using drake::systems::InputPortIndex;
using drake::systems::OutputPortIndex;
using drake::systems::BasicVector;
using drake::systems::Context;

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace c3 {
namespace systems {

// For simulated rollouts
class PdInputSource : public drake::systems::LeafSystem<double> {
public:
  explicit PdInputSource(MatrixXd x_hat, MatrixXd u_hat, int q_idx, 
          int v_idx, double dt, int N, VectorXd Kp, VectorXd Kd);

  const InputPort<double>& get_input_port_state() const {
    return this->get_input_port(state_input_port_);
  }

  const OutputPort<double>& get_output_port_u() const {
    return this->get_output_port(u_output_port_);
  }


private:
  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* output) const;

  MatrixXd x_hat_;
  MatrixXd u_hat_;

  int q_idx_;
  int v_idx_;
  double dt_;
  int N_;

  VectorXd Kp_;
  VectorXd Kd_;

  InputPortIndex state_input_port_;
  OutputPortIndex u_output_port_;

};



} // namespace systems
} // namespace c3