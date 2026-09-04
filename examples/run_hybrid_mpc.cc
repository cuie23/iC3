#include <algorithm>
#include <cmath>
#include <atomic>
#include <csignal>
#include <chrono>
#include <thread>
#include <iostream>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp> 
#include <Eigen/Dense>
#include <fstream>
#include <vector>

#include <drake/geometry/drake_visualizer.h>
#include <drake/geometry/meshcat_visualizer.h>
#include <drake/geometry/meshcat_visualizer_params.h>
#include <drake/multibody/meshcat/contact_visualizer.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/tree/multibody_element.h>
#include <drake/multibody/tree/rigid_body.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/primitives/constant_value_source.h>
#include <drake/systems/primitives/constant_vector_source.h>
#include <drake/systems/primitives/demultiplexer.h>
#include <drake/systems/primitives/multiplexer.h>
#include <drake/systems/primitives/zero_order_hold.h>
#include <drake/systems/lcm/lcm_publisher_system.h>
#include <drake/lcm/drake_lcm.h>
#include <drake/systems/primitives/vector_log_sink.h>
#include <drake/common/find_resource.h>
#include <drake/math/rotation_matrix.h>
#include <drake/math/roll_pitch_yaw.h>

#include <gflags/gflags.h>

#include "systems/common/quaternion_error_hessian.h"
#include "core/test/c3_cartpole_problem.hpp"
#include "examples/common_systems.hpp"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/iC3_options.h"
#include "systems/iC3.h"
#include "systems/MSiC3_options.h"
#include "systems/MSiC3.h"
#include "systems/MSiC3_parallel.h"

#include "systems/common/system_utils.hpp"
#include "systems/lcs_factory_system.h"
#include "systems/lcs_simulator.h"
#include "systems/manual_input.h"
#include "lcm/lcm_trajectory.h"
#include "tools/serialization_utils.h"

#include "core/lcs.h"

#include "c3/lcmt_timestamped_saved_traj.hpp"
#include "c3/lcmt_lqr_output.hpp"
#include "c3/lcmt_saved_traj.hpp"
#include "c3/lcmt_trajectory_block.hpp"

#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

#include <drake/systems/primitives/saturation.h>
#include <drake/systems/framework/leaf_system.h>

using c3::systems::C3Controller;
using c3::systems::C3ControllerOptions;
using c3::systems::iC3Options;
using c3::systems::MSiC3Options;
using c3::systems::LCSFactorySystem;
using c3::systems::LCSSimulator;
using c3::systems::ManualInput;

using c3::LcmTrajectory;

using drake::SortedPair;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using drake::geometry::GeometryId;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::ModelInstanceIndex;
using drake::systems::DiagramBuilder;
using drake::systems::rendering::MultibodyPositionToGeometryPose;
using drake::systems::lcm::LcmPublisherSystem;
using drake::systems::TriggerType;
using drake::systems::TriggerTypeSet;
using drake::math::RigidTransform;
using drake::math::RotationMatrix;
using drake::math::RollPitchYaw;

DEFINE_int32(ee_config, 1, "End effector configuration index, 1 = default");
DEFINE_int32(cube_model, 1, "Cube model index, 1 = default");
DEFINE_string(lcm_url, "udpm://239.255.76.67:7667?ttl=0",
              "LCM URL with IP, port, and TTL settings");
DEFINE_string(diagram_path, "",
              "Path to store the diagram (.ps) for the system. If empty, will "
              "be ignored");

class TrajToLcmSystem : public drake::systems::LeafSystem<double> {
 public:
  TrajToLcmSystem(std::vector<MatrixXd> traj_set)
      : traj_set_(traj_set) {
    this->DeclareAbstractOutputPort(
        "traj_message",
        &TrajToLcmSystem::CalcMessage);
  }

 private:
  void CalcMessage(
      const drake::systems::Context<double>& context,
      lcmt_timestamped_saved_traj* msg) const {

      LcmTrajectory lcm_traj;

      for (int i = 0; i < traj_set_.size(); i++) {
        c3::LcmTrajectory::Trajectory traj;
        std::string name = "iteration_" + std::to_string(i);

        MatrixXd traj_i = traj_set_.at(i);

        traj.traj_name = name;
        traj.datatypes = std::vector<std::string>(traj_i.rows(), "double");
        traj.datapoints = traj_i;


        VectorXd timestamps(traj_i.cols());
        for (int t = 0; t < traj_i.cols(); t++) {
          timestamps(t) = t;
        }
        traj.time_vector = timestamps.cast<double>();
        if (i == 0) {
          lcm_traj = LcmTrajectory({traj}, {name}, name, name, false);
        } else {
          lcm_traj.AddTrajectory(traj.traj_name, traj);
        }
      }
      msg->saved_traj = lcm_traj.GenerateLcmObject();
      msg->utime = context.get_time() * 1e6;   

  }
  

  std::vector<MatrixXd> traj_set_;
};

std::atomic<bool> g_run{true};
void SigIntHandler(int) { 
  g_run.store(false); 
}

int RunHybridMPCPlate(drake::lcm::DrakeLcm& lcm, MatrixXd z_hat) {}

int RunHybridMPCPointHand(drake::lcm::DrakeLcm& lcm, MatrixXd z_hat, int example) {
  // Load controller options and cost matrices.
  std::string ms_c3_options_file;
  std::string ms_ic3_options_file;
  std::string hybrid_mpc_options_file;

  if (example == 0) {
    ms_c3_options_file = "examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand.yaml";
    ms_ic3_options_file = "examples/resources/multifinger_hand/ms_ic3_options_point_hand.yaml";
    hybrid_mpc_options_file = "examples/resources/multifinger_hand/hybrid_mpc_options_pivot.yaml";
  } else if (example == 1) {
    ms_c3_options_file = "examples/resources/multifinger_hand/ms_c3_tracking_options_point_hand_180.yaml";
    ms_ic3_options_file = "examples/resources/multifinger_hand/ms_ic3_options_point_hand_180.yaml";
    hybrid_mpc_options_file = "examples/resources/multifinger_hand/hybrid_mpc_options_180.yaml";
  }

  std::string hand_config = "";
  if (FLAGS_ee_config >= 0) {
    hand_config = "_config_" + std::to_string(FLAGS_ee_config);
  }
  std::string cube_model = "";
  if (FLAGS_cube_model >= 0) {
    cube_model = "_" + std::to_string(FLAGS_cube_model);
  }

  C3ControllerOptions options = c3::systems::LoadC3ControllerOptions(ms_c3_options_file);
  MSiC3Options ms_ic3_options = drake::yaml::LoadYamlFile<MSiC3Options>(ms_ic3_options_file);
  HybridMpcOptions hybrid_mpc_options = drake::yaml::LoadYamlFile<HybridMpcOptions>(hybrid_mpc_options_file);


  // Build the plant and scene graph for the pivoting system.
  DiagramBuilder<double> plant_builder;
  auto [plant_for_lcs, scene_graph_for_lcs] =
      AddMultibodyPlantSceneGraph(&plant_builder, 0);
  Parser parser_for_lcs(&plant_for_lcs, &scene_graph_for_lcs);

  std::string hand_file_lcs;
	std::string cube_file_lcs;
  if (example == 0) {
    cube_file_lcs = "examples/resources/multifinger_hand/urdf/cube_for_lcs_heavy" + cube_model + ".sdf";
    hand_file_lcs = "examples/resources/multifinger_hand/urdf/simplified_hand_pivot" + hand_config + ".sdf";
  } else {
    if (ms_ic3_options.use_drake_sim == true) {
      cube_file_lcs = "examples/resources/multifinger_hand/urdf/cube_for_lcs" + cube_model + ".sdf";
    } else {
      cube_file_lcs = "examples/resources/multifinger_hand/urdf/cube_for_lcs_small_contacts.sdf";
    }
    hand_file_lcs = "examples/resources/multifinger_hand/urdf/simplified_hand_180" + hand_config + ".sdf";
  }

  // const std::string cube_file_lcs = "examples/resources/multifinger_hand/urdf/cylinder_for_lcs.sdf";
  const std::string ground_file_lcs = "examples/resources/multifinger_hand/ground.urdf";

  parser_for_lcs.AddModels(hand_file_lcs);
  parser_for_lcs.AddModels(cube_file_lcs);
  parser_for_lcs.AddModels(ground_file_lcs);

  RigidTransform<double> X_G_lcs = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0.0});

  RigidTransform<double> X_1_lcs = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0});
  RigidTransform<double> X_2_lcs = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0});
  RigidTransform<double> X_3_lcs = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0});

  plant_for_lcs.WeldFrames(plant_for_lcs.world_frame(),
                          plant_for_lcs.GetFrameByName("base_link_1"), X_1_lcs);
  plant_for_lcs.WeldFrames(plant_for_lcs.world_frame(),
                          plant_for_lcs.GetFrameByName("base_link_2"), X_2_lcs);
  plant_for_lcs.WeldFrames(plant_for_lcs.world_frame(),
                          plant_for_lcs.GetFrameByName("base_link_3"), X_3_lcs);
                                                  
  plant_for_lcs.WeldFrames(plant_for_lcs.world_frame(),
                          plant_for_lcs.GetFrameByName("ground"), X_G_lcs);

  plant_for_lcs.Finalize();

  // Build the plant diagram.
  auto plant_diagram = plant_builder.Build();


  // Build the plant and scene graph for the pivoting system.
  DiagramBuilder<double> plant_builder_rollout;
  auto [plant_rollout, scene_graph_rollout] =
      AddMultibodyPlantSceneGraph(&plant_builder_rollout, ms_ic3_options.drake_sim_dt);
  Parser parser_rollout(&plant_rollout, &scene_graph_rollout);

	std::string cube_file_rollout = "examples/resources/multifinger_hand/urdf/cube.sdf";
  if (ms_ic3_options.use_drake_sim == true) {
    cube_file_rollout = "examples/resources/multifinger_hand/urdf/cube.sdf";
  } else {
    cube_file_rollout = "examples/resources/multifinger_hand/urdf/cube_small_contacts.sdf";
  }
  // const std::string cube_file_rollout = "examples/resources/multifinger_hand/urdf/cylinder.sdf";
	const std::string ground_file_rollout = "examples/resources/multifinger_hand/ground.urdf";

  std::string hand_file_rollout;
  if (example == 0) {
    hand_file_rollout = "examples/resources/multifinger_hand/urdf/simplified_hand_pivot" + hand_config + ".sdf";
  } else {
    hand_file_rollout = "examples/resources/multifinger_hand/urdf/simplified_hand_180" + hand_config + ".sdf";
  }

  parser_rollout.AddModels(hand_file_rollout);
  parser_rollout.AddModels(cube_file_rollout);
  parser_rollout.AddModels(ground_file_rollout);

  RigidTransform<double> X_G_rollout = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0.0});

  RigidTransform<double> X_1_rollout = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0});
  RigidTransform<double> X_2_rollout = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0});
  RigidTransform<double> X_3_rollout = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0});

  plant_rollout.WeldFrames(plant_rollout.world_frame(),
                          plant_rollout.GetFrameByName("base_link_1"), X_1_rollout);
  plant_rollout.WeldFrames(plant_rollout.world_frame(),
                          plant_rollout.GetFrameByName("base_link_2"), X_2_rollout);
  plant_rollout.WeldFrames(plant_rollout.world_frame(),
                          plant_rollout.GetFrameByName("base_link_3"), X_3_rollout);                                                
  plant_rollout.WeldFrames(plant_rollout.world_frame(),
                          plant_rollout.GetFrameByName("ground"), X_G_rollout);

  plant_rollout.Finalize();

  // Build the plant diagram.
  auto plant_diagram_rollout = plant_builder_rollout.Build();


  // Retrieve collision geometries for relevant bodies.
  GeometryId ground_collision_geom = 
    plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("ground"))[0];

  std::vector<GeometryId> fingertip_collision_geoms;
  // Index, middle, ring, thumb
  fingertip_collision_geoms.push_back(
    plant_for_lcs.GetCollisionGeometriesForBody(
        plant_for_lcs.GetBodyByName("fingertip_1"))[0]);
  fingertip_collision_geoms.push_back(
    plant_for_lcs.GetCollisionGeometriesForBody(
        plant_for_lcs.GetBodyByName("fingertip_2"))[0]);
  fingertip_collision_geoms.push_back(
    plant_for_lcs.GetCollisionGeometriesForBody(
        plant_for_lcs.GetBodyByName("fingertip_3"))[0]);

	std::vector<GeometryId> cube_collision_geoms;
  for (int i = 0; i <= 8; i++) {
		cube_collision_geoms.push_back(
			plant_for_lcs.GetCollisionGeometriesForBody(
          plant_for_lcs.GetBodyByName("cube"))[i]);
	}

  // Define contact pairs for the LCS system.
  std::vector<SortedPair<GeometryId>> contact_pairs_finger;
  for (auto geom_id : fingertip_collision_geoms) {
    contact_pairs_finger.emplace_back(cube_collision_geoms[0], geom_id);
  }
  std::vector<SortedPair<GeometryId>> contact_pairs_box;
  int num_box_contacts = (example == 0) ? 8 : 4;
  for (int i = 1; i <= num_box_contacts; i++) {
    contact_pairs_box.emplace_back(cube_collision_geoms[i], ground_collision_geom);
  }
  std::vector<SortedPair<GeometryId>> contact_pairs;
  contact_pairs.insert(contact_pairs.end(), contact_pairs_finger.begin(), contact_pairs_finger.end());
  contact_pairs.insert(contact_pairs.end(), contact_pairs_box.begin(), contact_pairs_box.end());

  // Retrieve collision geometries for relevant bodies (ROLLOUT PLANT)
  GeometryId ground_collision_geom_rollout = 
    plant_rollout.GetCollisionGeometriesForBody(
          plant_rollout.GetBodyByName("ground"))[0];

  std::vector<GeometryId> fingertip_collision_geoms_rollout;
  // Index, middle, ring, thumb
  fingertip_collision_geoms_rollout.push_back(
    plant_rollout.GetCollisionGeometriesForBody(
        plant_rollout.GetBodyByName("fingertip_1"))[0]);
  fingertip_collision_geoms_rollout.push_back(
    plant_rollout.GetCollisionGeometriesForBody(
        plant_rollout.GetBodyByName("fingertip_2"))[0]);
  fingertip_collision_geoms_rollout.push_back(
    plant_rollout.GetCollisionGeometriesForBody(
        plant_rollout.GetBodyByName("fingertip_3"))[0]);

	std::vector<GeometryId> cube_collision_geoms_rollout;
  for (int i = 0; i <= 8; i++) {
		cube_collision_geoms_rollout.push_back(
			plant_rollout.GetCollisionGeometriesForBody(
          plant_rollout.GetBodyByName("cube"))[i]);
	}

  // Define contact pairs for the LCS system.
  std::vector<SortedPair<GeometryId>> contact_pairs_finger_rollout;
  for (auto geom_id : fingertip_collision_geoms_rollout) {
    contact_pairs_finger_rollout.emplace_back(cube_collision_geoms_rollout[0], geom_id);
  }
  std::vector<SortedPair<GeometryId>> contact_pairs_box_rollout;
  for (int i = 1; i <= num_box_contacts; i++) {
    contact_pairs_box_rollout.emplace_back(cube_collision_geoms_rollout[i], ground_collision_geom_rollout);
  }
  std::vector<SortedPair<GeometryId>> contact_pairs_rollout;
  contact_pairs_rollout.insert(contact_pairs_rollout.end(), contact_pairs_finger_rollout.begin(), contact_pairs_finger_rollout.end());
  contact_pairs_rollout.insert(contact_pairs_rollout.end(), contact_pairs_box_rollout.begin(), contact_pairs_box_rollout.end());

  // Build the main diagram.
  DiagramBuilder<double> builder;
  auto [plant, scene_graph] = AddMultibodyPlantSceneGraph(&builder, ms_ic3_options.drake_sim_dt);
  Parser parser(&plant, &scene_graph);

	std::string cube_file;
  if (ms_ic3_options.use_drake_sim == true) {
    cube_file = "examples/resources/multifinger_hand/urdf/cube.sdf";
  } else {
    cube_file = "examples/resources/multifinger_hand/urdf/cube_small_contacts.sdf";
  }
	const std::string ground_file = "examples/resources/multifinger_hand/ground.urdf";

  std::string hand_file;
  if (example == 0) {
    hand_file = "examples/resources/multifinger_hand/urdf/simplified_hand_pivot" + hand_config + ".sdf";
  } else {
    hand_file = "examples/resources/multifinger_hand/urdf/simplified_hand_180" + hand_config + ".sdf";
  }

  parser.AddModels(hand_file);
  parser.AddModels(cube_file);
  parser.AddModels(ground_file);

  RigidTransform<double> X_G = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0, 0, 0.0});

  RigidTransform<double> X_1 = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {0.0, 0, 0.0});
  RigidTransform<double> X_2 = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {-0.0, -0.0, 0.0});
  RigidTransform<double> X_3 = RigidTransform<double>(
    drake::math::RotationMatrix<double>(), {-0.0, 0.0, 0.0});

  plant.WeldFrames(plant.world_frame(),
                   plant.GetFrameByName("base_link_1"), X_1);
  plant.WeldFrames(plant.world_frame(),
                   plant.GetFrameByName("base_link_2"), X_2);
  plant.WeldFrames(plant.world_frame(),
                   plant.GetFrameByName("base_link_3"), X_3);
  // plant.WeldFrames(plant.world_frame(),
  //                  plant.GetFrameByName("base_link_4"), X_4);                                                  
  plant.WeldFrames(plant.world_frame(),
                   plant.GetFrameByName("ground"), X_G);

  plant.Finalize();

  // Create contexts for the plant and LCS factory system.
  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      plant_diagram->CreateDefaultContext();
  auto plant_lcs_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_for_lcs);
  auto& plant_for_lcs_context = plant_diagram->GetMutableSubsystemContext(
      plant_for_lcs, plant_diagram_context.get());
  auto plant_lcs_context_autodiff = plant_lcs_autodiff->CreateDefaultContext(); 

  std::unique_ptr<drake::systems::Context<double>> plant_diagram_rollout_context =
      plant_diagram_rollout->CreateDefaultContext();
  auto plant_rollout_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(plant_rollout);
  auto& plant_rollout_context = plant_diagram_rollout->GetMutableSubsystemContext(
      plant_rollout, plant_diagram_rollout_context.get());
  auto plant_rollout_context_autodiff = plant_rollout_autodiff->CreateDefaultContext(); 

  int example_idx = (example == 0) ? 2 : 1;

  if (options.resolve_contacts_to_lists.has_value() ||
      options.c3_options.resolve_contacts_to_lists.has_value()) {
    const auto& res_lists = options.resolve_contacts_to_lists.has_value()
                                ? options.resolve_contacts_to_lists.value()
                                : options.c3_options.resolve_contacts_to_lists.value();
    int idx = options.num_contacts_index.value_or(
        options.c3_options.num_contacts_index.value_or(0));
    const auto& res_list = res_lists[idx];
    std::vector<std::vector<SortedPair<GeometryId>>> contact_groups = {
        contact_pairs_finger, contact_pairs_box};
    std::vector<std::vector<SortedPair<GeometryId>>> contact_groups_rollout = {
        contact_pairs_finger_rollout, contact_pairs_box_rollout};
    contact_pairs = multibody::LCSFactory::ResolveContactPairs(
        plant_for_lcs, plant_for_lcs_context, contact_groups, res_list);
    contact_pairs_rollout = multibody::LCSFactory::ResolveContactPairs(
        plant_rollout, plant_rollout_context, contact_groups_rollout, res_list);
    options.lcs_factory_options.num_contacts = contact_pairs.size();
    if (options.mu_per_pair_type.has_value() || options.c3_options.mu_per_pair_type.has_value()) {
      const auto& mu_types = options.mu_per_pair_type.has_value()
                                 ? options.mu_per_pair_type.value()
                                 : options.c3_options.mu_per_pair_type.value();
      std::vector<double> new_mu;
      for (size_t g = 0; g < contact_groups.size(); ++g) {
        int n_active = (g < res_list.size())
                           ? std::min(res_list[g], static_cast<int>(contact_groups[g].size()))
                           : contact_groups[g].size();
        if (g < mu_types.size()) {
          new_mu.insert(new_mu.end(), n_active, mu_types[g]);
        }
      }
      options.lcs_factory_options.mu = new_mu;
    } else {
      options.lcs_factory_options.mu.resize(contact_pairs.size(), options.lcs_factory_options.mu[0]);
    }
  }

  std::unique_ptr<systems::MSiC3> ms_ic3_controller =
     std::make_unique<systems::MSiC3>(plant_for_lcs, *plant_lcs_autodiff, plant_rollout, *plant_rollout_autodiff, 
        *plant_diagram_rollout, std::move(plant_diagram_rollout_context), contact_pairs, contact_pairs_rollout, 
        options, ms_ic3_options, example_idx);
  
  int n_x = plant_for_lcs.num_positions() + plant_for_lcs.num_velocities();
  int n_u = plant_for_lcs.num_actuators();
  int n_lambda = (example == 0) ? 44 : 28;

  std::cout << "z hat " << z_hat.rows() << " x " << z_hat.cols() << std::endl;

  MatrixXd x_hat(n_x, z_hat.cols()+1);
  x_hat.leftCols(z_hat.cols()) = z_hat.middleRows(0, n_x);
  x_hat.col(x_hat.cols()-1) = x_hat.col(x_hat.cols()-2);



  MatrixXd lambda_hat = z_hat.middleRows(n_x, n_lambda);
  MatrixXd u_hat = z_hat.middleRows(n_x + n_lambda, n_u);


  Eigen::VectorXd x0(31);
  std::vector<double> x_init = *options.x_init;
  x0 = Eigen::Map<Eigen::VectorXd>(x_init.data(), x_init.size());

  vector<VectorXd> x0s;
  x0s.push_back(x0);
  vector<double> yaws = {0};
  
  auto [x_traj, u_traj, lambda_traj] = 
      ms_ic3_controller->DoHybridMPCTracking(x0s, x_hat, u_hat, lambda_hat, hybrid_mpc_options, 
              plant_for_lcs_context, *plant_lcs_context_autodiff, plant_rollout_context);

  std::cout << "x_hat final " << x_hat.col(x_hat.cols()-1).segment(0, plant_for_lcs.num_positions()).transpose() << std::endl;
  std::cout << "x final " << x_traj[0].col(x_traj[0].cols()-1).segment(0, plant_for_lcs.num_positions()).transpose() << std::endl;

  // Publishes input std::vector<MatrixXd> as a lcmt_timestamped_saved_traj
  auto traj_source_x = builder.AddSystem<TrajToLcmSystem>(x_traj);
  traj_source_x->set_name("traj_source_x");
  auto traj_source_u = builder.AddSystem<TrajToLcmSystem>(u_traj);
  traj_source_u->set_name("traj_source_u");

  auto traj_publisher_x = builder.AddSystem(
      LcmPublisherSystem::Make<c3::lcmt_timestamped_saved_traj>(
          "iC3_TRAJECTORY_X", &lcm,
          TriggerTypeSet({TriggerType::kForced})));

  auto traj_publisher_u = builder.AddSystem(
      LcmPublisherSystem::Make<c3::lcmt_timestamped_saved_traj>(
          "iC3_TRAJECTORY_U", &lcm,
          TriggerTypeSet({TriggerType::kForced})));

  builder.Connect(traj_source_x->get_output_port(),
                    traj_publisher_x->get_input_port());
  builder.Connect(traj_source_u->get_output_port(),
                    traj_publisher_u->get_input_port());

  // Build the diagram.
  auto diagram = builder.Build();

  if (!FLAGS_diagram_path.empty())
    c3::systems::common::DrawAndSaveDiagramGraph(*diagram, FLAGS_diagram_path);

  // Create a default context for the diagram.
  auto diagram_context = diagram->CreateDefaultContext();

  // Set the initial state of the system.
	auto& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());
  plant.SetPositionsAndVelocities(&plant_context, x0);

  std::signal(SIGINT, SigIntHandler);
  auto output = diagram->AllocateOutput();

  std::cout << "Publishing to lcm " << std::endl;

  const std::chrono::milliseconds period(50);
  while (g_run.load()) {
    diagram->CalcOutput(*diagram_context, output.get()); 
    diagram->ForcedPublish(*diagram_context);
    std::this_thread::sleep_for(period);
  }

  return 0;

}





int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::lcm::DrakeLcm lcm(FLAGS_lcm_url);

  //  ./bazel-bin/examples/run_hybrid_mpc

  std::string rollout_z_sol_file = "examples/resources/multifinger_hand/ic3_debug_data/rollout_z_sol.bin";
  vector<MatrixXd> z_sol_rollout =
      c3::utils::LoadTrajectoryData<vector<MatrixXd>>(
          rollout_z_sol_file);

  int iC3_iter = z_sol_rollout.size()-1;
  MatrixXd final_rollout = z_sol_rollout[iC3_iter];

  std::cout << "iC3 iter " << iC3_iter << std::endl;

  // TODO: hardcoded sizes
  if (final_rollout.rows() == 23 + 32 + 5) {
    std::cout << "plate" << std::endl;
    return RunHybridMPCPlate(lcm, final_rollout);
  } else if (final_rollout.rows() == 31 + 28 + 9) {
    std::cout << "180 yaw" << std::endl;
    return RunHybridMPCPointHand(lcm, final_rollout, 1);
  } else if (final_rollout.rows() == 31 + 44 + 9) {
    std::cout << "pivot" << std::endl;
    return RunHybridMPCPointHand(lcm, final_rollout, 0);
  } else {
    std::cerr
        << "Unknown experiment type: "
        << std::endl;
    return -1;
  }
}