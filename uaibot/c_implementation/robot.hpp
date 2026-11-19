#pragma once
// Optional owning facade. Physics, state types and pure functions remain usable
// directly from each model header. No type erasure or shared mutable state.
#include <utility>

#include "bluerov2_physics.hpp"
#include "manipulator_physics.hpp"
#include "uav_physics.hpp"

namespace dynamics {
struct AUVModel {
  using Parameters = bluerov2::Parameters;
  using State = bluerov2::State;
  static State initial(const Parameters&) { return {}; }
  static State velocity(const Eigen::VectorXd& u, const State& s,
                        const Parameters& p) {
    return bluerov2::step(u, s, p);
  }
  static State actuators(const Eigen::VectorXd& u, const State& s,
                         const Parameters& p) {
    return bluerov2::step_open_loop(u, s, p);
  }
};
struct UAVModel {
  using Parameters = uav::Parameters;
  using State = uav::State;
  static State initial(const Parameters&) { return {}; }
  static State velocity(const Eigen::VectorXd& u, const State& s,
                        const Parameters& p) {
    return uav::step(u, s, p);
  }
  static State actuators(const Eigen::VectorXd& u, const State& s,
                         const Parameters& p) {
    return uav::step_open_loop(u, s, p);
  }
};
struct ManipulatorModel {
  using Parameters = manipulator::Parameters;
  using State = manipulator::State;
  static State initial(const Parameters& p) {
    return manipulator::make_state(p);
  }
  static State velocity(const Eigen::VectorXd& u, const State& s,
                        const Parameters& p) {
    return manipulator::step(u, s, p);
  }
  static State actuators(const Eigen::VectorXd& u, const State& s,
                         const Parameters& p) {
    return manipulator::step_open_loop(u, s, p);
  }
};

template <class Model>
class Robot {
 public:
  using Parameters = typename Model::Parameters;
  using State = typename Model::State;
  Parameters parameters;
  State state;
  explicit Robot(Parameters p)
      : parameters(std::move(p)), state(Model::initial(parameters)) {}
  Robot(Parameters p, State s)
      : parameters(std::move(p)), state(std::move(s)) {}
  const State& step(const Eigen::VectorXd& target_velocity) {
    state = Model::velocity(target_velocity, state, parameters);
    return state;
  }
  const State& step_open_loop(const Eigen::VectorXd& actuator_setpoints) {
    state = Model::actuators(actuator_setpoints, state, parameters);
    return state;
  }
};
using AUV = Robot<AUVModel>;
using UAV = Robot<UAVModel>;
using Manipulator = Robot<ManipulatorModel>;
}  // namespace dynamics
