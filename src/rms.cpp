#include <cxxrl/montecarlo_onpolicy.hpp>
#include <cxxrl/montecarlo_ordinary_sampling.hpp>
#include <cxxrl/random_walk.hpp>

#include <cmath>
#include <iostream>

template <class Agent>
void proc (Agent & agent, int count, int sample) {
  using namespace cxxrl;

  agent.epsilon(0.3);
  agent.gamma(1);

  state_type<Agent> state_init;
  std::vector<double> v(sample);
  double mean;
  double rms;

  for (int i = 0; i < sample; ++i) {
    state_init.site(0);
    agent.loop(state_init, count);
    v[i] = agent.state_value(state_init);
  }

  mean = 0;
  for (int i = 0; i < sample; ++i) {
    mean += v[i];
  }
  mean /= sample;

  rms = 0;
  for (int i = 0; i < sample; ++i) {
    rms += (v[i] - mean) * (v[i] - mean);
  }
  rms = std::sqrt(rms / sample);
  std::cout << count << " " << rms << std::endl;
}

template <class Agent>
void proc2 (Agent && agent, int count, int sample) {
  for (int i = 0; i < count; ++i) {
    int min = 1 << i;
    int max = 1 << (i + 1);
    int d = std::max(1, (max - min) / 100);
    for (int j = min; j < max; j += d) {
      proc(agent,  j, sample);
    }
  }
}

int main (int argc, char ** argv) {
  using namespace cxxrl;

  using environment_type = random_walk<9>;

  std::string agent_name(argv[1]);
  int count = std::atoi(argv[2]);
  int sample= std::atoi(argv[3]);

  int n = 1;
  if (agent_name == "onpolicy") {
    proc2(montecarlo_onpolicy<environment_type>(), count, sample);
  } else if (agent_name == "ordinary-sampling") {
    proc2(montecarlo_ordinary_sampling<environment_type>(), count, sample);
  } else {
    std::cout << "unknown agent name." << std::endl;
  }
}
