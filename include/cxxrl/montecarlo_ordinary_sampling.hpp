#pragma once

#include <cxxrl/trait.hpp>

#include <array>
#include <cassert>
#include <random>
#include <limits>

namespace cxxrl {

template <class Environment>
struct montecarlo_ordinary_sampling {

  using environment_type = Environment;
  using state_type = typename Environment::state_type;
  using action_type = typename Environment::action_type;
  using scalar_type = typename Environment::scalar_type;

  montecarlo_ordinary_sampling () {
    random_seed();
    pi_.resize(state_type::max_index());
    Q_.resize(state_type::max_index() * action_type::max_index(), 0);
    C_.resize(state_type::max_index() * action_type::max_index(), 0);
  }

  scalar_type epsilon () const noexcept {
    return epsilon_;
  }

  void epsilon (scalar_type value) noexcept {
    assert(0 <= value && (value * action_type::max_index()) <= 1);
    epsilon_ = value;
  }

  scalar_type gamma () const noexcept {
    return gamma_;
  }

  void gamma (scalar_type value) noexcept {
    gamma_ = value;
  }

  void random_seed () noexcept {
    std::random_device device;
    random_generator_.seed(device());
  }

  void random_seed (int seed) noexcept {
    random_generator_.seed(seed);
  }

  void randomize_policy () {
    for (int i = 0; i < state_type::max_index(); ++i) { 
      pi_[i] = std::uniform_int_distribution<int>(0, action_type::max_index() - 1)(random_generator_);
    }
  }

  action_type policy (state_type const & s) const noexcept {
    return action_type(pi_[s.index()]);
  }

  scalar_type state_value (state_type const & s) const noexcept {
    return action_value(s, action_type(pi_[s.index()]));
  }

  scalar_type & action_value (state_type const & s, action_type const & a) noexcept {
    return action_value(s.index(), a.index());
  }

  scalar_type & action_value (int state, int action) noexcept {
    return Q_[action + action_type::max_index() * state];
  }

  scalar_type action_value (state_type const & s, action_type const & a) const noexcept {
    return Q_[a.index() + action_type::max_index() * s.index()];
  }

  scalar_type action_value (int state, int action) const noexcept {
    return Q_[action + action_type::max_index() * state];
  }

  void loop_once (state_type const & state_init, state_type & state_final) {

    S_.clear();
    A_.clear();
    R_.clear();

    S_.push_back(state_init.index());
    A_.push_back(policy_(S_[0]));

    while (!state_type(S_.back()).is_terminal()) {
      state_type s;
      R_.push_back(environment_trait<environment_type>::reward(S_.back(), s, A_.back()));
      S_.push_back(s.index());
      A_.push_back(policy_(S_.back()));
    }

    R_.push_back(environment_type::reward(S_.back(), state_final, A_.back()));

    scalar_type G = 0;
    scalar_type W = 1;
    int N = S_.size();

    for (int i = N - 1; i >= 0; --i) {
      G = R_[i] + gamma() * G;
      int j = A_[i] + action_type::max_index() * S_[i];
      C_[j] += 1;
      Q_[j] += (W / C_[j]) * (G - Q_[j]);

      int a = argmax_policy_(S_[i]);
      if (A_[i] != a) {
        pi_[S_[i]] = a;
        break;
      }

      if (pi_[S_[i]] == a) { 
        W /= (1 - epsilon() + epsilon() / action_type::max_index());
      } else {
        W /= (epsilon() / action_type::max_index());
      }

      pi_[S_[i]] = a;
    }

  }

  void loop (state_type const & state_init, int count) {
    assert(count > 0);

    randomize_policy();

    state_type state_final;
    for (int i = 0; i < count; ++i) {
      loop_once(state_init, state_final);
    }
  }

private:
  scalar_type gamma_ = 1.0;
  scalar_type epsilon_ = 0.01;
  std::mt19937 random_generator_;

  std::vector<int> S_;
  std::vector<int> A_;
  std::vector<int> pi_;
  std::vector<scalar_type> R_;
  std::vector<scalar_type> Q_;
  std::vector<int> C_;

  int policy_ (int state_index) noexcept {
    assert(0 <= state_index && state_index < state_type::max_index());

    scalar_type p = std::uniform_real_distribution<scalar_type>(0, 1)(random_generator_);
    scalar_type q = epsilon();
    scalar_type e = epsilon() / action_type::max_index();

    for (int i = 0; i < action_type::max_index(); ++i, q += e) {
      if (p < q) { return i; }
    }
    return pi_[state_index];
  }

  int argmax_policy_ (int state) {
    int argmax = 0;
    scalar_type max = -std::numeric_limits<scalar_type>::max();
    for (int j = 0; j < action_type::max_index(); ++j) {
      if (action_value(state, j) > max) {
        max = action_value(state, j);
        argmax = j;
      }
    }
    return argmax;
  }


};

}
