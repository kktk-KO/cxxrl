#pragma once

namespace cxxrl {

template <class T>
struct environment_trait {

  using scalar_type = typename T::scalar_type;
  using state_type = typename T::state_type;
  using action_type = typename T::action_type;

  static scalar_type reward (state_type const & prev, state_type & next, action_type const & action) noexcept {
    return T::reward(prev, next, action);
  }

  static scalar_type reward (int prev, int & next, int action) noexcept {
    state_type sprev = state_type(prev);
    state_type snext = state_type(next);
    action_type a = action_type(action);
    auto r = reward(sprev, snext, a);
    next = snext.index();
    return r;
  }
};

template <class T>
using state_type = typename T::state_type;

template <class T>
using action_type = typename T::action_type;

template <class T>
using scalar_type = typename T::scalar_type;
}
