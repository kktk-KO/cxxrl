#pragma once

namespace cxxrl {

template <class T>
struct state_trait {

  static int index (T const & t) noexcept {
    return t.index();
  }

  static void index (T & t, int value) noexcept {
    t.index(value);
  }

  static bool is_terminal (T const & t) noexcept {
    return t.is_terminal();
  }

  static constexpr int max_index () noexcept {
    return T::max_index();
  }
};

template <class T>
struct action_trait {

  static int index (T const & t) noexcept {
    return t.index();
  }

  static void index (T & t, int value) noexcept {
    t.index(value);
  }

  static constexpr int max_index () noexcept {
    return T::max_index();
  }
};

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
