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
using state_type = typename T::state_type;

template <class T>
using action_type = typename T::action_type;

template <class T>
using scalar_type = typename T::scalar_type;
}
