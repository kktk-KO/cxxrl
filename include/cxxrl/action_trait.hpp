#pragma once

namespace cxxrl {

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

}
