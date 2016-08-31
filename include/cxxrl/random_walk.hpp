#pragma once

#include <cstddef>
#include <cassert>

namespace cxxrl {

template <int N>
struct random_walk_state {
  static_assert(N > 2, "the number of sites should be greater than 2.");
  static_assert(N % 2 == 1, "the number of sites should be odd.");

  random_walk_state (int index_value = 0) noexcept {
    index(index_value);
  }

  random_walk_state & operator= (random_walk_state const & other) noexcept {
    site_ = other.site_;
    return *this;
  }

  int site () const noexcept {
    return site_;
  }

  void site (int value) noexcept {
    assert((-(N - 1) / 2) <= value && value <= ((N - 1) / 2));
    site_ = value;
  }

  bool is_left_end () const noexcept {
    return site_ == -((N - 1) / 2);
  }

  bool is_right_end () const noexcept {
    return site_ == ((N - 1) / 2);
  }

  bool is_terminal () const noexcept {
    return is_left_end() || is_right_end();
  }

  int index () const noexcept {
    return site_ + ((N - 1) / 2);
  }

  void index (int value) noexcept {
    assert(0 <= value && value < N);
    site_ = value - ((N - 1) / 2);
  }

  static constexpr int max_index () noexcept {
    return N;
  }

private:
  int site_ = 0;
};

template <int N>
struct random_walk_action {
  static_assert(N > 2, "the number of sites should be greater than 2.");
  static_assert(N % 2 == 1, "the number of sites should be odd.");

  enum Action {
    ActionLeft = 0,
    ActionRight = 1
  };

  random_walk_action (int index_value = 0) noexcept {
    index(index_value);
  }

  random_walk_action & operator= (random_walk_action const & other) noexcept {
    action_ = other.action_;
    return *this;
  }

  bool left () const noexcept {
    return action_ == ActionLeft;
  }

  bool right () const noexcept {
    return action_ == ActionRight;
  }

  int index () const noexcept {
    return action_;
  }

  void index (int value) noexcept {
    assert(0 <= value && value < 2);
    action_ = value;
  }

  static constexpr int max_index () noexcept {
    return 2;
  }

private:
  int action_;
};

template <int N, class ScalarType = double>
struct random_walk {
  static_assert(N > 2, "the number of sites should be greater than 2.");
  static_assert(N % 2 == 1, "the number of sites should be odd.");

  using state_type = random_walk_state<N>;
  using action_type = random_walk_action<N>;
  using scalar_type = ScalarType;

  static scalar_type reward (state_type const & prev, state_type & next, action_type const & action) {
    if (prev.is_left_end() || prev.is_right_end()) {
      next = prev;
      return 0;
    }

    if (action.left()) {
      next.site(prev.site() - 1);
    } else if (action.right()) {
      next.site(prev.site() + 1);
      if (next.is_right_end()) {
        return 1;
      }
    }
    return 0;
  }

};

}
