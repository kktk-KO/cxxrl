#pragma once

namespace cxxrl {

template <class T>
struct env_trait {
  using state = typename T::state;
  using action = typename T::action;
};

}
