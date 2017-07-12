#include "scorer.h"

namespace amunmt {

Scorer::Scorer(const God &god,
              const std::string& name,
              const YAML::Node& config, size_t tab)
: name_(name), config_(config), tab_(tab)
{
}

}
