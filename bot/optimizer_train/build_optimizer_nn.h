#include <../BuildOptimizerGenetic.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <vector>
#include "../utilities/mappings.h"

struct BuildOptimizerNN {
    py::scoped_interpreter guard;
    py::object;

    BuildOptimizerNN()
        : guard();

    std::vector<float> predictTimeToBuild(const BuildState& startState, std::vector < std::vector<std::pair<UNIT_TYPEID, int>> targets);
}