#include "build_optimizer_nn.h"
#include <iostream>

using namespace std;
using namespace sc2;

void BuildOptimizerNN::init() {
    // return;
    // pybind11::exec(R"(
    //     import sys
    //     sys.path.append("bot/optimizer_train")
    // )");
    // pybind11::module trainer = pybind11::module::import("optimizer_train");
    // predictFunction = trainer.attr("predict");
    cout << "Finished initializing NN" << endl;
}

vector<float> BuildOptimizerNN::predictTimeToBuild(const vector<pair<int, int>>& startingState, const BuildResources& startingResources, const vector < vector<pair<int, int>>>& targets) const {
    // return vector<float>(targets.size());
    // auto res = predictFunction(startingState, tuple<int,int>(startingResources.minerals, startingResources.vespene), targets);
    // return res.cast<std::vector<float>>();
    return vector<float>();
}
