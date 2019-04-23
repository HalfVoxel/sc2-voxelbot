#pragma once
#include "../ml/simulator.h"
#include <pybind11/pybind11.h>
#include "../mcts/mcts.h"
#include "../ml/mcts_sc2.h"
#include "../ml/simulator_context.h"

struct MCTSDebugger {
    pybind11::object visualize_fn;
    pybind11::object visualize_bar_fn;
    pybind11::object reset_fn;

    MCTSDebugger();
    MCTSDebugger(pybind11::object simulatorVisualizerModule);
    void visualize(SimulatorState& state);
    void debugInteractive(std::shared_ptr<MCTSState<int, SimulatorMCTSState>> state);
};
