#pragma once
#include <algorithm>
#include <vector>
#include "mcts_sc2.h"
#include "game_state_loader.h"
#include "simulator_context.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


struct EnvObserver {
    const AvailableUnitTypes& unitMappings;
    CoordinateRemapper coordinates;

    EnvObserver(const AvailableUnitTypes& unitMappings, CoordinateRemapper coordinates) : unitMappings(unitMappings), coordinates(coordinates) {}
    std::vector<pybind11::array_t<float>> observe(const SimulatorMCTSState& state, int playerID) const;
};


struct RLPlanningEnv {
    int trainingPlayer = 1;
    SimulatorMCTSState state;
    std::shared_ptr<SimulatorContext> simulator;
    const EnvObserver& observer;

    RLPlanningEnv(int trainingPlayer, SimulatorMCTSState state, const EnvObserver& observer) : trainingPlayer(trainingPlayer), state(state), simulator(state.state.simulator), observer(observer) {}

    std::pair<float, bool> step(int action);
    std::vector<pybind11::array_t<float>> observe();
    void print();
    std::string actionName(int action);
    std::pair<std::vector<std::vector<double>>, float> visualizationInfo();
};

struct RLEnvManager {
    EnvObserver observer;
    CombatPredictor combatPredictor;
    std::vector<SimulatorState> statePool;

    RLEnvManager(pybind11::object simulatorVisualizerModule, pybind11::object replayLoadFn, std::vector<std::string> binaryReplayFilePaths);
    RLPlanningEnv getEnv();
};
