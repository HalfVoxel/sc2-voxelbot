#pragma once
#include <libvoxelbot/combat/simulator.h>
#include "mcts_cache.h"
#include "sc2lib/sc2_lib.h"
#include <iostream>
#include<functional>
#include <vector>

struct SimulatorContext {
    CombatPredictor* combatPredictor;
    float simulationStartTime = 0;
    std::vector<sc2::Point2D> defaultPositions;
    std::array<std::vector<sc2::Point2D>, 2> extraDestinations;
    MCTSCache cache;
    bool debug = false;

    SimulatorContext(CombatPredictor* combatPredictor, std::vector<sc2::Point2D> defaultPositions, std::array<std::vector<sc2::Point2D>, 2> extraDestinations) : combatPredictor(combatPredictor), defaultPositions(defaultPositions), extraDestinations(extraDestinations) {
        assert(extraDestinations[0].size() == 3);
        assert(extraDestinations[1].size() == 3);
    }
};
