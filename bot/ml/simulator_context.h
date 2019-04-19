#pragma once
#include "../CombatPredictor.h"
#include "mcts_cache.h"
#include "sc2lib/sc2_lib.h"

#include<functional>
#include <vector>

struct SimulatorContext {
    CombatPredictor* combatPredictor;
    float simulationStartTime = 0;
    std::vector<sc2::Point2D> defaultPositions;
    MCTSCache cache;

    SimulatorContext(CombatPredictor* combatPredictor, std::vector<sc2::Point2D> defaultPositions) : combatPredictor(combatPredictor), defaultPositions(defaultPositions) {}
};
