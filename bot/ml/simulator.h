#pragma once
#include "../BuildOptimizerGenetic.h"
#include "../CombatPredictor.h"
#include <vector>
#include <functional>

extern int simulatorUnitIndexCounter;

struct SimulatorUnit {
    CombatUnit combat;
    sc2::Tag tag = sc2::NullTag;

    SimulatorUnit (CombatUnit combat) : combat(combat) {
        tag = simulatorUnitIndexCounter++;
    }
};

enum SimulatorOrderType {
    None,
    Attack,
};

struct SimulatorOrder {
    SimulatorOrderType type = SimulatorOrderType::None;
    sc2::Point2D target;

    SimulatorOrder() = default;
    SimulatorOrder(SimulatorOrderType type, sc2::Point2D target) : type(type), target(target) {}
};

struct SimulatorUnitGroup {
    int owner = 0;
    std::vector<SimulatorUnit> units;
    sc2::Point2D pos;
    sc2::Point2D previousPos;
    SimulatorOrder order;

    SimulatorUnitGroup () = default;

    SimulatorUnitGroup (sc2::Point2D pos, std::vector<SimulatorUnit> units) : units(units), pos(pos) {
        owner = units[0].combat.owner;
    }

    inline void execute(SimulatorOrder order) {
        this->order = order;
    }

    sc2::Point2D futurePosition(float deltaTime);
};

struct Simulator {
    CombatPredictor* combatPredictor;
    float simulationStartTime = 0;

    Simulator(CombatPredictor* combatPredictor) : combatPredictor(combatPredictor) {}
};

struct SimulatorState {
    Simulator& simulator;
    std::vector<BuildState> states;
    std::vector<SimulatorUnitGroup> groups;
    std::vector<BuildOrderState> buildOrders;
    
    float time() {
        return states[0].time;
    }

    SimulatorState (Simulator& simulator, std::vector<BuildState> states, std::vector<BuildOrderState> buildOrders) : simulator(simulator), states(states), buildOrders(buildOrders) {
        assert(states.size() == 2);
        assert(buildOrders.size() == 2);
    }

    void simulate (Simulator& simulator, float endTime);

    std::vector<SimulatorUnitGroup*> select(int player, std::function<bool(const SimulatorUnitGroup&)>* groupFilter, std::function<bool(const SimulatorUnit&)>* unitFilter);
    bool command(int player, std::function<bool(const SimulatorUnitGroup&)>* groupFilter, std::function<bool(const SimulatorUnit&)>* unitFilter, SimulatorOrder order);
    void command(const std::vector<SimulatorUnitGroup*>& selection, SimulatorOrder order);

    void filterDeadUnits();
    void filterDeadUnits(SimulatorUnitGroup* group);
private:
    void simulateGroupMovement(Simulator& simulator, float endTime);
    void simulateGroupCombat(Simulator& simulator, float endTime);
    void simulateBuildOrder (Simulator& simulator, float endTime);
    void mergeGroups (Simulator& simulator);
};
