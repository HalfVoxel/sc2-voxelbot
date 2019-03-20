#pragma once
#include "../BuildOptimizerGenetic.h"
#include "../CombatPredictor.h"
#include <vector>
#include <functional>

struct SimulatorContext;
extern sc2::Tag simulatorUnitIndexCounter;

inline bool isFakeTag (sc2::Tag tag) {
    // No units tags seem to set any bits above bit 32 (which always seems to be set)
    return tag & (1ULL << 40);
}

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

    SimulatorUnitGroup (sc2::Point2D pos, std::vector<SimulatorUnit> units) : units(units), pos(pos), previousPos(pos) {
        owner = units[0].combat.owner;
    }

    inline void execute(SimulatorOrder order) {
        this->order = order;
    }

    sc2::Point2D futurePosition(float deltaTime);
};

struct SimulatorState {
    SimulatorContext& simulator;
    std::vector<const BuildState*> states;
    std::vector<SimulatorUnitGroup> groups;
    std::vector<BuildOrderState> buildOrders;
    
    float time() {
        return states[0]->time;
    }

    SimulatorState (SimulatorContext& simulator, std::vector<const BuildState*> states, std::vector<BuildOrderState> buildOrders) : simulator(simulator), states(states), buildOrders(buildOrders) {
        assert(states.size() == 2);
        assert(buildOrders.size() == 2);
    }

    void simulate (SimulatorContext& simulator, float endTime);

    std::vector<SimulatorUnitGroup*> select(int player, std::function<bool(const SimulatorUnitGroup&)>* groupFilter, std::function<bool(const SimulatorUnit&)>* unitFilter);
    bool command(int player, std::function<bool(const SimulatorUnitGroup&)>* groupFilter, std::function<bool(const SimulatorUnit&)>* unitFilter, SimulatorOrder order);
    void command(const std::vector<SimulatorUnitGroup*>& selection, SimulatorOrder order, std::function<void(SimulatorUnitGroup&, SimulatorOrder)>* commandListener = nullptr);

    void filterDeadUnits();
    void filterDeadUnits(SimulatorUnitGroup* group);

    void addUnit(const sc2::Unit* unit);
    void addUnit(CombatUnit unit, sc2::Point2D pos);
    void addUnit(int owner, sc2::UNIT_TYPEID unit_type);
    void replaceUnit(int owner, sc2::UNIT_TYPEID unit_type, sc2::UNIT_TYPEID replacement);
    void assertValidState();
private:
    void simulateGroupMovement(SimulatorContext& simulator, float endTime);
    void simulateGroupCombat(SimulatorContext& simulator, float endTime);
    void simulateBuildOrder (SimulatorContext& simulator, float endTime);
    void mergeGroups (SimulatorContext& simulator);
};
