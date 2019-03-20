#include "../ml/mcts_sc2.h"
#include "../ml/simulator.h"
#include "../ml/simulator_context.h"
#include "../ml/mcts_cache.h"
#include "../CombatPredictor.h"
#include "../utilities/mappings.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

using namespace std;
using namespace sc2;

void assertIdentical(SimulatorState& s1, SimulatorState& s2) {
    assert(s1.time() == s2.time());
    assert(s1.states[0] == s2.states[0]);
    assert(s1.states[1] == s2.states[1]);
    assert(s1.buildOrders[0].buildIndex == s2.buildOrders[0].buildIndex);
    assert(s1.buildOrders[1].buildIndex == s2.buildOrders[1].buildIndex);
    assert(s1.groups.size() == s2.groups.size());
    for (int i = 0; i < s1.groups.size(); i++) {
        auto& g1 = s1.groups[i];
        auto& g2 = s1.groups[i];
        assert(g1.order.type == g2.order.type);
        assert(g1.order.target == g2.order.target);
        assert(g1.units.size() == g2.units.size());
        assert(g1.owner == g2.owner);
    }
}

void test() {
    CombatPredictor combatPredictor;
    combatPredictor.init();

    SimulatorContext simulator(&combatPredictor, { Point2D(0,0), Point2D(100, 100) });
    vector<UNIT_TYPEID> bo1 = { UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
    };
    vector<UNIT_TYPEID> bo2 = { UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_ZEALOT,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
     };
    SimulatorState state(simulator, {
        simulator.cache.copyState(BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PYLON, 6 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 0 }, { UNIT_TYPEID::PROTOSS_STALKER, 2 }, { UNIT_TYPEID::PROTOSS_PHOENIX, 9 }, { UNIT_TYPEID::PROTOSS_IMMORTAL, 4 }})),
        simulator.cache.copyState(BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 2 }, { UNIT_TYPEID::PROTOSS_PYLON, 6 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 }, { UNIT_TYPEID::PROTOSS_CARRIER, 2 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 1+2 }, { UNIT_TYPEID::PROTOSS_COLOSSUS, 2 }, { UNIT_TYPEID::PROTOSS_VOIDRAY, 2 }, { UNIT_TYPEID::PROTOSS_PHOTONCANNON, 6 }, { UNIT_TYPEID::PROTOSS_STARGATE, 1 } })),
    },
    {
        BuildOrderState(bo1),
        BuildOrderState(bo2),
    });
    // state.states[1]->resources.minerals = 10000;
    // state.states[1]->resources.vespene = 10000;

    state.groups.push_back(SimulatorUnitGroup(Point2D(10, 10), {
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(20, 10), {
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(20, 10), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(100, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(50, 10), {
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(80, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(5, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(100, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_NEXUS)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_STARGATE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(50, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_NEXUS)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(10, 10), {
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_NEXUS)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
    }));
    
    assert(state.command(1, &idleGroup, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 60))));
    assert(state.command(2, &idleGroup, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(40, 100))));
    assert(state.command(1, &idleGroup, &notArmyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(10, 60))));
    assert(state.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(60, 60))));
    assert(state.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));
    assert(state.command(2, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));


    SimulatorState state2 = state;
    SimulatorState state3 = state;
    state3.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 0)));

    state.simulate(simulator, 10);
    assert(state.states[0] != state2.states[0]);
    state2.simulate(simulator, 10);
    state3.simulate(simulator, 10);

    // Literally the same pointer
    assertIdentical(state, state2);
    assert(state.time() == 10);
    assert(state2.time() == 10);
    state.assertValidState();
    state2.assertValidState();

    cout << "Hashes " << state.states[0]->hash() << " " << state2.states[0]->hash() << endl;

    // Should trigger a combat
    cout << "S1" << endl;
    state.simulate(simulator, 100);
    cout << "S2" << endl;
    state2.simulate(simulator, 100);
    cout << "S3" << endl;
    state3.simulate(simulator, 50);
    assert(state3.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));
    cout << "S3" << endl;
    state3.simulate(simulator, 100);
    

    assertIdentical(state, state2);
    // assertIdentical(state, state3);
    assert(state.buildOrders[0].buildIndex > 0);
    assert(state.time() == 100);
    assert(state2.time() == 100);
    state.assertValidState();
    state2.assertValidState();
}

int main(int argc, char* argv[]) {
    initMappings();

    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        sys.path.append("bot/python")
    )");

    test();
}
