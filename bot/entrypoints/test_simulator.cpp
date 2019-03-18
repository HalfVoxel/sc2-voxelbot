#include "../ml/simulator.h"
#include "../utilities/predicates.h"
#include "../utilities/mappings.h"
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <random>
#include <ctime>
#include <sstream>
#include <iostream>
#include "../mcts/mcts.h"
#include "../ml/mcts_sc2.h"

using namespace std;
using namespace sc2;

pybind11::object visualize_fn;
pybind11::object visualize_bar_fn;


void visualizeSimulator(SimulatorState& state) {
    vector<vector<float>> units;
    for (auto& group : state.groups) {
        for (auto& u : group.units) {
            units.push_back(vector<float> { group.pos.x, group.pos.y, (float)u.combat.owner, (float)u.tag, (float)u.combat.type, u.combat.health + u.combat.shield, u.combat.health_max + u.combat.shield_max });
        }
    }

    visualize_fn(units, state.time(), healthFraction(state, 1), healthFraction(state, 2));
}

void stepSimulator(SimulatorState& state, Simulator& simulator, float endTime) {
    while(state.time() < endTime) {
        state.simulate(simulator, state.time() + 1);
        visualizeSimulator(state);
    }
}

void example_simulation(Simulator& simulator, SimulatorState& state) {
    assert(state.command(1, &idleGroup, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 60))));
    assert(state.groups[0].order.type == SimulatorOrderType::Attack);
    assert(state.groups[0].order.target == Point2D(100, 60));

    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;
    visualizeSimulator(state);

    stepSimulator(state, simulator, state.time() + 2);

    assert(state.command(2, &idleGroup, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(40, 100))));

    assert(state.command(1, &idleGroup, &notArmyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(10, 60))));
    assert(state.groups[0].order.type == SimulatorOrderType::Attack);
    assert(state.groups[0].order.target == Point2D(100, 60));

    stepSimulator(state, simulator, state.time() + 6);

    assert(state.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(60, 60))));

    stepSimulator(state, simulator, state.time() + 6);

    assert(state.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));
    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;

    stepSimulator(state, simulator, state.time() + 6);

    assert(state.command(2, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));

    stepSimulator(state, simulator, state.time() + 50);
    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;
}

void mcts(Simulator& simulator, SimulatorState& startState) {
    srand(time(0));

    unique_ptr<State<int, SimulatorMCTSState>> state = make_unique<State<int, SimulatorMCTSState>>(SimulatorMCTSState(startState));
    /*for (int i = 0; i < 100000; i++) {
        mcts<int, SimulatorMCTSState>(*state);
    }*/

    // exit(0);

    // state->getChild(6)->getChild(4)->getChild(0)->getChild(1)->getChild(4)->print(0, 2);
    // state->getChild(4)->getChild(4)->print(0, 2);

    auto* currentState = &*state;
    visualizeSimulator(currentState->internalState.state);
    int it = 0;
    while(true) {
        if (it % 2 == 0) {
            state = make_unique<State<int, SimulatorMCTSState>>(SimulatorMCTSState(currentState->internalState.state));
            simulator.simulationStartTime = state->internalState.state.time();

            // vector<vector<int>> stats(200, vector<int>(14));
            for (int i = 0; i < 20000; i++) {
                mcts<int, SimulatorMCTSState>(*state);
                /*for (auto& c : state->children) {
                    assert(c.action < 14);
                    stats[i/100][c.action] = c.state != nullptr ? c.state->visits : 0;
                }*/
            }
            // visualize_bar_fn(stats);

            if (it == 0) {
                state->print(0, 3);
            }
            currentState = &*state;
        }
        //auto action = ((it % 2) == 1) ? (it == 1 || it == 3 || it == 5 ? nonstd::make_optional<pair<int, State<int, SimulatorMCTSState>&>>(4, *currentState->getChild(4)) : nonstd::make_optional<pair<int, State<int, SimulatorMCTSState>&>>(0, *currentState->getChild(0))) : currentState->bestAction();
        auto action = currentState->bestAction();
        it++;
        if (action && currentState->getChild(action.value().first) != nullptr) {
            cout << "Action " << action.value().first << endl;
            currentState = &action.value().second;
            for (auto& g : currentState->internalState.state.groups) {
                cout << "Group action: " << g.order.type << " -> " << g.order.target.x << "," << g.order.target.y << " (player " << g.owner << " at " << g.pos.x << "," << g.pos.y << ")" << endl;
            }
            visualizeSimulator(currentState->internalState.state);
        } else {
            break;
        }
    }

    while(true) {
        stepSimulator(currentState->internalState.state, currentState->internalState.state.simulator, currentState->internalState.state.time() + 5);
    }
}

void test_simulator() {
    CombatPredictor combatPredictor;
    combatPredictor.init();

    Simulator simulator(&combatPredictor, { Point2D(0,0), Point2D(100, 100) });
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
        BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PYLON, 6 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 0 }, { UNIT_TYPEID::PROTOSS_STALKER, 2 }, { UNIT_TYPEID::PROTOSS_PHOENIX, 9 }, { UNIT_TYPEID::PROTOSS_IMMORTAL, 4 }}),
        BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 2 }, { UNIT_TYPEID::PROTOSS_PYLON, 6 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 }, { UNIT_TYPEID::PROTOSS_CARRIER, 2 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 1+2 }, { UNIT_TYPEID::PROTOSS_COLOSSUS, 2 }, { UNIT_TYPEID::PROTOSS_VOIDRAY, 2 }, { UNIT_TYPEID::PROTOSS_PHOTONCANNON, 6 }, { UNIT_TYPEID::PROTOSS_STARGATE, 1 } }),
    },
    {
        BuildOrderState(bo1),
        BuildOrderState(bo2),
    });
    state.states[1].resources.minerals = 10000;
    state.states[1].resources.vespene = 10000;

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

    mcts(simulator, state);
    // example_simulation(simulator, state);
}


int main(int argc, char* argv[]) {
    initMappings();

    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        sys.path.append("bot/python")
    )");

    auto simulatorVisualizer = pybind11::module::import("simulator_visualizer");
    visualize_fn = simulatorVisualizer.attr("visualize");
    visualize_bar_fn = simulatorVisualizer.attr("visualize_bar");

    test_simulator();
}