#include "../ml/simulator.h"
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/utilities/mappings.h>
#include <libvoxelbot/utilities/profiler.h>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <random>
#include <ctime>
#include <sstream>
#include <iostream>
#include "../mcts/mcts.h"
#include "../ml/mcts_sc2.h"
#include "../ml/simulator_context.h"

using namespace std;
using namespace sc2;

pybind11::object visualize_fn;
pybind11::object visualize_bar_fn;


void visualizeSimulator(SimulatorState& state) {
    vector<vector<double>> units;
    for (auto& group : state.groups) {
        for (auto& u : group.units) {
            units.push_back(vector<double> { group.pos.x, group.pos.y, (double)u.combat.owner, (double)u.tag, (double)u.combat.type, u.combat.health + u.combat.shield, u.combat.health_max + u.combat.shield_max });
        }
    }

    visualize_fn(units, state.time(), healthFraction(state, 1), healthFraction(state, 2));
}

void stepSimulator(SimulatorState& state, float endTime) {
    while(state.time() < endTime) {
        state.simulate(state.time() + 1);
        visualizeSimulator(state);
    }
}

void example_simulation(SimulatorContext& simulator, SimulatorState& state) {
    assert(state.command(1, &idleGroup, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 60))));
    assert(state.groups[0].order.type == SimulatorOrderType::Attack);
    assert(state.groups[0].order.target == Point2D(100, 60));

    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;
    visualizeSimulator(state);

    stepSimulator(state, state.time() + 2);

    assert(state.command(2, &idleGroup, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(40, 100))));

    assert(state.command(1, &idleGroup, &notArmyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(10, 60))));
    assert(state.groups[0].order.type == SimulatorOrderType::Attack);
    assert(state.groups[0].order.target == Point2D(100, 60));

    stepSimulator(state, state.time() + 6);

    assert(state.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(60, 60))));

    stepSimulator(state, state.time() + 6);

    assert(state.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));
    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;

    stepSimulator(state, state.time() + 6);

    assert(state.command(2, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));

    stepSimulator(state, state.time() + 50);
    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;
}

int cs = 0;
int ds = 0;
int cp = 0;
struct Blah {
    int x;
    int y;
    int z;
    float q;
    vector<float> meh;
    vector<float> meh2;
    vector<float> meh3;

    Blah(int x, int y) : x(x), y(y) {
        cs++;
        meh = vector<float>(100, 2);
        meh2 = vector<float>(99, 1);
        meh3 = vector<float>(50, 3);
    }

    // Blah(const Blah& x) {
    //     cp++;
    // }

    // ~Blah() {
    //     ds++;
    // }
};

void test_allocator() {
    {
        Stopwatch w;
        {
            BumpAllocator<Blah, 512> blahs;
            vector<Blah*> bs;
            for (int i = 0; i < 1000000; i++) {
                Blah* b = blahs.allocate(5, 3);
                b->z = 3;
                b->z = 6;
                bs.push_back(b);
            }
            blahs.clear();
        }
        w.stop();
        cout << "Bump: " << w.millis() << endl;
    }
    // assert(cs == ds);
    // assert(cp == 0);

    {
        Stopwatch w;
        vector<Blah*> bs;
        for (int i = 0; i < 1000000; i++) {
            Blah* b = new Blah(5, 3);
            b->z = 3;
            b->z = 6;
            bs.push_back(b);
        }
        for (auto b : bs) delete b;
        w.stop();
        cout << "raw ptr: " << w.millis() << endl;
    }

    {
        Stopwatch w;
        {
            vector<shared_ptr<Blah>> bs;
            for (int i = 0; i < 1000000; i++) {
                // shared_ptr<Blah> b = std::shared_ptr<Blah>(new Blah(5, 3));
                shared_ptr<Blah> b = std::make_shared<Blah>(5, 3);
                b->z = 3;
                b->z = 6;
                bs.push_back(b);
            }
        }
        w.stop();
        cout << "Shared ptr: " << w.millis() << endl;
    }
}

void mcts(SimulatorState& startState) {
    // for (int i = 0; i < 100; i++) {
    //     test_allocator();
    // }

    //srand(time(0));
    srand(0);

    {
        MCTSSearch<int, SimulatorMCTSState> search((SimulatorMCTSState(startState)));
        search.search(10000);
        cout << "Post search bump size " << search.stateAllocator.size() << " " << startState.simulator.lock()->cache.buildStateAllocator.size() << endl;
    }

    exit(0);

    // state->getChild(6)->getChild(4)->getChild(0)->getChild(1)->getChild(4)->print(0, 2);
    // state->getChild(4)->getChild(4)->print(0, 2);

    /*
    auto currentState = state;
    visualizeSimulator(currentState->internalState.state);
    int it = 0;
    while(true) {
        if (it % 2 == 0) {
            auto simulator = shared_ptr<SimulatorContext>(state->internalState.state.simulator);
            // Clear cache to excessive memory usage over time
            simulator->cache.clear();

            state = make_shared<MCTSState<int, SimulatorMCTSState>>(SimulatorMCTSState(currentState->internalState.state));
            simulator->simulationStartTime = state->internalState.state.time();

            vector<vector<float>> stats(200, vector<float>(14));
            vector<vector<float>> statsRave(200, vector<float>(14));
            for (int i = 0; i < 20000; i++) {
                mcts<int, SimulatorMCTSState>(*state);
                for (auto& c : state->children) {
                    assert(c.action < 14);
                    // stats[i/100][c.action] = c.state != nullptr ? c.state->wins / (0.01f + c.state->visits) : 0;
                    // statsRave[i/100][c.action] = c.state != nullptr ? c.state->raveWins / (0.01f + c.state->raveVisits) : 0;
                    stats[i/100][c.action] = c.state != nullptr ? c.state->visits : 0;
                    statsRave[i/100][c.action] = c.state != nullptr ? c.state->raveVisits : 0;
                }
            }
            // visualize_bar_fn(stats, statsRave);

            if (it == 0) {
                state->print(0, 3);
            }
            currentState = state;
        }
        //auto action = ((it % 2) == 1) ? (it == 1 || it == 3 || it == 5 ? nonstd::make_optional<pair<int, State<int, SimulatorMCTSState>&>>(4, *currentState->getChild(4)) : nonstd::make_optional<pair<int, State<int, SimulatorMCTSState>&>>(0, *currentState->getChild(0))) : currentState->bestAction();
        auto action = currentState->bestAction();
        it++;
        if (action && currentState->getChild(action.value().first) != nullptr) {
            cout << "Action " << action.value().first << endl;
            currentState = action.value().second;
            for (auto& g : currentState->internalState.state.groups) {
                cout << "Group action: " << g.order.type << " -> " << g.order.target.x << "," << g.order.target.y << " (player " << g.owner << " at " << g.pos.x << "," << g.pos.y << ")" << endl;
            }
            visualizeSimulator(currentState->internalState.state);
        } else {
            break;
        }
    }

    while(true) {
        stepSimulator(currentState->internalState.state, currentState->internalState.state.time() + 5);
    }
    */
}

void test_simulator() {
    CombatPredictor combatPredictor;
    combatPredictor.init();

    array<vector<Point2D>, 2> extraDestinations = {{ {}, {} }};
    extraDestinations[0].push_back(Point2D(100/2, 100/2));
    extraDestinations[0].push_back(Point2D(100, 100));
    extraDestinations[0].push_back(Point2D(0, 0));
    extraDestinations[1].push_back(Point2D(100/2, 100/2));
    extraDestinations[1].push_back(Point2D(100, 100));
    extraDestinations[1].push_back(Point2D(0, 0));
    
    auto simulator = make_shared<SimulatorContext>(&combatPredictor, vector<Point2D>{ Point2D(0,0), Point2D(100, 100) }, extraDestinations);
    BuildOrder bo1 = { UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
    };
    BuildOrder bo2 = { UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_ZEALOT,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
     };
    SimulatorState state(simulator, vector<BuildState>{
        BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PYLON, 6 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 0 }, { UNIT_TYPEID::PROTOSS_STALKER, 2 }, { UNIT_TYPEID::PROTOSS_PHOENIX, 9 }, { UNIT_TYPEID::PROTOSS_IMMORTAL, 4 }}),
        BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 2 }, { UNIT_TYPEID::PROTOSS_PYLON, 6 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 }, { UNIT_TYPEID::PROTOSS_CARRIER, 2 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 1+2 }, { UNIT_TYPEID::PROTOSS_COLOSSUS, 2 }, { UNIT_TYPEID::PROTOSS_VOIDRAY, 2 }, { UNIT_TYPEID::PROTOSS_PHOTONCANNON, 6 }, { UNIT_TYPEID::PROTOSS_STARGATE, 1 } }),
    },
    {
        BuildOrderState(make_shared<BuildOrder>(bo1)),
        BuildOrderState(make_shared<BuildOrder>(bo2)),
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

    mcts(state);
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
