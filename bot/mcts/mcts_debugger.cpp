#include "mcts_debugger.h"
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/utilities/mappings.h>
#include <pybind11/embed.h>
#include <random>
#include <ctime>
#include <vector>
#include <stack>
#include <sstream>
#include <iostream>
#include <libvoxelbot/utilities/python_utils.h>

using namespace std;
using namespace sc2;

MCTSDebugger::MCTSDebugger() {
    lock_guard<mutex> lock(python_thread_mutex);
#if !DISABLE_PYTHON
    pybind11::exec(R"(
        import sys
        if "bot/python" not in sys.path:
            sys.path.append("bot/python")
    )");
    auto simulatorVisualizer = pybind11::module::import("simulator_visualizer");
    visualize_fn = simulatorVisualizer.attr("visualize");
    visualize_bar_fn = simulatorVisualizer.attr("visualize_bar");
    reset_fn = simulatorVisualizer.attr("reset_visualizer");
#endif
}

MCTSDebugger::MCTSDebugger(pybind11::object simulatorVisualizerModule) {
#if !DISABLE_PYTHON
    visualize_fn = simulatorVisualizerModule.attr("visualize");
    visualize_bar_fn = simulatorVisualizerModule.attr("visualize_bar");
    reset_fn = simulatorVisualizerModule.attr("reset_visualizer");
#endif
}

void MCTSDebugger::visualize(SimulatorState& state) {
#if !DISABLE_PYTHON
    vector<vector<double>> units;
    for (auto& group : state.groups) {
        for (auto& u : group.units) {
            units.push_back(vector<double> { group.pos.x, group.pos.y, (double)u.combat.owner, (double)u.tag, (double)u.combat.type, u.combat.health + u.combat.shield, u.combat.health_max + u.combat.shield_max });
        }
    }

    lock_guard<mutex> lock(python_thread_mutex);
    // visualize_fn(units, state.time(), healthFraction(state, 1), healthFraction(state, 2));
    visualize_fn(units, state.time(), state.rewards[0], state.rewards[1]);
#endif
}

void MCTSDebugger::debugInteractive(MCTSSearch<int, SimulatorMCTSState>& search) {
    reset_fn();

    MCTSState<int, SimulatorMCTSState>* state = search.root;
    stack<MCTSState<int, SimulatorMCTSState>*> stateStack;

    auto sim = shared_ptr<SimulatorContext>(state->internalState.state.simulator);

    string green = "\x1b[38;2;0;255;0m";
    string greenish = "\x1b[38;2;93;173;110m";
    string red = "\x1b[38;2;255;0;0m";
    string grey = "\x1b[38;2;193;184;192m";
    string reset = "\033[0m";
    string clear_line = "\033[0K";
    while(true) {
        visualize(state->internalState.state);

        if (state->internalState.player == 0) cout << green;
        else cout << red;

        if (state->children.size() == 0) {
            cout << "Expanding child" << endl;
            state->expand();
        }

        // Make sure all children are intantiated
        for (int i = state->children.size() - 1; i >= 0; i--) {
            if (state->children[i].state == nullptr) {
                bool origDebug = sim->debug;
                sim->debug = false;
                state->instantiateAction(search, i);
                sim->debug = origDebug;
            }
        }

        cout << "Select child" << endl;

        for (auto& c : state->children) {
            cout << c.action << ": " << MCTSActionName((MCTSAction)c.action) << " " << (c.state != nullptr ? c.state->visits : 0) << endl;
        }

        cout << reset;

        string command;
        if (!(cin >> command)) continue;

        if (command == "exit") return;
        if (command == "debug") {
            sim->debug = !sim->debug;
            cout << "Debug mode: " << sim->debug << endl;
            continue;
        }
        if (command == "b" || command == "back") {
            if (stateStack.size() > 0) {
                state = stateStack.top();
                stateStack.pop();
            }
            continue;
        }

        stringstream ss(command);
        int chosenAction;
        if (!(ss >> chosenAction)) continue;

        MCTSState<int, SimulatorMCTSState>* child;
        if (sim->debug) {
            auto nextState = state->internalState.step(chosenAction);
            if (!nextState.second) {
                cout << red << "Invalid action" << reset << endl;
                continue;
            }

            child = search.stateAllocator.allocate(std::move(nextState.first));
        } else {
            child = state->getChild(chosenAction);
            if (child == nullptr) continue;
        }
        
        stateStack.push(state);
        state = child;
    }
}
