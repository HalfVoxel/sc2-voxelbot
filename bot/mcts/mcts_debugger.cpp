#include "mcts_debugger.h"
#include "../utilities/predicates.h"
#include "../utilities/mappings.h"
#include <pybind11/embed.h>
#include <random>
#include <ctime>
#include <vector>
#include <stack>
#include <sstream>
#include <iostream>
#include "../utilities/python_utils.h"

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
#endif
}

MCTSDebugger::MCTSDebugger(pybind11::object simulatorVisualizerModule) {
#if !DISABLE_PYTHON
    visualize_fn = simulatorVisualizerModule.attr("visualize");
    visualize_bar_fn = simulatorVisualizerModule.attr("visualize_bar");
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
    visualize_fn(units, state.time(), healthFraction(state, 1), healthFraction(state, 2));
#endif
}

void MCTSDebugger::debugInteractive(MCTSState<int, SimulatorMCTSState>* startState) {
    MCTSState<int, SimulatorMCTSState>* state = startState;
    stack<MCTSState<int, SimulatorMCTSState>*> stateStack;

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
                state->instantiateAction(i);
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

        auto* child = state->getChild(chosenAction);
        if (child == nullptr) continue;
        
        stateStack.push(state);
        state = child;
    }
}
