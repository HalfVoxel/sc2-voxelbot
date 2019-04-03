#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "optional.hpp"
#include "../utilities/profiler.h"

template <class A, class T>
struct MCTSState;

template <class A, class T>
struct MCTSChild {
    A action;
    float prior;
    MCTSState<A,T>* state;
};

template <class A, class T>
struct MCTSState {
    T internalState;
    std::vector<MCTSChild<A,T>> children;
    int visits = 0;
    float wins = 0;
    int raveVisits = 0;
    float raveWins = 0;

    MCTSState (const T& state) : internalState(state) {
    }

    MCTSChild<A,T>& select();
    bool instantiateAction(int actionIndex);
    void expand();
    float rollout() const;
    void print(int padding=0, int maxDepth = 100000) const;
    MCTSState<A,T>* getChild(A action);
    nonstd::optional<std::pair<A, MCTSState<A,T>&>> bestAction() const;
    ~MCTSState();
    MCTSState(const MCTSState&) = delete;
};

template <class A, class T>
MCTSState<A,T>::~MCTSState<A,T>() {
    for (auto& c : children) {
        delete c.state;
    }
}

template <class A, class T>
MCTSState<A,T>* MCTSState<A,T>::getChild(A action) {
    for (auto& c : children) {
        if (c.action == action) return c.state;
    }
    return nullptr;
}

template <class A, class T>
void MCTSState<A,T>::print(int padding, int maxDepth) const {

    for (int i = 0; i < padding; i++) std::cout << "|\t";
    auto p = std::cout.precision();
    std::cout << internalState.to_string() << " (" << wins << "/" << visits << " = " << std::setprecision(2) << (wins/visits) << std::setprecision(p) << ")" << std::endl;
    if (maxDepth > 0) {
        for (auto& child : children) {
            if (child.state != nullptr) {
                std::cout << child.action << ": ";
                child.state->print(padding+1, maxDepth - 1);
            }
        }
    }
}

template <class A, class T>
MCTSChild<A,T>& MCTSState<A,T>::select() {
    while(true) {
        const float c = 1;
        // float exploration = c * sqrt(visits);
        float exploration = c * sqrt(log(visits));
        float raveExploration = c * sqrt(log(raveVisits));
        float bestScore = -1000;
        int bestAction = -1;
        for (int i = 0; i < children.size(); i++) {
            const auto& child = children[i];
            float score;
            if (children[i].state == nullptr) {
                score = child.prior + exploration;
                // std::cout << "A " << score << std::endl;
            } else {
                float priorStrength = 4;
                float r = (child.state->wins + (1 - child.prior) * priorStrength) / (child.state->visits + priorStrength);
                float raveR = (child.state->raveWins + ( 1 - child.prior) * priorStrength) / (child.state->raveVisits + priorStrength);
                // score = (1 - r) + child.prior * exploration / sqrt(1 + child.state->visits);
                // score = (1 - r) + exploration / sqrt(1 + child.state->visits);
                
                float UCTScore = (1 - r) + exploration / sqrt(1 + child.state->visits);
                float RaveScore = (1 - raveR) + raveExploration / sqrt(1 + child.state->raveVisits);

                float alpha = 5000/(5000 + child.state->visits);
                score = (1 - alpha) * UCTScore + alpha * RaveScore;
                // std::cout << "B " << score << std::endl;
            }

            if (score > bestScore) {
                bestScore = score;
                bestAction = i;
            }
        }

        assert(bestAction != -1);

        if (children[bestAction].state == nullptr) {
            if (!instantiateAction(bestAction)) continue;
        }

        return children[bestAction];
    }
}

template <class A, class T>
bool MCTSState<A,T>::instantiateAction(int actionIndex) {
    // std::cout << "Exploring" << std::endl;
    auto ret = internalState.step(children[actionIndex].action);
    if (ret.second) {
        children[actionIndex].state = new MCTSState(std::move(ret.first));
        return true;
    } else {
        // Invalid action, remove this child node
        children[actionIndex] = std::move(*children.rbegin());
        children.pop_back();
        return false;
    }
}

template <class A, class T>
void MCTSState<A, T>::expand() {
    std::vector<std::pair<A, float>> moves = internalState.generateMoves();
    if (moves.size() == 0) {
        // throw std::exception();
        return;
    }
    children = std::vector<MCTSChild<A, T>>(moves.size());
    for (int i = 0; i < children.size(); i++) {
        children[i] = { moves[i].first, moves[i].second, nullptr };
    }
}

template <class A, class T>
float MCTSState<A, T>::rollout() const {
    return internalState.rollout();
}

extern float tRollout;
extern float tExpand;
extern float tSelect;

struct MCTSPropagationResult {
    float wins = 0;
    int rollouts = 0;
    std::vector<bool> usedActions;
};

// TODO: needlessly evals root note
template <class A, class T>
MCTSPropagationResult mcts(MCTSState<A,T>& node) {
    MCTSPropagationResult result;
    if (node.visits == 0) {
        Stopwatch w;
        result.rollouts = 2;
        result.usedActions = std::vector<bool>(10);
        for (int i = 0; i < result.rollouts; i++) {
            result.wins += node.rollout();
        }
        // std::cout << "Evald " << node.internalState.to_string() << ": " << wins << " of " << rollouts << std::endl;
        w.stop();
        tRollout += w.millis();
    } else {
        if (node.children.size() == 0) {
            Stopwatch w;
            node.expand();
            w.stop();
            tExpand += w.millis();
        }

        if (node.children.size() == 0) {
            // Terminal node
            result.rollouts = 1;
            result.wins = node.rollout();
            result.usedActions = std::vector<bool>(10);
        } else {
            Stopwatch w;
            // std::cout << node.internalState.to_string() << " -> ";
            auto& child = node.select();
            w.stop();
            tSelect += w.millis();
            result = mcts(*child.state);
            result.wins = result.rollouts - result.wins;
            result.usedActions[child.action] = true;
        }
    }

    node.wins += result.wins;
    node.visits += result.rollouts;
    node.raveWins += result.wins;
    node.raveVisits += result.rollouts;
    for (int i = node.children.size() - 1; i >= 0; i--) {
        auto& c = node.children[i];
        if (result.usedActions[c.action]) {
            if (c.state == nullptr) {
                if (!node.instantiateAction(i)) continue;
            }

            c.state->raveWins += result.wins;
            c.state->raveVisits += result.rollouts;
        }
    }

    if ((node.visits % (8*1000)) == 0) {
        std::cout << "MCTS timings " << tRollout << " " << tExpand << " " << tSelect << std::endl;
    }
    return result;
}

template <class A, class T>
nonstd::optional<std::pair<A, MCTSState<A,T>&>> MCTSState<A, T>::bestAction() const {
    int bestAction = -1;
    int bestScore = -1000;
    for (int i = 0; i < children.size(); i++) {
        if (children[i].state != nullptr) {
            if (children[i].state->visits > bestScore || (children[i].state->visits == bestScore && children[i].state->wins < children[bestAction].state->wins)) {
                bestScore = children[i].state->visits;
                bestAction = i;
            }
        }
    }

    std::cout << bestScore << std::endl;

    if (bestAction == -1) {
        return {};
    }
    return nonstd::make_optional<std::pair<A, MCTSState<A,T>&>> (children[bestAction].action, *children[bestAction].state);
}
