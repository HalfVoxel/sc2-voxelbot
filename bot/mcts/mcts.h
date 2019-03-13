#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "optional.hpp"

template <class A, class T>
struct State;

template <class A, class T>
struct Child {
    A action;
    float prior;
    State<A,T>* state;
};

template <class A, class T>
struct State {
    T internalState;
    std::vector<Child<A,T>> children;
    int visits = 0;
    float wins = 0;

    State (const T& state) : internalState(state) {
    }

    State& select();
    void expand();
    float rollout() const;
    void print(int padding=0, int maxDepth = 100000) const;
    State<A,T>* getChild(A action);
    nonstd::optional<std::pair<A, State<A,T>&>> bestAction() const;
};

template <class A, class T>
State<A,T>* State<A,T>::getChild(A action) {
    for (auto& c : children) {
        if (c.action == action) return c.state;
    }
    return nullptr;
}

template <class A, class T>
void State<A,T>::print(int padding, int maxDepth) const {

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
State<A,T>& State<A,T>::select() {
    while(true) {
        const float c = 1;
        // float exploration = c * sqrt(visits);
        float exploration = c * sqrt(log(visits));
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
                // score = (1 - r) + child.prior * exploration / sqrt(1 + child.state->visits);
                // score = (1 - r) + exploration / sqrt(1 + child.state->visits);
                score = (1 - r) + exploration / sqrt(1 + child.state->visits);
                // std::cout << "B " << score << std::endl;
            }

            if (score > bestScore) {
                bestScore = score;
                bestAction = i;
            }
        }

        if (children[bestAction].state == nullptr) {
            // std::cout << "Exploring" << std::endl;
            auto ret = internalState.step(children[bestAction].action);
            if (ret.second) {
                children[bestAction].state = new State(std::move(ret.first));
            } else {
                // Invalid action, remove this child node
                children[bestAction] = std::move(*children.rbegin());
                children.pop_back();
                continue;
            }
        }

        return *children[bestAction].state;
    }
}

template <class A, class T>
void State<A, T>::expand() {
    std::vector<std::pair<A, float>> moves = internalState.generateMoves();
    if (moves.size() == 0) {
        // throw std::exception();
        return;
    }
    children = std::vector<Child<A, T>>(moves.size());
    for (int i = 0; i < children.size(); i++) {
        children[i] = { moves[i].first, moves[i].second, nullptr };
    }
}

template <class A, class T>
float State<A, T>::rollout() const {
    return internalState.rollout();
}

// TODO: needlessly evals root note
template <class A, class T>
std::pair<float, int> mcts(State<A,T>& node) {
    std::pair<float, int> result;
    if (node.visits == 0) {
        float wins = 0;
        int rollouts = 3;
        for (int i = 0; i < rollouts; i++) {
            wins += node.rollout();
        }
        // std::cout << "Evald " << node.internalState.to_string() << ": " << wins << " of " << rollouts << std::endl;
        result = std::make_pair(wins, rollouts);
    } else {
        if (node.children.size() == 0) {
            node.expand();
        }

        if (node.children.size() == 0) {
            // Terminal node
            result = std::make_pair(node.rollout(), 1);
        } else {
            // std::cout << node.internalState.to_string() << " -> ";
            result = mcts(node.select());
            result = std::make_pair(result.second - result.first, result.second);
        }
    }

    node.wins += result.first;
    node.visits += result.second;
    return result;
}

template <class A, class T>
nonstd::optional<std::pair<A, State<A,T>&>> State<A, T>::bestAction() const {
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
    return nonstd::make_optional<std::pair<A, State<A,T>&>> (children[bestAction].action, *children[bestAction].state);
}
