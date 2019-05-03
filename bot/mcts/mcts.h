#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include "optional.hpp"
#include "../utilities/profiler.h"
#include "../utilities/bump_allocator.h"
#include "variance_estimator.h"

static const bool ENABLE_RAVE = false;
static const bool ENABLE_SCORE_NORMALIZATION = true;

template <class A, class T>
struct MCTSState;

template <class A, class T>
struct MCTSSearch;

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
    VarianceEstimator varianceEstimator;

    MCTSState (T state) : internalState(state) {
    }

    MCTSChild<A,T>& select(MCTSSearch<A,T>& context);
    bool instantiateAction(MCTSSearch<A,T>& context, int actionIndex);
    void expand();
    float rollout() const;
    void print(int padding=0, int maxDepth = 100000) const;
    MCTSState<A,T>* getChild(A action);
    nonstd::optional<std::pair<A, MCTSState<A,T>*>> bestAction() const;
    MCTSState(const MCTSState&) = delete;
};

extern float tRollout;
extern float tExpand;
extern float tSelect;

struct MCTSPropagationResult {
    float wins = 0;
    int rollouts = 0;
    std::vector<bool> usedActions;
};

template <class A, class T>
struct MCTSSearch {
    BumpAllocator<MCTSState<A,T>> stateAllocator;
    MCTSState<A,T>* root;

    MCTSSearch(const T& state) {
        root = stateAllocator.allocate(state);
    }

    // TODO: needlessly evals root note
    MCTSPropagationResult mcts(MCTSState<A,T>& node, int depth) {
        MCTSPropagationResult result;
        if (node.visits == 0 || depth > 30) {
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
                auto& child = node.select(*this);
                w.stop();
                tSelect += w.millis();
                result = mcts(*child.state, depth + 1);
                result.wins = result.rollouts - result.wins;
                result.usedActions[child.action] = true;
            }
        }

        node.wins += result.wins;
        node.visits += result.rollouts;
        node.raveWins += result.wins;
        node.raveVisits += result.rollouts;
        for (int i = 0; i < result.rollouts; i++) {
            node.varianceEstimator.add(result.wins/result.rollouts);
        }

        if (ENABLE_RAVE) {
            for (int i = (int)node.children.size() - 1; i >= 0; i--) {
                auto& c = node.children[i];
                if (result.usedActions[c.action]) {
                    if (c.state == nullptr) {
                        if (!node.instantiateAction(*this, i)) continue;
                    }

                    c.state->raveWins += result.wins;
                    c.state->raveVisits += result.rollouts;
                }
            }
        }

        if ((node.visits % (8*1000)) == 0) {
            // std::cout << "MCTS timings " << tRollout << " " << tExpand << " " << tSelect << std::endl;
        }
        return result;
    }

    void search(int iterations) {
        for (int i = 0; i < iterations; i++) mcts(*root, 0);
    }
};

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
    std::cout << internalState.to_string() << " (" << wins << "/" << visits << " = " << std::setprecision(2) << (wins/visits) << std::setprecision(p) << ", var=" << varianceEstimator.variance() << ")" << std::endl;
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
MCTSChild<A,T>& MCTSState<A,T>::select(MCTSSearch<A,T>& context) {
    float varianceMultiplier = 1.0f / (0.001f + sqrt(varianceEstimator.variance()));
    while(true) {
        const float c = 2.0f;
        // float exploration = c * sqrt(visits);
        float exploration = c * sqrt(log(visits));
        float raveExploration = c * sqrt(log(raveVisits));
        float bestScore = -1000;
        int bestAction = -1;
        for (size_t i = 0; i < children.size(); i++) {
            const auto& child = children[i];
            float score;
            if (children[i].state == nullptr) {
                float r = child.prior;
                if (ENABLE_SCORE_NORMALIZATION) r *= varianceMultiplier;
                score = r + exploration;
                // std::cout << "A " << score << std::endl;
            } else {
                // if (ENABLE_SCORE_NORMALIZATION) {
                //     float r = child.state->wins / child.state->visits;
                //     r *= normalizationFactor;
                //     float UCTScore = -r + exploration / sqrt(1 + child.state->visits);
                // }
                float priorStrength = 4;
                float r = (child.state->wins - child.prior * priorStrength) / (child.state->visits + priorStrength);
                // score = (1 - r) + child.prior * exploration / sqrt(1 + child.state->visits);
                // score = (1 - r) + exploration / sqrt(1 + child.state->visits);
                
                // float UCTScore = (1 - r) + exploration / sqrt(1 + child.state->visits);
                if (ENABLE_SCORE_NORMALIZATION) r *= varianceMultiplier;
                float UCTScore = -r + exploration / sqrt(1 + child.state->visits);
                
                if (ENABLE_RAVE) {
                    float raveR = (child.state->raveWins + ( 1 - child.prior) * priorStrength) / (child.state->raveVisits + priorStrength);
                    float RaveScore = (1 - raveR) + raveExploration / sqrt(1 + child.state->raveVisits);

                    float alpha = 5000/(5000 + child.state->visits);
                    score = (1 - alpha) * UCTScore + alpha * RaveScore;
                } else {
                    score = UCTScore;
                }
                // std::cout << "B " << score << std::endl;
            }

            if (score > bestScore) {
                bestScore = score;
                bestAction = i;
            }
        }

        assert(bestAction != -1);

        if (children[bestAction].state == nullptr) {
            if (!instantiateAction(context, bestAction)) continue;
        }

        return children[bestAction];
    }
}

template <class A, class T>
bool MCTSState<A,T>::instantiateAction(MCTSSearch<A,T>& context, int actionIndex) {
    // std::cout << "Exploring" << std::endl;
    auto ret = internalState.step(children[actionIndex].action);
    if (ret.second) {
        // children[actionIndex].state = std::make_shared<MCTSState>(std::move(ret.first));
        children[actionIndex].state = context.stateAllocator.allocate(std::move(ret.first));
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
    for (size_t i = 0; i < children.size(); i++) {
        children[i] = { moves[i].first, moves[i].second, nullptr };
    }
}

template <class A, class T>
float MCTSState<A, T>::rollout() const {
    return internalState.rollout();
}


template <class A, class T>
nonstd::optional<std::pair<A, MCTSState<A,T>*>> MCTSState<A, T>::bestAction() const {
    int bestAction = -1;
    int bestScore = -1000;
    for (size_t i = 0; i < children.size(); i++) {
        if (children[i].state != nullptr) {
            if (children[i].state->visits > bestScore || (children[i].state->visits == bestScore && children[i].state->wins < children[bestAction].state->wins)) {
                bestScore = children[i].state->visits;
                bestAction = i;
            }
        }
    }

    // std::cout << bestScore << std::endl;

    if (bestAction == -1) {
        return {};
    }
    return nonstd::make_optional<std::pair<A, MCTSState<A,T>*>> (children[bestAction].action, children[bestAction].state);
}
