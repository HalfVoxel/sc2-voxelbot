#include "../mcts/mcts.h"
#include <vector>
#include <random>
#include <sstream>
#include <ctime>
#include <array>
#include <iostream>

using namespace std;


struct TestState {
    int player = 0;
    int state = 0;
    int count = 0;

    TestState step(int action) {
        TestState res = *this;
        res.internalStep(action);
        return res;
    }

    void internalStep(int action) {
        state = state + (action * (player == 0 ? 1 : -1));
        player = 1 - player;
        // if (action == 0) state += 100;
        count++;
    }

    vector<pair<int, float>> generateMoves() {
        vector<pair<int, float>> moves;
        for (int i = 0; i < 10; i++) {
            // moves.push_back({ i, min(1.0f, (i/45.0f) * 0.1f + 0.9f * 1/10.0f + 0.01f * (rand() % 10))});
            moves.push_back({ i, 0.5f });
        }
        return moves;
    }

    float rollout() const {
        // cout << state << endl;
        TestState res = *this;
        res.count = 0;
        while(res.count < 4) {
            res.internalStep(1 + (rand() % 9));
        }
        cout << "Eval " << state << " -> " << res.state << endl;
        return (res.state > 0) == (player == 0) ? 1 : 0;
    }

    string to_string() const {
        stringstream ss;
        ss << state;
        return ss.str();
    }
};

struct TicTacToeState {
    int player = 0;
    vector<vector<int>> state = vector<vector<int>>(3, vector<int>(3, 0));
    int count = 0;

    pair<TicTacToeState, bool> step(int action) {
        TicTacToeState res = *this;
        res.internalStep(action);
        return make_pair(res, true);
    }

    void internalStep(int action) {
        int r = action / 3;
        int c = action % 3;
        assert(state[r][c] == 0);
        state[r][c] = player + 1;
        // state = state + (action * (player == 0 ? 1 : -1));
        player = 1 - player;
        // if (action == 0) state += 100;
        count++;
    }

    int isWin() const {
        for(int r = 0; r < 3; r++) {
            if (state[r][0] == state[r][1] && state[r][0] == state[r][2] && state[r][0] != 0) {
                return state[r][0];
            }
        }

        for(int c = 0; c < 3; c++) {
            if (state[0][c] == state[1][c] && state[0][c] == state[2][c] && state[0][c] != 0) {
                return state[0][c];
            }
        }

        // Diagonal 1
        if (state[0][0] == state[1][1] && state[0][0] == state[2][2] && state[0][0] != 0) {
            return state[0][0];
        }

        // Diagonal 2
        if (state[0][2] == state[1][1] && state[0][2] == state[2][0] && state[0][2] != 0) {
            return state[0][2];
        }

        return 0;
    }

    vector<pair<int, float>> generateMoves() {
        vector<pair<int, float>> moves;
        if (isWin()) return moves;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] == 0) {
                    // moves.push_back({ i, min(1.0f, (i/45.0f) * 0.1f + 0.9f * 1/10.0f + 0.01f * (rand() % 10))});
                    moves.push_back({ i*3 + j, 0.5f });
                }
            }
        }
        return moves;
    }

    array<float, 2> rollout() const {
        TicTacToeState res = *this;
        res.count = 0;

        while(true) {
            int w = res.isWin();
            if (w != 0) return {{ (float)(int)(w == 1), (float)(int)(w == 2) }};

            int possibleActions = 0;
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    possibleActions += (res.state[r][c] == 0);

            // Tie
            if (possibleActions == 0) return {{ 0.5f, 0.5f }};

            int a = rand() % possibleActions;
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    if (res.state[r][c] == 0) {
                        if (a == 0) {
                            res.internalStep(3*r + c);
                            break;
                        }
                        a--;
                    }
                }
            }
            assert(a == 0);
        }

        assert(false);
    }

    string to_string() const {
        stringstream ss;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (state[i][j] == 0) {
                    ss << "   ";
                }
                if (state[i][j] == 1) {
                    ss << " X ";
                }
                if (state[i][j] == 2) {
                    ss << " O ";
                }
            }
            ss << endl;
        }
        return ss.str();
    }
};

int main () {
    srand(time(0));

    MCTSSearch<int, TicTacToeState> search((TicTacToeState()));
    search.search(10000);

    search.root->print(0, 2);

    MCTSState<int, TicTacToeState>* state = search.root;
    while(true) {
        auto action = state->bestAction();
        if (action) {
            cout << "Action " << action.value().first << endl;
            state = action.value().second;
            cout << state->internalState.to_string();
        } else {
            break;
        }
    }
    return 0;
}
