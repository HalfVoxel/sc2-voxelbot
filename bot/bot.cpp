#include "Bot.h"
using namespace BOT;
using namespace std;

void BOT::Bot::OnGameStart() {
    tree = std::make_unique<SequenceNode>(SequenceNode{
        new SelectorNode{},
        new SelectorNode{},
    });
}

void BOT::Bot::OnStep() {
    Status status = tree->Tick();
    int i = 1;
}
