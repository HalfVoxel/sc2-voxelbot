#include "Bot.h"
#include "UnitNodes.h"
using namespace BOT;
using namespace std;


Bot bot = Bot();

void Bot::OnGameStart() {
    tree = std::make_unique<SequenceNode>(SequenceNode{
        new BuildUnit(sc2::ABILITY_ID::TRAIN_SCV, sc2::UNIT_TYPEID::TERRAN_COMMANDCENTER)
    });
}

void Bot::OnStep() {
    tree->Tick();
}
