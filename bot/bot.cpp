#include "Bot.h"
#include "UnitNodes.h"
using namespace BOT;
using namespace std;
using namespace sc2;

Bot bot = Bot();

void Bot::OnGameStart() {
    tree = unique_ptr<TreeNode>(new ParallelNode{
        new BuildUnit(sc2::ABILITY_ID::TRAIN_SCV, sc2::UNIT_TYPEID::TERRAN_COMMANDCENTER),
        new SelectorNode {
            new Not(new ShouldBuildSupply()),
            new BuildStructure(ABILITY_ID::BUILD_SUPPLYDEPOT, UNIT_TYPEID::TERRAN_SCV)
        },
        new SequenceNode {
            new SelectorNode {
                new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS),
                new BuildStructure(ABILITY_ID::BUILD_BARRACKS, UNIT_TYPEID::TERRAN_SCV)
            },
            new SelectorNode {
                new HasUnit(UNIT_TYPEID::TERRAN_REFINERY, 1),
                new BuildGas()
            },
            new SelectorNode {
                new HasUnit(UNIT_TYPEID::TERRAN_FACTORY),
                new BuildStructure(ABILITY_ID::BUILD_FACTORY, UNIT_TYPEID::TERRAN_SCV)
            },
            new SelectorNode {
                new HasUnit(UNIT_TYPEID::TERRAN_STARPORT,1),
                new BuildStructure(ABILITY_ID::BUILD_STARPORT, UNIT_TYPEID::TERRAN_SCV)
            },
        },
        new AssignHarvesters(UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::HARVEST_GATHER, UNIT_TYPEID::TERRAN_REFINERY)
    });
}

void Bot::OnStep() {
    tree->Tick();
}
