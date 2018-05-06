#include "Bot.h"
#include "UnitNodes.h"
#include "sc2lib/sc2_lib.h"
using namespace BOT;
using namespace std;
using namespace sc2;

Bot bot = Bot();

void Bot::OnGameStart() {
    game_info_ = Observation()->GetGameInfo();
    expansions_ = search::CalculateExpansionLocations(Observation(), Query());
    startLocation_ = Observation()->GetStartLocation();
    staging_location_ = startLocation_;

    tree = unique_ptr<TreeNode>(new ParallelNode{
        new SelectorNode{
            new HasUnit(UNIT_TYPEID::TERRAN_SCV, bot.max_worker_count_),
            new BuildUnit(UNIT_TYPEID::TERRAN_SCV),
        },
        new SelectorNode{
            new Not(new ShouldBuildSupply()),
            new BuildStructure(UNIT_TYPEID::TERRAN_SUPPLYDEPOT)
        },
        new SelectorNode{
            new ShouldExpand(UNIT_TYPEID::TERRAN_REFINERY),
            new Expand(UNIT_TYPEID::TERRAN_COMMANDCENTER)
        },
        new SequenceNode{
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_SUPPLYDEPOT),
                new BuildStructure(UNIT_TYPEID::TERRAN_SUPPLYDEPOT)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS),
                new BuildStructure(UNIT_TYPEID::TERRAN_BARRACKS)
            },
            new SelectorNode{
                new Not(new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS)),
                new HasUnit(UNIT_TYPEID::TERRAN_REFINERY, 1),
                new BuildGas(UNIT_TYPEID::TERRAN_REFINERY),
            },
            new SelectorNode{
                new Not(new HasUnit(UNIT_TYPEID::TERRAN_FACTORY)),
                new HasUnit(UNIT_TYPEID::TERRAN_REFINERY, 2),
                new BuildGas(UNIT_TYPEID::TERRAN_REFINERY)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_FACTORY),
                new BuildStructure(UNIT_TYPEID::TERRAN_FACTORY)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_STARPORT, 1),
                new BuildStructure(UNIT_TYPEID::TERRAN_STARPORT)
            },
        },
        new AssignHarvesters(UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::HARVEST_GATHER,
                             UNIT_TYPEID::TERRAN_REFINERY)
    });
}

void Bot::OnStep() {
    tree->Tick();
}
