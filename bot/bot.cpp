#include "Bot.h"
#include "UnitNodes.h"
#include "sc2lib/sc2_lib.h"
#include <cmath>        
using namespace BOT;
using namespace std;
using namespace sc2;

Bot bot = Bot();

void Bot::OnGameStart() {
    game_info_ = Observation()->GetGameInfo();
    expansions_ = search::CalculateExpansionLocations(Observation(), Query());
    startLocation_ = Observation()->GetStartLocation();
    staging_location_ = startLocation_;
    size_t size = game_info_.placement_grid.data.size(); //Placementgrid 0 == pathing grid 1
    std::vector<int> diff((size));
    for (int i = 0; i < size; ++i) {
        if (game_info_.placement_grid.data[i] == 0 && game_info_.pathing_grid.data[i] == 0) {
            diff[i] = 1;
            Point2D p = GetMapCoordinate(i);
            bot.Debug()->DebugSphereOut(Point3D(p.x + 0.5, p.y - 0.5, startLocation_.z), 0.5, Colors::Red);
        }
    }

    for (int i = 0; i < size; ++i) {
        if (diff[i] == 1) {
            for (int j = 0; j < size; ++j) {
                Point2D p = GetMapCoordinate(j);
                if (Distance2D(GetMapCoordinate(i), p) <= 2 && game_info_.placement_grid.data[j] !=
                    0) {
                    bot.Debug()->DebugSphereOut(Point3D(p.x + 0.5, p.y - 0.5, startLocation_.z),
                                                0.5, Colors::Blue);
                    diff[j] = 2;
                }
            }
        }
    }
   
    bot.Debug()->SendDebug();
    tree = unique_ptr<TreeNode>(new ParallelNode{
        new SelectorNode{
            new ShouldExpand(UNIT_TYPEID::TERRAN_REFINERY),
            new Expand(UNIT_TYPEID::TERRAN_COMMANDCENTER)
        },
        new SelectorNode{
            new HasUnit(UNIT_TYPEID::TERRAN_SCV, bot.max_worker_count_),
            new BuildUnit(UNIT_TYPEID::TERRAN_SCV),
        },
        new SelectorNode{
            new Not(new ShouldBuildSupply()),
            new BuildStructure(UNIT_TYPEID::TERRAN_SUPPLYDEPOT)
        },
        new SequenceNode{
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS),
                new BuildStructure(UNIT_TYPEID::TERRAN_BARRACKS)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_REFINERY, 1),
                new Not(new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS, 1)),
                new BuildGas(UNIT_TYPEID::TERRAN_REFINERY),
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_FACTORY),
                new BuildStructure(UNIT_TYPEID::TERRAN_FACTORY)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_STARPORT, 1),
                new BuildStructure(UNIT_TYPEID::TERRAN_STARPORT)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_BARRACKSTECHLAB, 1),
                new BuildAddon(ABILITY_ID::BUILD_TECHLAB_BARRACKS, bot.barrack_types)
            },
            new HasUnit(UNIT_TYPEID::TERRAN_COMMANDCENTER, 2),
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_REFINERY, 2),
                new BuildGas(UNIT_TYPEID::TERRAN_REFINERY)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS, 5),
                new BuildStructure(UNIT_TYPEID::TERRAN_BARRACKS),
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_FACTORYTECHLAB, 1),
                new BuildAddon(ABILITY_ID::BUILD_TECHLAB_FACTORY, bot.factory_types)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_BARRACKSREACTOR, 4),
                new BuildAddon(ABILITY_ID::BUILD_REACTOR_BARRACKS, bot.barrack_types)
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_STARPORTREACTOR, 1),
                new BuildAddon(ABILITY_ID::BUILD_REACTOR_STARPORT, bot.starport_types)
            },
        },
        new SequenceNode{
            new HasUnit(UNIT_TYPEID::TERRAN_MARINE, 20),
            new SimpleAttackMove()
        },
        new AssignHarvesters(UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::HARVEST_GATHER,
                             UNIT_TYPEID::TERRAN_REFINERY),
        new SequenceNode{
            new HasUnit(UNIT_TYPEID::TERRAN_COMMANDCENTER, 2),
            new BuildUnit(UNIT_TYPEID::TERRAN_MARAUDER),
            new BuildUnit(UNIT_TYPEID::TERRAN_MEDIVAC),
            new BuildUnit(UNIT_TYPEID::TERRAN_SIEGETANK)
        },
        new BuildUnit(UNIT_TYPEID::TERRAN_MARINE)
    });
}

int Bot::GetPositionIndex(int x, int y) {
    return x + game_info_.width * (game_info_.height-y);
}

Point2D Bot::GetMapCoordinate(int i){
    return Point2D(i % game_info_.width, game_info_.height - i / game_info_.width);
}

int Bot::ManhattanDistance(Point2D p1, Point2D p2) {
    return std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}

void Bot::OnStep() {
    tree->Tick();
}
