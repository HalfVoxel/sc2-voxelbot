#include "Bot.h"
#include "StrategicNodes.h"
#include "sc2lib/sc2_lib.h"
// #include "sc2renderer/sc2_renderer.h"
#include "Renderer.h"
#include <cmath>
#include <iostream>
#include "TacticalNodes.h"
#include "SDL.h"
#include "Influence.h"
#include "Predicates.h"

using namespace BOT;
using namespace std;
using namespace sc2;

Bot bot = Bot();
InfluenceMap pathing_grid;
InfluenceMap placement_grid;
InfluenceMap mp;
InfluenceMap mp2;

void Bot::OnGameStart() {
    game_info_ = Observation()->GetGameInfo();
    expansions_ = search::CalculateExpansionLocations(Observation(), Query());
    startLocation_ = Observation()->GetStartLocation();
    staging_location_ = startLocation_;
    buildingPlacement.OnGameStart();

    tactical_manager = new TacticalManager(buildingPlacement.wallPlacement);
    tree = unique_ptr<TreeNode>(new ParallelNode{
        new SequenceNode{
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
        new AssignHarvesters(UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::HARVEST_GATHER,
                             UNIT_TYPEID::TERRAN_REFINERY),
        new SequenceNode{
            new Not(new ShouldExpand(UNIT_TYPEID::TERRAN_REFINERY)),
            new SequenceNode{
                new HasUnit(UNIT_TYPEID::TERRAN_COMMANDCENTER, 2),
                new BuildUnit(UNIT_TYPEID::TERRAN_MARAUDER),
                new BuildUnit(UNIT_TYPEID::TERRAN_MEDIVAC),
                new BuildUnit(UNIT_TYPEID::TERRAN_SIEGETANK)
            }

        },
        new SequenceNode{
            new Not(new ShouldExpand(UNIT_TYPEID::TERRAN_REFINERY)),
            new BuildUnit(UNIT_TYPEID::TERRAN_MARINE)
        }
    });

    armyTree = unique_ptr<TreeNode>(new ParallelNode{
        new ControlSupplyDepots(),
        new SimpleArmyPosition(),
        new SequenceNode{
            new HasUnit(UNIT_TYPEID::TERRAN_MARINE, 40),
            new SimpleAttackMove()
        },
    });

    pathing_grid = InfluenceMap(game_info_.pathing_grid);
    placement_grid = InfluenceMap(game_info_.placement_grid);
    mp = InfluenceMap(pathing_grid.w, pathing_grid.h);
    mp2 = InfluenceMap(pathing_grid.w, pathing_grid.h);
}

time_t t0;

void DebugUnitPositions() {
    Units units = bot.Observation()->GetUnits(Unit::Alliance::Self);
    for (auto unit : units) {
        bot.Debug()->DebugSphereOut(unit->pos, 0.5, Colors::Green);
    }
}

void Bot::OnGameLoading() {
    InitializeRenderer("Starcraft II Bot", 50, 50, 600, 512);
    Render();
}

int ticks = 0;
void Bot::OnStep() {
    if (ticks == 0) t0 = time(0);
    ticks++;
    if ((ticks % 100) == 0) {
        cout << "FPS: " << (int)(ticks/(double)(time(0) - t0)) << endl;
    }
    tree->Tick();
    armyTree->Tick();

    const int InfluenceFrameInterval = 10;
    if ((ticks % InfluenceFrameInterval) == 0) {
        for (auto unit : Observation()->GetUnits(Unit::Alliance::Self)) {
            if (IsStructure(Observation())(*unit)) {
                mp.addInfluence(0.5, unit->pos);
                mp2.addInfluence(0.5, unit->pos);
            } else {
                mp.addInfluence(0.2, unit->pos);
                mp2.addInfluence(0.2, unit->pos);
            }
        }
        mp.propagateSum(exp(-4), 1.0, pathing_grid);

        mp2.propagateMax(0.2, 0.3, pathing_grid);

        mp.render(0, 0, 2);
        mp2.render(mp.w*2+5, 0, 2);
        Render();
    }

    cameraController.OnStep();

    // DebugUnitPositions();
    Debug()->SendDebug();
}

void Bot::OnGameEnd() {
    Shutdown();
}

void Bot::OnUnitDestroyed(const Unit* unit) {
    tactical_manager->OnUnitDestroyed(unit);
}

void Bot::OnUnitCreated(const Unit* unit) {
    tactical_manager->OnUnitCreated(unit);
}

void Bot::OnNydusDetected() {
    tactical_manager->OnNydusDetected();
}

void Bot::OnNuclearLaunchDetected() {
    tactical_manager->OnNuclearLaunchDetected();
}

void Bot::OnUnitEnterVision(const Unit* unit) {
    tactical_manager->OnUnitEnterVision(unit);
}

int Bot::GetPositionIndex(int x, int y) {
    return x + game_info_.width * (game_info_.height - y);
}

Point2D Bot::GetMapCoordinate(int i) {
    return Point2D(i % game_info_.width, game_info_.height - i / game_info_.width);
}

int Bot::ManhattanDistance(Point2D p1, Point2D p2) {
    return std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}


