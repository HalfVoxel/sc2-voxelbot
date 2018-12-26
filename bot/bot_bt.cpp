#if FALSE
#include "Bot.h"
#include "behaviortree/StrategicNodes.h"
#include "sc2lib/sc2_lib.h"
// #include "sc2renderer/sc2_renderer.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include "behaviortree/MicroNodes.h"
#include "utilities/pathfinding.h"
#include "utilities/predicates.h"
#include "utilities/renderer.h"
#include "SDL.h"
#include "ScoutingManager.h"
#include "behaviortree/TacticalNodes.h"
#include "buildingPlacement.h"

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <fstream>
#include "cereal/cereal.hpp"

using Clock = std::chrono::high_resolution_clock;

using namespace BOT;
using namespace std;
using namespace sc2;

Bot bot = Bot();
Agent& agent = bot;
map<const Unit*, AvailableAbilities> availableAbilities;
map<const Unit*, AvailableAbilities> availableAbilitiesExcludingCosts;

// TODO: Should move this to a better place
bool IsAbilityReady(const Unit* unit, ABILITY_ID ability) {
    for (auto& a : availableAbilities[unit].abilities) {
        if (a.ability_id == ability)
            return true;
    }
    return false;
}

bool IsAbilityReadyExcludingCosts(const Unit* unit, ABILITY_ID ability) {
    for (auto& a : availableAbilitiesExcludingCosts[unit].abilities) {
        if (a.ability_id == ability)
            return true;
    }
    return false;
}

// clang-format off
void Bot::OnGameStart() {
    // Debug()->DebugEnemyControl();
    // Debug()->DebugShowMap();
    // Debug()->DebugGiveAllTech();
    // Debug()->DebugGiveAllUpgrades();

    initMappings(Observation());
    deductionManager = DeductionManager();

    game_info_ = Observation()->GetGameInfo();
    expansions_ = search::CalculateExpansionLocations(Observation(), Query());
    startLocation_ = Observation()->GetStartLocation();
    staging_location_ = startLocation_;
    dependencyAnalyzer.analyze();
    deductionManager.OnGameStart();
    ourDeductionManager.OnGameStart();
    buildingPlacement.OnGameStart();
    tree = unique_ptr<TreeNode>(new ParallelNode{
        new SequenceNode{
            new ShouldExpand(UNIT_TYPEID::TERRAN_REFINERY),
            new Expand(UNIT_TYPEID::TERRAN_COMMANDCENTER, [](auto) { return 5; })
        },
        new SelectorNode{
            new HasUnit(UNIT_TYPEID::TERRAN_ORBITALCOMMAND, 2),
            new Build(UNIT_TYPEID::TERRAN_ORBITALCOMMAND, [](auto) { return 11; }),
        },
        new SelectorNode{
            new HasUnit(UNIT_TYPEID::TERRAN_SCV, bot.max_worker_count_),
            new Build(UNIT_TYPEID::TERRAN_SCV, SCVScore),
        },
        new SelectorNode{
            new Not(new ShouldBuildSupply()),
            new Construct(UNIT_TYPEID::TERRAN_SUPPLYDEPOT, [](auto) { return 9; })
        },
        new SequenceNode{
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS),
                new Construct(UNIT_TYPEID::TERRAN_BARRACKS, [](auto) { return 2; })
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_REFINERY, 1),
                new Not(new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS, 1)),
                new BuildGas(UNIT_TYPEID::TERRAN_REFINERY, [](auto) { return 2; }),
            },
            new HasUnit(UNIT_TYPEID::TERRAN_COMMANDCENTER, 2),
            new ParallelNode{
                    new SelectorNode{
                            new HasUnit(UNIT_TYPEID::TERRAN_BARRACKSTECHLAB, 1),
                            new Addon(ABILITY_ID::BUILD_TECHLAB_BARRACKS, bot.barrack_types, [](auto) { return 6; })
                    },
                    new SelectorNode{
                            new HasUnit(UNIT_TYPEID::TERRAN_FACTORY),
                            new Construct(UNIT_TYPEID::TERRAN_FACTORY, [](auto) { return 2; })
                    },
                    new SelectorNode{
                            new HasUnit(UNIT_TYPEID::TERRAN_STARPORT, 1),
                            new Construct(UNIT_TYPEID::TERRAN_STARPORT, [](auto) { return 2; })
                    }
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_REFINERY, 2),
                new BuildGas(UNIT_TYPEID::TERRAN_REFINERY, [](auto) { return 2; })
            },
            new SelectorNode{
                new HasUnit(UNIT_TYPEID::TERRAN_BARRACKS, 5),
                new Construct(UNIT_TYPEID::TERRAN_BARRACKS, [](auto) { return 0.5; }),
            },
            new ParallelNode{
                    new SelectorNode{
                            new HasUnit(UNIT_TYPEID::TERRAN_FACTORYTECHLAB, 1),
                            new Addon(ABILITY_ID::BUILD_TECHLAB_FACTORY, bot.factory_types, [](auto) { return 2; })
                    },
                    new SelectorNode{
                            new HasUnit(UNIT_TYPEID::TERRAN_BARRACKSREACTOR, 4),
                            new Addon(ABILITY_ID::BUILD_REACTOR_BARRACKS, bot.barrack_types, [](auto) { return 2; })
                    },
                    new SelectorNode{
                            new HasUnit(UNIT_TYPEID::TERRAN_STARPORTREACTOR, 1),
                            new Addon(ABILITY_ID::BUILD_REACTOR_STARPORT, bot.starport_types, [](auto) { return 2; })
                    },
            },
        },
        new AssignHarvesters(UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::HARVEST_GATHER,
                             UNIT_TYPEID::TERRAN_REFINERY),
        new ParallelNode{
            new Build(UNIT_TYPEID::TERRAN_MARAUDER, DefaultScore),
            new Build(UNIT_TYPEID::TERRAN_MEDIVAC, DefaultScore),
            new Build(UNIT_TYPEID::TERRAN_SIEGETANK, DefaultScore),
            new Build(UNIT_TYPEID::TERRAN_MARINE, DefaultScore),
            new Build(UNIT_TYPEID::TERRAN_CYCLONE, DefaultScore),
            new Build(UNIT_TYPEID::TERRAN_LIBERATOR, DefaultScore),
            new Build(UNIT_TYPEID::TERRAN_VIKINGFIGHTER, DefaultScore),
            new Build(UNIT_TYPEID::TERRAN_BANSHEE, DefaultScore),
        }
    });
   
    armyTree = unique_ptr<ControlFlowNode>(new ParallelNode{
        new ControlSupplyDepots()
    });

    researchTree = shared_ptr<ControlFlowNode>(new ParallelNode{
        new SequenceNode{new HasUnit(UNIT_TYPEID::TERRAN_BARRACKSTECHLAB),
                         new Research(UPGRADE_ID::STIMPACK, [](auto) { return 9; })
        }
    });

    tacticalManager = new TacticalManager(armyTree ,buildingPlacement.wallPlacement);
    scoutingManager = new ScoutingManager();

    influenceManager.Init();
}
// clang-format on

time_t t0;

void DebugUnitPositions() {
    Units units = bot.Observation()->GetUnits(Unit::Alliance::Self);
    for (auto unit : units) {
        bot.Debug()->DebugSphereOut(unit->pos, 0.5, Colors::Green);
    }
}

void Bot::OnGameLoading() {
    InitializeRenderer("Starcraft II Bot", 50, 50, 256 * 3 + 20, 256 * 4 + 30);
    Render();
}

int ticks = 0;
bool test = false;
set<BuffID> seenBuffs;
uint32_t lastEffectID;

void Bot::OnStep() {
    auto ourUnits = agent.Observation()->GetUnits(Unit::Alliance::Self);
    auto enemyUnits = agent.Observation()->GetUnits(Unit::Alliance::Enemy);
    auto abilities = agent.Query()->GetAbilitiesForUnits(ourUnits, false);

    deductionManager.Observe(enemyUnits);
    ourDeductionManager.Observe(ourUnits);

    availableAbilities.clear();
    for (int i = 0; i < ourUnits.size(); i++) {
        availableAbilities[ourUnits[i]] = abilities[i];
    }

    for (int i = 0; i < enemyUnits.size(); i++) {
        vector<BuffID> buffs = enemyUnits[i]->buffs;
        for (auto buff : buffs) {
            if (seenBuffs.find(buff) == seenBuffs.end()) {
                seenBuffs.insert(buff);
                cout << "Observed buff: " << BuffIDToName(buff) << endl;
            }
        }

        if (enemyUnits[i]->is_burrowed) {
            cout << "Found out an enemy is burrowed!!!" << endl;
        }
    }

    const auto& effectDatas = agent.Observation()->GetEffectData();
    for (auto& eff : agent.Observation()->GetEffects()) {
        if (lastEffectID != eff.effect_id) {
            lastEffectID = eff.effect_id;
            const EffectData& effectData = effectDatas[eff.effect_id];
            cout << "Seen effect: " << effectData.friendly_name << endl;
        }
    }

    abilities = agent.Query()->GetAbilitiesForUnits(ourUnits, true);
    availableAbilitiesExcludingCosts.clear();
    for (int i = 0; i < ourUnits.size(); i++) {
        availableAbilitiesExcludingCosts[ourUnits[i]] = abilities[i];
    }

    if (ticks == 0)
        t0 = time(0);
    ticks++;
    if ((ticks % 100) == 0) {
        //cout << "FPS: " << (int)(ticks/(double)(time(0) - t0)) << endl;
    }
    tree->Tick();
    armyTree->Tick();
    researchTree->Tick();
    if ((ticks % 10) == 0) {
        TickMicro();
    }

    spendingManager.OnStep();
    influenceManager.OnStep();
    scoutingManager->OnStep();
    tacticalManager->OnStep();
    // cameraController.OnStep();
    // DebugUnitPositions();
    Debug()->SendDebug();
}

void Bot::OnGameEnd() {
    auto ourUnits = Observation()->GetUnits(Unit::Alliance::Self, IsStructure(Observation()));
    auto enemyUnits = Observation()->GetUnits(Unit::Alliance::Enemy, IsStructure(Observation()));
    if (ourUnits.size() > enemyUnits.size()) {
        cout << "Victory" << endl;
    } else {
        cout << "Defeat" << endl;
    }
    Shutdown();
}

void Bot::OnUnitDestroyed(const Unit* unit) {
    tacticalManager->OnUnitDestroyed(unit);
}

void Bot::OnUnitCreated(const Unit* unit) {
    tacticalManager->OnUnitCreated(unit);
}

void Bot::OnNydusDetected() {
    tacticalManager->OnNydusDetected();
}

void Bot::OnNuclearLaunchDetected() {
    tacticalManager->OnNuclearLaunchDetected();
}

void Bot::OnUnitEnterVision(const Unit* unit) {
    tacticalManager->OnUnitEnterVision(unit);
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

#endif