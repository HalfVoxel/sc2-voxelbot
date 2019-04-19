#include "Bot.h"
#include "behaviortree/StrategicNodes.h"
#include "sc2lib/sc2_lib.h"
// #include "sc2renderer/sc2_renderer.h"
#include <chrono>
#include <future>
#include <thread>
#include <cmath>
#include <iostream>
//#include <pybind11/embed.h>
//#include <pybind11/stl.h>
#include <limits>
#include <map>
#include <random>
#include "ml/simulator.h"
#include "ml/simulator_context.h"
#include "utilities/profiler.h"
#include "behaviortree/MicroNodes.h"
#include "utilities/pathfinding.h"
#include "utilities/predicates.h"
#include "utilities/renderer.h"
#include "SDL.h"
#include "ml/mcts_sc2.h"
#include "ScoutingManager.h"
#include "behaviortree/TacticalNodes.h"
#include "BuildingPlacement.h"
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <fstream>
#include "cereal/cereal.hpp"
#include "mcts/mcts_debugger.h"
#include "build_order_helpers.h"

using Clock = std::chrono::high_resolution_clock;

using namespace BOT;
using namespace std;
using namespace sc2;

Bot* bot = nullptr;
Agent* agent = nullptr;

map<const Unit*, AvailableAbilities> availableAbilities;
map<const Unit*, AvailableAbilities> availableAbilitiesExcludingCosts;
bool mctsDebug = false;

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

void runMCTS();

// clang-format off
void Bot::OnGameStart() {
    clearFields();
    // Debug()->DebugEnemyControl();
    // Debug()->DebugShowMap();
    // Debug()->DebugGiveAllTech();
    // Debug()->DebugGiveAllUpgrades();

#if !DISABLE_PYTHON
    buildTimePredictor.init();
#endif
    initMappings(Observation());
    deductionManager = DeductionManager();

    game_info_ = Observation()->GetGameInfo();
    expansions_ = search::CalculateExpansionLocations(Observation(), Query());
    startLocation_ = Observation()->GetStartLocation();
    staging_location_ = startLocation_;
    dependencyAnalyzer.analyze();
    int ourID = Observation()->GetPlayerID();
    int opponentID = 3 - ourID;
    assert(opponentID == 1 || opponentID == 2);
    deductionManager.OnGameStart(opponentID);
    ourDeductionManager.OnGameStart(ourID);
    buildingPlacement.OnGameStart();
    
#if !DISABLE_PYTHON
    mlMovement.OnGameStart();
#endif
    scoutingManager = new ScoutingManager();

    influenceManager.Init();

    combatPredictor.init();

    armyTree = shared_ptr<ControlFlowNode>(new ParallelNode{
        new ControlSupplyDepots(),
        new AssignHarvesters(UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::HARVEST_GATHER, UNIT_TYPEID::TERRAN_REFINERY),
        new AssignHarvesters(UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::HARVEST_GATHER, UNIT_TYPEID::PROTOSS_ASSIMILATOR),
    });
    tacticalManager = new TacticalManager(armyTree, buildingPlacement.wallPlacement);
}
// clang-format on

time_t t0;

void DebugUnitPositions() {
    Units units = agent->Observation()->GetUnits(Unit::Alliance::Self);
    for (auto unit : units) {
        agent->Debug()->DebugSphereOut(unit->pos, 0.5, Colors::Green);
    }
}

void Bot::OnGameLoading() {
    InitializeRenderer("Starcraft II Bot", 50, 50, 256 * 3 + 20, 256 * 4 + 30);
    Render();
}

int ticks = 0;
std::future<tuple<BuildOrder, BuildState, float, vector<pair<UNIT_TYPEID,int>>, BuildOrderTracker>> currentBuildOrderFuture;
int currentBuildOrderFutureTick;
BuildOrder currentBuildOrder;
int currentBuildOrderIndex;
vector<bool> buildOrderItemsDone;
BuildOrderTracker buildOrderTracker;
vector<pair<UNIT_TYPEID,int>> lastCounter;
float currentBuildOrderTime;
BuildState lastStartingState;
float enemyScaling = 1;
BuildOrder emptyBO;

void Bot::clearFields() {
    availableAbilities.clear();
    availableAbilitiesExcludingCosts.clear();
    mctsDebug = false;

    ticks = 0;
    constructionPreparation.clear();
    currentBuildOrderFuture = {};
    currentBuildOrderFutureTick = 0;
    currentBuildOrder = BuildOrder();
    currentBuildOrderIndex = 0;
    buildOrderItemsDone.clear();
    lastCounter.clear();
    currentBuildOrderTime = 0;
    buildOrderTracker = {};
    lastStartingState = {};
    enemyScaling = 1;

    mOurUnits.clear();
    mNeutralUnits.clear();
    mEnemyUnits.clear();
    mAllOurUnits.clear();
    constructionPreparation.clear();
    wallPlacements.clear();
}

SimulatorState createSimulatorState(shared_ptr<SimulatorContext> mctsSimulator) {
    auto ourUnits = agent->Observation()->GetUnits(Unit::Alliance::Self);
    auto enemyUnits = agent->Observation()->GetUnits(Unit::Alliance::Enemy);

    vector<pair<CombatUnit, Point2D>> enemyUnitPositions = bot->deductionManager.SampleUnitPositions(1);

    BuildState ourBuildState(agent->Observation(), Unit::Alliance::Self, Race::Protoss, BuildResources(agent->Observation()->GetMinerals(), agent->Observation()->GetVespene()), 0);
    BuildState enemyBuildState;
    for (auto& u : enemyUnitPositions) {
        enemyBuildState.addUnits(u.first.type, 1);
    }
    enemyBuildState.time = ourBuildState.time;

    BuildOrderState ourBO = BuildOrderState(make_shared<BuildOrder>(currentBuildOrder));
    BuildOrderState enemyBO = BuildOrderState(make_shared<BuildOrder>(emptyBO));
    ourBO.buildIndex = currentBuildOrderIndex;

    vector<shared_ptr<const BuildState>> startingStates(2);
    vector<BuildOrderState> buildOrders;

    int ourID = agent->Observation()->GetPlayerID();
    int opponentID = 3 - ourID;

    startingStates[ourID - 1] = mctsSimulator->cache.copyState(ourBuildState);
    startingStates[opponentID - 1] = mctsSimulator->cache.copyState(enemyBuildState);
    buildOrders.push_back(ourID == 1 ? ourBO : enemyBO);
    buildOrders.push_back(ourID == 1 ? enemyBO : ourBO);

    SimulatorState startState(mctsSimulator, startingStates, buildOrders);
    for (auto& u : enemyUnitPositions) {
        startState.addUnit(u.first, u.second);
    }
    // TODO blacklist projectiles and such
    for (auto* u : ourUnits) {
        if (isAddon(u->unit_type)) continue;
        if (u->build_progress < 1) continue;

        startState.addUnit(u);
    }

    for (int k = 1; k <= 2; k++) {
        BuildState validation = k == ourID ? ourBuildState : enemyBuildState;
        map<UNIT_TYPEID, int> unitCounts;
        for (auto& g : startState.groups) {
            if (g.owner == k) {
                for (auto& u : g.units) {
                    unitCounts[u.combat.type] += 1;
                }
            }
        }

        for (auto p : unitCounts) {
            for (auto& u : validation.units) {
                if (u.type == p.first) {
                    if (u.units != p.second) {
                        cerr << "Mismatch in unit counts " << UnitTypeToName(u.type) << " " << u.units << " " << p.second << endl;
                        cerr << "For player " << k << endl;
                        for (auto& u2 : validation.units) {
                            cout << "Has " << UnitTypeToName(u2.type) << " " << u2.units << endl;
                        }
                        for (auto p : unitCounts) {
                            cout << "Expected " << UnitTypeToName(p.first) << " " << p.second << endl;
                        }
                        exit(1);
                    }
                }
            }
        }
    }

    return startState;
};

float tmcts;
float tmctsprep;

float tRollout = 0;
float tExpand = 0;
float tSelect = 0;

MCTSDebugger* debugger;

void runMCTS () {
    cout << "Running mcts..." << endl;
    Stopwatch w1;
    Stopwatch w2;
    Point2D defaultEnemyPosition = agent->Observation()->GetGameInfo().enemy_start_locations[0];
    Point2D ourDefaultPosition = agent->Observation()->GetStartLocation();
    auto mctsSimulator = make_shared<SimulatorContext>(&bot->combatPredictor, vector<Point2D>{ ourDefaultPosition, defaultEnemyPosition });
    auto state = createSimulatorState(mctsSimulator);
    int ourPlayerIndex = agent->Observation()->GetPlayerID() - 1;
    w2.stop();
    std::unique_ptr<MCTSState<int, SimulatorMCTSState>> mctsState = findBestActions(state, ourPlayerIndex);
    // cout << "Executing mcts action..." << endl;
    w1.stop();
    tmcts += w1.millis();
    tmctsprep += w2.millis();

    std::function<void(SimulatorUnitGroup&, SimulatorOrder)> listener = [&](SimulatorUnitGroup& group, SimulatorOrder order) {
        vector<const Unit*> realUnits;
        MCTSGroup* tacticalGroup = (MCTSGroup*)bot->tacticalManager->CreateGroup(GroupType::MCTS);
        for (auto& u : group.units) {
            if (!isFakeTag(u.tag)) {
                const Unit* realUnit = agent->Observation()->GetUnit(u.tag);
                if (realUnit != nullptr) {
                    realUnits.push_back(realUnit);
                } else {
                    cerr << "Could not find unit with tag " << u.tag << endl;
                }
            }
        }

        tacticalGroup->target = Point2DI((int)order.target.x, (int)order.target.y);
        bot->tacticalManager->TransferUnits(realUnits, tacticalGroup);
        // agent->Actions()->UnitCommand(realUnits, ABILITY_ID::ATTACK, order.target);
    };

    // This is a bit hacky.
    // Group merging also takes into account the groups' current destinations, so merging the groups before taking the actions will not merge all that can be merged.
    // So we first execute the actions, merge the groups and then execute the actions again with a listener
    // The executeAction method can safely be executed multiple times.
    mctsState->internalState.executeAction((MCTSAction)mctsState->bestAction().value().first, nullptr);
    mctsState->internalState.state.mergeGroups();
    mctsState->internalState.executeAction((MCTSAction)mctsState->bestAction().value().first, &listener);
    cout << "MCTS done " << tmcts << " " << tmctsprep << endl;

    if (debugger == nullptr) debugger = new MCTSDebugger();

    if (mctsDebug) {
        debugger->debugInteractive(&*mctsState);
    } else {
        // debugger->visualize(mctsState->internalState.state);
    }
}

void Bot::refreshAbilities() {
    availableAbilities.clear();
    auto& ourUnits = this->ourUnits();
    auto abilities = Query()->GetAbilitiesForUnits(ourUnits, false);
    for (int i = 0; i < ourUnits.size(); i++) {
        availableAbilities[ourUnits[i]] = abilities[i];
    }

    abilities = Query()->GetAbilitiesForUnits(ourUnits, true);
    availableAbilitiesExcludingCosts.clear();
    for (int i = 0; i < ourUnits.size(); i++) {
        availableAbilitiesExcludingCosts[ourUnits[i]] = abilities[i];
    }
}

void Bot::OnStep() {
    refreshUnits();
    auto& ourUnits = this->ourUnits();
    auto& enemyUnits = this->enemyUnits();
    refreshAbilities();


    deductionManager.Observe(enemyUnits);
    ourDeductionManager.Observe(ourUnits);

    for (auto& msg : Observation()->GetChatMessages()) {
        if (msg.message == "mcts") {
            mctsDebug = !mctsDebug;
        }
    }

    if (ticks == 0)
        t0 = time(0);
    ticks++;
    if ((ticks % 100) == 0) {
        //cout << "FPS: " << (int)(ticks/(double)(time(0) - t0)) << endl;
    }

    // mlMovement.Tick(Observation());
    bool shouldRecalculateBuildOrder = (ticks % 20000) == 1 || ((ticks % 200) == 1 && currentBuildOrderIndex > currentBuildOrder.size() * 0.8f);
    if ((ticks >= currentBuildOrderFutureTick || shouldRecalculateBuildOrder) && currentBuildOrderFuture.valid()) {
        currentBuildOrderFutureTick = 100000000;
        cout << "Updating build order" << endl;
        currentBuildOrderFuture.wait();
        float buildOrderTime;
        tie(currentBuildOrder, lastStartingState, buildOrderTime, lastCounter, buildOrderTracker) = currentBuildOrderFuture.get();
        currentBuildOrderTime = buildOrderTime;

        if (buildOrderTime < 120) {
            enemyScaling = min(10.0f, enemyScaling * 1.2f);
        } else if (buildOrderTime > 400) {
            enemyScaling = max(1.0f, enemyScaling * 0.9f);
        }

        if (currentBuildOrder.size() < 5) {
            enemyScaling = min(10.0f, enemyScaling * 1.2f);
        } else if (currentBuildOrder.size() > 20) {
            enemyScaling = max(1.0f, enemyScaling * 0.9f);
        }
        cout << "Enemy scaling " << enemyScaling << endl;
        
        /*currentBuildOrder = {
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_PYLON,
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_GATEWAY,
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_GATEWAY,
            UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_ASSIMILATOR,
            UNIT_TYPEID::PROTOSS_PYLON,
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_PYLON,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_PYLON,
            UNIT_TYPEID::PROTOSS_PROBE,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_STARGATE,
            UNIT_TYPEID::PROTOSS_PYLON,
            UNIT_TYPEID::PROTOSS_ASSIMILATOR,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_PHOENIX,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_PYLON,
            UNIT_TYPEID::PROTOSS_PHOENIX,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_PHOENIX,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
            UNIT_TYPEID::PROTOSS_ZEALOT,
            UNIT_TYPEID::PROTOSS_FLEETBEACON,
            UNIT_TYPEID::PROTOSS_PYLON,
            UNIT_TYPEID::PROTOSS_OBSERVER,
            UNIT_TYPEID::PROTOSS_CARRIER,
            UNIT_TYPEID::PROTOSS_IMMORTAL,
        };*/
    }

    if (shouldRecalculateBuildOrder) {
    // if ((ticks == 1 || ticks == 2)) {
        CombatState startingState;
        for (auto u : deductionManager.ApproximateArmy(1.5f)) {
            cout << "Expected enemy unit: " << UnitTypeToName(u.first) << " " << u.second << endl;
            for (int i = 0; i < u.second; i++) startingState.units.push_back(makeUnit(1, u.first));
        }

        /*auto knownEnemyUnits = deductionManager.GetKnownUnits();
        for (auto u : knownEnemyUnits) {
            for (int i = 0; i < u.second; i++) {
                startingState.units.push_back(makeUnit(1, u.first));
            }
        }
        
        for (int i = 0; i < 10; i++) {
            startingState.units.push_back(makeUnit(1, UNIT_TYPEID::ZERG_ROACH));
        }
        for (int i = 0; i < 10; i++) {
            startingState.units.push_back(makeUnit(1, UNIT_TYPEID::ZERG_ZERGLING));
        }
        for (int i = 0; i < 3; i++) {
            startingState.units.push_back(makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK));
        }*/

        for (int i = 0; i < ourUnits.size(); i++) {
            if (isArmy(ourUnits[i]->unit_type)) {
                startingState.units.push_back(makeUnit(2, ourUnits[i]->unit_type));
            }
        }

        map<UNIT_TYPEID, int> targetUnitsCount;
        BuildState buildOrderStartingState(Observation(), Unit::Alliance::Self, Race::Protoss, BuildResources(Observation()->GetMinerals(), Observation()->GetVespene()), 0);
        for (const auto& u : buildOrderStartingState.units) {
            targetUnitsCount[u.type] += u.units;
        }
        for (const auto& e : buildOrderStartingState.events) {
            if (e.type == BuildEventType::FinishedUnit) {
                auto createdUnit = abilityToUnit(e.ability);
                if (createdUnit != UNIT_TYPEID::INVALID && !isAddon(createdUnit)) {
                    targetUnitsCount[createdUnit]++;
                }
            }
        }

        for (auto& b : buildOrderStartingState.baseInfos) {
            cout << "Base minerals " << b.remainingMinerals << endl;
        }

        // buildOrderTracker needs to know about units that existed and were in progress when the build order was calculated
        BuildOrderTracker tracker;
        tracker.knownUnits = mAllOurUnits;
        for (auto& e : buildOrderStartingState.events) {
            if (e.type == BuildEventType::FinishedUnit) {
                UNIT_TYPEID u = abilityToUnit(e.ability);
                tracker.ignoreUnit(u, 1);
            }
        }

        currentBuildOrderFutureTick = ticks;
        currentBuildOrderFuture = std::async(std::launch::async, [=]{
            // return make_tuple(vector<UNIT_TYPEID>(0), buildOrderStartingState, 0.0f, vector<pair<UNIT_TYPEID,int>>(0));
            // Make sure most buildings are built even though they are currently under construction.
            // The buildTimePredictor cannot take buildings under construction into account.
            cout << "Async..." << endl;
            auto futureState = buildOrderStartingState;
            futureState.simulate(futureState.time + 40);
            futureState.resources = buildOrderStartingState.resources;

            Stopwatch watch;
            map<UNIT_TYPEID, int> targetUnitsCount2;
            for (auto u : targetUnitsCount) {
                if (isArmy(u.first)) targetUnitsCount2[u.first] = u.second;
                else targetUnitsCount2[u.first] = u.second;
            }

            cout << "Finding best composition" << endl;
            auto* buildTimePredictorPtr = &buildTimePredictor;
#if DISABLE_PYTHON
            buildTimePredictorPtr = nullptr;
#endif

            auto bestCounter = findBestCompositionGenetic(combatPredictor, availableUnitTypesProtoss, startingState, buildTimePredictorPtr, &futureState, &lastCounter);

            // auto bestCounter = findBestCompositionGenetic(combatPredictor, availableUnitTypesProtoss, startingState, nullptr, &futureState, &lastCounter);
            /*vector<pair<UNIT_TYPEID, int>> bestCounter = {
                { UNIT_TYPEID::TERRAN_MARINE, 30 },
                { UNIT_TYPEID::TERRAN_SIEGETANK, 10 },
            };*/

            cout << "Best counter" << endl;
            for (auto c : bestCounter) {
                cout << "\t" << UnitTypeToName(c.first) << " " << c.second << endl;
            }

            int totalTargetCount = 0;
            for (auto c : bestCounter) {
                targetUnitsCount2[c.first] += c.second * 2;
                totalTargetCount += c.second * 2;
            }

            // Add some extra units until we are going to produce at least N units
            const int MinAdditionalUnitsToProduce = 8;
            while(totalTargetCount < MinAdditionalUnitsToProduce) {
                UNIT_TYPEID rndUnit;
                int weight = 0;
                for (auto& p : targetUnitsCount2) {
                    if (p.second > 0 && isArmy(p.first)) {
                        weight += p.second;
                        if ((rand() % weight) <= p.second) rndUnit = p.first;
                    }
                }

                if (weight == 0) break;
                targetUnitsCount2[rndUnit] += 1;
                totalTargetCount += 1;
            }

            vector<pair<UNIT_TYPEID, int>> targetUnits;
            for (auto p : targetUnitsCount2) targetUnits.push_back(p);

            cout << "Additional units to build" << endl;
            for (auto p : targetUnitsCount2) {
                int delta = p.second;
                if (targetUnitsCount.count(p.first)) delta -= targetUnitsCount.at(p.first);
                if (delta != 0) {
                    cout << UnitTypeToName(p.first) << ": " << delta << endl;
                }
            }
            cout << "---" << endl;

            auto buildOrder = findBestBuildOrderGenetic(buildOrderStartingState, targetUnits, &currentBuildOrder);
            auto state2 = buildOrderStartingState;
            state2.simulateBuildOrder(buildOrder);
            watch.stop();
            cout << "Time " << watch.millis() << endl;
            BuildOrderTracker trackerTmp = tracker;
            trackerTmp.setBuildOrder(buildOrder);
            return make_tuple(buildOrder, buildOrderStartingState, state2.time, bestCounter, trackerTmp);
        });
    }

    if ((ticks % 200) == 1) {
        runMCTS();
    }

    auto boExTuple = executeBuildOrder(ourUnits, lastStartingState, buildOrderTracker, Observation()->GetMinerals(), spendingManager);
    currentBuildOrderIndex = boExTuple.first;

    // tree->Tick();
    armyTree->Tick();
    // researchTree->Tick();
    if ((ticks % 10) == 0) {
        TickMicro();
    }

    BuildState spendingManagerState(Observation(), Unit::Alliance::Self, Race::Protoss, BuildResources(Observation()->GetMinerals(), Observation()->GetVespene()), 0);
    spendingManager.OnStep(spendingManagerState);
    debugBuildOrder(lastStartingState, currentBuildOrder, boExTuple.second);
    influenceManager.OnStep();
    // scoutingManager->OnStep();
    tacticalManager->OnStep();

    // cameraController.OnStep();
    // DebugUnitPositions();

    Debug()->SendDebug();

    Actions()->SendActions();
}

void Bot::OnGameEnd() {
    Control()->SaveReplay("saved_replays/latest.SC2Replay");
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
