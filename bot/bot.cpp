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
#include <libvoxelbot/utilities/profiler.h>
#include "behaviortree/MicroNodes.h"
#include <libvoxelbot/utilities/pathfinding.h>
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/utilities/renderer.h>
#include "SDL.h"
#include "ml/mcts_sc2.h"
#include "ScoutingManager.h"
#include "behaviortree/TacticalNodes.h"
#include "BuildingPlacement.h"
#include <libvoxelbot/utilities/cereal_json.h>
#include <cereal/types/string.hpp>
#include <fstream>
#include "cereal/cereal.hpp"
#include "mcts/mcts_debugger.h"
#include "build_order_helpers.h"
#include <libvoxelbot/common/unit_lists.h>
#include <libvoxelbot/utilities/build_state_serialization.h>
#include <libvoxelbot/buildorder/tracker.h>

using Clock = std::chrono::high_resolution_clock;

using namespace BOT;
using namespace std;
using namespace sc2;

Bot* bot = nullptr;
Agent* agent = nullptr;

map<const Unit*, AvailableAbilities> availableAbilities;
map<const Unit*, AvailableAbilities> availableAbilitiesExcludingCosts;
bool mctsDebug = false;
bool autoCamera = false;
bool botDisabled = false;

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

    influenceManager.Init(&renderer);

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
    renderer = MapRenderer("Starcraft II Bot", 50, 50, 256 * 3 + 20, 256 * 4 + 30);
    renderer.present();
}

int ticks = 0;
std::future<tuple<BuildOrder, BuildState, float, vector<pair<UNIT_TYPEID,int>>, BuildOrderTracker>> currentBuildOrderFuture;
int currentBuildOrderFutureTick;
int currentBuildOrderIndex;
vector<bool> buildOrderItemsDone;
BuildOrderTracker buildOrderTracker;
vector<pair<UNIT_TYPEID,int>> lastCounter;
float currentBuildOrderTime;
BuildState lastStartingState;
float enemyScaling = 1;
BuildOrder emptyBO;
bool saveNextBO;

void Bot::clearFields() {
    availableAbilities.clear();
    availableAbilitiesExcludingCosts.clear();
    mctsDebug = false;

    ticks = 0;
    constructionPreparation.clear();
    currentBuildOrderFuture = {};
    currentBuildOrderFutureTick = 0;
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

    BuildOrderState ourBO = BuildOrderState(make_shared<BuildOrder>(buildOrderTracker.buildOrder));
    BuildOrderState enemyBO = BuildOrderState(make_shared<BuildOrder>(emptyBO));
    ourBO.buildIndex = currentBuildOrderIndex;

    vector<BuildState> startingStates(2);
    vector<BuildOrderState> buildOrders;

    int ourID = agent->Observation()->GetPlayerID();
    int opponentID = 3 - ourID;

    startingStates[ourID - 1] = ourBuildState;
    startingStates[opponentID - 1] = enemyBuildState;
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

    // If units 
    for (auto& g1 : startState.groups) {
        for (auto& g2 : startState.groups) {
            if (g1.owner != g2.owner && DistanceSquared2D(g1.pos, g2.pos) < 8*8) {
                // for (auto& u : g1.units) {
                //     timeToBeAbleToAttack(mctsSimulator->combatPredictor.defaultEnvironment, u, )
                // }
                g1.combatTime = 4;
            }
        }
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
                        assert(false);
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

MCTSSearchSC2 lastMCTSSearch;

void mctsActionListener(SimulatorUnitGroup& group, SimulatorOrder order) {
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
}

void runMCTSOffBeat(int depth) {
    if (!lastMCTSSearch.search) return;

    auto mctsState = lastMCTSSearch.search->root;

    // 2 steps per depth step because we need to skip over the opponent's turn
    // Assume the opponent took what we though was the most reasonable action
    // (todo, maybe look for the closest matching state?)
    for (int i = 0; i < depth * 2; i++) {
        auto best = mctsState->bestAction();

        // MCTS tree didn't contain anything this far. Return without taking an action
        if (!best) return;

        mctsState = best.value().second;
    }
    
    if (!mctsState->bestAction()) return;

    auto action = (MCTSAction)mctsState->bestAction().value().first;

    std::function<void(SimulatorUnitGroup&, SimulatorOrder)> listener = mctsActionListener;

    cout << "Offbeat action " << depth << " " << MCTSActionName(action) << endl;

    // Construct a new state that matches the current one?
    auto updatedState = SimulatorMCTSState(createSimulatorState(lastMCTSSearch.simulator), agent->Observation()->GetPlayerID() - 1);
    updatedState.executeAction(action, nullptr);
    updatedState.state.mergeGroups();
    updatedState.executeAction(action, &listener);
}

void runMCTS () {
    bool hasAnyMilitaryUnits = false;
    for (auto* u : bot->ourUnits()) hasAnyMilitaryUnits |= isArmy(u->unit_type);

    if (!hasAnyMilitaryUnits) {
        cout << "Skipping mcts because there are no military units" << endl;
        lastMCTSSearch = {};
        return;
    }

    cout << "Running mcts..." << endl;
    Stopwatch w1;
    Stopwatch w2;
    Point2D defaultEnemyPosition = agent->Observation()->GetGameInfo().enemy_start_locations[0];
    Point2D ourDefaultPosition = agent->Observation()->GetStartLocation();
    int ourPlayerIndex = agent->Observation()->GetPlayerID() - 1;

    array<vector<Point2D>, 2> extraDestinations = {{ {}, {} }};
    extraDestinations[ourPlayerIndex].push_back(Point2D(168/2, 168/2));
    extraDestinations[ourPlayerIndex].push_back(bot->deductionManager.sortedExpansionLocations[2]);
    extraDestinations[ourPlayerIndex].push_back(bot->deductionManager.sortedExpansionLocations[3]);

    extraDestinations[1 - ourPlayerIndex].push_back(Point2D(168/2, 168/2));
    extraDestinations[1 - ourPlayerIndex].push_back(bot->ourDeductionManager.sortedExpansionLocations[2]);
    extraDestinations[1 - ourPlayerIndex].push_back(bot->ourDeductionManager.sortedExpansionLocations[3]);


    auto mctsSimulator = make_shared<SimulatorContext>(&bot->combatPredictor, vector<Point2D>{ ourDefaultPosition, defaultEnemyPosition }, extraDestinations);
    auto state = createSimulatorState(mctsSimulator);

    w2.stop();
    MCTSSearchSC2 search = findBestActions(state, ourPlayerIndex);
    // cout << "Executing mcts action..." << endl;
    w1.stop();
    tmcts += w1.millis();
    tmctsprep += w2.millis();

    // This is a bit hacky.
    // Group merging also takes into account the groups' current destinations, so merging the groups before taking the actions will not merge all that can be merged.
    // So we first execute the actions, merge the groups and then execute the actions again with a listener
    // The executeAction method can safely be executed multiple times.
    auto best = search.search->root->bestAction();
    if (!best) {
        cout << "No possible action, one player has probably lost" << endl;
        lastMCTSSearch = {};
        return;
    }
    MCTSAction action = (MCTSAction)best.value().first;
    cout << "Primary action " << MCTSActionName(action) << endl;
    search.search->root->internalState.executeAction(action, nullptr);
    search.search->root->internalState.state.mergeGroups();
    std::function<void(SimulatorUnitGroup&, SimulatorOrder)> listener = mctsActionListener;
    search.search->root->internalState.executeAction(action, &listener);

    cout << "MCTS done " << tmcts << " " << tmctsprep << endl;

    if (debugger == nullptr) debugger = new MCTSDebugger();

    if (mctsDebug) {
        debugger->debugInteractive(*search.search);
    } else {
        // debugger->visualize(mctsState->internalState.state);
    }

    // Ensures the simulator and search data still lives, otherwise lastMCTSState.state.simulator can become a bad weak ptr
    lastMCTSSearch = move(search);
}

void Bot::refreshAbilities() {
    availableAbilities.clear();
    auto& ourUnits = this->ourUnits();
    auto abilities = Query()->GetAbilitiesForUnits(ourUnits, false);
    for (size_t i = 0; i < ourUnits.size(); i++) {
        availableAbilities[ourUnits[i]] = abilities[i];
    }

    abilities = Query()->GetAbilitiesForUnits(ourUnits, true);
    availableAbilitiesExcludingCosts.clear();
    for (size_t i = 0; i < ourUnits.size(); i++) {
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
        if (msg.message == "cam") {
            autoCamera = !autoCamera;
        }
        if (msg.message == "fow") {
            Debug()->DebugShowMap();
        }
        if (msg.message == "control") {
            Debug()->DebugEnemyControl();
        }
        if (msg.message == "tech") {
            Debug()->DebugGiveAllTech();
        }
        if (msg.message == "upgrades") {
            Debug()->DebugGiveAllUpgrades();
        }
        if (msg.message == "spawn enemy") {
            Debug()->DebugCreateUnit(UNIT_TYPEID::PROTOSS_STALKER, Observation()->GetCameraPos(), 3 - Observation()->GetPlayerID());
        }
        if (msg.message == "spawn ally") {
            Debug()->DebugCreateUnit(UNIT_TYPEID::PROTOSS_STALKER, Observation()->GetCameraPos(), Observation()->GetPlayerID());
        }
        if (msg.message == "save bo") {
            saveNextBO = true;
        }
        if (msg.message == "disable") {
            botDisabled = !botDisabled;
        }
    }

    if (ticks == 0)
        t0 = time(0);
    ticks++;
    if ((ticks % 100) == 0) {
        //cout << "FPS: " << (int)(ticks/(double)(time(0) - t0)) << endl;
    }

    // mlMovement.Tick(Observation());
    bool shouldRecalculateBuildOrder = (ticks % 20000) == 1 || ((ticks % 200) == 1 && currentBuildOrderIndex > buildOrderTracker.buildOrder.size() * 0.8f);
    if ((ticks >= currentBuildOrderFutureTick || shouldRecalculateBuildOrder) && currentBuildOrderFuture.valid()) {
        currentBuildOrderFutureTick = 100000000;
        cout << "Updating build order" << endl;
        currentBuildOrderFuture.wait();
        float buildOrderTime;
        BuildOrder currentBuildOrder;
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
        RecalculateBuildOrder();
    }

    if (!botDisabled) {
        // Run MCTS regularly or when we have just seen a large change in the game (since the last mcts)
        bool largeChange = observationChangeCheck.isLargeObservationChange(deductionManager);
        if ((ticks % 200) == 1 || largeChange) {
            if (largeChange) cout << "Detected large observation change. Running MCTS" << endl;
            runMCTS();
            observationChangeCheck.reset(deductionManager);
        }

        if ((ticks % 200) == 136) {
            runMCTSOffBeat(1);
        }
    }
    
    auto boExTuple = executeBuildOrder(Observation(), ourUnits, lastStartingState, buildOrderTracker, Observation()->GetMinerals(), spendingManager, saveNextBO);
    currentBuildOrderIndex = boExTuple.first;

    // tree->Tick();
    if (!botDisabled) {
        armyTree->Tick();
    }

    // researchTree->Tick();
    if ((ticks % 10) == 0) {
        TickMicro();
    }

    BuildState spendingManagerState(Observation(), Unit::Alliance::Self, Race::Protoss, BuildResources(Observation()->GetMinerals(), Observation()->GetVespene()), 0);
    spendingManager.OnStep(spendingManagerState);
    {
        BuildState currentState(agent->Observation(), Unit::Alliance::Self, Race::Protoss, BuildResources(agent->Observation()->GetMinerals(), agent->Observation()->GetVespene()), 0);
        debugBuildOrderMasked(currentState, buildOrderTracker.buildOrder, boExTuple.second);
    }

    influenceManager.OnStep();
    // scoutingManager->OnStep();
    // tacticalManager->OnStep();

    if (autoCamera) {
        cameraController.OnStep();
    }
    // DebugUnitPositions();

    Debug()->SendDebug();

    Actions()->SendActions();
    renderer.present();
}

void Bot::OnGameEnd() {
    string path = resultSavePath != "" ? resultSavePath : "saved_replays/latest.SC2Replay";
    Control()->SaveReplay(path);
    auto ourUnits = Observation()->GetUnits(Unit::Alliance::Self, IsStructure(Observation()));
    auto enemyUnits = Observation()->GetUnits(Unit::Alliance::Enemy, IsStructure(Observation()));
    ofstream info(path + ".result");
    if (ourUnits.size() > enemyUnits.size()) {
        info << "victory" << endl;
        cout << "Victory" << endl;
    } else {
        info << "defeat" << endl;
        cout << "Defeat" << endl;
    }
    info.close();
    renderer.shutdown();
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

void printCounter(ArmyComposition counter) {
    cout << "Best counter" << endl;
    for (auto c : counter.unitCounts) {
        cout << "\t" << UnitTypeToName(c.first) << " " << c.second << endl;
    }
    cout << "With upgrades:";
    for (auto u : counter.upgrades) {
        cout << " " << UpgradeIDToName(u);
    }
    cout << endl;
}

map<UNIT_TYPEID, int> calculateStartingUnitCounts(BuildState startingState) {
    map<UNIT_TYPEID, int> startingUnitCounts;
    for (const auto& u : startingState.units) {
        startingUnitCounts[u.type] += u.units;
    }
    for (const auto& e : startingState.events) {
        if (e.type == BuildEventType::FinishedUnit) {
            auto createdUnit = abilityToUnit(e.ability);
            if (createdUnit != UNIT_TYPEID::INVALID && !isAddon(createdUnit)) {
                startingUnitCounts[createdUnit]++;
            }
        }
    }

    return startingUnitCounts;
}

// Note: always assumes player id 1=enemy, 2=player
const CombatEnvironment* determineCurrentCombatEnvironment(const CombatPredictor& combatPredictor, CombatUpgrades upgrades, float time) {
    auto* env = &combatPredictor.combineCombatEnvironment(&combatPredictor.defaultCombatEnvironment, upgrades, 2);
    // TODO Use deduction manager to fill in what upgrades we think the enemy might reasonable have at this time?
    CombatUpgrades enemyUpgrades;
    if (time > 60*5) {
        enemyUpgrades.add(UPGRADE_ID::TERRANINFANTRYWEAPONSLEVEL1);
        enemyUpgrades.add(UPGRADE_ID::ZERGMELEEWEAPONSLEVEL1);
        enemyUpgrades.add(UPGRADE_ID::PROTOSSGROUNDWEAPONSLEVEL1);
    }
    if (time > 60*10) {
        enemyUpgrades.add(UPGRADE_ID::TERRANINFANTRYARMORSLEVEL1);
        enemyUpgrades.add(UPGRADE_ID::ZERGGROUNDARMORSLEVEL1);
        enemyUpgrades.add(UPGRADE_ID::PROTOSSGROUNDARMORSLEVEL1);
    }
    env = &combatPredictor.combineCombatEnvironment(env, enemyUpgrades, 1);
    return env;
}

void ensureMinimumNumberOfUnitsInCounter(map<UNIT_TYPEID, int>& targetUnitsCount, int minimumUnitCount) {
    for (auto& p : targetUnitsCount) {
        minimumUnitCount -= p.second;
    }

    while(minimumUnitCount > 0) {
        UNIT_TYPEID rndUnit;
        int weight = 0;
        for (auto& p : targetUnitsCount) {
            if (p.second > 0 && isArmy(p.first)) {
                weight += p.second;
                if ((rand() % weight) <= p.second) rndUnit = p.first;
            }
        }

        if (weight == 0) break;
        targetUnitsCount[rndUnit] += 1;
        minimumUnitCount--;
    }
}

vector<pair<BuildOrderItem, int>> convertCounterToBuildTarget(const map<UNIT_TYPEID, int>& startingUnits, const ArmyComposition& bestCounter) {
    map<UNIT_TYPEID, int> targetUnitsCount;

    for (auto c : bestCounter.unitCounts) {
        targetUnitsCount[c.first] += c.second * 1.0;
    }

    // Add some extra units until we are going to produce at least N units
    const int MinAdditionalUnitsToProduce = 8;

    ensureMinimumNumberOfUnitsInCounter(targetUnitsCount, MinAdditionalUnitsToProduce);

    for (auto u : startingUnits) {
        targetUnitsCount[u.first] += u.second;
        // if (isArmy(u.first)) targetUnitsCount[u.first] += u.second;
        // else targetUnitsCount[u.first] += u.second;
    }


    vector<pair<BuildOrderItem, int>> buildOrderTarget;
    for (auto p : targetUnitsCount) buildOrderTarget.push_back({ BuildOrderItem(p.first), p.second });
    for (auto u : bestCounter.upgrades) buildOrderTarget.push_back({ BuildOrderItem(u), 1 });
    return buildOrderTarget;
}

void Bot::RecalculateBuildOrder() {

    CombatState startingState;
    for (auto u : deductionManager.ApproximateArmy(1.0f)) {
        cout << "Expected enemy unit: " << UnitTypeToName(u.first) << " " << u.second << endl;
        for (int i = 0; i < u.second; i++) startingState.units.push_back(makeUnit(1, u.first));
    }

    for (auto* unit : ourUnits()) {
        if (isArmy(unit->unit_type)) {
            startingState.units.push_back(makeUnit(2, unit->unit_type));
        }
    }

    BuildState buildOrderStartingState(Observation(), Unit::Alliance::Self, Race::Protoss, BuildResources(Observation()->GetMinerals(), Observation()->GetVespene()), 0);
    map<UNIT_TYPEID, int> startingUnitCounts = calculateStartingUnitCounts(buildOrderStartingState);

    // buildOrderTracker needs to know about units that existed and were in progress when the build order was calculated
    BuildOrderTracker tracker;
    tracker.knownUnits = mAllOurUnits;
    for (auto& e : buildOrderStartingState.events) {
        if (e.type == BuildEventType::FinishedUnit) {
            UNIT_TYPEID u = abilityToUnit(e.ability);
            tracker.ignoreUnit(u, 1);
        }
    }

    // Add our current upgrades to the build order state
    startingState.environment = determineCurrentCombatEnvironment(combatPredictor, buildOrderStartingState.upgrades, ticksToSeconds(Observation()->GetGameLoop()));

    currentBuildOrderFutureTick = ticks;
    float currentTime = ticksToSeconds(Observation()->GetGameLoop());
    currentBuildOrderFuture = std::async(std::launch::async, [=]{
        // Make sure most buildings are built even though they are currently under construction.
        // The buildTimePredictor cannot take buildings under construction into account.
        cout << "Async..." << endl;

        auto* buildTimePredictorPtr = &buildTimePredictor;
#if DISABLE_PYTHON
        buildTimePredictorPtr = nullptr;
#endif

        CompositionSearchSettings settings(combatPredictor, getAvailableUnitsForRace(Race::Protoss, UnitCategory::ArmyCompositionOptions), buildTimePredictorPtr);
        settings.availableTime = 3 * 60;

        auto futureState = buildOrderStartingState;
        futureState.simulate(futureState.time + 40);
        futureState.resources = buildOrderStartingState.resources;
        auto bestCounter1 = findBestCompositionGenetic(startingState, settings, &futureState, nullptr);

        vector<pair<BuildOrderItem, int>> targetUnits1 = convertCounterToBuildTarget(startingUnitCounts, bestCounter1);

        auto currentBO = buildOrderTracker.buildOrder;
        auto buildOrder1 = findBestBuildOrderGenetic(buildOrderStartingState, targetUnits1, &currentBO);

        auto tmpStartingState = startingState;
        remove_if(tmpStartingState.units.begin(), tmpStartingState.units.end(), [](auto& unit) { return unit.owner == 2; });

        settings.availableTime = 4 * 60;
        futureState = buildOrderStartingState;
        futureState.simulateBuildOrder(buildOrder1, nullptr, false);
        auto resources = futureState.resources;
        futureState.simulate(futureState.time + 40);
        futureState.resources = resources;
        // TODO: Exclude current units here?
        auto bestCounter2 = findBestCompositionGenetic(tmpStartingState, settings, &futureState, nullptr);

        // Final counter
        auto bestCounter = bestCounter1;
        bestCounter.combine(bestCounter2);
        
        printCounter(bestCounter);

        vector<pair<BuildOrderItem, int>> targetUnits = convertCounterToBuildTarget(startingUnitCounts, bestCounter);

        /*if (currentTime < 4*60) {
            targetUnits = {
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT), 4 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_ADEPT), 4 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER), 4 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_SENTRY), 1 },
                { BuildOrderItem(UPGRADE_ID::WARPGATERESEARCH), 1 },
            };
        } else if (currentTime < 6*60) {
            targetUnits = {
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT), 8 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER), 4 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_SENTRY), 2 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL), 8 },
                { BuildOrderItem(UPGRADE_ID::PROTOSSGROUNDARMORSLEVEL1), 1 },
            };
        } else {
            targetUnits = {
                { BuildOrderItem(UPGRADE_ID::EXTENDEDTHERMALLANCE), 1 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT), 2 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_COLOSSUS), 6 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_SENTRY), 2 },
                { BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL), 8 },
            };
        }*/

        auto buildOrder = findBestBuildOrderGenetic(buildOrderStartingState, targetUnits, &currentBO);
        auto state2 = buildOrderStartingState;
        state2.simulateBuildOrder(buildOrder);
        BuildOrderTracker trackerTmp = tracker;
        trackerTmp.setBuildOrder(buildOrder);
        return make_tuple(buildOrder, buildOrderStartingState, state2.time, bestCounter.unitCounts, trackerTmp);
    });
}
