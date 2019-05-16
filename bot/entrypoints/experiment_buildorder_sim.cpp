#include <libvoxelbot/buildorder/optimizer.h>
#include "../build_order_helpers.h"
#include "sc2lib/sc2_lib.h"
#include "../behaviortree/TacticalNodes.h"
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/buildorder/tracker.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <thread>
#include <fstream>
#include <sstream>
#include "../Bot.h"

using namespace std;
using namespace sc2;
using namespace BOT;

const char* ParaSiteLE = "Ladder/ParaSiteLE.SC2Map";

struct BuildOrderTimings {
    vector<int> buildOrder;
    vector<bool> chronoBoosted;
    vector<float> expectedTimings;
    vector<float> realTimings;
};

void compareBuildOrder(BuildState startingState, BuildOrder buildOrder, vector<float> doneTimes) {
    vector<float> expectedTimings(buildOrder.size());

    BuildState state = startingState;
    // ss << "Time: " << (int)round(buildOrderTime) << endl;
    state.simulateBuildOrder(buildOrder, [&](int i) {
        expectedTimings[i] = state.time;
        // string name = buildOrder[i].isUnitType() ? getUnitData(buildOrder[i].typeID()).name : UpgradeIDToName(buildOrder[i].upgradeID());
        // ss << (int)state.time << "\t" << doneTimes[i] << "\t" << name;
        // ss << endl;
    });

    BuildOrderTimings timings;
    timings.expectedTimings = expectedTimings;
    timings.realTimings = doneTimes;
    timings.buildOrder = vector<int>(buildOrder.size());
    timings.chronoBoosted = vector<bool>(buildOrder.size());
    for (size_t i = 0; i < buildOrder.size(); i++) {
        timings.buildOrder[i] = (int)buildOrder[i].rawType();
        timings.chronoBoosted[i] = buildOrder[i].chronoBoosted;
    }

    pybind11::module::import("experiment_buildorder_sim").attr("save")(timings.buildOrder, timings.chronoBoosted, timings.expectedTimings, timings.realTimings);
}

vector<UNIT_TYPEID> unitTypesProtossMilitary = {
    UNIT_TYPEID::PROTOSS_ADEPT,
    // UNIT_TYPEID::PROTOSS_ADEPTPHASESHIFT,
    // UNIT_TYPEID::PROTOSS_ARCHON, // TODO: Special case creation rule
    // UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_CARRIER,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    // UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
    // UNIT_TYPEID::PROTOSS_DARKSHRINE,
    UNIT_TYPEID::PROTOSS_DARKTEMPLAR,
    UNIT_TYPEID::PROTOSS_DISRUPTOR,
    // UNIT_TYPEID::PROTOSS_DISRUPTORPHASED,
    // UNIT_TYPEID::PROTOSS_FLEETBEACON,
    // UNIT_TYPEID::PROTOSS_FORGE,
    // UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_HIGHTEMPLAR,
    UNIT_TYPEID::PROTOSS_IMMORTAL,
    // UNIT_TYPEID::PROTOSS_INTERCEPTOR,
    // UNIT_TYPEID::PROTOSS_MOTHERSHIP, // TODO: Mothership cannot be created for some reason (no unit has the required ability)
    // UNIT_TYPEID::PROTOSS_MOTHERSHIPCORE,
    // UNIT_TYPEID::PROTOSS_NEXUS,
    UNIT_TYPEID::PROTOSS_OBSERVER,
    UNIT_TYPEID::PROTOSS_ORACLE,
    // UNIT_TYPEID::PROTOSS_ORACLESTASISTRAP,
    // UNIT_TYPEID::PROTOSS_PHOENIX,
    // UNIT_TYPEID::PROTOSS_PHOTONCANNON,
    UNIT_TYPEID::PROTOSS_PROBE,
    // UNIT_TYPEID::PROTOSS_PYLON,
    // UNIT_TYPEID::PROTOSS_PYLONOVERCHARGED,
    // UNIT_TYPEID::PROTOSS_ROBOTICSBAY,
    // UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
    UNIT_TYPEID::PROTOSS_SENTRY,
    // UNIT_TYPEID::PROTOSS_SHIELDBATTERY,
    UNIT_TYPEID::PROTOSS_STALKER,
    // UNIT_TYPEID::PROTOSS_STARGATE,
    UNIT_TYPEID::PROTOSS_TEMPEST,
    // UNIT_TYPEID::PROTOSS_TEMPLARARCHIVE,
    // UNIT_TYPEID::PROTOSS_TWILIGHTCOUNCIL,
    UNIT_TYPEID::PROTOSS_VOIDRAY,
    // UNIT_TYPEID::PROTOSS_WARPGATE,
    UNIT_TYPEID::PROTOSS_WARPPRISM,
    // UNIT_TYPEID::PROTOSS_WARPPRISMPHASING,
    UNIT_TYPEID::PROTOSS_ZEALOT,
};

class ExperimentBuildOrderAgent : public Bot {
    // std::shared_ptr<ControlFlowNode> behaviorTree;
    BuildState lastStartingState;
    BuildOrder currentBuildOrder;
    vector<float> doneBuildOrderTimes;
    bool hasQuit = false;
    BuildOrderTracker tracker;

   public:
    void OnGameLoading() {
    }

    void OnGameStart() override {
        hasQuit = false;

        Bot::OnGameStart();

        // initMappings();
        // BuildOptimizerNN buildTimePredictor;
        // buildTimePredictor.init();
        // predictor.init();

        // Debug()->DebugEnemyControl();
        // Debug()->DebugShowMap();

        // behaviorTree = shared_ptr<ControlFlowNode>(new ParallelNode{
        //     // new ControlSupplyDepots(),
        //     new AssignHarvesters(UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::HARVEST_GATHER, UNIT_TYPEID::TERRAN_REFINERY),
        //     new AssignHarvesters(UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::HARVEST_GATHER, UNIT_TYPEID::PROTOSS_ASSIMILATOR),
        // });

        BuildState startState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 } });
        startState.resources.minerals = 50;
        startState.resources.vespene = 0;
        
        startState.race = Race::Protoss;
        startState.chronoInfo.addNexusWithEnergy(startState.time, 50);
        // Initial delay before harvesters start mining properly
        startState.makeUnitsBusy(UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::INVALID, 12);
        for (int i = 0; i < 12; i++) startState.addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, 4, UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::INVALID));

        cout << "Finding build order " << endl;
        BuildOptimizerParams params;
        params.iterations = 512;
        params.allowChronoBoost = false;
        vector<pair<UNIT_TYPEID, int>> targetUnitCounts;

        int numUnits = 1 + (rand() % 5);
        for (int i = 0; i < numUnits; i++) {
            targetUnitCounts.push_back({ unitTypesProtossMilitary[rand() % unitTypesProtossMilitary.size()], rand() % 15 });
        }
        
        // { { UNIT_TYPEID::PROTOSS_ZEALOT, 8 }, { UNIT_TYPEID::PROTOSS_STALKER, 10 }, { UNIT_TYPEID::PROTOSS_IMMORTAL, 5 } }

        vector<pair<BuildOrderItem, int>> targetItems;
        for (auto p : targetUnitCounts) targetItems.push_back({ BuildOrderItem(p.first), p.second });

        auto boTuple = findBestBuildOrderGenetic(startState, targetItems, nullptr, params);

        lastStartingState = startState;
        currentBuildOrder = boTuple.first;

        tracker = BuildOrderTracker();
        tracker.setBuildOrder(boTuple.first);
        
        cout << "Calculated build order" << endl;

        doneBuildOrderTimes = vector<float>(currentBuildOrder.size(), -1);
    }

    void OnStep() override {
        if (hasQuit) return;

        refreshUnits();
        refreshAbilities();
        auto& ourUnits = this->ourUnits();
        auto abilities = Query()->GetAbilitiesForUnits(ourUnits, false);

        armyTree->Tick();

        bool serialize = false;
        auto doneBuildOrderActions = executeBuildOrder(Observation(), ourUnits, lastStartingState, tracker, Observation()->GetMinerals(), spendingManager, serialize).second;
        bool anyFalse = false;
        // cout << "Tick " << Observation()->GetGameLoop() << " " << ticksToSeconds(Observation()->GetGameLoop()) << endl;
        for (size_t i = 0; i < doneBuildOrderActions.size(); i++) {
            if (doneBuildOrderActions[i] && doneBuildOrderTimes[i] < 0) {
                cout << "Done at " << Observation()->GetGameLoop() << " " << ticksToSeconds(Observation()->GetGameLoop()) << endl;
                doneBuildOrderTimes[i] = ticksToSeconds(Observation()->GetGameLoop());
            }
            if (!doneBuildOrderActions[i]) anyFalse = true;
        }

        if (!anyFalse) {
            // Build order complete
            compareBuildOrder(lastStartingState, currentBuildOrder, doneBuildOrderTimes);
            Debug()->DebugEndGame();
            Debug()->SendDebug();
            Actions()->SendActions();
            hasQuit = true;
            return;
        }

        BuildState spendingManagerState(Observation(), Unit::Alliance::Self, Race::Protoss, BuildResources(Observation()->GetMinerals(), Observation()->GetVespene()), 0);
        spendingManager.OnStep(spendingManagerState);

        debugBuildOrder(lastStartingState, currentBuildOrder, doneBuildOrderTimes);

        influenceManager.OnStep();

        if ((Observation()->GetGameLoop() % 100) == 0) {
            for (auto* u : ourUnits) {
                if (isArmy(u->unit_type)) {
                    Actions()->UnitCommand(u, ABILITY_ID::ATTACK, Point2D(84, 84));
                }
            }
        }

        Debug()->SendDebug();
        Actions()->SendActions();
        
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
};

int main(int argc, char* argv[]) {
    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        print(sys.path)
        sys.path.append("bot/python")
    )");

    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    srand(time(0));

    coordinator.SetMultithreaded(true);

    bot = new ExperimentBuildOrderAgent();
    agent = bot;
    coordinator.SetParticipants({
        CreateParticipant(Race::Protoss, bot),
        CreateComputer(Race::Protoss, Difficulty::VeryEasy),
    });

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();

    for (int i = 0; i < 100; i++) {
        coordinator.StartGame(ParaSiteLE);

        while (coordinator.Update()) {
        }
        break;
    }
    return 0;
}
