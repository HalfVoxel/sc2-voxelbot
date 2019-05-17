#include <fstream>
#include <iostream>
#include <queue>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "../Bot.h"
#include <libvoxelbot/combat/simulator.h>
#include "../CompositionAnalyzer.h"
#include <libvoxelbot/utilities/mappings.h>
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"
#include <random>
#include <sstream>

using namespace sc2;
using namespace std;

static const char* EmptyMap = "Test/units.SC2Map";

void printSide(const SideResult& side) {
    for (auto unitCount : side.unitCounts) {
        cout << "\t" << unitCount.count << "x " << UnitTypeToName(unitCount.type) << (isFlying(unitCount.type) ? " (flying)" : "") << endl;
    }
}

static vector<UNIT_TYPEID> unitTypesZergMilitary = {
    UNIT_TYPEID::ZERG_BANELING,
    // UNIT_TYPEID::ZERG_BROODLORD,
    UNIT_TYPEID::ZERG_CORRUPTOR,
    // UNIT_TYPEID::ZERG_DRONE,
    // UNIT_TYPEID::ZERG_EVOLUTIONCHAMBER,
    // UNIT_TYPEID::ZERG_EXTRACTOR,
    // UNIT_TYPEID::ZERG_GREATERSPIRE,
    // UNIT_TYPEID::ZERG_HATCHERY,
    // UNIT_TYPEID::ZERG_HIVE,
    UNIT_TYPEID::ZERG_HYDRALISK,
    // UNIT_TYPEID::ZERG_HYDRALISKDEN,
    // UNIT_TYPEID::ZERG_INFESTATIONPIT,
    // UNIT_TYPEID::ZERG_INFESTOR,
    // UNIT_TYPEID::ZERG_LAIR,
    // UNIT_TYPEID::ZERG_LURKERDENMP,
    // UNIT_TYPEID::ZERG_LURKERMP,
    UNIT_TYPEID::ZERG_MUTALISK,
    // UNIT_TYPEID::ZERG_NYDUSCANAL,
    // UNIT_TYPEID::ZERG_NYDUSNETWORK,
    // UNIT_TYPEID::ZERG_OVERLORD,
    // UNIT_TYPEID::ZERG_OVERLORDTRANSPORT,
    // UNIT_TYPEID::ZERG_OVERSEER,
    UNIT_TYPEID::ZERG_QUEEN,
    UNIT_TYPEID::ZERG_RAVAGER,
    UNIT_TYPEID::ZERG_ROACH,
    // UNIT_TYPEID::ZERG_ROACHWARREN,
    // UNIT_TYPEID::ZERG_SPAWNINGPOOL,
    UNIT_TYPEID::ZERG_SPINECRAWLER,
    // UNIT_TYPEID::ZERG_SPIRE,
    UNIT_TYPEID::ZERG_SPORECRAWLER,
    // UNIT_TYPEID::ZERG_SWARMHOSTMP,
    UNIT_TYPEID::ZERG_ULTRALISK,
    // UNIT_TYPEID::ZERG_ULTRALISKCAVERN,
    UNIT_TYPEID::ZERG_VIPER,
    UNIT_TYPEID::ZERG_ZERGLING,
};

static vector<UNIT_TYPEID> unitTypesTerranMilitary = {
    // UNIT_TYPEID::TERRAN_ARMORY,
    UNIT_TYPEID::TERRAN_BANSHEE,
    // UNIT_TYPEID::TERRAN_BARRACKS,
    // UNIT_TYPEID::TERRAN_BARRACKSREACTOR,
    // UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
    UNIT_TYPEID::TERRAN_BATTLECRUISER,
    // UNIT_TYPEID::TERRAN_BUNKER,
    // UNIT_TYPEID::TERRAN_COMMANDCENTER,
    UNIT_TYPEID::TERRAN_CYCLONE,
    // UNIT_TYPEID::TERRAN_ENGINEERINGBAY,
    // UNIT_TYPEID::TERRAN_FACTORY,
    // UNIT_TYPEID::TERRAN_FACTORYREACTOR,
    // UNIT_TYPEID::TERRAN_FACTORYTECHLAB,
    // UNIT_TYPEID::TERRAN_FUSIONCORE,
    UNIT_TYPEID::TERRAN_GHOST,
    // UNIT_TYPEID::TERRAN_GHOSTACADEMY,
    UNIT_TYPEID::TERRAN_HELLION,
    UNIT_TYPEID::TERRAN_HELLIONTANK,
    UNIT_TYPEID::TERRAN_LIBERATOR,
    UNIT_TYPEID::TERRAN_MARAUDER,
    UNIT_TYPEID::TERRAN_MARINE,
    UNIT_TYPEID::TERRAN_MEDIVAC,
    UNIT_TYPEID::TERRAN_MISSILETURRET,
    // UNIT_TYPEID::TERRAN_MULE,
    // UNIT_TYPEID::TERRAN_ORBITALCOMMAND,
    // UNIT_TYPEID::TERRAN_PLANETARYFORTRESS,
    UNIT_TYPEID::TERRAN_RAVEN,
    UNIT_TYPEID::TERRAN_REAPER,
    // UNIT_TYPEID::TERRAN_REFINERY,
    UNIT_TYPEID::TERRAN_SCV,
    // UNIT_TYPEID::TERRAN_SENSORTOWER,
    UNIT_TYPEID::TERRAN_SIEGETANK,
    // UNIT_TYPEID::TERRAN_STARPORT,
    // UNIT_TYPEID::TERRAN_STARPORTREACTOR,
    // UNIT_TYPEID::TERRAN_STARPORTTECHLAB,
    // UNIT_TYPEID::TERRAN_SUPPLYDEPOT,
    UNIT_TYPEID::TERRAN_THOR,
    UNIT_TYPEID::TERRAN_VIKINGFIGHTER,
    UNIT_TYPEID::TERRAN_WIDOWMINE,
};

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

CombatState sampleCombatState(default_random_engine& rnd) {
    CombatState state;
    for (int owner = 1; owner <= 2; owner++) {
        uniform_int_distribution<int> raceDist(0, 2);
        Race race = (Race)raceDist(rnd);

        vector<UNIT_TYPEID>& units = race == Race::Protoss ? unitTypesProtossMilitary : (race == Race::Zerg ? unitTypesZergMilitary : unitTypesTerranMilitary);

        int numUnits = uniform_int_distribution<int>(5, 20)(rnd);

        while(numUnits > 0) {
            int n = min(numUnits, uniform_int_distribution<int>(5, 10)(rnd));
            UNIT_TYPEID unitType = units[uniform_int_distribution<int>(0, units.size()-1)(rnd)];
            for (int i = 0; i < n; i++) state.units.push_back(makeUnit(owner, unitType));
            numUnits -= n;
        }
    }

    return state;
}

// Model
// Σ Pi + Σ PiSij
class CompositionAnalyzer2 : public sc2::Agent {
    CombatPredictor predictor;

    vector<CombatState> combatStates;
    int combatIndex = -1;
    vector<const Unit*> combatUnits;
    int waitForUnits = -1;
    float combatStartTime;

   public:
    void OnGameLoading() {
    }

    void OnGameStart() override {
        initMappings();
        BuildOptimizerNN buildTimePredictor;
        buildTimePredictor.init();
        predictor.init();
        Debug()->DebugEnemyControl();
        Debug()->DebugShowMap();

        combatStates = {
        { {
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),

            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
		} },
        { {
            makeUnit(1, UNIT_TYPEID::TERRAN_SIEGETANKSIEGED),
            makeUnit(1, UNIT_TYPEID::TERRAN_SIEGETANKSIEGED),
            makeUnit(1, UNIT_TYPEID::TERRAN_SIEGETANKSIEGED),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),

            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(2, UNIT_TYPEID::ZERG_ROACH),
		} },
        { {
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),

            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
		} },

        { {
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(1, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(1, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(1, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(1, UNIT_TYPEID::TERRAN_BANSHEE),

            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
		} },

        { {
            makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
            makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
            makeUnit(1, UNIT_TYPEID::ZERG_ROACH),

            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
		} },

        { {
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),


            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
		} },

        { {
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),


            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
		} },

        { {
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),


            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
		} },

        { {
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),


            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_MARAUDER),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
            makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
		} },
        };

        default_random_engine rnd;
        while(combatStates.size() < 100) {
            combatStates.push_back(sampleCombatState(rnd));
        }

        for (auto& u : combatStates[2].units) {
            cout << u.health << endl;
        }
    }

    unique_ptr<CombatRecorder> recorder = nullptr;
    Point2D p1 = Point2D(40 - 20, 20);
    Point2D p2 = Point2D(40 + 20, 20);
    Point2D pm = Point2D(40 - 20 + 5, 20);

    void createState(const CombatState& state, float offset = 0) {
        for (auto& u : state.units) {
            Debug()->DebugCreateUnit(u.type, u.owner == 1 ? p1 : p2, u.owner);
        }
    }

    void OnStep() override {
        if (Observation()->GetGameLoop() == 10) {
            for (auto* u : Observation()->GetUnits()) {
                Debug()->DebugKillUnit(u);
            }
            waitForUnits = 0;
        }
        
        for (auto& message : Observation()->GetChatMessages()) {
            cout << "Read message '" << message.message << "'" << endl;
            if (message.message == "rec") {
                if (recorder != nullptr) {
                    Actions()->SendChat("Finished recording");
                    recorder->finalize();
                    recorder = nullptr;
                } else {
                    Actions()->SendChat("Starting recording");
                    recorder = make_unique<CombatRecorder>();
                }
            }
        }

        if (recorder != nullptr && (Observation()->GetGameLoop() % 10) == 0) {
            recorder->tick(Observation());
        }

        if (waitForUnits == 2) {
            for (auto* u : Observation()->GetUnits()) {
                combatUnits.push_back(u);
                if (u->owner == 2) {
                    Actions()->UnitCommand(u, ABILITY_ID::ATTACK, p1);
                }
            }
            waitForUnits = 0;
            assert(combatUnits.size() > 0);

            recorder = make_unique<CombatRecorder>();
            waitForUnits = 3;
        } else if (waitForUnits == 1) {
            waitForUnits++;
        } else if (waitForUnits == 0) {
            bool anyAlive = false;
            for (auto* u : combatUnits) anyAlive |= u->is_alive;

            if ((Observation()->GetGameLoop() % 25) == 0) {
                for (auto* u : Observation()->GetUnits()) {
                    combatUnits.push_back(u);
                    Actions()->UnitCommand(u, ABILITY_ID::ATTACK, pm);
                }
            }

            if (!anyAlive || ticksToSeconds(Observation()->GetGameLoop() - combatStartTime) > 60) {
                if (recorder != nullptr) {
                    vector<CombatSettings> allSettings(6);
                    allSettings[1].enableMeleeBlocking = false;
                    allSettings[2].enableSplash = false;
                    allSettings[3].enableSurroundLimits = false;
                    allSettings[0].debug = true;
                    allSettings[4].enableTimingAdjustment = false;
                    allSettings[5].assumeReasonablePositioning = false;
                
                    stringstream ss;
                    ss << "experiment_results/combat/test" << combatIndex << "_real.csv";
                    recorder->finalize(ss.str());
                    recorder = nullptr;

                    for (size_t si = 0; si < allSettings.size(); si++) {
                        CombatRecording simRecording;
                        predictor.predict_engage(combatStates[combatIndex], allSettings[si], &simRecording, 1);
                        stringstream ss2;
                        ss2 << "experiment_results/combat/test" << combatIndex << "_sim_" << si << ".csv";
                        simRecording.writeCSV(ss2.str());
                    }
                }

                for (auto* u : Observation()->GetUnits()) {
                    Debug()->DebugKillUnit(u);
                }
                combatUnits.clear();

                combatIndex++;
                if (combatIndex >= (int)combatStates.size()) {
                    Debug()->DebugEndGame();
                    Debug()->SendDebug();
                    return;
                }
                createState(combatStates[combatIndex]);
                waitForUnits = 1;
                combatStartTime = Observation()->GetGameLoop();
            }
        } else if (waitForUnits == 3) {
            bool anyInCombat = false;
            for (const Unit* u : Observation()->GetUnits()) {
                if (u->engaged_target_tag != NullTag) anyInCombat = true;
            }

            if (anyInCombat || ticksToSeconds(Observation()->GetGameLoop() - combatStartTime) > 30) {
                for (auto* u : Observation()->GetUnits()) {
                    combatUnits.push_back(u);
                    Actions()->UnitCommand(u, ABILITY_ID::ATTACK, pm);
                }
                waitForUnits = 0;
            }
        }

        Actions()->SendActions();
        Debug()->SendDebug();
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

    coordinator.SetMultithreaded(true);

    initMappings();
    
    CompositionAnalyzer2 bot;
    agent = &bot;
    coordinator.SetParticipants({ CreateParticipant(Race::Terran, &bot) });

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    bot.OnGameLoading();
    coordinator.StartGame(EmptyMap);

    while (coordinator.Update() && !do_break) {
    }
    return 0;
}
