#include "Mappings.h"
#include <iostream>
#include "sc2api/sc2_api.h"

using namespace std;
using namespace sc2;

/** Maps an ability to the unit that primarily uses it.
 * In particular this is defined for BUILD_* and TRAIN_* abilities.
 */
std::vector<UNIT_TYPEID> abilityToCasterUnit(ABILITY_ID ability) {
    switch (ability) {
        case ABILITY_ID::RESEARCH_HISECAUTOTRACKING:
        case ABILITY_ID::RESEARCH_TERRANSTRUCTUREARMORUPGRADE:
        case ABILITY_ID::RESEARCH_TERRANINFANTRYWEAPONS:
        case ABILITY_ID::RESEARCH_TERRANINFANTRYARMOR:
            return { UNIT_TYPEID::TERRAN_ENGINEERINGBAY };
        case ABILITY_ID::RESEARCH_INFERNALPREIGNITER:
        case ABILITY_ID::RESEARCH_RAPIDFIRELAUNCHERS:
        case ABILITY_ID::RESEARCH_SMARTSERVOS:
        case ABILITY_ID::RESEARCH_DRILLINGCLAWS:
        case ABILITY_ID::RESEARCH_BANSHEEHYPERFLIGHTROTORS:
            return { UNIT_TYPEID::TERRAN_FACTORYTECHLAB };
        case ABILITY_ID::RESEARCH_STIMPACK:
        case ABILITY_ID::RESEARCH_COMBATSHIELD:
        case ABILITY_ID::RESEARCH_CONCUSSIVESHELLS:
            return { UNIT_TYPEID::TERRAN_BARRACKSTECHLAB };
        case ABILITY_ID::RESEARCH_TERRANSHIPWEAPONS:
        case ABILITY_ID::RESEARCH_TERRANVEHICLEANDSHIPPLATING:
        case ABILITY_ID::RESEARCH_TERRANVEHICLEWEAPONS:
            return { UNIT_TYPEID::TERRAN_ARMORY };
        case ABILITY_ID::RESEARCH_HIGHCAPACITYFUELTANKS:
        case ABILITY_ID::RESEARCH_RAVENCORVIDREACTOR:
        case ABILITY_ID::RESEARCH_ADVANCEDBALLISTICS:
        case ABILITY_ID::RESEARCH_BANSHEECLOAKINGFIELD:
            return { UNIT_TYPEID::TERRAN_STARPORTTECHLAB };
        case ABILITY_ID::RESEARCH_BATTLECRUISERWEAPONREFIT:
            return { UNIT_TYPEID::TERRAN_FUSIONCORE };
        case ABILITY_ID::RESEARCH_PERSONALCLOAKING:
            return { UNIT_TYPEID::TERRAN_GHOSTACADEMY };
        case ABILITY_ID::MORPH_PLANETARYFORTRESS:
        case ABILITY_ID::MORPH_ORBITALCOMMAND:
            return { UNIT_TYPEID::TERRAN_COMMANDCENTER };
        case ABILITY_ID::BUILD_ARMORY:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_ASSIMILATOR:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_BANELINGNEST:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_BARRACKS:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_BUNKER:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_COMMANDCENTER:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_CREEPTUMOR:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_CREEPTUMOR_QUEEN:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_CREEPTUMOR_TUMOR:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_CYBERNETICSCORE:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_DARKSHRINE:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_ENGINEERINGBAY:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_EVOLUTIONCHAMBER:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_EXTRACTOR:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_FACTORY:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_FLEETBEACON:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_FORGE:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_FUSIONCORE:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_GATEWAY:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_GHOSTACADEMY:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_HATCHERY:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_HYDRALISKDEN:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_INFESTATIONPIT:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_INTERCEPTORS:
            return { UNIT_TYPEID::PROTOSS_CARRIER };
        case ABILITY_ID::BUILD_MISSILETURRET:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_NEXUS:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_NUKE:
            return { UNIT_TYPEID::TERRAN_GHOSTACADEMY };
        case ABILITY_ID::BUILD_NYDUSNETWORK:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_NYDUSWORM:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_PHOTONCANNON:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_PYLON:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_REACTOR:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_REACTOR_BARRACKS:
            return { UNIT_TYPEID::TERRAN_BARRACKS };
        case ABILITY_ID::BUILD_REACTOR_FACTORY:
            return { UNIT_TYPEID::TERRAN_FACTORY };
        case ABILITY_ID::BUILD_REACTOR_STARPORT:
            return { UNIT_TYPEID::TERRAN_STARPORT };
        case ABILITY_ID::BUILD_REFINERY:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_ROACHWARREN:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_ROBOTICSBAY:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_ROBOTICSFACILITY:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_SENSORTOWER:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_SHIELDBATTERY:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_SPAWNINGPOOL:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_SPINECRAWLER:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_SPIRE:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_SPORECRAWLER:
            return { UNIT_TYPEID::ZERG_DRONE };
        case ABILITY_ID::BUILD_STARGATE:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_STARPORT:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_STASISTRAP:
            return { UNIT_TYPEID::PROTOSS_ORACLE };
        case ABILITY_ID::BUILD_SUPPLYDEPOT:
            return { UNIT_TYPEID::TERRAN_SCV };
        case ABILITY_ID::BUILD_TECHLAB:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_TECHLAB_BARRACKS:
            return { UNIT_TYPEID::TERRAN_BARRACKS };
        case ABILITY_ID::BUILD_TECHLAB_FACTORY:
            return { UNIT_TYPEID::TERRAN_FACTORY };
        case ABILITY_ID::BUILD_TECHLAB_STARPORT:
            return { UNIT_TYPEID::TERRAN_STARPORT };
        case ABILITY_ID::BUILD_TEMPLARARCHIVE:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_TWILIGHTCOUNCIL:
            return { UNIT_TYPEID::PROTOSS_PROBE };
        case ABILITY_ID::BUILD_ULTRALISKCAVERN:
            return { UNIT_TYPEID::ZERG_DRONE };

        case ABILITY_ID::TRAINWARP_ADEPT:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAINWARP_DARKTEMPLAR:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAINWARP_HIGHTEMPLAR:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAINWARP_SENTRY:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAINWARP_STALKER:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAINWARP_ZEALOT:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAIN_ADEPT:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAIN_BANELING:
            return { UNIT_TYPEID::ZERG_BANELING };
        case ABILITY_ID::TRAIN_BANSHEE:
            return { UNIT_TYPEID::TERRAN_STARPORTTECHLAB };
        case ABILITY_ID::TRAIN_BATTLECRUISER:
            return { UNIT_TYPEID::TERRAN_STARPORTTECHLAB };
        case ABILITY_ID::TRAIN_CARRIER:
            return { UNIT_TYPEID::PROTOSS_STARGATE };
        case ABILITY_ID::TRAIN_COLOSSUS:
            return { UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY };
        case ABILITY_ID::TRAIN_CORRUPTOR:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_CYCLONE:
            return { UNIT_TYPEID::TERRAN_FACTORY };
        case ABILITY_ID::TRAIN_DARKTEMPLAR:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAIN_DISRUPTOR:
            return { UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY };
        case ABILITY_ID::TRAIN_DRONE:
            return { UNIT_TYPEID::PROTOSS_NEXUS };
        case ABILITY_ID::TRAIN_GHOST:
            return { UNIT_TYPEID::TERRAN_BARRACKSTECHLAB };
        case ABILITY_ID::TRAIN_HELLBAT:
            return { UNIT_TYPEID::TERRAN_FACTORY };
        case ABILITY_ID::TRAIN_HELLION:
            return { UNIT_TYPEID::TERRAN_FACTORY };
        case ABILITY_ID::TRAIN_HIGHTEMPLAR:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAIN_HYDRALISK:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_IMMORTAL:
            return { UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY };
        case ABILITY_ID::TRAIN_INFESTOR:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_LIBERATOR:
            return { UNIT_TYPEID::TERRAN_STARPORT };
        case ABILITY_ID::TRAIN_MARAUDER:
            return { UNIT_TYPEID::TERRAN_BARRACKS };
        case ABILITY_ID::TRAIN_MARINE:
            return { UNIT_TYPEID::TERRAN_BARRACKS };
        case ABILITY_ID::TRAIN_MEDIVAC:
            return { UNIT_TYPEID::TERRAN_STARPORT };
        case ABILITY_ID::TRAIN_MOTHERSHIP:
            return { UNIT_TYPEID::PROTOSS_NEXUS };
        case ABILITY_ID::TRAIN_MOTHERSHIPCORE:
            return { UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY };
        case ABILITY_ID::TRAIN_MUTALISK:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_OBSERVER:
            return { UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY };
        case ABILITY_ID::TRAIN_ORACLE:
            return { UNIT_TYPEID::PROTOSS_STARGATE };
        case ABILITY_ID::TRAIN_OVERLORD:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_PHOENIX:
            return { UNIT_TYPEID::PROTOSS_STARGATE };
        case ABILITY_ID::TRAIN_PROBE:
            return { UNIT_TYPEID::PROTOSS_NEXUS };
        case ABILITY_ID::TRAIN_QUEEN:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_RAVEN:
            return { UNIT_TYPEID::TERRAN_STARPORT };
        case ABILITY_ID::TRAIN_REAPER:
            return { UNIT_TYPEID::TERRAN_BARRACKS };
        case ABILITY_ID::TRAIN_ROACH:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_SCV:
            return { UNIT_TYPEID::TERRAN_COMMANDCENTER, UNIT_TYPEID::TERRAN_ORBITALCOMMAND, UNIT_TYPEID::TERRAN_PLANETARYFORTRESS };
        case ABILITY_ID::TRAIN_SENTRY:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAIN_SIEGETANK:
            return { UNIT_TYPEID::TERRAN_FACTORY };
        case ABILITY_ID::TRAIN_STALKER:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAIN_SWARMHOST:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_TEMPEST:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_THOR:
            return { UNIT_TYPEID::TERRAN_FACTORYTECHLAB };
        case ABILITY_ID::TRAIN_ULTRALISK:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_VIKINGFIGHTER:
            return { UNIT_TYPEID::TERRAN_STARPORT };
        case ABILITY_ID::TRAIN_VIPER:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_VOIDRAY:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::TRAIN_WARPPRISM:
            return { UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY };
        case ABILITY_ID::TRAIN_WIDOWMINE:
            return { UNIT_TYPEID::TERRAN_FACTORYREACTOR };
        case ABILITY_ID::TRAIN_ZEALOT:
            return { UNIT_TYPEID::PROTOSS_GATEWAY };
        case ABILITY_ID::TRAIN_ZERGLING:
            return { UNIT_TYPEID::INVALID };
        default:
            cerr << AbilityTypeToName(ability) << endl;
            throw invalid_argument("Not a train or build ability");
    }
}

/** Maps an ability to the unit that is built or trained by that ability.
 * In particular this is defined for BUILD_* abilities.
 */
UNIT_TYPEID abilityToUnit(ABILITY_ID ability) {
    switch (ability) {
        case ABILITY_ID::BUILD_ARMORY:
            return UNIT_TYPEID::TERRAN_ARMORY;
        case ABILITY_ID::BUILD_ASSIMILATOR:
            return UNIT_TYPEID::PROTOSS_ASSIMILATOR;
        case ABILITY_ID::BUILD_BANELINGNEST:
            return UNIT_TYPEID::ZERG_BANELINGNEST;
        case ABILITY_ID::BUILD_BARRACKS:
            return UNIT_TYPEID::TERRAN_BARRACKS;
        case ABILITY_ID::BUILD_BUNKER:
            return UNIT_TYPEID::TERRAN_BUNKER;
        case ABILITY_ID::BUILD_COMMANDCENTER:
            return UNIT_TYPEID::TERRAN_COMMANDCENTER;
        case ABILITY_ID::BUILD_CREEPTUMOR:
            return UNIT_TYPEID::ZERG_CREEPTUMOR;
        case ABILITY_ID::BUILD_CREEPTUMOR_QUEEN:
            return UNIT_TYPEID::ZERG_CREEPTUMORQUEEN;
        case ABILITY_ID::BUILD_CREEPTUMOR_TUMOR:
            return UNIT_TYPEID::ZERG_CREEPTUMOR;
        case ABILITY_ID::BUILD_CYBERNETICSCORE:
            return UNIT_TYPEID::PROTOSS_CYBERNETICSCORE;
        case ABILITY_ID::BUILD_DARKSHRINE:
            return UNIT_TYPEID::PROTOSS_DARKSHRINE;
        case ABILITY_ID::BUILD_ENGINEERINGBAY:
            return UNIT_TYPEID::TERRAN_ENGINEERINGBAY;
        case ABILITY_ID::BUILD_EVOLUTIONCHAMBER:
            return UNIT_TYPEID::ZERG_EVOLUTIONCHAMBER;
        case ABILITY_ID::BUILD_EXTRACTOR:
            return UNIT_TYPEID::ZERG_EXTRACTOR;
        case ABILITY_ID::BUILD_FACTORY:
            return UNIT_TYPEID::TERRAN_FACTORY;
        case ABILITY_ID::BUILD_FLEETBEACON:
            return UNIT_TYPEID::PROTOSS_FLEETBEACON;
        case ABILITY_ID::BUILD_FORGE:
            return UNIT_TYPEID::PROTOSS_FORGE;
        case ABILITY_ID::BUILD_FUSIONCORE:
            return UNIT_TYPEID::TERRAN_FUSIONCORE;
        case ABILITY_ID::BUILD_GATEWAY:
            return UNIT_TYPEID::PROTOSS_GATEWAY;
        case ABILITY_ID::BUILD_GHOSTACADEMY:
            return UNIT_TYPEID::TERRAN_GHOSTACADEMY;
        case ABILITY_ID::BUILD_HATCHERY:
            return UNIT_TYPEID::ZERG_HATCHERY;
        case ABILITY_ID::BUILD_HYDRALISKDEN:
            return UNIT_TYPEID::ZERG_HYDRALISKDEN;
        case ABILITY_ID::BUILD_INFESTATIONPIT:
            return UNIT_TYPEID::ZERG_INFESTATIONPIT;
        case ABILITY_ID::BUILD_INTERCEPTORS:
            return UNIT_TYPEID::PROTOSS_INTERCEPTOR;
        case ABILITY_ID::BUILD_MISSILETURRET:
            return UNIT_TYPEID::TERRAN_MISSILETURRET;
        case ABILITY_ID::BUILD_NEXUS:
            return UNIT_TYPEID::PROTOSS_NEXUS;
        case ABILITY_ID::BUILD_NUKE:
            return UNIT_TYPEID::TERRAN_NUKE;
        case ABILITY_ID::BUILD_NYDUSNETWORK:
            return UNIT_TYPEID::ZERG_NYDUSNETWORK;
        case ABILITY_ID::BUILD_NYDUSWORM:
            return UNIT_TYPEID::ZERG_NYDUSCANAL;
        case ABILITY_ID::BUILD_PHOTONCANNON:
            return UNIT_TYPEID::PROTOSS_PHOTONCANNON;
        case ABILITY_ID::BUILD_PYLON:
            return UNIT_TYPEID::PROTOSS_PYLON;
        case ABILITY_ID::BUILD_REACTOR:
            return UNIT_TYPEID::TERRAN_REACTOR;
        case ABILITY_ID::BUILD_REACTOR_BARRACKS:
            return UNIT_TYPEID::TERRAN_BARRACKSREACTOR;
        case ABILITY_ID::BUILD_REACTOR_FACTORY:
            return UNIT_TYPEID::TERRAN_FACTORYREACTOR;
        case ABILITY_ID::BUILD_REACTOR_STARPORT:
            return UNIT_TYPEID::TERRAN_STARPORTREACTOR;
        case ABILITY_ID::BUILD_REFINERY:
            return UNIT_TYPEID::TERRAN_REFINERY;
        case ABILITY_ID::BUILD_ROACHWARREN:
            return UNIT_TYPEID::ZERG_ROACHWARREN;
        case ABILITY_ID::BUILD_ROBOTICSBAY:
            return UNIT_TYPEID::PROTOSS_ROBOTICSBAY;
        case ABILITY_ID::BUILD_ROBOTICSFACILITY:
            return UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY;
        case ABILITY_ID::BUILD_SENSORTOWER:
            return UNIT_TYPEID::TERRAN_SENSORTOWER;
        case ABILITY_ID::BUILD_SHIELDBATTERY:
            return UNIT_TYPEID::PROTOSS_SHIELDBATTERY;
        case ABILITY_ID::BUILD_SPAWNINGPOOL:
            return UNIT_TYPEID::ZERG_SPAWNINGPOOL;
        case ABILITY_ID::BUILD_SPINECRAWLER:
            return UNIT_TYPEID::ZERG_SPINECRAWLER;
        case ABILITY_ID::BUILD_SPIRE:
            return UNIT_TYPEID::ZERG_SPIRE;
        case ABILITY_ID::BUILD_SPORECRAWLER:
            return UNIT_TYPEID::ZERG_SPORECRAWLER;
        case ABILITY_ID::BUILD_STARGATE:
            return UNIT_TYPEID::PROTOSS_STARGATE;
        case ABILITY_ID::BUILD_STARPORT:
            return UNIT_TYPEID::TERRAN_STARPORT;
        case ABILITY_ID::BUILD_STASISTRAP:
            return UNIT_TYPEID::PROTOSS_ORACLESTASISTRAP;
        case ABILITY_ID::BUILD_SUPPLYDEPOT:
            return UNIT_TYPEID::TERRAN_SUPPLYDEPOT;
        case ABILITY_ID::BUILD_TECHLAB:
            return UNIT_TYPEID::TERRAN_TECHLAB;
        case ABILITY_ID::BUILD_TECHLAB_BARRACKS:
            return UNIT_TYPEID::TERRAN_BARRACKSTECHLAB;
        case ABILITY_ID::BUILD_TECHLAB_FACTORY:
            return UNIT_TYPEID::TERRAN_FACTORYTECHLAB;
        case ABILITY_ID::BUILD_TECHLAB_STARPORT:
            return UNIT_TYPEID::TERRAN_STARPORTTECHLAB;
        case ABILITY_ID::BUILD_TEMPLARARCHIVE:
            return UNIT_TYPEID::PROTOSS_TEMPLARARCHIVE;
        case ABILITY_ID::BUILD_TWILIGHTCOUNCIL:
            return UNIT_TYPEID::PROTOSS_TWILIGHTCOUNCIL;
        case ABILITY_ID::BUILD_ULTRALISKCAVERN:
            return UNIT_TYPEID::ZERG_ULTRALISKCAVERN;
        default:
            return UNIT_TYPEID::INVALID;
    }
}

UNIT_TYPEID simplifyUnitType(UNIT_TYPEID type) {
    // TODO: Extend
    switch (type) {
        case UNIT_TYPEID::TERRAN_SUPPLYDEPOTLOWERED:
            return UNIT_TYPEID::TERRAN_SUPPLYDEPOT;
        case UNIT_TYPEID::TERRAN_BARRACKSFLYING:
        case UNIT_TYPEID::TERRAN_BARRACKSREACTOR:
        case UNIT_TYPEID::TERRAN_BARRACKSTECHLAB:
            return UNIT_TYPEID::TERRAN_BARRACKS;
        case UNIT_TYPEID::TERRAN_FACTORYFLYING:
        case UNIT_TYPEID::TERRAN_FACTORYREACTOR:
        case UNIT_TYPEID::TERRAN_FACTORYTECHLAB:
            return UNIT_TYPEID::TERRAN_FACTORY;
        case UNIT_TYPEID::TERRAN_STARPORTFLYING:
        case UNIT_TYPEID::TERRAN_STARPORTREACTOR:
        case UNIT_TYPEID::TERRAN_STARPORTTECHLAB:
            return UNIT_TYPEID::TERRAN_STARPORT;
        case UNIT_TYPEID::TERRAN_COMMANDCENTER:
        case UNIT_TYPEID::TERRAN_COMMANDCENTERFLYING:
        case UNIT_TYPEID::TERRAN_ORBITALCOMMAND:
        case UNIT_TYPEID::TERRAN_ORBITALCOMMANDFLYING:
        case UNIT_TYPEID::TERRAN_PLANETARYFORTRESS:
            return UNIT_TYPEID::TERRAN_COMMANDCENTER;
        case UNIT_TYPEID::TERRAN_LIBERATORAG:
            return UNIT_TYPEID::TERRAN_LIBERATOR;
        default:
            return type;
    }
}
