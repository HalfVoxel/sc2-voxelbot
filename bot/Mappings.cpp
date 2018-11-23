#include "Mappings.h"
#include <iostream>
#include "Predicates.h"
#include "generated/abilities.h"
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_api.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "stdutils.h"

using namespace std;
using namespace sc2;

vector<vector<UNIT_TYPEID>> mAbilityToCasterUnit;
vector<UNIT_TYPEID> mAbilityToCreatedUnit;
vector<vector<UNIT_TYPEID>> mCanBecome;
vector<vector<UNIT_TYPEID>> mHasBeen;
vector<vector<ABILITY_ID>> mUnitTypeHasAbilities;
bool mappingInitialized = false;

UNIT_TYPEID canonicalize(UNIT_TYPEID unitType) {
    const auto& unitTypes = bot.Observation()->GetUnitTypeData();
    const UnitTypeData& unitTypeData = unitTypes[(int)unitType];

    // Use canonical representation (e.g SUPPLY_DEPOT instead of SUPPLY_DEPOT_LOWERED)
    if (unitTypeData.unit_alias != UNIT_TYPEID::INVALID) {
        return unitTypeData.unit_alias;
    }
    return unitType;
}

const vector<UNIT_TYPEID>& hasBeen(sc2::UNIT_TYPEID type) {
    return mHasBeen[(int)type];
}

const vector<UNIT_TYPEID>& canBecome(sc2::UNIT_TYPEID type) {
    return mCanBecome[(int)type];
}

const vector<ABILITY_ID>& unitAbilities(UNIT_TYPEID type) {
    return mUnitTypeHasAbilities[(int)type];
}

void initMappings(const ObservationInterface* observation) {
    if (mappingInitialized)
        return;
    mappingInitialized = true;

    cout << "A" << endl;
    int maxAbilityID = 0;
    for (auto pair : unit_type_has_ability) {
        maxAbilityID = max(maxAbilityID, pair.second);
    }
    mAbilityToCasterUnit = vector<vector<UNIT_TYPEID>>(maxAbilityID + 1);
    for (auto pair : unit_type_has_ability) {
        mAbilityToCasterUnit[pair.second].push_back((UNIT_TYPEID)pair.first);
    }

    cout << "B" << endl;

    const sc2::UnitTypes& unitTypes = observation->GetUnitTypeData();
    const auto abilities = observation->GetAbilityData();

    mUnitTypeHasAbilities = vector<vector<ABILITY_ID>>(unitTypes.size());
    for (auto pair : unit_type_has_ability) {
        mUnitTypeHasAbilities[pair.first].push_back((ABILITY_ID)pair.second);
    }

    mAbilityToCreatedUnit = vector<UNIT_TYPEID>(abilities.size(), UNIT_TYPEID::INVALID);
    for (auto type : unitTypes) {
        mAbilityToCreatedUnit[type.ability_id] = (UNIT_TYPEID)type.unit_type_id;
    }

    cout << "C" << endl;

    mCanBecome = vector<vector<UNIT_TYPEID>>(unitTypes.size());
    mHasBeen = vector<vector<UNIT_TYPEID>>(unitTypes.size());
    for (int i = 0; i < unitTypes.size(); i++) {
        const UnitTypeData& unitTypeData = unitTypes[i];
        mHasBeen[i].push_back(unitTypeData.unit_type_id);

        if (unitTypeData.unit_alias != UNIT_TYPEID::INVALID)
            continue;

        // Handle building upgrades
        if (isStructure(unitTypeData)) {
            auto currentTypeData = unitTypeData;
            while (true) {
                auto createdByAlternatives = abilityToCasterUnit(currentTypeData.ability_id);
                if (createdByAlternatives.size() == 0 || createdByAlternatives[0] == UNIT_TYPEID::INVALID) {
                    cout << "Cannot determine how " << currentTypeData.name << " was created (ability: " << AbilityTypeToName(currentTypeData.ability_id) << endl;
                    break;
                }
                UNIT_TYPEID createdBy = createdByAlternatives[0];

                currentTypeData = unitTypes[(int)createdBy];
                if (isStructure(currentTypeData) || createdBy == UNIT_TYPEID::ZERG_DRONE) {
                    if (contains(mHasBeen[i], createdBy)) {
                        cout << "Loop in dependency tree for " << currentTypeData.name << endl;
                        break;
                    }
                    mHasBeen[i].push_back(createdBy);

                    if (createdBy == UNIT_TYPEID::ZERG_DRONE) {
                        break;
                    }
                } else {
                    break;
                }
            }
        } else {
            // Special case the relevant units
            switch ((UNIT_TYPEID)unitTypeData.unit_type_id) {
                case UNIT_TYPEID::ZERG_RAVAGER:
                    mHasBeen[i].push_back(UNIT_TYPEID::ZERG_ROACH);
                    break;
                case UNIT_TYPEID::ZERG_OVERSEER:
                    mHasBeen[i].push_back(UNIT_TYPEID::ZERG_OVERLORD);
                    break;
                case UNIT_TYPEID::ZERG_BANELING:
                    mHasBeen[i].push_back(UNIT_TYPEID::ZERG_ZERGLING);
                    break;
                case UNIT_TYPEID::ZERG_BROODLORD:
                    mHasBeen[i].push_back(UNIT_TYPEID::ZERG_CORRUPTOR);
                    break;
                case UNIT_TYPEID::ZERG_LURKERMP:
                    mHasBeen[i].push_back(UNIT_TYPEID::ZERG_HYDRALISK);
                    break;
                    // TODO: Tech labs?
                default:
                    break;
            }
        }

        for (UNIT_TYPEID previous : mHasBeen[i]) {
            auto& canBecomeArr = mCanBecome[(int)previous];
            if (previous != (UNIT_TYPEID)i && find(canBecomeArr.begin(), canBecomeArr.end(), unitTypeData.unit_type_id) == canBecomeArr.end()) {
                canBecomeArr.push_back(unitTypeData.unit_type_id);
            }
        }
    }

    cout << "D" << endl;

    for (int i = 0; i < unitTypes.size(); i++) {
        if (mCanBecome[i].size() == 0)
            continue;

        // Sort by lowest cost first
        sort(mCanBecome[i].begin(), mCanBecome[i].end(), [&](UNIT_TYPEID a, UNIT_TYPEID b) {
            return unitTypes[(int)a].mineral_cost + unitTypes[(int)a].vespene_cost < unitTypes[(int)b].mineral_cost + unitTypes[(int)b].vespene_cost;
        });
        for (auto p : mCanBecome[i])
            cout << UnitTypeToName(p) << " (" << unitTypes[(int)p].mineral_cost << ", " << unitTypes[(int)p].vespene_cost << "), ";
        cout << endl;
    }

    cout << "E" << endl;
}

/** Maps an ability to the unit that primarily uses it.
 * In particular this is defined for BUILD_* and TRAIN_* abilities.
 */
std::vector<UNIT_TYPEID> abilityToCasterUnit(ABILITY_ID ability) {
    switch (ability) {
        case ABILITY_ID::BUILD_TECHLAB:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_TECHLAB_BARRACKS:
            return { UNIT_TYPEID::TERRAN_BARRACKS };
        case ABILITY_ID::BUILD_TECHLAB_FACTORY:
            return { UNIT_TYPEID::TERRAN_FACTORY };
        case ABILITY_ID::BUILD_TECHLAB_STARPORT:
            return { UNIT_TYPEID::TERRAN_STARPORT };
        case ABILITY_ID::BUILD_REACTOR:
            return { UNIT_TYPEID::INVALID };
        case ABILITY_ID::BUILD_REACTOR_BARRACKS:
            return { UNIT_TYPEID::TERRAN_BARRACKS };
        case ABILITY_ID::BUILD_REACTOR_FACTORY:
            return { UNIT_TYPEID::TERRAN_FACTORY };
        case ABILITY_ID::BUILD_REACTOR_STARPORT:
            return { UNIT_TYPEID::TERRAN_STARPORT };
        default:
            break;
    }

    if ((int)ability >= mAbilityToCasterUnit.size()) {
        return { UNIT_TYPEID::INVALID };
    }
    return mAbilityToCasterUnit[(int)ability];
    /*
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
            return { UNIT_TYPEID::PROTOSS_PROBE };
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
            return { UNIT_TYPEID::TERRAN_SCV };
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
            return { UNIT_TYPEID::ZERG_DRONE };
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

        case ABILITY_ID::MORPH_GREATERSPIRE:
            return { UNIT_TYPEID::ZERG_SPIRE };

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
            return { UNIT_TYPEID::ZERG_HATCHERY, UNIT_TYPEID::ZERG_LAIR, UNIT_TYPEID::ZERG_HIVE };
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
            return { UNIT_TYPEID::INVALID };
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
    }*/
}

float maxHealth(sc2::UNIT_TYPEID type) {
    return unit_type_initial_health[(int)type].first;
}

float maxShield(sc2::UNIT_TYPEID type) {
    return unit_type_initial_health[(int)type].second;
}

bool isFlying(sc2::UNIT_TYPEID type) {
    return unit_type_is_flying[(int)type];
}

float unitRadius(sc2::UNIT_TYPEID type) {
    return unit_type_radius[(int)type];
}

/** Maps an ability to the unit that is built or trained by that ability.
 * In particular this is defined for BUILD_* abilities.
 */
UNIT_TYPEID abilityToUnit(ABILITY_ID ability) {
    return mAbilityToCreatedUnit[(int)ability];
    /*switch (ability) {
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
        case ABILITY_ID::MORPH_GREATERSPIRE:
            return UNIT_TYPEID::ZERG_GREATERSPIRE;
        default:
            return UNIT_TYPEID::INVALID;
    }*/
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

/** Maps a buff to approximately which ability that causes it */
ABILITY_ID buffToAbility(BUFF_ID type) {
    switch (type) {
        case BUFF_ID::INVALID:
            return ABILITY_ID::INVALID;
        case BUFF_ID::GRAVITONBEAM:
            return ABILITY_ID::EFFECT_GRAVITONBEAM;
        case BUFF_ID::GHOSTCLOAK:
            return ABILITY_ID::BEHAVIOR_CLOAKON_GHOST;
        case BUFF_ID::BANSHEECLOAK:
            return ABILITY_ID::BEHAVIOR_CLOAKON_BANSHEE;
        case BUFF_ID::POWERUSERWARPABLE:
            return ABILITY_ID::INVALID;
        case BUFF_ID::QUEENSPAWNLARVATIMER:
            return ABILITY_ID::INVALID;
        case BUFF_ID::GHOSTHOLDFIRE:
            return ABILITY_ID::INVALID;
        case BUFF_ID::GHOSTHOLDFIREB:
            return ABILITY_ID::INVALID;
        case BUFF_ID::EMPDECLOAK:
            return ABILITY_ID::EFFECT_EMP;
        case BUFF_ID::FUNGALGROWTH:
            return ABILITY_ID::EFFECT_FUNGALGROWTH;
        case BUFF_ID::GUARDIANSHIELD:
            return ABILITY_ID::INVALID;
        case BUFF_ID::TIMEWARPPRODUCTION:
            return ABILITY_ID::EFFECT_TIMEWARP;
        case BUFF_ID::NEURALPARASITE:
            return ABILITY_ID::EFFECT_NEURALPARASITE;
        case BUFF_ID::STIMPACKMARAUDER:
            return ABILITY_ID::EFFECT_STIM_MARAUDER;
        case BUFF_ID::SUPPLYDROP:
            return ABILITY_ID::INVALID;
        case BUFF_ID::STIMPACK:
            return ABILITY_ID::EFFECT_STIM_MARINE;
        case BUFF_ID::PSISTORM:
            return ABILITY_ID::EFFECT_PSISTORM;
        case BUFF_ID::CLOAKFIELDEFFECT:
            return ABILITY_ID::INVALID;
        case BUFF_ID::CHARGING:
            return ABILITY_ID::EFFECT_CHARGE;
        case BUFF_ID::SLOW:
            return ABILITY_ID::INVALID;
        case BUFF_ID::CONTAMINATED:
            return ABILITY_ID::EFFECT_CONTAMINATE;
        case BUFF_ID::BLINDINGCLOUDSTRUCTURE:
            return ABILITY_ID::EFFECT_BLINDINGCLOUD;
        case BUFF_ID::ORACLEREVELATION:
            return ABILITY_ID::EFFECT_ORACLEREVELATION;
        case BUFF_ID::VIPERCONSUMESTRUCTURE:
            return ABILITY_ID::EFFECT_VIPERCONSUME;
        case BUFF_ID::BLINDINGCLOUD:
            return ABILITY_ID::EFFECT_BLINDINGCLOUD;
        case BUFF_ID::MEDIVACSPEEDBOOST:
            return ABILITY_ID::EFFECT_MEDIVACIGNITEAFTERBURNERS;
        case BUFF_ID::PURIFY:
            return ABILITY_ID::EFFECT_PURIFICATIONNOVA;
        case BUFF_ID::ORACLEWEAPON:
            return ABILITY_ID::INVALID;
        case BUFF_ID::IMMORTALOVERLOAD:
            return ABILITY_ID::EFFECT_IMMORTALBARRIER;
        case BUFF_ID::LOCKON:
            return ABILITY_ID::EFFECT_LOCKON;
        case BUFF_ID::SEEKERMISSILE:
            return ABILITY_ID::EFFECT_HUNTERSEEKERMISSILE;
        case BUFF_ID::TEMPORALFIELD:
            return ABILITY_ID::INVALID;
        case BUFF_ID::VOIDRAYSWARMDAMAGEBOOST:
            return ABILITY_ID::EFFECT_VOIDRAYPRISMATICALIGNMENT;
        case BUFF_ID::ORACLESTASISTRAPTARGET:
            return ABILITY_ID::BUILD_STASISTRAP;
        case BUFF_ID::PARASITICBOMB:
            return ABILITY_ID::EFFECT_PARASITICBOMB;
        case BUFF_ID::PARASITICBOMBUNITKU:
            return ABILITY_ID::EFFECT_PARASITICBOMB;
        case BUFF_ID::PARASITICBOMBSECONDARYUNITSEARCH:
            return ABILITY_ID::EFFECT_PARASITICBOMB;
        case BUFF_ID::LURKERHOLDFIREB:
            return ABILITY_ID::BURROWDOWN_LURKER;
        case BUFF_ID::CHANNELSNIPECOMBAT:
            return ABILITY_ID::INVALID;
        case BUFF_ID::TEMPESTDISRUPTIONBLASTSTUNBEHAVIOR:
            return ABILITY_ID::EFFECT_TEMPESTDISRUPTIONBLAST;
        case BUFF_ID::CARRYMINERALFIELDMINERALS:
            return ABILITY_ID::INVALID;
        case BUFF_ID::CARRYHIGHYIELDMINERALFIELDMINERALS:
            return ABILITY_ID::INVALID;
        case BUFF_ID::CARRYHARVESTABLEVESPENEGEYSERGAS:
            return ABILITY_ID::INVALID;
        case BUFF_ID::CARRYHARVESTABLEVESPENEGEYSERGASPROTOSS:
            return ABILITY_ID::INVALID;
        case BUFF_ID::CARRYHARVESTABLEVESPENEGEYSERGASZERG:
            return ABILITY_ID::INVALID;
    }
}
