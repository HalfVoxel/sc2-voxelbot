#include "predicates.h"
#include "mappings.h"

using namespace sc2;

bool IsAttackable::operator()(const Unit& unit) {
    switch (unit.unit_type.ToType()) {
        case UNIT_TYPEID::ZERG_OVERLORD:
            return false;
        case UNIT_TYPEID::ZERG_OVERSEER:
            return false;
        case UNIT_TYPEID::PROTOSS_OBSERVER:
            return false;
        default:
            return true;
    }
}

bool IsFlying::operator()(const Unit& unit) {
    return unit.is_flying;
}

bool IsArmy::operator()(const Unit& unit) {
    auto attributes = observation_->GetUnitTypeData().at(unit.unit_type).attributes;
    for (const auto& attribute : attributes) {
        if (attribute == Attribute::Structure) {
            return false;
        }
    }
    switch (unit.unit_type.ToType()) {
        case UNIT_TYPEID::ZERG_OVERLORD:
            return false;
        case UNIT_TYPEID::PROTOSS_PROBE:
            return false;
        case UNIT_TYPEID::ZERG_DRONE:
            return false;
        case UNIT_TYPEID::TERRAN_SCV:
            return false;
        case UNIT_TYPEID::ZERG_QUEEN:
            return false;
        case UNIT_TYPEID::ZERG_LARVA:
            return false;
        case UNIT_TYPEID::ZERG_EGG:
            return false;
        case UNIT_TYPEID::TERRAN_MULE:
            return false;
        case UNIT_TYPEID::TERRAN_NUKE:
            return false;
        default:
            return true;
    }
}

bool isArmy(UNIT_TYPEID type) {
    auto& unitData = getUnitData(type);
    auto attributes = unitData.attributes;
    for (const auto& attribute : attributes) {
        if (attribute == Attribute::Structure) {
            return false;
        }
    }
    switch (type) {
        case UNIT_TYPEID::ZERG_OVERLORD:
            return false;
        case UNIT_TYPEID::PROTOSS_PROBE:
            return false;
        case UNIT_TYPEID::ZERG_DRONE:
            return false;
        case UNIT_TYPEID::TERRAN_SCV:
            return false;
        case UNIT_TYPEID::ZERG_QUEEN:
            return false;
        case UNIT_TYPEID::ZERG_LARVA:
            return false;
        case UNIT_TYPEID::ZERG_EGG:
            return false;
        case UNIT_TYPEID::TERRAN_MULE:
            return false;
        case UNIT_TYPEID::TERRAN_NUKE:
            return false;
        default:
            return true;
    }
}

bool IsTownHall::operator()(const Unit& unit) {
    switch (unit.unit_type.ToType()) {
        case UNIT_TYPEID::ZERG_HATCHERY:
            return true;
        case UNIT_TYPEID::ZERG_LAIR:
            return true;
        case UNIT_TYPEID::ZERG_HIVE:
            return true;
        case UNIT_TYPEID::TERRAN_COMMANDCENTER:
            return true;
        case UNIT_TYPEID::TERRAN_ORBITALCOMMAND:
            return true;
        case UNIT_TYPEID::TERRAN_ORBITALCOMMANDFLYING:
            return true;
        case UNIT_TYPEID::TERRAN_PLANETARYFORTRESS:
            return true;
        case UNIT_TYPEID::PROTOSS_NEXUS:
            return true;
        default:
            return false;
    }
}

bool IsVespeneGeyser::operator()(const Unit& unit) {
    switch (unit.unit_type.ToType()) {
        case UNIT_TYPEID::NEUTRAL_VESPENEGEYSER:
        case UNIT_TYPEID::NEUTRAL_SPACEPLATFORMGEYSER:
        case UNIT_TYPEID::NEUTRAL_PROTOSSVESPENEGEYSER:
        case UNIT_TYPEID::NEUTRAL_PURIFIERVESPENEGEYSER:
        case UNIT_TYPEID::NEUTRAL_RICHVESPENEGEYSER:
        case UNIT_TYPEID::NEUTRAL_SHAKURASVESPENEGEYSER:
            return true;
        default:
            return false;
    }
}

bool IsStructure::operator()(const Unit& unit) {
    return isStructure(observation_->GetUnitTypeData().at(unit.unit_type));
}

bool isStructure(const UnitTypeData& unitType) {
    auto& attributes = unitType.attributes;
    bool is_structure = false;
    for (const auto& attribute : attributes) {
        if (attribute == Attribute::Structure) {
            is_structure = true;
        }
    }
    return is_structure;
}

bool isStructure(UNIT_TYPEID type) {
    // TODO: Cache in mappings?
    return isStructure(getUnitData(type));
}
