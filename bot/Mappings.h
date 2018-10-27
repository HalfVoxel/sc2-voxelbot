#pragma once
#include "sc2api/sc2_api.h"

void initMappings(const sc2::ObservationInterface* observation);
std::vector<sc2::UNIT_TYPEID> abilityToCasterUnit(sc2::ABILITY_ID ability);
sc2::UNIT_TYPEID abilityToUnit(sc2::ABILITY_ID ability);
sc2::UNIT_TYPEID simplifyUnitType(sc2::UNIT_TYPEID type);
std::vector<sc2::UNIT_TYPEID> hasBeen(sc2::UNIT_TYPEID type);
std::vector<sc2::UNIT_TYPEID> canBecome(sc2::UNIT_TYPEID type);
