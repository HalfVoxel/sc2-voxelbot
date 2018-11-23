#pragma once
#include "sc2api/sc2_api.h"

void initMappings(const sc2::ObservationInterface* observation);
sc2::UNIT_TYPEID canonicalize(sc2::UNIT_TYPEID unitType);
std::vector<sc2::UNIT_TYPEID> abilityToCasterUnit(sc2::ABILITY_ID ability);
sc2::UNIT_TYPEID abilityToUnit(sc2::ABILITY_ID ability);
sc2::UNIT_TYPEID simplifyUnitType(sc2::UNIT_TYPEID type);
const std::vector<sc2::UNIT_TYPEID>& hasBeen(sc2::UNIT_TYPEID type);
const std::vector<sc2::UNIT_TYPEID>& canBecome(sc2::UNIT_TYPEID type);
float maxHealth(sc2::UNIT_TYPEID type);
float maxShield(sc2::UNIT_TYPEID type);
bool isFlying(sc2::UNIT_TYPEID type);
float unitRadius(sc2::UNIT_TYPEID type);
const std::vector<sc2::ABILITY_ID>& unitAbilities(sc2::UNIT_TYPEID type);