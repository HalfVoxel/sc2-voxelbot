#pragma once
#include "sc2api/sc2_api.h"

std::vector<sc2::UNIT_TYPEID> abilityToCasterUnit(sc2::ABILITY_ID ability);
sc2::UNIT_TYPEID abilityToUnit(sc2::ABILITY_ID ability);
sc2::UNIT_TYPEID simplifyUnitType(sc2::UNIT_TYPEID);