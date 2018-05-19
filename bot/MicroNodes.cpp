#include "MicroNodes.h"
#include "Bot.h"
#include "Mappings.h"
#include <iostream>

using namespace std;
using namespace BOT;
using namespace sc2;

bool IsAbilityReady (const Unit* unit, ABILITY_ID ability) {
	for (auto a : agent.Query()->GetAbilitiesForUnit(unit, false).abilities) {
		if (a.ability_id == ability) return true;
	}
	return false;
}

const Unit* BestTarget(function<double(const Unit*)> score, Point2D from, double range, double scoreThreshold = 0) {
	const Unit* best = nullptr;
	double bestScore = scoreThreshold;
	for (auto unit : agent.Observation()->GetUnits(Unit::Alliance::Enemy)) {
		if (Distance2D(from, unit->pos) < range) {
			double s = score(unit);
			if (s >= bestScore) {
				best = unit;
				bestScore = s;
			}
		}
	}
	return best;
}

Status MicroBattleCruiser::OnTick() {
	auto unit = GetUnit();
	auto ability = agent.Observation()->GetAbilityData()[(int)ABILITY_ID::EFFECT_YAMATOGUN];
	if (IsAbilityReady(unit, ABILITY_ID::EFFECT_YAMATOGUN)) {
		double damage = 300;
		// Find the enemy unit which we can deal the most damage to.
		// Ignore units which we deal less than 50% of the maximum damage to though.
		// For example we don't want to use it on a zergling.
		auto target = BestTarget([&](auto unit) { return unit->health - max(unit->health - damage, 0.0); }, unit->pos, ability.cast_range, 0.5*damage);
		if (target != nullptr) {
			agent.Actions()->UnitCommand(unit, ABILITY_ID::EFFECT_YAMATOGUN, target);
		}
	}
	
	return Success;
}

map<const Unit*, MicroNode*> microNodes;

void TickMicro () {
	cout << "Ticking..." << endl;
	for (auto* unit : agent.Observation()->GetUnits(Unit::Alliance::Self)) {
		auto& node = microNodes[unit];
		if (node == nullptr) {
			cout << "Create " << UnitTypeToName(unit->unit_type) << endl;
			switch(simplifyUnitType(unit->unit_type)) {
			case UNIT_TYPEID::TERRAN_BATTLECRUISER:
				cout << "Created..." << endl;
				node = new MicroBattleCruiser(unit);
				break;
			default:
				continue;
			}
		}
		node->Tick();
	}
}