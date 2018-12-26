#include "MicroNodes.h"
#include <ctime>
#include <iostream>
#include <map>
#include "../utilities/mappings.h"
#include "../utilities/predicates.h"
#include "../Bot.h"

using namespace std;
using namespace BOT;
using namespace sc2;

vector<const Unit*> enemyUnits;

bool IsChangeling(const Unit* unit) {
    return unit->unit_type == UNIT_TYPEID::ZERG_CHANGELINGMARINE || unit->unit_type == UNIT_TYPEID::ZERG_CHANGELINGMARINESHIELD;
}

const Unit* BestTarget(function<double(const Unit*)> score, Point2D from, double range, double scoreThreshold = -1) {
    const Unit* best = nullptr;
    double bestScore = scoreThreshold;
    for (auto unit : enemyUnits) {
        if (Distance2D(from, unit->pos) < range && !IsChangeling(unit)) {
            double s = score(unit);
            if (s > bestScore) {
                best = unit;
                bestScore = s;
            }
        }
    }
    return best;
}

double SumUnits(function<double(const Unit*)> score, Point2D around, double range) {
    double sum = 0;
    for (auto unit : enemyUnits) {
        if (Distance2D(around, unit->pos) < range && !IsChangeling(unit)) {
            sum += score(unit);
        }
    }
    return sum;
}

Status MicroTank::OnTick() {
    auto unit = GetUnit();
    auto ability = agent.Observation()->GetAbilityData()[(int)ABILITY_ID::EFFECT_YAMATOGUN];
    if (IsAbilityReady(unit, ABILITY_ID::MORPH_SIEGEMODE)) {
        auto target = BestTarget([&](auto u) { return u->is_flying ? 0 : 1; }, unit->pos, 13.0, 0);

        if (target != nullptr) {
            agent.Actions()->UnitCommand(unit, ABILITY_ID::MORPH_SIEGEMODE);
        }
    }

    if (IsAbilityReady(unit, ABILITY_ID::MORPH_UNSIEGE)) {
        auto target = BestTarget([&](auto u) { return u->is_flying ? 0 : 1; }, unit->pos, 13.0, 0);

        if (target == nullptr) {
            agent.Actions()->UnitCommand(unit, ABILITY_ID::MORPH_UNSIEGE);
        }
    }

    return Success;
}

Status MicroBattleCruiser::OnTick() {
    auto unit = GetUnit();
    auto ability = agent.Observation()->GetAbilityData()[(int)ABILITY_ID::EFFECT_YAMATOGUN];
    if (IsAbilityReady(unit, ABILITY_ID::EFFECT_YAMATOGUN)) {
        const double damage = 300;
        // Find the enemy unit which we can deal the most damage to.
        // Ignore units which we deal less than 50% of the maximum damage to though.
        // For example we don't want to use it on a zergling.
        auto target = BestTarget([&](auto u) { return u->health - max(u->health - damage, 0.0); }, unit->pos, ability.cast_range, 0.5 * damage);
        if (target != nullptr) {
            agent.Actions()->UnitCommand(unit, ABILITY_ID::EFFECT_YAMATOGUN, target);
        }
    }

    return Success;
}

Status MicroLiberator::OnTick() {
    const double defender_radius = 5;
    auto unit = GetUnit();
    auto observation = agent.Observation();
    if (IsAbilityReady(unit, ABILITY_ID::MORPH_LIBERATORAGMODE)) {
        auto ability = observation->GetAbilityData()[(int)ABILITY_ID::MORPH_LIBERATORAGMODE];
        // Center the defender circle on a unit such that the total health of all units in the circle
        // is as high as possible. Only allow targets which return a positive sum.
        auto target = BestTarget([&](auto u) { return SumUnits([&](auto u2) { return u2->is_flying || IsStructure(observation)(*u2) ? 0.0 : u2->health; }, u->pos, defender_radius); }, unit->pos, ability.cast_range, 0);

        if (target != nullptr) {
            defensive_point = target->pos;
            agent.Actions()->UnitCommand(unit, ABILITY_ID::MORPH_LIBERATORAGMODE, defensive_point);
        }
    } else if (IsAbilityReady(unit, ABILITY_ID::MORPH_LIBERATORAAMODE)) {
        // If there are no units in our current defender circle, then go to anti-air mode instead.
        double score = SumUnits([&](auto u2) { return u2->is_flying || IsStructure(observation)(*u2) ? 0.0 : u2->health; }, defensive_point, defender_radius);
        if (score == 0) {
            agent.Actions()->UnitCommand(unit, ABILITY_ID::MORPH_LIBERATORAAMODE);
        }
    }

    return Success;
}

vector<UNIT_TYPEID> mineral_fields = { UNIT_TYPEID::NEUTRAL_BATTLESTATIONMINERALFIELD, UNIT_TYPEID::NEUTRAL_BATTLESTATIONMINERALFIELD750, UNIT_TYPEID::NEUTRAL_LABMINERALFIELD, UNIT_TYPEID::NEUTRAL_LABMINERALFIELD750, UNIT_TYPEID::NEUTRAL_MINERALFIELD, UNIT_TYPEID::NEUTRAL_MINERALFIELD750, UNIT_TYPEID::NEUTRAL_PURIFIERMINERALFIELD, UNIT_TYPEID::NEUTRAL_PURIFIERMINERALFIELD750, UNIT_TYPEID::NEUTRAL_PURIFIERRICHMINERALFIELD, UNIT_TYPEID::NEUTRAL_PURIFIERRICHMINERALFIELD750, UNIT_TYPEID::NEUTRAL_RICHMINERALFIELD, UNIT_TYPEID::NEUTRAL_RICHMINERALFIELD750 };

Status MicroOrbitalCommand::OnTick() {
    auto unit = GetUnit();
    auto ability = agent.Observation()->GetAbilityData()[(int)ABILITY_ID::EFFECT_SCAN];
    if (IsAbilityReady(unit, ABILITY_ID::EFFECT_SCAN) && unit->energy > 125) {
        auto p = bot.influenceManager.scanningMap.argmax();
        if (bot.influenceManager.scanningMap(p) > 0.5) {
            // Scan!
            Point2D p2D = Point2D(p.x, p.y);
            agent.Actions()->UnitCommand(unit, ABILITY_ID::EFFECT_SCAN, p2D);
            bot.influenceManager.scanningMap.setInfluenceInCircle(0, 13, p2D);
        }
    }

    if (IsAbilityReady(unit, ABILITY_ID::EFFECT_CALLDOWNMULE) && unit->energy > 100 && unit->assigned_harvesters < unit->ideal_harvesters - 2) {
        const Unit* closestMinerals = nullptr;
        for (auto other : bot.Observation()->GetUnits(Unit::Alliance::Neutral)) {
            if (find(mineral_fields.begin(), mineral_fields.end(), other->unit_type) != mineral_fields.end() && (closestMinerals == nullptr || Distance2D(unit->pos, other->pos) < Distance2D(unit->pos, closestMinerals->pos))) {
                closestMinerals = other;
            }
        }

        if (closestMinerals != nullptr) {
            agent.Actions()->UnitCommand(unit, ABILITY_ID::EFFECT_CALLDOWNMULE, closestMinerals);
        }
    }

    return Success;
}

map<const Unit*, MicroNode*> microNodes;

void TickMicro() {
    // Cache for performance reasons
    enemyUnits = agent.Observation()->GetUnits(Unit::Alliance::Enemy);

    for (auto* unit : agent.Observation()->GetUnits(Unit::Alliance::Self)) {
        auto& node = microNodes[unit];
        if (node == nullptr) {
            switch (simplifyUnitType(unit->unit_type)) {
                case UNIT_TYPEID::TERRAN_BATTLECRUISER:
                    node = new MicroBattleCruiser(unit);
                    break;
                case UNIT_TYPEID::TERRAN_LIBERATOR:
                    node = new MicroLiberator(unit);
                    break;
                case UNIT_TYPEID::TERRAN_ORBITALCOMMAND:
                    node = new MicroOrbitalCommand(unit);
                    break;
                case UNIT_TYPEID::TERRAN_SIEGETANK:
                    node = new MicroTank(unit);
                    break;
                default:
                    continue;
            }
        }
        node->Tick();
    }
}