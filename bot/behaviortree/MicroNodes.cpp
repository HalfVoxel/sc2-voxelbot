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
    auto ability = agent->Observation()->GetAbilityData()[(int)ABILITY_ID::EFFECT_YAMATOGUN];
    if (IsAbilityReady(unit, ABILITY_ID::MORPH_SIEGEMODE)) {
        auto target = BestTarget([&](auto u) { return u->is_flying ? 0 : 1; }, unit->pos, 13.0, 0);

        if (target != nullptr) {
            agent->Actions()->UnitCommand(unit, ABILITY_ID::MORPH_SIEGEMODE);
        }
    }

    if (IsAbilityReady(unit, ABILITY_ID::MORPH_UNSIEGE)) {
        auto target = BestTarget([&](auto u) { return u->is_flying ? 0 : 1; }, unit->pos, 13.0, 0);

        if (target == nullptr) {
            agent->Actions()->UnitCommand(unit, ABILITY_ID::MORPH_UNSIEGE);
        }
    }

    return Success;
}

Status MicroBattleCruiser::OnTick() {
    auto unit = GetUnit();
    auto ability = agent->Observation()->GetAbilityData()[(int)ABILITY_ID::EFFECT_YAMATOGUN];
    if (IsAbilityReady(unit, ABILITY_ID::EFFECT_YAMATOGUN)) {
        const double damage = 300;
        // Find the enemy unit which we can deal the most damage to.
        // Ignore units which we deal less than 50% of the maximum damage to though.
        // For example we don't want to use it on a zergling.
        auto target = BestTarget([&](auto u) { return u->health - max(u->health - damage, 0.0); }, unit->pos, ability.cast_range, 0.5 * damage);
        if (target != nullptr) {
            agent->Actions()->UnitCommand(unit, ABILITY_ID::EFFECT_YAMATOGUN, target);
        }
    }

    return Success;
}

Status MicroLiberator::OnTick() {
    const double defender_radius = 5;
    auto unit = GetUnit();
    auto observation = agent->Observation();
    if (IsAbilityReady(unit, ABILITY_ID::MORPH_LIBERATORAGMODE)) {
        auto ability = observation->GetAbilityData()[(int)ABILITY_ID::MORPH_LIBERATORAGMODE];
        // Center the defender circle on a unit such that the total health of all units in the circle
        // is as high as possible. Only allow targets which return a positive sum.
        auto target = BestTarget([&](auto u) { return SumUnits([&](auto u2) { return u2->is_flying || IsStructure(observation)(*u2) ? 0.0 : u2->health; }, u->pos, defender_radius); }, unit->pos, ability.cast_range, 0);

        if (target != nullptr) {
            defensive_point = target->pos;
            agent->Actions()->UnitCommand(unit, ABILITY_ID::MORPH_LIBERATORAGMODE, defensive_point);
        }
    } else if (IsAbilityReady(unit, ABILITY_ID::MORPH_LIBERATORAAMODE)) {
        // If there are no units in our current defender circle, then go to anti-air mode instead.
        double score = SumUnits([&](auto u2) { return u2->is_flying || IsStructure(observation)(*u2) ? 0.0 : u2->health; }, defensive_point, defender_radius);
        if (score == 0) {
            agent->Actions()->UnitCommand(unit, ABILITY_ID::MORPH_LIBERATORAAMODE);
        }
    }

    return Success;
}

vector<UNIT_TYPEID> mineral_fields = { UNIT_TYPEID::NEUTRAL_BATTLESTATIONMINERALFIELD, UNIT_TYPEID::NEUTRAL_BATTLESTATIONMINERALFIELD750, UNIT_TYPEID::NEUTRAL_LABMINERALFIELD, UNIT_TYPEID::NEUTRAL_LABMINERALFIELD750, UNIT_TYPEID::NEUTRAL_MINERALFIELD, UNIT_TYPEID::NEUTRAL_MINERALFIELD750, UNIT_TYPEID::NEUTRAL_PURIFIERMINERALFIELD, UNIT_TYPEID::NEUTRAL_PURIFIERMINERALFIELD750, UNIT_TYPEID::NEUTRAL_PURIFIERRICHMINERALFIELD, UNIT_TYPEID::NEUTRAL_PURIFIERRICHMINERALFIELD750, UNIT_TYPEID::NEUTRAL_RICHMINERALFIELD, UNIT_TYPEID::NEUTRAL_RICHMINERALFIELD750 };

Status MicroOrbitalCommand::OnTick() {
    auto unit = GetUnit();
    auto ability = agent->Observation()->GetAbilityData()[(int)ABILITY_ID::EFFECT_SCAN];
    if (IsAbilityReady(unit, ABILITY_ID::EFFECT_SCAN) && unit->energy > 125) {
        auto p = bot->influenceManager.scanningMap.argmax();
        if (bot->influenceManager.scanningMap(p) > 0.5) {
            // Scan!
            Point2D p2D = Point2D(p.x, p.y);
            agent->Actions()->UnitCommand(unit, ABILITY_ID::EFFECT_SCAN, p2D);
            bot->influenceManager.scanningMap.setInfluenceInCircle(0, 13, p2D);
        }
    }

    if (IsAbilityReady(unit, ABILITY_ID::EFFECT_CALLDOWNMULE) && unit->energy > 100 && unit->assigned_harvesters < unit->ideal_harvesters - 2) {
        const Unit* closestMinerals = nullptr;
        for (auto other : bot->Observation()->GetUnits(Unit::Alliance::Neutral)) {
            if (find(mineral_fields.begin(), mineral_fields.end(), other->unit_type) != mineral_fields.end() && (closestMinerals == nullptr || Distance2D(unit->pos, other->pos) < Distance2D(unit->pos, closestMinerals->pos))) {
                closestMinerals = other;
            }
        }

        if (closestMinerals != nullptr) {
            agent->Actions()->UnitCommand(unit, ABILITY_ID::EFFECT_CALLDOWNMULE, closestMinerals);
        }
    }

    return Success;
}

Status MicroNexus::OnTick() {
    // Note: incorrect ID in API
    ABILITY_ID chronoboostAbility = (ABILITY_ID)3755;
    auto unit = GetUnit();
    auto ability = agent->Observation()->GetAbilityData()[(int)chronoboostAbility];
    if (IsAbilityReady(unit, chronoboostAbility)) {
        for (auto other : bot->ourUnits()) {
            if (other->owner == unit->owner && isStructure(other->unit_type) && other->build_progress == 1 && other->orders.size() > 0 && other->orders[0].progress < 0.2f && other->buffs.size() == 0) {
                agent->Actions()->UnitCommand(unit, chronoboostAbility, other);
                break;
            }
        }
    }

    return Success;
}

static int lastGuardianShieldTick = 0;

Status MicroSentry::OnTick() {
    auto unit = GetUnit();
    // If the guardian shield ability is ready and we didn't just enable a guardian shield somewhere.
    // Observations may take a tick to update so we want to make sure we base the decision on up to date information
    if (agent->Observation()->GetGameLoop() - lastGuardianShieldTick > 2 && IsAbilityReady(unit, ABILITY_ID::EFFECT_GUARDIANSHIELD)) {
        float nearbyEnemyRangedDPS = 0;
        float nearbyAllyDPS = 0;
        int nearbyNotBuffed = 0;
        bool anyEngaged = false;
        const auto& env = bot->combatPredictor.defaultCombatEnvironment;
        for (auto other : bot->ourUnits()) {
            if (other->engaged_target_tag != NullTag) anyEngaged = true;
            if (DistanceSquared2D(unit->pos, other->pos) < 4.5f*4.5f && !hasBuff(other, BUFF_ID::GUARDIANSHIELD)) {
                nearbyNotBuffed++;
            }
            if (DistanceSquared2D(unit->pos, other->pos) < 6.0f*6.0f) {
                nearbyAllyDPS += env.calculateDPS(1, other->unit_type, false);
            }
        }
        for (auto other : bot->enemyUnits()) {
            if (DistanceSquared2D(unit->pos, other->pos) < 8.0f*8.0f) {
                if (!isMelee(other->unit_type)) nearbyEnemyRangedDPS += env.calculateDPS(1, other->unit_type, false);
            }
        }

        if (nearbyNotBuffed >= 3 && anyEngaged && nearbyEnemyRangedDPS > nearbyAllyDPS*0.2f) {
            lastGuardianShieldTick = agent->Observation()->GetGameLoop();
            agent->Actions()->UnitCommand(unit, ABILITY_ID::EFFECT_GUARDIANSHIELD);
        }
    }

    return Success;
}

map<const Unit*, MicroNode*> microNodes;

void TickMicro() {
    // Cache for performance reasons
    enemyUnits = bot->enemyUnits();

    for (auto* unit : bot->ourUnits()) {
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
                case UNIT_TYPEID::PROTOSS_NEXUS:
                    node = new MicroNexus(unit);
                    break;
                case UNIT_TYPEID::PROTOSS_SENTRY:
                    node = new MicroSentry(unit);
                    break;
                default:
                    continue;
            }
        }
        node->Tick();
    }
}
