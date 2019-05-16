#include "TacticalNodes.h"
#include <libvoxelbot/utilities/pathfinding.h>
#include <libvoxelbot/utilities/predicates.h>
#include "StrategicNodes.h"
#include "../Bot.h"

using namespace BOT;
using namespace sc2;
using namespace std;

BOT::Status ControlSupplyDepots::OnTick() {  //Just so we dont get stuck in base. This is probably overkill in terms of computation
    Units enemies = bot->Observation()->GetUnits(Unit::Alliance::Enemy);
    for (auto unit : bot->Observation()->GetUnits(Unit::Alliance::Self, IsUnits(bot->supply_depot_types))) {
        bool enemyNear = false;
        for (auto enemy : enemies) {
            if (!enemy->is_flying && Distance2D(unit->pos, enemy->pos) < 8 && !(enemy->unit_type == UNIT_TYPEID::ZERG_CHANGELINGMARINE || enemy->unit_type == UNIT_TYPEID::ZERG_CHANGELINGMARINESHIELD)) {
                enemyNear = true;
            }
        }
        if (!enemyNear && unit->unit_type == UNIT_TYPEID::TERRAN_SUPPLYDEPOT) {
            bot->Actions()->UnitCommand(unit, ABILITY_ID::MORPH_SUPPLYDEPOT_LOWER);
        }
        if (enemyNear && unit->unit_type == UNIT_TYPEID::TERRAN_SUPPLYDEPOTLOWERED) {
            bot->Actions()->UnitCommand(unit, ABILITY_ID::MORPH_SUPPLYDEPOT_RAISE);
        }
    }

    return Success;
}

BOT::Status GroupPosition::OnTick() {
    auto group = GetGroup();
    Point2DI request_target_position = bot->tacticalManager->RequestTargetPosition(group);
    Point2D preferred_army_position = Point2D(request_target_position.x, request_target_position.y);
    for (auto const& unit : group->units) {
        Point2D p = Point2D(unit->pos.x, unit->pos.y);
        if (Distance2D(preferred_army_position, p) > 3 &&
            (unit->orders.size() == 0 || Distance2D(preferred_army_position, unit->orders[0].target_pos) > 1)) {
            bot->Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, preferred_army_position);
        }
    }
    return Success;
}

static Point2D NormalizeVector(Point2D v) {
    float magn = sqrt(v.x*v.x + v.y*v.y);
    return magn > 0 ? v / magn : Point2D(0,0);
}

bool isCombatRetreat(const UnitGroup* group, Point2D movementTarget) {
    Point2D meanPos = Point2D(0,0);

    for (auto unit : group->units) {
        meanPos += unit->pos;
    }
    if (group->units.size() > 0) meanPos /= group->units.size();

    auto movementDirection = movementTarget - meanPos;
    auto normalizedMovementDirection = NormalizeVector(movementDirection);
    const float DistanceThreshold = 14;
    int inRange = 0;
    int inRangeAndDirection = 0;
    for (auto* unit : bot->enemyUnits()) {
        if (DistanceSquared2D(meanPos, unit->pos) < DistanceThreshold*DistanceThreshold) {
            inRange++;
            auto unitDirection = unit->pos - meanPos;
            // Dot product
            float distanceAlongDirection = unitDirection.x*normalizedMovementDirection.x + unitDirection.y*normalizedMovementDirection.y;
            if (distanceAlongDirection > 1) {
                inRangeAndDirection++;
            }
        }
    }

    // True if the desired movement direction is not in the direction of any enemies
    bool shouldAttack = inRangeAndDirection > inRange * 0.2f;

    // Point is close enough to be sort of in the middle of the group.
    // This is not a retreat.
    if (DistanceSquared2D(meanPos, movementTarget) < 7*7) shouldAttack = true;

    // Also attack move if there are no enemies in range at all
    shouldAttack |= inRange == 0;

    // TODO: Should force attack if MCTS action is AttackClosestEnemy?

    return !shouldAttack;
}

BOT::Status InCombat::OnTick() {
    auto group = GetGroup();
    auto movementTarget = bot->tacticalManager->RequestTargetPosition(group);
    bool retreat = isCombatRetreat(group, Point2D(movementTarget.x, movementTarget.y));
    if (!retreat) {
        for (auto unit : group->units) {
            if (!unit->orders.empty() && unit->orders[0].target_unit_tag != NullTag) {
                const Unit* enemy = bot->Observation()->GetUnit(unit->orders[0].target_unit_tag);
                if (enemy && !isChangeling(enemy->unit_type)) {
                    group->SetCombatPosition(new Point2D(enemy->pos.x, enemy->pos.y));
                    bot->Debug()->DebugLineOut(unit->pos, enemy->pos, Colors::Red);
                    return Success;
                }
            }
        }
    }
    group->SetCombatPosition(nullptr);
    return Failure;
}

BOT::Status TacticalMove::OnTick() {
    auto group = GetGroup();
    if (!group->units.empty()) {
        Point3D from = group->GetPosition();
        auto movementTarget = bot->tacticalManager->RequestTargetPosition(group);
        if (pathingTicker % 100 == 0) {
            bool anyGroundUnits = false;
            for (auto& u : group->units) if (!u->is_flying) anyGroundUnits = true;

            if (anyGroundUnits) {
                currentPath = getPath(Point2DI((int)from.x, (int)from.y), movementTarget, bot->influenceManager.pathing_cost_finite);
            } else {
                // Flying units can take the direct path
                currentPath = { movementTarget };
            }
        }

        bool retreat = isCombatRetreat(group, Point2D(movementTarget.x, movementTarget.y));
        auto game_info = bot->Observation()->GetGameInfo();
        auto ability = retreat ? ABILITY_ID::MOVE : ABILITY_ID::ATTACK;

        for (int i = 0; i < std::min(40, (int)currentPath.size() - 1); i++) {
            bot->Debug()->DebugLineOut(Point3D(currentPath[i].x, currentPath[i].y, from.z + 1), Point3D(currentPath[i + 1].x, currentPath[i + 1].y, from.z + 1), Colors::White);
        }
        
        // Only move units at most every second frame
        // Orders sometimes seem to take 2 frames to show up in the API so multiple redundant actions might be issued
        // if an order was given every frame.
        if (!currentPath.empty() && (pathingTicker % 2) == 0) {
            // bot->Debug()->DebugLineOut(from, Point3D(currentPath[0].x, currentPath[0].y, from.z), Colors::White);
            while(true) {
                auto target_pos = Point2D(currentPath[0].x, currentPath[0].y);
                bool positionReached = true;
                std::vector<const Unit*> unitsToOrder;
                
                for (auto* unit : group->units) {
                    int allowedDist = 3 + 2 * sqrt(group->units.size()) + bot->combatPredictor.defaultCombatEnvironment.attackRange(unit->owner, unit->unit_type);
                    bool withinDistance = DistanceSquared2D(unit->pos, target_pos) < allowedDist*allowedDist;
                    if (unit->orders.empty() || DistanceSquared2D(target_pos, unit->orders[0].target_pos) > 1 || unit->orders[0].ability_id != ability || (!withinDistance && pathingTicker % 250 == 0)) {
                        unitsToOrder.push_back(unit);
                    }
                    if (!withinDistance) {
                        positionReached = false;
                    }
                }

                if (positionReached) {
                    if (currentPath.size() > 1) {
                        currentPath.erase(currentPath.begin());
                        // Check again
                        continue;
                    } else if (pathingTicker % 100 != 0) {
                        // If we are at the end of the path then only allow actions every 100 ticks (≈5 seconds).
                        // The orders will be constantly completed so the above code will try to give them orders all the time.
                        break;
                    }
                }

                if (unitsToOrder.size() > 0) {
                    bot->Actions()->UnitCommand(unitsToOrder, ability, target_pos);
                }
                break;
            }
        }
        pathingTicker++;
    }
    return Success;
}

map<const Unit*, float> lastTargetReassignment;

BOT::Status GroupAttackMove::OnTick() {
    auto group = GetGroup();
    map<const Unit*, float> alreadyAssignedDamage;

    if (group->IsInCombat()) {
        auto target_pos = *group->combatPosition;
        const auto& env = bot->combatPredictor.defaultCombatEnvironment;
        int playerID = bot->Observation()->GetPlayerID();
        float currentTime = ticksToSeconds(agent->Observation()->GetGameLoop());
        bool redirectAttacksTick = (agent->Observation()->GetGameLoop() % 10) == 0;
        bool hasGround = false;
        bool hasAir = false;
        for (auto* unit : group->units) {
            hasGround |= !unit->is_flying;
            hasAir |= unit->is_flying;
        }
        for (auto* unit : group->units) {

            // See https://liquipedia.net/starcraft2/Damage_Point
            const float HighestPossibleDamagePoint = 0.3f;

            CombatUnit cUnit = CombatUnit(*unit);
            auto& unitInfo = env.getCombatInfo(cUnit);
            float slowestWeaponInterval = unitInfo.attackInterval();

            // Note: cooldown is in game ticks
            // Weapon speed also seems to be in normal game time
            float cooldownInSeconds = unit->weapon_cooldown/16.0f;

            if (cooldownInSeconds > slowestWeaponInterval) {
                cout << "Invalid cooldown" << " " << slowestWeaponInterval << " " << cooldownInSeconds << " " << getUnitData(cUnit.type).name << endl;
            }

            // Note: sometimes the units can actually have cooldown values a little bit higher than the listed cooldowns for the weapons...
            // not sure why that happens.
            assert(cooldownInSeconds <= slowestWeaponInterval + 0.2f);

            Tag targetTag = !unit->orders.empty() ? unit->engaged_target_tag : NullTag;
            if ((unit->orders.empty() || Distance2D(target_pos, unit->orders[unit->orders.size() - 1].target_pos) > 1 || (cooldownInSeconds > 0.1 && redirectAttacksTick)) && cooldownInSeconds <= max(0.0f, slowestWeaponInterval - HighestPossibleDamagePoint)) {
                float range = env.attackRange(cUnit);

                float bestScore = -1000000;
                const Unit* bestTarget = nullptr;
                // Find best target in range
                for (auto* enemy : bot->enemyUnits()) {
                    float dist = DistanceSquared2D(unit->pos, enemy->pos);
                    CombatUnit cUnitEnemy = CombatUnit(*enemy);

                    if (enemy->is_alive && enemy->display_type == Unit::DisplayType::Visible && dist < range*range*1.2f*1.2f) {
                        // Calculate a score which has the unit dps/second
                        // or in other words the average amount we will reduce the enemy dps with if we attack this enemy
                        float dps = env.calculateDPS(cUnit, cUnitEnemy);
                        if (dps == 0) continue;

                        auto& info = env.getCombatInfo(cUnitEnemy);
                        float score = dps * max(1.0f, max(info.groundWeapon.splash, info.airWeapon.splash)) * bot->combatPredictor.targetScore(cUnitEnemy, hasGround, hasAir);
                        if (score == 0) {
                            // Some unit that cannot attack, for example a building
                            score = 0.01 * dps;
                        }

                        float oneAttackDamage = dps * unitInfo.attackInterval();
                        // Divide by the health to get dps/second
                        // Clamp the health to at least one attack so that we don't just redirect our attack to a unit with a tiny amount of health even
                        // though we could do a lot more useful damage to another unit.
                        // Take for example Tempests vs Tanks + 1 marine with a very small amount of health.
                        score /= max(oneAttackDamage, enemy->health + enemy->shield);

                        if (isVespeneHarvester(cUnitEnemy.type)) score *= 0.25f;
                        if (isTownHall(cUnitEnemy.type)) score *= 2.0f;

                        // Check if the enemy is alive and if it will be alive after all the previous units have attacked it
                        // Also include a small margin to account for unknown buffs and healing and stuff
                        bool willBeAlive = enemy->health + enemy->shield - alreadyAssignedDamage[enemy] > -5;
                        if (!willBeAlive) score *= 0.1f;

                        // Penalty for moving outside the range
                        if (dist > range*range) {
                            score *= 0.25f;
                            if (isMelee(unit->unit_type)) continue;
                        }

                        if (score > bestScore) {
                            bestTarget = enemy;
                            bestScore = score;
                        }
                    }
                }

                if (bestTarget != nullptr) {
                    // Make sure we only reassign the target when absolutely necessary, and at most once every 2 seconds
                    // otherwise the unit's rotation speed might become a limiting factor
                    if (unit->orders.empty() || (unit->orders[0].target_unit_tag != bestTarget->tag && currentTime - lastTargetReassignment[unit] > 2)) {
                        if (cooldownInSeconds >= slowestWeaponInterval - HighestPossibleDamagePoint) {
                            cerr << "Possibly redirecting target in the middle of an attack" << endl;
                            assert(false);
                        }
                        targetTag = bestTarget->tag;
                        bot->Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, bestTarget);
                        lastTargetReassignment[unit] = ticksToSeconds(agent->Observation()->GetGameLoop());
                    }
                } else if (unit->orders.empty() || Distance2D(target_pos, unit->orders[unit->orders.size() - 1].target_pos) > 1) {
                    bot->Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, target_pos);
                }
            }

            // Make the rest of the units know that this unit is going to deal damage to a target very soon
            if (targetTag != NullTag && unit->weapon_cooldown < 0.5f) {
                auto* targetUnit = agent->Observation()->GetUnit(targetTag);
                if (targetUnit != nullptr) {
                    bot->Debug()->DebugLineOut(unit->pos + Point3D(0, 0, 0.1), targetUnit->pos + Point3D(0, 0, 0.1), Colors::Red);
                    CombatUnit cUnitEnemy = CombatUnit(*targetUnit);
                    float dps = env.calculateDPS(cUnit, cUnitEnemy);
                    float oneAttackDamage = dps * unitInfo.attackInterval();
                    alreadyAssignedDamage[targetUnit] += oneAttackDamage;
                }
            }
        }
        return Running;
    }
    return Success;
}
