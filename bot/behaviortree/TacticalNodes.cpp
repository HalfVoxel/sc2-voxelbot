#include "TacticalNodes.h"
#include "../utilities/pathfinding.h"
#include "../utilities/predicates.h"
#include "StrategicNodes.h"
#include "../Bot.h"

using namespace BOT;
using namespace sc2;

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

        // Only move units at most every second frame
        // Orders sometimes seem to take 2 frames to show up in the API so multiple redundant actions might be issued
        // if an order was given every frame.
        if (!currentPath.empty() && (pathingTicker % 2) == 0) {
            for (int i = 0; i < std::min(40, (int)currentPath.size() - 1); i++) {
                bot->Debug()->DebugLineOut(Point3D(currentPath[i].x, currentPath[i].y, from.z + 1), Point3D(currentPath[i + 1].x, currentPath[i + 1].y, from.z + 1), Colors::White);
            }
            // bot->Debug()->DebugLineOut(from, Point3D(currentPath[0].x, currentPath[0].y, from.z), Colors::White);
            while(true) {
                auto target_pos = Point2D(currentPath[0].x, currentPath[0].y);
                bool positionReached = true;
                std::vector<const Unit*> unitsToOrder;
                int allowedDist = 5 + 2 * sqrt(group->units.size());
                
                for (auto* unit : group->units) {
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

BOT::Status GroupAttackMove::OnTick() {
    auto group = GetGroup();
    if (group->IsInCombat()) {
        auto target_pos = *group->combatPosition;
        for (auto* unit : group->units) {
            if (unit->orders.empty() || Distance2D(target_pos, unit->orders[unit->orders.size() - 1].target_pos) > 1) {
                bot->Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, target_pos);
            }
        }
        return Running;
    }
    return Success;
}
