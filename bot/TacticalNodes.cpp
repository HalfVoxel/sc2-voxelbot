#include "TacticalNodes.h"
#include "Pathfinding.h"
#include "Predicates.h"
#include "StrategicNodes.h"
#include "bot.h"

using namespace BOT;
using namespace sc2;

BOT::Status ControlSupplyDepots::OnTick() {  //Just so we dont get stuck in base. This is probably overkill in terms of computation
    Units enemies = bot.Observation()->GetUnits(Unit::Alliance::Enemy);
    for (auto unit : bot.Observation()->GetUnits(Unit::Alliance::Self, IsUnits(bot.supply_depot_types))) {
        bool enemyNear = false;
        for (auto enemy : enemies) {
            if (!enemy->is_flying && Distance2D(unit->pos, enemy->pos) < 8 && !(enemy->unit_type == UNIT_TYPEID::ZERG_CHANGELINGMARINE || enemy->unit_type == UNIT_TYPEID::ZERG_CHANGELINGMARINESHIELD)) {
                enemyNear = true;
            }
        }
        if (!enemyNear && unit->unit_type == UNIT_TYPEID::TERRAN_SUPPLYDEPOT) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::MORPH_SUPPLYDEPOT_LOWER);
        }
        if (enemyNear && unit->unit_type == UNIT_TYPEID::TERRAN_SUPPLYDEPOTLOWERED) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::MORPH_SUPPLYDEPOT_RAISE);
        }
    }

    return Success;
}

BOT::Status GroupPosition::OnTick() {
    auto group = GetGroup();
    Point2DI request_target_position = bot.tacticalManager->RequestTargetPosition(group);
    Point2D preferred_army_position = Point2D(request_target_position.x, request_target_position.y);
    for (auto const& unit : group->units) {
        Point2D p = Point2D(unit->pos.x, unit->pos.y);
        if (Distance2D(preferred_army_position, p) > 3 &&
            (unit->orders.size() == 0 || Distance2D(preferred_army_position, unit->orders[0].target_pos) > 1)) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, preferred_army_position);
        }
    }
    return Success;
}

BOT::Status InCombat::OnTick() {
    auto group = GetGroup();
    for (auto unit : group->units) {
        if (!unit->orders.empty() && unit->orders[0].target_unit_tag != NullTag) {
            const Unit* enemy = bot.Observation()->GetUnit(unit->orders[0].target_unit_tag);
            if (enemy && !(enemy->unit_type == UNIT_TYPEID::ZERG_CHANGELINGMARINE || enemy->unit_type == UNIT_TYPEID::ZERG_CHANGELINGMARINESHIELD)) {
                group->SetCombatPosition(new Point2D(enemy->pos.x, enemy->pos.y));
                bot.Debug()->DebugLineOut(unit->pos, enemy->pos, Colors::Red);
                return Success;
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
        if (pathingTicker % 100 == 0) {
            currentPath = getPath(Point2DI((int)from.x, (int)from.y), bot.tacticalManager->RequestTargetPosition(group), bot.influenceManager.pathing_cost);
        }
        auto game_info = bot.Observation()->GetGameInfo();
        if (!currentPath.empty()) {
            for (int i = 0; i < std::min(40, (int)currentPath.size()); i++) {
                bot.Debug()->DebugLineOut(Point3D(currentPath[i].x, currentPath[i].y, from.z), Point3D(currentPath[i + 1].x, currentPath[i + 1].y, from.z), Colors::White);
            }
            // bot.Debug()->DebugLineOut(from, Point3D(currentPath[0].x, currentPath[0].y, from.z), Colors::White);
            auto target_pos = Point2D(currentPath[0].x, currentPath[0].y);
            bool positionReached = true;
            int allowedDist = 7;
            for (auto* unit : group->units) {
                bool withinDistance = Distance2D(unit->pos, target_pos) < allowedDist;
                if (!withinDistance && (unit->orders.empty() || Distance2D(target_pos, unit->orders[0].target_pos) > 1 || (pathingTicker % 250 == 0))) {
                    bot.Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, target_pos);
                }
                if (!withinDistance) {
                    positionReached = false;
                }
            }
            if (positionReached) {
                currentPath.erase(currentPath.begin());
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
                bot.Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, target_pos);
            }
        }
        return Running;
    }
    return Success;
}