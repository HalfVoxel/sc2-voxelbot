#include "TacticalNodes.h"
#include "Bot.h"
#include "Predicates.h"
#include "StrategicNodes.h"
#include "Pathfinding.h"

using namespace BOT;
using namespace sc2;

BOT::Status ControlSupplyDepots::OnTick() { //Just so we dont get stuck in base. This is probably overkill in terms of computation
    Units enemies = bot.Observation()->GetUnits(Unit::Alliance::Enemy);
    for (auto unit : bot.Observation()->GetUnits(Unit::Alliance::Self, IsUnits(bot.supply_depot_types))) {
        bool enemyNear = false;
        for (auto enemy : enemies) {
            if (!enemy->is_flying && Distance2D(unit->pos, enemy->pos) < 8) {
                enemyNear = true; 
            }
        }
        if(!enemyNear && unit->unit_type == UNIT_TYPEID::TERRAN_SUPPLYDEPOT) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::MORPH_SUPPLYDEPOT_LOWER);
        } 
        if(enemyNear && unit->unit_type == UNIT_TYPEID::TERRAN_SUPPLYDEPOTLOWERED){
            bot.Actions()->UnitCommand(unit, ABILITY_ID::MORPH_SUPPLYDEPOT_RAISE);
        }
    }

    return Success;
}

BOT::Status GroupPosition::OnTick() {
    auto group = static_cast<UnitGroup*>(context);
    Point2DI request_target_position = bot.tacticalManager->RequestTargetPosition(group);
    Point2D preferred_army_position = Point2D(request_target_position.x, request_target_position.y);
    for (auto const& unit : GetGroup()->units) {
        Point2D p = Point2D(unit->pos.x, unit->pos.y);
        if (Distance2D(preferred_army_position, p) > 3 &&
            (unit->orders.size() == 0 || Distance2D(preferred_army_position, unit->orders[0].target_pos) > 1)) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, preferred_army_position);
        }
    }
    return Success;
}


BOT::Status TacticalMove::OnTick() {
    auto group = static_cast<UnitGroup*>(context);
    if (!group->units.empty()) {
        Point3D from = group->GetPosition();
        if (pathingTicker % 100 == 0) {
            currentPath = getPath(Point2DI(from.x, from.y), bot.tacticalManager->RequestTargetPosition(group), bot.influenceManager.pathing_cost);
        }
        auto game_info = bot.Observation()->GetGameInfo();
        if (!currentPath.empty()) {
            bot.Debug()->DebugLineOut(from, Point3D(currentPath[0].x, currentPath[0].y, from.z), Colors::White);
            auto target_pos = Point2D(currentPath[0].x, currentPath[0].y);
            bool positionReached = true;
            int allowedDist = 5;
            for (auto* unit : group->units) {
                bool withinDistance = Distance2D(unit->pos, target_pos) < allowedDist;
                if (!withinDistance && (unit->orders.size() == 0 || Distance2D(target_pos, unit->orders[0].target_pos) > 1)) {
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


