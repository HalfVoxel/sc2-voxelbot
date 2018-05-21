#include "TacticalNodes.h"
#include "Bot.h"
#include "Predicates.h"
#include "StrategicNodes.h"

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
    Point2D preferred_army_position = bot.tacticalManager->GetPreferredArmyPosition();
    for (auto const& unit : GetGroup()->units) {
        Point2D p = Point2D(unit->pos.x, unit->pos.y);
        if (Distance2D(preferred_army_position, p) > 3 && 
            (unit->orders.size() == 0 || Distance2D(preferred_army_position, unit->orders[0].target_pos) > 1)) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, preferred_army_position);
        }
    }
    return Success;
}

BOT::Status GroupAttackMove::OnTick() {

    auto group = static_cast<UnitGroup*>(context);
    auto game_info = bot.Observation()->GetGameInfo();
    if (game_info.enemy_start_locations.empty()) {
        return Failure;
    }
    auto target_posI = bot.influenceManager.enemyDensity.argmax();
    auto target_pos = Point2D(target_posI.x, target_posI.y);
    //game_info.enemy_start_locations.front();

    for (auto* unit : group->units) {
        if ((unit->orders.size() == 0 || Distance2D(target_pos, unit->orders[0].target_pos) > 1)) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, target_pos);
        }
    }
    return Success;
}


