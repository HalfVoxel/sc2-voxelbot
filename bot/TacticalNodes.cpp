#include "TacticalNodes.h"
#include "Bot.h"
#include "Predicates.h"

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

BOT::Status SimpleArmyPosition::OnTick() {
    Point2D preferred_army_position = bot.tactical_manager->GetPreferredArmyPosition();
    for(auto const& unit : bot.Observation()->GetUnits(Unit::Self, IsArmy(bot.Observation()))){
        Point2D p = Point2D(unit->pos.x, unit->pos.y);
        if(Distance2D(preferred_army_position, p) > 3 && unit->orders.size() == 0){
            bot.Actions()->UnitCommand(unit, ABILITY_ID::SMART, preferred_army_position);
            return Success;
        }
    }
    return Success;
}

int tick = 0;
BOT::Status SimpleAttackMove::OnTick() {
    tick++;
    if ((tick % 100) != 0) return Running;

    //If unit isn't doing anything make it attack.
    Units units = bot.Observation()->GetUnits(Unit::Self, IsArmy(bot.Observation()));
    auto game_info = bot.Observation()->GetGameInfo();
    if (game_info.enemy_start_locations.empty()) {
        return Failure;
    }
    auto target_pos = game_info.enemy_start_locations.front();

    // Just add an attack order
    // TODO: Might be queued
    for (auto* unit : units) {
        bot.Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, target_pos);
    }
    return Success;
}

BOT::Status IsUnderAttack::OnTick() {
    return Failure;
}
