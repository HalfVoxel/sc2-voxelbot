#include "TacticalNodes.h"
#include "Bot.h"

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