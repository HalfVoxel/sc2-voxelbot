#include "ScoutingNodes.h"
#include "Bot.h"
using namespace sc2;
using namespace BOT;

BOT::Status ScoutingBehavior::OnTick() {
    auto group = GetGroup();
    for (auto* unit : group->units) {
        Point2D point = bot.scoutingManager->RequestScoutingPosition(group);
        if (unit->orders.size() == 0 || Distance2D(point, unit->orders[0].target_pos) > 1) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::SMART, point);
        }
    }
    return Success;
}
