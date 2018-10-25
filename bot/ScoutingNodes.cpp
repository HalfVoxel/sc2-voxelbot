#include "ScoutingNodes.h"
#include "Predicates.h"
#include "bot.h"
using namespace sc2;
using namespace BOT;

BOT::Status ScoutingBehavior::OnTick() {
    auto group = GetGroup();
    timer++;
    for (auto* unit : group->units) {
        Point2D point = bot.scoutingManager->RequestScoutingPosition(group);
        bot.Debug()->DebugLineOut(unit->pos, Point3D(point.x, point.y, unit->pos.z));
        if (unit->orders.size() == 0 || Distance2D(point, unit->orders[0].target_pos) > 1) {
            bot.Actions()->UnitCommand(unit, ABILITY_ID::SMART, point);
        }

        if (timer > 100) {
            bool enemiesNearby = false;
            const double range = 10;
            for (auto enemy : bot.Observation()->GetUnits(Unit::Alliance::Enemy, IsArmy(bot.Observation()))) {
                if (Distance2D(unit->pos, enemy->pos) < range) {
                    enemiesNearby = true;
                }
            }

            if (enemiesNearby) {
                bot.scoutingManager->MarkUnreachable(unit->pos, point);
                timer = 0;
            }
        }
    }
    return Success;
}
