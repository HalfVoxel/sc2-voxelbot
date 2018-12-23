#include "ScoutingManager.h"
#include "InfluenceManager.h"
#include "bot.h"

using namespace sc2;

void ScoutingManager::OnStep() {
    const sc2::ObservationInterface* observation = bot.Observation();
    int scoutCount = (observation->GetFoodUsed() > 16 ? 1 : 0) + (observation->GetFoodUsed() / 200) * 4;

    if (scoutCount > scoutAssignments.size()) {
        UnitGroup* unit_group = bot.tacticalManager->CreateGroup(Scout);
        if (unit_group != nullptr) {
            auto p = bot.influenceManager.scoutingMap.argmax();
            scoutAssignments[unit_group] = Point2D(p.x, p.y);
        }
    }
}

void ScoutingManager::ScoutDestroyed(UnitGroup* group) {
    scoutAssignments.erase(scoutAssignments.find(group));
}

Point2D ScoutingManager::RequestScoutingPosition(UnitGroup* group) {
    //return scoutAssignments.at(group);
    auto p = bot.influenceManager.scoutingMap.argmax();
    return Point2D(p.x, p.y);
}

void ScoutingManager::MarkUnreachable(Point2D unitPosition, Point2D unreachablePoint) {
    double dist = Distance2D(unitPosition, unreachablePoint);
    if (dist < 60 && dist > 10) {
        bot.influenceManager.scanningMap.addInfluenceInDecayingCircle(1, 10, unreachablePoint);
    }

    bot.influenceManager.scoutingMap.setInfluenceInCircle(0, 20, unreachablePoint);
}
