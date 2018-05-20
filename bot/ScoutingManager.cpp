#include "ScoutingManager.h"
#include "InfluenceManager.h"
#include "Bot.h"

using namespace sc2;

void ScoutingManager::OnStep() {
    const sc2::ObservationInterface* observation = bot.Observation();
    int scoutCount = 1 + (observation->GetFoodUsed() / 200) * 5;
   
    if(scoutCount > scoutAssignments.size()){
        UnitGroup* unit_group = bot.tacticalManager->CreateGroup(Scout);
        scoutAssignments.insert_or_assign(unit_group, bot.game_info_.enemy_start_locations[0]);
    }
}

void ScoutingManager::ScoutDestroyed(UnitGroup* group){
    scoutAssignments.erase(scoutAssignments.find(group));
}

Point2D ScoutingManager::RequestScoutingPosition(UnitGroup* group){
    return scoutAssignments.at(group);
}
