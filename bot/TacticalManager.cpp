#include "TacticalManager.h"
#include "Bot.h"
#include "Predicates.h"

using namespace std;

void TacticalManager::OnUnitDestroyed(const Unit* unit) {
    if (unit->alliance == Unit::Alliance::Self) {
        if (unit->unit_type == UNIT_TYPEID::TERRAN_SCV) {
            availableWorkers.remove_if([unit](Unit x) { return x.tag == unit->tag; });
        } else if (IsArmy(bot.Observation()).operator()(*unit)) {
            availableArmy.remove_if([unit](Unit x) { return x.tag == unit->tag; });
        }
    } else if (unit->alliance == Unit::Alliance::Enemy) {
        if (IsArmy(bot.Observation()).operator()(*unit)) {
            knownEnemies.remove_if([unit](Unit x) { return x.tag == unit->tag; });
        }
    }
}

void TacticalManager::OnUnitCreated(const Unit* unit) {
    if (unit->alliance == Unit::Alliance::Self) {
        if (unit->unit_type == UNIT_TYPEID::TERRAN_SCV && !IsCarryingVespene(*unit)) {
            availableWorkers.push_back(*unit);
        } else if (IsArmy(bot.Observation()).operator()(*unit)) {
            availableArmy.push_back(*unit);
        }
    }
}


void TacticalManager::OnUnitEnterVision(const Unit* unit) {
    if (unit->alliance == Unit::Alliance::Enemy) {
        if (IsArmy(bot.Observation()).operator()(*unit)) {
            auto found = find_if(knownEnemies.begin(), knownEnemies.end(), [unit](Unit x) { return x.tag == unit->tag; });
            if (found == knownEnemies.end()) {
                knownEnemies.push_back(*unit);
            }
        }
    }
}


void TacticalManager::OnNydusDetected() {

}

void TacticalManager::OnNuclearLaunchDetected() {

}

Point2D TacticalManager::GetPreferredArmyPosition() {
    Point2D p = bot.staging_location_ + (wallPlacement - bot.staging_location_) * (bot.staging_location_ == bot.startLocation_ ? 0.75 : 0.35);
    Units units = bot.Observation()->GetUnits(Unit::Alliance::Self, IsTownHall());
    if(units.size() > 2) {
        
    }
    bot.Debug()->DebugSphereOut(Point3D(p.x, p.y, bot.startLocation_.z), 0.5, Colors::Green);
    bot.Debug()->SendDebug();
    return p;
}
