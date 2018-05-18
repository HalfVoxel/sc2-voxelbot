#pragma once
#include "sc2api/sc2_api.h"
#include <list>
using namespace sc2;

class TacticalManager {
private:
    Point2D wallPlacement;
    std::list<Unit> availableArmy;
    std::list<Unit> availableWorkers;
    std::list<Unit> knownEnemies;

    std::vector<UNIT_TYPEID> bio_types = { UNIT_TYPEID::TERRAN_MARINE, UNIT_TYPEID::TERRAN_MARAUDER, UNIT_TYPEID::TERRAN_GHOST, UNIT_TYPEID::TERRAN_REAPER /*reaper*/ };
    std::vector<UNIT_TYPEID> widow_mine_types = { UNIT_TYPEID::TERRAN_WIDOWMINE, UNIT_TYPEID::TERRAN_WIDOWMINEBURROWED };
    std::vector<UNIT_TYPEID> siege_tank_types = { UNIT_TYPEID::TERRAN_SIEGETANK, UNIT_TYPEID::TERRAN_SIEGETANKSIEGED };
    std::vector<UNIT_TYPEID> viking_types = { UNIT_TYPEID::TERRAN_VIKINGASSAULT, UNIT_TYPEID::TERRAN_VIKINGFIGHTER };
    std::vector<UNIT_TYPEID> hellion_types = { UNIT_TYPEID::TERRAN_HELLION, UNIT_TYPEID::TERRAN_HELLIONTANK };
    std::vector<UNIT_TYPEID> liberator_types = { UNIT_TYPEID::TERRAN_LIBERATOR, UNIT_TYPEID::TERRAN_LIBERATORAG };
    std::vector<UNIT_TYPEID> thor_types = { UNIT_TYPEID::TERRAN_THOR, UNIT_TYPEID::TERRAN_THORAP };

public:
    Point2D preferredArmyPosition;
    void OnUnitDestroyed(const Unit* unit);

    void OnUnitCreated(const Unit* unit);

    void OnNydusDetected();

    void OnNuclearLaunchDetected();
    Point2D GetPreferredArmyPosition();

    void OnUnitEnterVision(const Unit* unit);

    TacticalManager(Point2D wallPlacement): wallPlacement(wallPlacement) {
    }
};
