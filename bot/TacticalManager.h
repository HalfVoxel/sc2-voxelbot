#pragma once
#include "sc2api/sc2_api.h"
#include <list>
#include "Group.h"

class TacticalManager {
private:
    sc2::Point2D wallPlacement;
    UnitGroup* main;
    std::list<const sc2::Unit*> availableWorkers;
    std::list<const sc2::Unit*> knownEnemies;
    std::vector<UnitGroup*> groups;
    std::shared_ptr<BOT::ControlFlowNode> armyTree;
    std::map<UnitGroup*, sc2::Point2DI> groupAssignments;

    std::vector<sc2::UNIT_TYPEID> bio_types = { sc2::UNIT_TYPEID::TERRAN_MARINE, sc2::UNIT_TYPEID::TERRAN_MARAUDER, sc2::UNIT_TYPEID::TERRAN_GHOST, sc2::UNIT_TYPEID::TERRAN_REAPER /*reaper*/ };
    std::vector<sc2::UNIT_TYPEID> widow_mine_types = { sc2::UNIT_TYPEID::TERRAN_WIDOWMINE, sc2::UNIT_TYPEID::TERRAN_WIDOWMINEBURROWED };
    std::vector<sc2::UNIT_TYPEID> siege_tank_types = { sc2::UNIT_TYPEID::TERRAN_SIEGETANK, sc2::UNIT_TYPEID::TERRAN_SIEGETANKSIEGED };
    std::vector<sc2::UNIT_TYPEID> viking_types = { sc2::UNIT_TYPEID::TERRAN_VIKINGASSAULT, sc2::UNIT_TYPEID::TERRAN_VIKINGFIGHTER };
    std::vector<sc2::UNIT_TYPEID> hellion_types = { sc2::UNIT_TYPEID::TERRAN_HELLION, sc2::UNIT_TYPEID::TERRAN_HELLIONTANK };
    std::vector<sc2::UNIT_TYPEID> liberator_types = { sc2::UNIT_TYPEID::TERRAN_LIBERATOR, sc2::UNIT_TYPEID::TERRAN_LIBERATORAG };
    std::vector<sc2::UNIT_TYPEID> thor_types = { sc2::UNIT_TYPEID::TERRAN_THOR, sc2::UNIT_TYPEID::TERRAN_THORAP };

public:
    sc2::Point2D preferredArmyPosition;

    void OnUnitDestroyed(const sc2::Unit* unit);

    void OnUnitCreated(const sc2::Unit* unit);

    void OnNydusDetected();

    void OnNuclearLaunchDetected();
    sc2::Point2D GetPreferredArmyPosition();
    UnitGroup* CreateGroup(GroupType type);
    sc2::Point2DI RequestTargetPosition(UnitGroup* group);

    void OnUnitEnterVision(const sc2::Unit* unit);

    TacticalManager(std::shared_ptr<BOT::ControlFlowNode> armyTree, sc2::Point2D wallPlacement);
    void OnStep();
};
