#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "BuildingPlacement.h"

namespace BOT {

using namespace sc2;
class Bot : public sc2::Agent {
public:
    std::vector<UNIT_TYPEID> barrack_types = {
        UNIT_TYPEID::TERRAN_BARRACKSFLYING, UNIT_TYPEID::TERRAN_BARRACKS
    };
    std::vector<UNIT_TYPEID> factory_types = {
        UNIT_TYPEID::TERRAN_FACTORYFLYING, UNIT_TYPEID::TERRAN_FACTORY
    };
    std::vector<UNIT_TYPEID> starport_types = {
        UNIT_TYPEID::TERRAN_STARPORTFLYING, UNIT_TYPEID::TERRAN_STARPORT
    };
    std::vector<UNIT_TYPEID> supply_depot_types = {
        UNIT_TYPEID::TERRAN_SUPPLYDEPOT, UNIT_TYPEID::TERRAN_SUPPLYDEPOTLOWERED
    };
    std::vector<UNIT_TYPEID> bio_types = {
        UNIT_TYPEID::TERRAN_MARINE, UNIT_TYPEID::TERRAN_MARAUDER, UNIT_TYPEID::TERRAN_GHOST,
        UNIT_TYPEID::TERRAN_REAPER /*reaper*/
    };
    std::vector<UNIT_TYPEID> widow_mine_types = {
        UNIT_TYPEID::TERRAN_WIDOWMINE, UNIT_TYPEID::TERRAN_WIDOWMINEBURROWED
    };
    std::vector<UNIT_TYPEID> siege_tank_types = {
        UNIT_TYPEID::TERRAN_SIEGETANK, UNIT_TYPEID::TERRAN_SIEGETANKSIEGED
    };
    std::vector<UNIT_TYPEID> viking_types = {
        UNIT_TYPEID::TERRAN_VIKINGASSAULT, UNIT_TYPEID::TERRAN_VIKINGFIGHTER
    };
    std::vector<UNIT_TYPEID> hellion_types = {
        UNIT_TYPEID::TERRAN_HELLION, UNIT_TYPEID::TERRAN_HELLIONTANK
    };
    std::vector<UNIT_TYPEID> production_types = {
        UNIT_TYPEID::TERRAN_COMMANDCENTER, UNIT_TYPEID::TERRAN_BARRACKS,
        UNIT_TYPEID::TERRAN_FACTORY, UNIT_TYPEID::TERRAN_STARPORT
    };
    std::vector<Point2D> wallPlacements;


    int max_worker_count_ = 73;
    sc2::GameInfo game_info_;
    std::vector<sc2::Point3D> expansions_;
    sc2::Point3D startLocation_;
    sc2::Point3D staging_location_;
    BuildingPlacement buildingPlacement;
    std::vector<Point2D>* FindWallPlacements(size_t size);
    void OnGameStart() override final;
    int GetPositionIndex(int x, int y);
    Point2D GetMapCoordinate(int i);
    int ManhattanDistance(Point2D p1, Point2D p2);
    void OnStep() override final;
    // void OnUnitDestroyed(const sc2::Unit* unit) override
    Point2D* Rotate(Point2D p, float degrees);

private:
    std::unique_ptr<TreeNode> tree;

    std::unique_ptr<TreeNode> armyTree;
};


};  // namespace BOT

extern BOT::Bot bot;