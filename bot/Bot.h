#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "BuildingPlacement.h"
#include "TacticalManager.h"
#include "CameraController.h"
#include "InfluenceManager.h"

const float kCameraWidth = 24.0f;
const int kFeatureLayerSize = 80;
const int kPixelDrawSize = 3;
const int kDrawSize = kFeatureLayerSize * kPixelDrawSize;

extern int ticks;

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
   
    std::vector<UNIT_TYPEID> production_types = {
        UNIT_TYPEID::TERRAN_COMMANDCENTER, UNIT_TYPEID::TERRAN_BARRACKS,
        UNIT_TYPEID::TERRAN_FACTORY, UNIT_TYPEID::TERRAN_STARPORT
    };
    std::vector<Point2D> wallPlacements;


    CameraController cameraController;
    TacticalManager* tactical_manager;
    InfluenceManager influenceManager;
    int max_worker_count_ = 73;
    sc2::GameInfo game_info_;
    std::vector<sc2::Point3D> expansions_;
    sc2::Point3D startLocation_;
    sc2::Point3D staging_location_;
    BuildingPlacement buildingPlacement;
    std::vector<Point2D>* FindWallPlacements(size_t size);
    
    int GetPositionIndex(int x, int y);
    Point2D GetMapCoordinate(int i);
    int ManhattanDistance(Point2D p1, Point2D p2);
    void OnGameLoading();
    void OnGameStart() override final;

    void OnStep() override final;

    void OnGameEnd() override final;

    void OnUnitDestroyed(const Unit* unit) override final;

    void OnUnitCreated(const Unit* unit) override final;

    void OnNydusDetected() override;

    void OnNuclearLaunchDetected() override final;

    void OnUnitEnterVision(const Unit* unit) override final;

    Point2D* Rotate(Point2D p, float degrees);
private:
    std::unique_ptr<TreeNode> tree;

    std::unique_ptr<TreeNode> armyTree;
};


};  // namespace BOT

extern BOT::Bot bot;