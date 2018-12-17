#pragma once
#include "behaviortree/BehaviorTree.h"
#include "ScoutingManager.h"
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "BuildingPlacement.h"
#include "TacticalManager.h"
#include "CameraController.h"
#include "InfluenceManager.h"
#include "SpendingManager.h"
#include "DeductionManager.h"
#include "DependencyAnalyzer.h"
#include "BuildOptimizerGenetic.h"
#include "build_optimizer_nn.h"
#include "CombatPredictor.h"

extern int ticks;

bool IsAbilityReady (const sc2::Unit* unit, sc2::ABILITY_ID ability);
bool IsAbilityReadyExcludingCosts (const sc2::Unit* unit, sc2::ABILITY_ID ability);

namespace BOT {

class Bot : public sc2::Agent {
public:
    std::vector<sc2::UNIT_TYPEID> barrack_types = {
        sc2::UNIT_TYPEID::TERRAN_BARRACKSFLYING, sc2::UNIT_TYPEID::TERRAN_BARRACKS
    };
    std::vector<sc2::UNIT_TYPEID> factory_types = {
        sc2::UNIT_TYPEID::TERRAN_FACTORYFLYING, sc2::UNIT_TYPEID::TERRAN_FACTORY
    };
    std::vector<sc2::UNIT_TYPEID> starport_types = {
        sc2::UNIT_TYPEID::TERRAN_STARPORTFLYING, sc2::UNIT_TYPEID::TERRAN_STARPORT
    };
    std::vector<sc2::UNIT_TYPEID> supply_depot_types = {
        sc2::UNIT_TYPEID::TERRAN_SUPPLYDEPOT, sc2::UNIT_TYPEID::TERRAN_SUPPLYDEPOTLOWERED
    };
   
    std::vector<sc2::UNIT_TYPEID> production_types = {
        sc2::UNIT_TYPEID::TERRAN_COMMANDCENTER, sc2::UNIT_TYPEID::TERRAN_ORBITALCOMMAND,
        sc2::UNIT_TYPEID::TERRAN_PLANETARYFORTRESS, sc2::UNIT_TYPEID::TERRAN_BARRACKS,
        sc2::UNIT_TYPEID::TERRAN_FACTORY, sc2::UNIT_TYPEID::TERRAN_STARPORT
    };

    std::vector<sc2::UNIT_TYPEID> researchBuildingTypes = {
        sc2::UNIT_TYPEID::TERRAN_COMMANDCENTER,
        sc2::UNIT_TYPEID::TERRAN_ENGINEERINGBAY,
        sc2::UNIT_TYPEID::TERRAN_ARMORY,
        sc2::UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
        sc2::UNIT_TYPEID::TERRAN_FACTORYTECHLAB,
        sc2::UNIT_TYPEID::TERRAN_STARPORTTECHLAB,
        sc2::UNIT_TYPEID::TERRAN_GHOSTACADEMY,
        sc2::UNIT_TYPEID::TERRAN_FUSIONCORE
    };
    std::vector<sc2::Point2D> wallPlacements;

    SpendingManager spendingManager;
    CameraController cameraController;
    TacticalManager* tacticalManager;
    InfluenceManager influenceManager;

    CombatPredictor combatPredictor;
    BuildOptimizerNN buildTimePredictor;

    int max_worker_count_ = 73;
    sc2::GameInfo game_info_;
    std::vector<sc2::Point3D> expansions_;
    sc2::Point3D startLocation_;
    sc2::Point3D staging_location_;
    BuildingPlacement buildingPlacement;
    ScoutingManager* scoutingManager;
    DeductionManager deductionManager;
    DeductionManager ourDeductionManager;
    DependencyAnalyzer dependencyAnalyzer;
    std::shared_ptr<ControlFlowNode> researchTree;
    std::vector<sc2::Point2D>* FindWallPlacements(size_t size);
    
    int GetPositionIndex(int x, int y);
    sc2::Point2D GetMapCoordinate(int i);
    int ManhattanDistance(sc2::Point2D p1, sc2::Point2D p2);
    void OnGameLoading();
    void OnGameStart() override final;

    void OnStep() override final;

    void OnGameEnd() override final;

    void OnUnitDestroyed(const sc2::Unit* unit) override final;

    void OnUnitCreated(const sc2::Unit* unit) override final;

    void OnNydusDetected() override;

    void OnNuclearLaunchDetected() override final;

    void OnUnitEnterVision(const sc2::Unit* unit) override final;

    sc2::Point2D* Rotate(sc2::Point2D p, float degrees);
private:
    std::unique_ptr<TreeNode> tree;

    std::shared_ptr<ControlFlowNode> armyTree;
};


};  // namespace BOT

extern BOT::Bot bot;
extern sc2::Agent& agent;