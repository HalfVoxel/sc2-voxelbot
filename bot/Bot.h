#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"

namespace BOT {

class Bot : public sc2::Agent {
   public:
    int max_worker_count_ = 75;
    sc2::GameInfo game_info_;
    std::vector<sc2::Point3D> expansions_;
    sc2::Point3D startLocation_;
    sc2::Point3D staging_location_;
    void OnGameStart() override final;
    void OnStep() override final;
    // void OnUnitDestroyed(const sc2::Unit* unit) override;

   private:
    std::unique_ptr<TreeNode> tree;
};

};  // namespace BOT

extern BOT::Bot bot;