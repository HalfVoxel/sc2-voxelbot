#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"
#include <vector>

enum GroupType {Scout, Strike, Drop};
class UnitGroup : public BOT::Context{
   
public:
    GroupType type;
    UnitGroup(GroupType type);
    std::vector<const sc2::Unit*> units;
    std::shared_ptr<BOT::TreeNode> behavior;

    bool IsDestroyed();
    bool IsFunctional();
};

class ScoutGroup: public UnitGroup{
public:
    ScoutGroup(const sc2::Unit* unit);
};


class StrikeGroup : public UnitGroup {
public:
    StrikeGroup(std::vector<const sc2::Unit*> units);
};