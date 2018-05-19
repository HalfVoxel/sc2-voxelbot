#pragma once
#include "TacticalNodes.h"

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