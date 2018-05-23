#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"
#include <vector>

enum GroupType {Main, Scout, Strike, Sub, Drop};
class UnitGroup : public BOT::Context{
   
public:
    GroupType type;
    UnitGroup(GroupType type);
    std::vector<const sc2::Unit*> units;
    std::shared_ptr<BOT::TreeNode> behavior;
    bool isActive = false;

    virtual bool IsDestroyed() const;
    virtual bool IsFunctional() const;
    void AddUnit(const sc2::Unit* unit);
    void TransferUnits(UnitGroup* group);
    void TransferUnits(UnitGroup* group, int n);
    void RemoveUnit(const sc2::Unit* unit);
    bool ContainsUnit(const sc2::Unit* unit);

    sc2::Point3D GetPosition();

};

class MainUnitGroup : public UnitGroup {
public:
    MainUnitGroup();
    bool IsDestroyed()  const override { return false; }
    bool IsFunctional() const override { return true; }
};

class ScoutGroup: public UnitGroup {
public:
    ScoutGroup(const sc2::Unit* unit);
};


class StrikeGroup : public UnitGroup {
public:
    StrikeGroup();
};