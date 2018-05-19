#include "Group.h"
#include "TacticalNodes.h"

using namespace sc2;
using namespace std;

UnitGroup::UnitGroup(GroupType type) : type(type) {}

bool UnitGroup::IsDestroyed() {
    return false;
}

bool UnitGroup::IsFunctional() {
    return false;
}

ScoutGroup::ScoutGroup(const Unit* unit) : UnitGroup(GroupType::Scout){
    behavior = std::make_shared<ScoutingBehavior>(this); //TODO:Scouting behavior
    units.push_back(unit);
}

StrikeGroup::StrikeGroup(vector<const Unit*> units) : UnitGroup(GroupType::Strike) {
    behavior = std::make_shared<GroupAttackMove>(this);
}
