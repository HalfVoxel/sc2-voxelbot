#include "Group.h"
#include "TacticalNodes.h"

using namespace sc2;
using namespace std;

UnitGroup::UnitGroup(GroupType type) : type(type) {}

bool UnitGroup::IsDestroyed() const {
    return units.empty();
}

bool UnitGroup::IsFunctional() const {
    return units.empty();
}

void UnitGroup::AddUnit(const sc2::Unit* unit) {
    if (!ContainsUnit(unit)) {
       units.push_back(unit);
    }
}

void UnitGroup::RemoveUnit(const sc2::Unit* unit) {
    remove_if(units.begin(), units.end(), [unit](const Unit* x) { return x->tag == unit->tag; });
}

bool UnitGroup::ContainsUnit(const sc2::Unit* unit){
    auto found = find_if(units.begin(), units.end(), [unit](const Unit* x) {return x->tag == unit->tag; });
    return found != units.end();
}

MainUnitGroup::MainUnitGroup() : UnitGroup(GroupType::Main) {
    behavior = std::make_shared<MainGroupBehavior>(this);
    isActive = true;
}


ScoutGroup::ScoutGroup(const Unit* unit) : UnitGroup(GroupType::Scout){
    behavior = std::make_shared<ScoutingBehavior>(this); //TODO:Scouting behavior
    AddUnit(unit);
    isActive = true;
}

StrikeGroup::StrikeGroup(vector<const Unit*> units) : UnitGroup(GroupType::Strike) {
    behavior = std::make_shared<GroupAttackMove>(this);
}
