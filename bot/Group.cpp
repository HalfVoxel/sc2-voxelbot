#include "Group.h"
#include "TacticalNodes.h"
#include "ScoutingNodes.h"
#include <numeric>

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

void UnitGroup::TransferUnits(UnitGroup* group) {
    copy(units.begin(), units.end(), back_inserter(group->units));
    units.clear();
}

void UnitGroup::TransferUnits(UnitGroup* group, int n) {
    copy(units.begin(), units.begin()+n, back_inserter(group->units));
    units.clear();
}

void UnitGroup::RemoveUnit(const sc2::Unit* unit) {
    units.erase(remove_if(units.begin(), units.end(), [unit](const Unit* x) { return x->tag == unit->tag; }), units.end());
}

bool UnitGroup::ContainsUnit(const sc2::Unit* unit){
    auto found = find_if(units.begin(), units.end(), [unit](const Unit* x) {return x->tag == unit->tag; });
    return found != units.end();
}

void UnitGroup::SetCombatPosition(const sc2::Point2D* point2_d) {
    if (point2_d) {
        combatPosition = new Point2D(point2_d->x, point2_d->y);
    } else {
        combatPosition = nullptr;
    }
}

bool UnitGroup::IsInCombat() {
    return combatPosition ? true : false;
}

sc2::Point3D UnitGroup::GetPosition() {
    Point3D sum = accumulate(units.begin(), units.end(), Point3D(0, 0, 0), [](Point3D a, auto b) {return a + b->pos; });
    return sum / units.size();
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

StrikeGroup::StrikeGroup() : UnitGroup(GroupType::Strike) {
    behavior = std::make_shared<StrikeGroupBehavior>(this);
}
