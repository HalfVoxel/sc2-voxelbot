#include "TacticalManager.h"
#include "utilities/predicates.h"
#include "Bot.h"
#include <iostream>

using namespace std;
using namespace sc2;

TacticalManager::TacticalManager(std::shared_ptr<BOT::ControlFlowNode> armyTree, sc2::Point2D wallPlacement)
    : wallPlacement(wallPlacement), armyTree(armyTree) {
    main = CreateGroup(GroupType::Main);
}

void TacticalManager::OnStep() {
    //if(main->units.size() > 40) {
    int foodInMain = 0;
    for (auto u : main->units) {
        if (isArmy(u->unit_type)) foodInMain += getUnitData(u->unit_type).food_required;
    }

    if (foodInMain > 60) {
        cout << "Army food " << bot.Observation()->GetFoodArmy() << " Main food: " << foodInMain << " Army size: " << main->units.size() << endl;
        groupAssignments[CreateGroup(Strike)] = bot.influenceManager.enemyDensity.argmax();
    }
}

void TacticalManager::OnUnitDestroyed(const Unit* unit) {
    if (unit->alliance == Unit::Alliance::Self) {
        if (unit->unit_type == UNIT_TYPEID::TERRAN_SCV) {
            availableWorkers.remove_if([unit](const Unit* x) { return x->tag == unit->tag; });
        }
        for (auto group : groups) {
            if (group->ContainsUnit(unit)) {
                group->RemoveUnit(unit);
                if (!group->IsFunctional()) {
                    for (auto unit : group->units) {
                        main->AddUnit(unit);
                    }
                }
                if (group->IsDestroyed() || !group->IsFunctional()) {
                    armyTree->Remove(group->behavior);
                    if (group->type == Scout) {
                        bot.scoutingManager->ScoutDestroyed(group);
                    }
                }
                break;
            }
        }
        groups.erase(remove_if(groups.begin(), groups.end(), [](const UnitGroup* x) { return x->IsDestroyed() || !x->IsFunctional(); }), groups.end());
    } else if (unit->alliance == Unit::Alliance::Enemy) {
        if (IsArmy(bot.Observation()).operator()(*unit)) {
            knownEnemies.remove_if([unit](const Unit* x) { return x->tag == unit->tag; });
        }
    }
}

void TacticalManager::OnUnitCreated(const Unit* unit) {
    if (unit->alliance == Unit::Alliance::Self) {
        if (unit->unit_type == UNIT_TYPEID::TERRAN_SCV && !IsCarryingVespene(*unit)) {
            availableWorkers.push_back(unit);
        } else if (IsArmy(bot.Observation()).operator()(*unit)) {
            main->AddUnit(unit);
        }
    }
}

void TacticalManager::OnUnitEnterVision(const Unit* unit) {
    if (unit->alliance == Unit::Alliance::Enemy) {
        if (IsArmy(bot.Observation()).operator()(*unit)) {
            auto found = find_if(knownEnemies.begin(), knownEnemies.end(), [unit](const Unit* x) {
                return x->tag == unit->tag;
            });
            if (found == knownEnemies.end()) {
                knownEnemies.push_back(unit);
            }
        }
    }
}

void TacticalManager::OnNydusDetected() {
}

void TacticalManager::OnNuclearLaunchDetected() {
}

Point2D TacticalManager::GetPreferredArmyPosition() {
    // Point2D p = bot.staging_location_ + (wallPlacement - bot.staging_location_) * (bot.staging_location_ == bot.startLocation_ ? 0.75 : 0.35);
    Point2D p = preferredArmyPosition;
    bot.Debug()->DebugSphereOut(Point3D(p.x, p.y, bot.startLocation_.z), 0.5, Colors::Green);
    return p;
}

UnitGroup* TacticalManager::CreateGroup(GroupType type) {
    UnitGroup* group;
    if (type == GroupType::Main) {
        group = new MainUnitGroup();
    } else if (type == GroupType::Scout) {
        if (!main->units.empty()) {
            //TODO: Choose marines over other units
            group = new ScoutGroup(main->units[0]);
            main->RemoveUnit(main->units[0]);
        } else if (availableWorkers.size() > 0) {
            group = new ScoutGroup(availableWorkers.back());
            availableWorkers.pop_back();
        } else {
            return nullptr;
        }
    } else if (type == GroupType::Strike) {
        group = new StrikeGroup();
        main->TransferUnits(group);
    } else {
        throw std::invalid_argument("type");
    }

    // armyTree->Add(group->behavior);
    groups.push_back(group);
    return group;
}

Point2DI TacticalManager::RequestTargetPosition(UnitGroup* group) {
    if (group->type == Main) {
        Point2D p = GetPreferredArmyPosition();
        return Point2DI(p.x, p.y);
    }
    Point2DI point = groupAssignments[group];
    if (Distance2D(group->GetPosition(), Point2D(point.x, point.y)) < 4) {
        groupAssignments[group] = bot.influenceManager.enemyDensity.argmax();
    }
    return groupAssignments[group];
}
