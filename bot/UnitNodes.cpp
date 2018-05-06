#include "UnitNodes.h"
#include "bot.h"
#include "bot_examples.h"
#include "Predicates.h"

using namespace BOT;
using namespace std;
using namespace sc2;

bool GetRandomUnit(const Unit*& unit_out, const ObservationInterface* observation, UnitTypeID unit_type) {
    Units my_units = observation->GetUnits(Unit::Alliance::Self);
    std::random_shuffle(my_units.begin(), my_units.end()); // Doesn't work, or doesn't work well.
    for (const auto unit : my_units) {
        if (unit->unit_type == unit_type) {
            unit_out = unit;
            return true;
        }
    }
    return false;
}

Status BuildUnit::OnTick() {
    const ObservationInterface* observation = bot.Observation();

    //If we are at supply cap, don't build anymore units, unless its an overlord.
    if (observation->GetFoodUsed() >= observation->GetFoodCap() && abilityType != ABILITY_ID::TRAIN_OVERLORD) {
        return Status::Failure;
    }

    const UnitTypeData& unitTypeData = observation->GetUnitTypeData(false)[unitType];

    if (observation->GetMinerals() < unitTypeData.mineral_cost) {
        return Status::Failure;
    }

    Units units = observation->GetUnits(Unit::Alliance::Self);
    for (auto unit : units) {
        if (unit->unit_type != unitType) {
            continue;
        }

        if (!unit->orders.empty()) {
            continue;
        }

        if (unit->build_progress != 1) {
            continue;
        }

        bot.Actions()->UnitCommand(unit, abilityType);
        return Status::Success;
    }

    return Status::Failure;
}


Status BuildStructure::OnTick() {
    const ObservationInterface* observation = bot.Observation();

    // If a unit already is building a supply structure of this type, do nothing.
    // Also get an scv to build the structure.
    const Unit* builderUnit = nullptr;
    Units units = observation->GetUnits(Unit::Alliance::Self);
    for (const auto& unit : units) {
        for (const auto& order : unit->orders) {
            if (order.ability_id == abilityType) {
                return Status::Running;
            }
        }

        if (unit->unit_type == builderUnitType) {
           builderUnit = unit;
        }
    }

    if (builderUnit == nullptr) {
        return Status::Failure;
    }

    if (location != NullTag) {
        // TODO: Sort units based on distance to location

        // Build at a specific position
        const Unit* target = observation->GetUnit(location);

        // Check to see if unit can build there
        if (bot.Query()->Placement(abilityType, target->pos)) {
            bot.Actions()->UnitCommand(builderUnit, abilityType, target);
            return Success;
        } else {
            return Failure;
        }
    } else {
        // Such random placement
        float rx = GetRandomScalar();
        float ry = GetRandomScalar();

        bot.Actions()->UnitCommand(builderUnit,
            abilityType,
            Point2D(builderUnit->pos.x + rx * 15.0f, builderUnit->pos.y + ry * 15.0f));

        return Status::Success;
    }
}

int countUnits(std::function<bool(const Unit*)> predicate) {
    Units units = bot.Observation()->GetUnits(Unit::Alliance::Self);
    return count_if(units.begin(), units.end(), predicate);
}

Status HasUnit::OnTick() {
    return countUnits([this](const Unit* unit) { return unit->unit_type == this->unit; }) >= count ? Status::Success : Status::Failure;
}

Status ShouldBuildSupply::OnTick() {
    auto observation = bot.Observation();
    return observation->GetFoodUsed() >= observation->GetFoodCap() - 2 ? Success : Failure;
}

Status BuildGas::OnTick() {
    if (child != nullptr) {
        return child->Tick();
    }

    auto abilityType = ABILITY_ID::BUILD_REFINERY;
    auto observation = bot.Observation();
    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    if (bases.empty()) return Failure;

    auto baseLocation = bases[0]->pos;
    Units geysers = observation->GetUnits(Unit::Alliance::Neutral, IsVespeneGeyser());

    // Only search within this radius
    float minimumDistance = 15.0f;
    Tag closestGeyser = 0;
    for (const auto& geyser : geysers) {
        float current_distance = Distance2D(baseLocation, geyser->pos);
        if (current_distance < minimumDistance) {
            if (bot.Query()->Placement(abilityType, geyser->pos)) {
                minimumDistance = current_distance;
                closestGeyser = geyser->tag;
            }
        }
    }

    // In the case where there are no more available geysers nearby
    if (closestGeyser == 0) {
        return Failure;
    }

    // TODO: Maybe a way to replan this?
    child = unique_ptr<TreeNode>(new BuildStructure(abilityType, UNIT_TYPEID::TERRAN_SCV, closestGeyser));
    return Tick();
}
