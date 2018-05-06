#include "UnitNodes.h"
#include "bot.h"
#include "Predicates.h"

using namespace BOT;
using namespace std;
using namespace sc2;

bool GetRandomUnit(const Unit*& unit_out, const ObservationInterface* observation,
                   UnitTypeID unit_type) {
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
    if (observation->GetFoodUsed() >= observation->GetFoodCap() && abilityType != ABILITY_ID::
        TRAIN_OVERLORD) {
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

Status BuildStructure::PlaceBuilding(AbilityID ability_type_for_structure, UnitTypeID unit_type, Point2D location, bool isExpansion = false) {

    const ObservationInterface* observation = bot.Observation();
    Units workers = observation->GetUnits(Unit::Alliance::Self, IsUnit(unit_type));

    //if we have no workers Don't build
    if (workers.empty()) {
        return Status::Failure;
    }

    // Check to see if there is already a worker heading out to build it
    for (const auto& worker : workers) {
        for (const auto& order : worker->orders) {
            if (order.ability_id == ability_type_for_structure) {
                return Status::Failure;
            }
        }
    }

    // If no worker is already building one, get a random worker to build one
    const Unit* unit = GetRandomEntry(workers);

    // Check to see if unit can make it there
    if (bot.Query()->PathingDistance(unit, location) < 0.1f) {
        return Status::Failure;
    }
    if (!isExpansion) {
        for (const auto& expansion : bot.expansions_) {
            if (Distance2D(location, Point2D(expansion.x, expansion.y)) < 7) {
                return Status::Failure;
            }
        }
    }
    // Check to see if unit can build there
    if (bot.Query()->Placement(ability_type_for_structure, location)) {
        bot.Actions()->UnitCommand(unit, ability_type_for_structure, location);
        return Status::Success;
    }
    return Status::Failure;

}

Status BuildStructure::PlaceBuilding(ABILITY_ID ability, UNIT_TYPEID unitType, Tag loc) {
    const ObservationInterface* observation = bot.Observation();

    // If a unit already is building a supply structure of this type, do nothing.
    // Also get an scv to build the structure.
    const Unit* builderUnit = nullptr;
    Units units = observation->GetUnits(Unit::Alliance::Self);
    for (const auto& unit : units) {
        for (const auto& order : unit->orders) {
            if (order.ability_id == ability) {
                return Status::Running;
            }
        }

        if (unit->unit_type == unitType) {
            builderUnit = unit;
        }
    }

    if (builderUnit == nullptr) {
        return Status::Failure;
    }

    if (loc != NullTag) {
        // TODO: Sort units based on distance to location

        // Build at a specific position
        const Unit* target = observation->GetUnit(loc);

        // Check to see if unit can build there
        if (bot.Query()->Placement(ability, target->pos)) {
            bot.Actions()->UnitCommand(builderUnit, ability, target);
            return Success;
        } else {
            return Failure;
        }
    } else {
        // Such random placement
        float rx = GetRandomScalar();
        float ry = GetRandomScalar();

        bot.Actions()->UnitCommand(builderUnit,
                                   ability,
                                   Point2D(builderUnit->pos.x + rx * 15.0f,
                                           builderUnit->pos.y + ry * 15.0f));

        return Status::Success;
    }
}

Status BuildStructure::OnTick() {
    return PlaceBuilding(abilityType, builderUnitType, location);
}

int countUnits(std::function<bool(const Unit*)> predicate) {
    Units units = bot.Observation()->GetUnits(Unit::Alliance::Self);
    return count_if(units.begin(), units.end(), predicate);
}

Status HasUnit::OnTick() {
    return countUnits([this](const Unit* unit) { return unit->unit_type == this->unit; }) >= count
               ? Status::Success
               : Status::Failure;
}

Status ShouldBuildSupply::OnTick() {
    auto observation = bot.Observation();
    return observation->GetFoodUsed() >= observation->GetFoodCap() - 2 ? Success : Failure;
}

Status ShouldExpand::OnTick() {    
    const ObservationInterface* observation = bot.Observation();
    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    //Don't have more active bases than we can provide workers for
    if (GetExpectedWorkers(gasType) > bot.max_worker_count_) {
        return Status::Failure;
    }

    // If we have extra workers around, try and build another Hatch.
    if (GetExpectedWorkers(gasType) < observation->GetFoodWorkers() - 10) {
        return Status::Success; 
    }
    //Only build another Hatch if we are floating extra minerals
    if (observation->GetMinerals() > std::min<size_t>(bases.size() * 400, 1200)) {
        return Status::Success; 
    }

    return Status::Failure;
}

int ShouldExpand::GetExpectedWorkers(UNIT_TYPEID vespene_building_type) {
    const ObservationInterface* observation = bot.Observation();
    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    Units geysers = observation->GetUnits(Unit::Alliance::Self, IsUnit(vespene_building_type));
    int expected_workers = 0;
    for (const auto& base : bases) {
        if (base->build_progress != 1) {
            continue;
        }
        expected_workers += base->ideal_harvesters;
    }

    for (const auto& geyser : geysers) {
        if (geyser->vespene_contents > 0) {
            if (geyser->build_progress != 1) {
                continue;
            }
            expected_workers += geyser->ideal_harvesters;
        }
    }

    return expected_workers;
}



Status BuildGas::OnTick() {
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
    
    return PlaceBuilding(abilityType, builderUnitType, closestGeyser);
}

BOT::Status Expand::OnTick() {
    const ObservationInterface* observation = bot.Observation();
    float minimum_distance = std::numeric_limits<float>::max();
    Point3D closest_expansion;
    for (const auto& expansion : bot.expansions_) {
        float current_distance = Distance2D(bot.startLocation_, expansion);
        if (current_distance < .01f) {
            continue;
        }

        if (current_distance < minimum_distance) {
            if (bot.Query()->Placement(abilityType, expansion)) {
                closest_expansion = expansion;
                minimum_distance = current_distance;
            }
        }
    }
    Status place_building = PlaceBuilding(abilityType, builderUnitType, closest_expansion, true);
    //only update staging location up till 3 bases.
    if (place_building == Status::Success && observation->GetUnits(Unit::Self, IsTownHall()).size() < 4) {
        bot.staging_location_ = Point3D(((bot.staging_location_.x + closest_expansion.x) / 2), ((bot.staging_location_.y + closest_expansion.y) / 2),
            ((bot.staging_location_.z + closest_expansion.z) / 2));
    }
    return place_building;
}


