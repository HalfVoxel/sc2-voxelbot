#include "StrategicNodes.h"
#include "bot.h"
#include "Predicates.h"
#include "Mappings.h"
#include <iostream>

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

Status Build::OnTick() {
    const ObservationInterface* observation = bot.Observation();

    // Figure out which ability is used to build the unit and which building/unit it is built from.
    const UnitTypeData& unitTypeData = observation->GetUnitTypeData(false)[unitType];

    auto abilityType = unitTypeData.ability_id;
    // Usually a building
    auto builderUnitType = abilityToCasterUnit(unitTypeData.ability_id);

    //If we are at supply cap, don't build anymore units, unless its an overlord.
    if (unitTypeData.food_required != 0 && observation->GetFoodUsed() >= observation->GetFoodCap() && abilityType != ABILITY_ID::TRAIN_OVERLORD) {
        return Status::Failure;
    }

    if (observation->GetMinerals() < unitTypeData.mineral_cost) {
        return Status::Failure;
    }

    Units units = observation->GetUnits(Unit::Alliance::Self);
    for (auto unit : units) {
        if (unit->unit_type != builderUnitType) {
            continue;
        }
       
        bool hasReactor = false;
        if (observation->GetUnit(unit->add_on_tag) != nullptr) {
            UNIT_TYPEID addonType = observation->GetUnit(unit->add_on_tag)->unit_type.ToType();
            hasReactor = (addonType == UNIT_TYPEID::TERRAN_BARRACKSREACTOR ||
                          addonType == UNIT_TYPEID::TERRAN_STARPORTREACTOR ||
                          addonType == UNIT_TYPEID::TERRAN_FACTORYREACTOR);
        }

        if (unit->build_progress != 1) {
            continue;
        }

        if(unit->orders.size() > (hasReactor ? 1 : 0)) {
            continue;
        }

        bot.Actions()->UnitCommand(unit, abilityType);
        return Status::Success;
    }

    return Status::Failure;
}

Status BuildStructure::PlaceBuilding(UnitTypeID unitType, Point2D location, bool isExpansion = false) {

    const ObservationInterface* observation = bot.Observation();

    const UnitTypeData& unitTypeData = observation->GetUnitTypeData(false)[unitType];

    if (observation->GetMinerals() < unitTypeData.mineral_cost) {
        return Status::Failure;
    }

    auto ability = unitTypeData.ability_id;
    auto builderUnitType = abilityToCasterUnit(unitTypeData.ability_id);

    Units workers = observation->GetUnits(Unit::Alliance::Self, IsUnit(builderUnitType));

    //if we have no workers Don't build
    if (workers.empty()) {
        return Status::Failure;
    }

    // Check to see if there is already a worker heading out to build it
    for (const auto& worker : workers) {
        for (const auto& order : worker->orders) {
            if (order.ability_id == ability) {
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
    if (bot.Query()->Placement(ability, location)) {
        bot.Actions()->UnitCommand(unit, ability, location);
        return Status::Success;
    }
    return Status::Failure;

}

Status BuildStructure::PlaceBuilding(UnitTypeID unitType, Tag loc) {
    const ObservationInterface* observation = bot.Observation();

    const UnitTypeData& unitTypeData = observation->GetUnitTypeData(false)[unitType];

    if (observation->GetMinerals() < unitTypeData.mineral_cost) {
        return Status::Failure;
    }

    auto ability = unitTypeData.ability_id;
    auto builderUnitType = abilityToCasterUnit(unitTypeData.ability_id);

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

        if (unit->unit_type == builderUnitType && (unit->orders.empty() || unit->orders.at(0).ability_id == ABILITY_ID::HARVEST_GATHER)) {
            builderUnit = unit;
            break;
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
        auto p = bot.buildingPlacement.GetReasonablePlacement(unitType);

        bot.Actions()->UnitCommand(builderUnit, ability, p);
        return Status::Success;
    }
}

Status BuildStructure::OnTick() {
    return PlaceBuilding(unitType, location);
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
    double productionModifier = bot.Observation()->GetUnits(Unit::Alliance::Self, IsUnits(bot.production_types)).size() * 1.0;

    for (auto unit : bot.Observation()->GetUnits(Unit::Alliance::Self, IsUnits(bot.production_types))) {
        if (unit->orders.size() > 0) productionModifier += 1.5;
    }

    int expectedAdditionalSupply = 0;
    const int SUPPLY_DEPOT_SUPPLY = 8;
    const int COMMAND_CENTER_SUPPLY = 16;
    for (auto unit : bot.Observation()->GetUnits(Unit::Alliance::Self)) {
        for (auto order : unit->orders) {
            if (order.ability_id == ABILITY_ID::BUILD_SUPPLYDEPOT) {
                expectedAdditionalSupply += SUPPLY_DEPOT_SUPPLY;
            }
            if (order.ability_id == ABILITY_ID::BUILD_COMMANDCENTER) {
                expectedAdditionalSupply += COMMAND_CENTER_SUPPLY;
            }
        }
    }

    int expectedCap = observation->GetFoodCap() + expectedAdditionalSupply;
    if (expectedCap >= 200) return Failure;

    double expectedUse = observation->GetFoodUsed() + 1 + productionModifier;
    return expectedUse >= expectedCap ? Success : Failure;
}

Status ShouldExpand::OnTick() {    
    const ObservationInterface* observation = bot.Observation();
    int commsBuilding = 0;
    for (auto unit : bot.Observation()->GetUnits(Unit::Alliance::Self)) {
        for (auto order : unit->orders) {
            if (order.ability_id == ABILITY_ID::BUILD_COMMANDCENTER) {
                commsBuilding += 1;
            }
        }
    }

    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    //Don't have more active bases than we can provide workers for
    if (GetExpectedWorkers(gasType) > bot.max_worker_count_) {
        return Status::Failure;
    }

    // If we have extra workers around, try and build another Hatch.
    if (GetExpectedWorkers(gasType) < observation->GetFoodWorkers() - 10) {
        return commsBuilding == 0 ? Status::Success : Failure; 
    }
    //Only build another Hatch if we are floating extra minerals
    if (observation->GetMinerals() > std::min<size_t>(bases.size() * 400, 1200)) {
        return commsBuilding == 0 ? Status::Success : Failure;
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

    auto abilityType = observation->GetUnitTypeData(false)[unitType].ability_id;

    auto baseLocation = bases[0]->pos;
    Units geysers = observation->GetUnits(Unit::Alliance::Neutral, IsVespeneGeyser());

    // Only search within this radius
    float minimumDistance = 15.0f;
    Tag closestGeyser = NullTag;
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
    if (closestGeyser == NullTag) {
        return Failure;
    }
    
    return PlaceBuilding(unitType, closestGeyser);
}

BOT::Status Expand::OnTick() {
    const ObservationInterface* observation = bot.Observation();
    auto abilityType = observation->GetUnitTypeData(false)[unitType].ability_id;

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
    Status place_building = PlaceBuilding(unitType, closest_expansion, true);
    //only update staging location up till 3 bases.
    if (place_building == Status::Success && observation->GetUnits(Unit::Self, IsTownHall()).size() < 4) {
        bot.staging_location_ = closest_expansion;
    }
    return place_building;
}


Status BuildAddon::TryBuildAddon(AbilityID ability_type_for_structure, Tag base_structure) {
    float rx = GetRandomScalar();
    float ry = GetRandomScalar();
    const Unit* unit = bot.Observation()->GetUnit(base_structure);

    if (unit->build_progress != 1) {
        return Status::Failure;
    }

    Point2D build_location = Point2D(unit->pos.x + rx * 15, unit->pos.y + ry * 15);

    Units units = bot.Observation()->GetUnits(Unit::Self, IsStructure(bot.Observation()));

    if (bot.Query()->Placement(ability_type_for_structure, unit->pos, unit)) {
        bot.Actions()->UnitCommand(unit, ability_type_for_structure);
        return Status::Success;
    }

    float distance = std::numeric_limits<float>::max();
    for (const auto& u : units) {
        float d = Distance2D(u->pos, build_location);
        if (d < distance) {
            distance = d;
        }
    }
    if (distance < 6) {
        return Status::Failure;
    }

    if (bot.Query()->Placement(ability_type_for_structure, build_location, unit)) {
        bot.Actions()->UnitCommand(unit, ability_type_for_structure, build_location);
        return Status::Success;
    }
    return Status::Failure;

}

BOT::Status HasUpgrade::OnTick() {
    for(auto const i : bot.Observation()->GetUpgrades()){
        if(upgrade == i){
            return Success;
        }
    }
    for(auto const unit : bot.Observation()->GetUnits(Unit::Self, IsUnits(buildingTypes))){
        if(!unit->orders.empty()  && unit->orders[0].ability_id == upgradeBuild){
            return Running;
        }
    }
    return Failure;
}

BOT::Status BuildAddon::OnTick() {
    Units buildings = bot.Observation()->GetUnits(Unit::Self, IsUnits(buildingTypes));
    for (const auto& building : buildings) {
        if (!building->orders.empty() || building->build_progress != 1) {
            continue;
        }
        if (bot.Observation()->GetUnit(building->add_on_tag) == nullptr) {
            return TryBuildAddon(abilityType, building->tag);
        }
    }
    return Status::Failure;
}

