#include "StrategicNodes.h"
#include <iostream>
#include "../utilities/mappings.h"
#include "../utilities/predicates.h"
#include "../Bot.h"
#include "../utilities/stdutils.h"

using namespace BOT;
using namespace std;
using namespace sc2;

ConstructionPreparationMovement::ConstructionPreparationMovement(const sc2::Unit* worker, sc2::ABILITY_ID target, sc2::Point2D constructionSpot) : worker(worker), target(target), constructionSpot(constructionSpot) {
    tickStarted = agent->Observation()->GetGameLoop();
}

bool ConstructionPreparationMovement::isValid() {
    return worker != nullptr && worker->is_alive && worker->orders.size() > 0 && (agent->Observation()->GetGameLoop() == tickStarted || (worker->orders[0].ability_id == sc2::ABILITY_ID::MOVE && DistanceSquared2D(constructionSpot, worker->orders[0].target_pos) < 1 && agent->Observation()->GetGameLoop() - tickStarted < 22*10));
}

bool GetRandomUnit(const Unit*& unit_out, const ObservationInterface* observation,
                   UnitTypeID unit_type) {
    auto my_units = bot->ourUnits();
    std::random_shuffle(my_units.begin(), my_units.end());  // Doesn't work, or doesn't work well.
    for (const auto unit : my_units) {
        if (unit->unit_type == unit_type) {
            unit_out = unit;
            return true;
        }
    }
    return false;
}

ABILITY_ID getWarpGateAbility(ABILITY_ID ability) {
    switch(ability) {
        case ABILITY_ID::TRAIN_ADEPT:
            return ABILITY_ID::TRAINWARP_ADEPT;
        case ABILITY_ID::TRAIN_DARKTEMPLAR:
            return ABILITY_ID::TRAINWARP_DARKTEMPLAR;
        case ABILITY_ID::TRAIN_HIGHTEMPLAR:
            return ABILITY_ID::TRAINWARP_HIGHTEMPLAR;
        case ABILITY_ID::TRAIN_SENTRY:
            return ABILITY_ID::TRAINWARP_SENTRY;
        case ABILITY_ID::TRAIN_STALKER:
            return ABILITY_ID::TRAINWARP_STALKER;
        case ABILITY_ID::TRAIN_ZEALOT:
            return ABILITY_ID::TRAINWARP_ZEALOT;
        default:
            assert(false);
    }
}

Status Build::OnTick() {
    const ObservationInterface* observation = bot->Observation();

    // Figure out which ability is used to build the unit and which building/unit it is built from.
    const UnitTypeData& unitTypeData = getUnitData(unitType);

    auto abilityType = unitTypeData.ability_id;
    // Usually a building
    const auto& builderUnitType = abilityToCasterUnit(unitTypeData.ability_id);

    Units units = observation->GetUnits(Unit::Alliance::Self, IsStructure(observation));

    bool unitIsAddon = isAddon(unitType);

    const Unit* bestCaster = nullptr;
    float bestCasterScore = -10000;

    // Note: incorrect ID in API
    const ABILITY_ID ChronoBoostAbility = (ABILITY_ID)3755;
    const BUFF_ID ChronoBoostBuff = (BUFF_ID)281;

    for (auto* unit : units) {
        if (!contains(builderUnitType, (UNIT_TYPEID)unit->unit_type)) {
            continue;
        }

        bool hasReactor = false;
        if (observation->GetUnit(unit->add_on_tag) != nullptr) {
            // Cannot build multiple addons on a building
            if (unitIsAddon) continue;

            UNIT_TYPEID addonType = observation->GetUnit(unit->add_on_tag)->unit_type.ToType();
            hasReactor = (addonType == UNIT_TYPEID::TERRAN_BARRACKSREACTOR ||
                          addonType == UNIT_TYPEID::TERRAN_STARPORTREACTOR ||
                          addonType == UNIT_TYPEID::TERRAN_FACTORYREACTOR);
        }

        if (unit->build_progress != 1) {
            continue;
        }

        if (unit->orders.size() > (hasReactor ? 1 : 0)) {
            continue;
        }

        if (unit->unit_type == UNIT_TYPEID::PROTOSS_WARPGATE) {
            auto warpAbility = getWarpGateAbility(abilityType);

            if (!IsAbilityReady(unit, warpAbility)) {
                continue;
            }
        } else {
            if (!IsAbilityReady(unit, abilityType)) {
                continue;
            }

            // Prevent building any units in gateways that can be morphed to warpgates (otherwise they will never have time to do it)
            if (unit->unit_type == UNIT_TYPEID::PROTOSS_GATEWAY && IsAbilityReadyExcludingCosts(unit, ABILITY_ID::MORPH_WARPGATE)) {
                continue;
            }
        }

        float score = 0;
        if (unit->unit_type == UNIT_TYPEID::PROTOSS_WARPGATE) score += 1;
        if (contains(unit->buffs, BuffID(ChronoBoostBuff))) {
            score += 1;
        }
        if (hasReactor) score += 1;

        if (score > bestCasterScore) {
            bestCaster = unit;
            bestCasterScore = score;
        }
    }

    if (bestCaster != nullptr) {
        auto* unit = bestCaster;

        // Note: before C++20 lambda captures with [=] do actually capture *this by reference
        // We don't want that, so store this field in a field that can be actually copied into the lambda
        bool tryToUseChronoBoost = this->tryToUseChronoBoost;
        UNIT_TYPEID unitType = this->unitType;
        // Try to find unit that can 
        

        bot->spendingManager.AddAction(score(unitType), CostOfUnit(unitType), [=]() {
            if (unit->unit_type == UNIT_TYPEID::PROTOSS_WARPGATE) {
                auto warpAbility = getWarpGateAbility(abilityType);
                auto point = bot->buildingPlacement.GetReasonablePlacement(unitType, warpAbility, true);
                bot->Actions()->UnitCommand(unit, warpAbility, point);
            } else {
                bot->Actions()->UnitCommand(unit, abilityType);
            }

            if (tryToUseChronoBoost && !contains(unit->buffs, BuffID(ChronoBoostBuff))) {
                // Find a nexus that can use chrono boost and then use it
                for (auto* possibleNexus : bot->ourUnits()) {
                    if (possibleNexus->unit_type == UNIT_TYPEID::PROTOSS_NEXUS && IsAbilityReady(possibleNexus, ChronoBoostAbility)) {
                        bot->Actions()->UnitCommand(possibleNexus, ChronoBoostAbility, unit);
                        break;
                    }
                }
            }
        });

        return Status::Running;
    }

    return Status::Failure;
}

BOT::Status Research::OnTick() {
    const ObservationInterface* observation = bot->Observation();

    // Figure out which ability is used to build the unit and which building/unit it is built from.
    const UpgradeData& data = observation->GetUpgradeData(false)[research];

    auto abilityType = data.ability_id;
    // Usually a building
    auto builderUnitType = abilityToCasterUnit(data.ability_id);

    for (auto const unit : bot->Observation()->GetUnits(Unit::Self, IsUnits(bot->researchBuildingTypes))) {
        if (!unit->orders.empty() && unit->orders[0].ability_id == abilityType) {
            return Running;
        }
    }

    Units units = observation->GetUnits(Unit::Alliance::Self, IsStructure(observation));
    for (auto unit : units) {
        if (std::find(builderUnitType.begin(), builderUnitType.end(), unit->unit_type) == builderUnitType.end()) {
            continue;
        }

        if (unit->build_progress != 1 || unit->orders.size() > 0) {
            continue;
        }

        abilityType = GetGeneralizedAbilityID(abilityType, *observation);
        if (!IsAbilityReady(unit, abilityType)) {
            continue;
        }
        

        bot->spendingManager.AddAction(score(research), CostOfUpgrade(research), [=]() {
            bot->Actions()->UnitCommand(unit, abilityType);
        });

        return Status::Running;
    }

    return Status::Failure;
}


const ConstructionPreparationMovement* findConstructionPreparation(ABILITY_ID ability) {
    for (int i = bot->constructionPreparation.size() - 1; i >= 0; i--) {
        auto& prep = bot->constructionPreparation[i];
        if (!prep.isValid()) {
            cout << "Removed invalid prep " << AbilityTypeToName(prep.worker->orders[0].ability_id) << " " << (agent->Observation()->GetGameLoop() - prep.tickStarted) << " " << prep.constructionSpot.x << " " << prep.worker->orders[0].target_pos.x << endl;
            bot->constructionPreparation.erase(bot->constructionPreparation.begin() + i);
            continue;
        }

        if (prep.target == ability) {
            return &prep;
        }
    }

    return nullptr;
}

/** Finds the closest unit of the specified types to the location.
 * If any of those units is already using the specified ability then (nullptr, true) will be returned (the second element will be false in other cases). Pass ABILITY_ID::INVALID to disable this.
 * Only returns workers that are currently harvesting resources or not doing anything
 */
pair<const Unit*, bool> findClosestWorker(const vector<UNIT_TYPEID> unitTypes, ABILITY_ID ability, Point2D location) {
    // If no worker is already building one, get a random worker to build one
    const Unit* unit = nullptr;
    
    Units workers = agent->Observation()->GetUnits(Unit::Alliance::Self, IsUnits(unitTypes));

    // Try first to just get the closest unit
    // but if the closest unit could not reach the point then do a slower check
    // which actually checks reachability for the units.
    // If everything fails then return failure
    for (int k = 0; k <= 1; k++) {
        float closestDistance = 100000;
        for (auto* u : workers) {
            // Check to see if there is already a worker heading out to do the same ability
            for (const auto& order : u->orders) {
                if (order.ability_id == ability) {
                    return { nullptr, true };
                }
            }
                
            float d = DistanceSquared2D(u->pos, location);
            if (carriesResources(u)) d *= 2;

            // Only return workers that are currently harvesting resources or not doing anything
            if (u->orders.size() > 0 && u->orders[0].ability_id != ABILITY_ID::HARVEST_GATHER) continue;

            if (d < closestDistance) {
                // Check to see if unit can make it there
                if (k == 1 && bot->Query()->PathingDistance(u, location) == 0) {
                    continue;
                }
                closestDistance = d;
                unit = u;
            }
        }

        // Check to see if unit can make it there
        if (unit != nullptr && bot->Query()->PathingDistance(unit, location) != 0) {
            break;
        }
    }

    return { unit, false };
}

Status Construct::PlaceBuilding(UnitTypeID unitType, Point2D location, bool isExpansion = false) {
    const ObservationInterface* observation = bot->Observation();

    const UnitTypeData& unitTypeData = observation->GetUnitTypeData(false)[unitType];

    auto ability = unitTypeData.ability_id;

    auto builderUnitType = abilityToCasterUnit(unitTypeData.ability_id);    

    auto closestWorker = findClosestWorker(builderUnitType, ability, location);
    if (closestWorker.second) return Status::Running;
    const Unit* unit = closestWorker.first;
    
    if (unit == nullptr) {
        return Status::Failure;
    }

    if (!isExpansion) {
        for (const auto& expansion : bot->expansions_) {
            if (Distance2D(location, Point2D(expansion.x, expansion.y)) < 7) {
                return Status::Failure;
            }
        }
    }
    // Check to see if unit can build there
    if (bot->Query()->Placement(ability, location)) {
        bot->spendingManager.AddAction(score(unitType), CostOfUnit(unitType), [=]() {
            bot->Actions()->UnitCommand(unit, ability, location);
        });
        return Status::Success;
    }
    return Status::Failure;
}

Status Construct::PlaceBuilding(UnitTypeID unitType, Tag loc) {
    const ObservationInterface* observation = bot->Observation();

    const UnitTypeData& unitTypeData = getUnitData(unitType);
    // auto& units = bot->ourUnits();

    // Check if the tech requirement is available
    /*if (unitTypeData.tech_requirement != UNIT_TYPEID::INVALID) {
        auto req = unitTypeData.tech_requirement;
        bool hasRequirement = false;
        for (const auto& unit : units) {
            if (unit->build_progress < 1) continue;

            if (unit->unit_type == req) {
                hasRequirement = true;
                break;
            }
            for (auto t : getUnitData(unit->unit_type).tech_alias) {
                if (t == req) {
                    hasRequirement = true;
                    break;
                }
            }
        }

        if (!hasRequirement) {
            return Status::Failure;
        }
    }*/

    auto ability = unitTypeData.ability_id;
    auto builderUnitType = abilityToCasterUnit(unitTypeData.ability_id);

    if (loc != NullTag) {
        // TODO: Sort units based on distance to location

        // Build at a specific position
        const Unit* target = observation->GetUnit(loc);

        auto closestWorker = findClosestWorker(builderUnitType, ability, target->pos);
        if (closestWorker.second) return Status::Running;
        const Unit* builderUnit = closestWorker.first;
        if (builderUnit == nullptr) return Status::Failure;

        // Check to see if unit can build there
        if (bot->Query()->Placement(ability, target->pos)) {
            bot->spendingManager.AddAction(score(unitType), CostOfUnit(unitType), [=]() {
                bot->Actions()->UnitCommand(builderUnit, ability, target);
            });
            return Running;
        } else {
            return Failure;
        }
    } else {
        auto* prep = findConstructionPreparation(ability);
        if (prep != nullptr) {
            // Note: have to copy here because the prep pointer may be invalidated later
            const Unit* builderUnit = prep->worker;
            auto p = prep->constructionSpot;

            agent->Debug()->DebugSphereOut(Point3D(p.x, p.y, builderUnit->pos.z), 2, Colors::Green);

            // Already got a worker and a location
            bot->spendingManager.AddAction(score(unitType), CostOfUnit(unitType), [=]() {
                bot->Actions()->UnitCommand(builderUnit, ability, p);
            });
        } else {
            auto p = bot->buildingPlacement.GetReasonablePlacement(unitType);

            auto closestWorker = findClosestWorker(builderUnitType, ability, p);
            if (closestWorker.second) return Status::Running;
            const Unit* builderUnit = closestWorker.first;
            if (builderUnit == nullptr) return Status::Failure;

            float estimatedTimeRequiredForPreparations = bot->Query()->PathingDistance(builderUnit, p) / getUnitData(builderUnit->unit_type).movement_speed;

            if (estimatedTimeRequiredForPreparations == 0) {
                // No path to the point, clear last known good placements to ensure it picks a new spot next time
                bot->buildingPlacement.clearLastKnownGoodPlacements();
                cout << "Clearing last good placements because point was not reachable" << endl;
            }

            // Convert from normal game speed to faster (used in LotV)
            estimatedTimeRequiredForPreparations /= 1.4f;

            estimatedTimeRequiredForPreparations -= 0.2f;

            agent->Debug()->DebugSphereOut(Point3D(p.x, p.y, builderUnit->pos.z), 2, Colors::Red);

            bot->spendingManager.AddAction(score(unitType), CostOfUnit(unitType), [=]() {
                bot->Actions()->UnitCommand(builderUnit, ability, p);
            }, estimatedTimeRequiredForPreparations, [=]() {
                bot->constructionPreparation.emplace_back(builderUnit, ability, p);
                bot->Actions()->UnitCommand(builderUnit, ABILITY_ID::MOVE, p);
            });
        }
        return Status::Running;
    }
}

Status Construct::OnTick() {
    return PlaceBuilding(unitType, location);
}

int countUnits(std::function<bool(const Unit*)> predicate) {
    auto& units = bot->ourUnits();
    return count_if(units.begin(), units.end(), predicate);
}

Status HasUnit::OnTick() {
    return countUnits([this](const Unit* unit) { return simplifyUnitType(unit->unit_type) == this->unit || unit->unit_type == this->unit; }) >= count
               ? Status::Success
               : Status::Failure;
}

Status ShouldBuildSupply::OnTick() {
    auto observation = bot->Observation();
    double productionModifier = bot->Observation()->GetUnits(Unit::Alliance::Self, IsUnits(bot->production_types)).size() * 1.0;

    for (auto unit : bot->Observation()->GetUnits(Unit::Alliance::Self, IsUnits(bot->production_types))) {
        if (unit->orders.size() > 0)
            productionModifier += 1.5;
    }

    int expectedAdditionalSupply = 0;
    const int SUPPLY_DEPOT_SUPPLY = 8;
    const int COMMAND_CENTER_SUPPLY = 1;
    for (auto unit : bot->ourUnits()) {
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
    if (expectedCap >= 200)
        return Failure;

    double expectedUse = observation->GetFoodUsed() + 1 + productionModifier;
    return expectedUse >= expectedCap ? Success : Failure;
}

Status ShouldExpand::OnTick() {
    const ObservationInterface* observation = bot->Observation();
    int commsBuilding = 0;
    for (auto unit : bot->ourUnits()) {
        for (auto order : unit->orders) {
            if (order.ability_id == ABILITY_ID::BUILD_COMMANDCENTER) {
                commsBuilding += 1;
            }
        }
    }

    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    //Don't have more active bases than we can provide workers for
    if (GetExpectedWorkers(gasType) > bot->max_worker_count_) {
        return Status::Failure;
    }

    // If we have extra workers around, try and build another Hatch.
    if (observation->GetFoodWorkers() > GetExpectedWorkers(gasType) + 10) {
        return commsBuilding == 0 ? Status::Success : Failure;
    }
    //Only build another Hatch if we are floating extra minerals
    if (observation->GetMinerals() > std::min<size_t>(bases.size() * 400, 1200)) {
        return commsBuilding == 0 ? Status::Success : Failure;
    }

    return Status::Failure;
}

int ShouldExpand::GetExpectedWorkers(UNIT_TYPEID vespene_building_type) {
    const ObservationInterface* observation = bot->Observation();
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
    auto observation = bot->Observation();
    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    if (bases.empty())
        return Failure;

    auto abilityType = getUnitData(unitType).ability_id;

    Units geysers = observation->GetUnits(Unit::Alliance::Neutral, IsVespeneGeyser());

    // Only search within this radius
    const float distanceThreshold = 15.0f;
    float minimumDistance = distanceThreshold;
    Tag closestGeyser = NullTag;
    for (const auto* base : bases) {
        auto baseLocation = base->pos;
        for (const auto& geyser : geysers) {
            float current_distance = Distance2D(baseLocation, geyser->pos);

            // Discourage building near bases that are not yet finished
            if (base->build_progress < 1) current_distance = max(current_distance, distanceThreshold - 0.01f);

            if (current_distance < minimumDistance) {
                if (bot->Query()->Placement(abilityType, geyser->pos)) {
                    minimumDistance = current_distance;
                    closestGeyser = geyser->tag;
                }
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
    const ObservationInterface* observation = bot->Observation();
    auto abilityType = observation->GetUnitTypeData(false)[unitType].ability_id;

    float minimum_distance = std::numeric_limits<float>::max();
    Point3D closest_expansion;
    for (const auto& expansion : bot->expansions_) {
        float current_distance = Distance2D(bot->startLocation_, expansion);
        if (current_distance < .01f) {
            continue;
        }

        if (current_distance < minimum_distance) {
            if (bot->Query()->Placement(abilityType, expansion)) {
                closest_expansion = expansion;
                minimum_distance = current_distance;
            }
        }
    }
    Status place_building = PlaceBuilding(unitType, closest_expansion, true);
    //only update staging location up till 3 bases.
    if (place_building == Status::Success && observation->GetUnits(Unit::Self, IsTownHall()).size() < 4) {
        bot->staging_location_ = closest_expansion;
    }
    return place_building;
}

Status Addon::TryBuildAddon(AbilityID ability_type_for_structure, Tag base_structure) {
    float rx = GetRandomScalar();
    float ry = GetRandomScalar();
    const Unit* unit = bot->Observation()->GetUnit(base_structure);

    if (unit->build_progress != 1) {
        return Status::Failure;
    }

    Point2D build_location = Point2D(unit->pos.x + rx * 15, unit->pos.y + ry * 15);

    Units units = bot->Observation()->GetUnits(Unit::Self, IsStructure(bot->Observation()));
    auto unitType = abilityToUnit(ability_type_for_structure);

    if (bot->Query()->Placement(ability_type_for_structure, unit->pos, unit)) {
        bot->spendingManager.AddAction(score(unitType), CostOfUnit(unitType), [=]() {
            bot->Actions()->UnitCommand(unit, ability_type_for_structure);
        });
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

    if (bot->Query()->Placement(ability_type_for_structure, build_location, unit)) {
        bot->spendingManager.AddAction(score(unitType), CostOfUnit(unitType), [=]() {
            bot->Actions()->UnitCommand(unit, ability_type_for_structure, build_location);
        });

        return Status::Success;
    }
    return Status::Failure;
}

HasUpgrade::HasUpgrade(sc2::UpgradeID upgrade)
    : upgrade(upgrade) {
    const UpgradeData& data = bot->Observation()->GetUpgradeData(false)[upgrade];

    upgradeBuild = data.ability_id;
    // Usually a building
    auto builderUnitType = abilityToCasterUnit(data.ability_id);
}

BOT::Status HasUpgrade::OnTick() {
    for (auto const i : agent->Observation()->GetUpgrades()) {
        if (upgrade == i) {
            return Success;
        }
    }
    for (auto const unit : bot->ourUnits()) {
        // Note: Upgrades with levels use generalized ability IDs
        if (!unit->orders.empty() && unit->orders[0].ability_id == upgradeBuild) {
            return Running;
        }
    }
    return Failure;
}

BOT::Status Addon::OnTick() {
    Units buildings = bot->Observation()->GetUnits(Unit::Self, IsUnits(buildingTypes));
    for (const auto& building : buildings) {
        if (!building->orders.empty() || building->build_progress != 1) {
            continue;
        }
        if (agent->Observation()->GetUnit(building->add_on_tag) == nullptr) {
            return TryBuildAddon(abilityType, building->tag);
        }
    }
    return Status::Failure;
}
