#include "Predicates.h"
#include "UnitNodes.h"
#include "bot.h"

using namespace BOT;
using namespace std;
using namespace sc2;

BOT::Status AssignHarvesters::OnTick() {
    Units workers = bot.Observation()->GetUnits(Unit::Alliance::Self, IsUnit(workerUnitType));
    for (const auto& worker : workers) {
        if (worker->orders.empty()) {
            MineIdleWorkers(worker, ABILITY_ID::HARVEST_GATHER, UNIT_TYPEID::TERRAN_REFINERY);
        }
    }
    return ManageWorkers(workerUnitType, abilityType, gasBuildingType);
}

// To ensure that we do not over or under saturate any base.
Status AssignHarvesters::ManageWorkers(UNIT_TYPEID worker_type, AbilityID worker_gather_command, UNIT_TYPEID vespene_building_type) {
    const ObservationInterface* observation = bot.Observation();
    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    Units geysers = observation->GetUnits(Unit::Alliance::Self, IsUnit(vespene_building_type));

    if (bases.empty()) {
        return Status::Failure;
    }

    for (const auto& base : bases) {
        // If we have already mined out or still building here skip the base.
        if (base->ideal_harvesters == 0 || base->build_progress != 1) {
            continue;
        }
        // if base is
        if (base->assigned_harvesters > base->ideal_harvesters) {
            Units workers = observation->GetUnits(Unit::Alliance::Self, IsUnit(worker_type));

            for (const auto& worker : workers) {
                if (!worker->orders.empty()) {
                    if (worker->orders.front().target_unit_tag == base->tag) {
                        // This should allow them to be picked up by mineidleworkers()
                        return MineIdleWorkers(worker, worker_gather_command, vespene_building_type);
                    }
                }
            }
        }
    }

    Units workers = observation->GetUnits(Unit::Alliance::Self, IsUnit(worker_type));
    for (const auto& geyser : geysers) {
        if (geyser->ideal_harvesters == 0 || geyser->build_progress != 1) {
            continue;
        }
        if (geyser->assigned_harvesters > geyser->ideal_harvesters) {
            for (const auto& worker : workers) {
                if (!worker->orders.empty()) {
                    if (worker->orders.front().target_unit_tag == geyser->tag) {
                        // This should allow them to be picked up by mineidleworkers()
                        return MineIdleWorkers(worker, worker_gather_command, vespene_building_type);
                    }
                }
            }
        } else if (geyser->assigned_harvesters < geyser->ideal_harvesters) {
            for (const auto& worker : workers) {
                if (!worker->orders.empty()) {
                    // This should move a worker that isn't mining gas to gas
                    const Unit* target = observation->GetUnit(worker->orders.front().target_unit_tag);
                    if (target == nullptr) {
                        continue;
                    }
                    if (target->unit_type != vespene_building_type) {
                        // This should allow them to be picked up by mineidleworkers()
                        return MineIdleWorkers(worker, worker_gather_command, vespene_building_type);
                    }
                }
            }
        }
    }
    return Status::Success;
}

// Mine the nearest mineral to Town hall.
// If we don't do this, probes may mine from other patches if they stray too far from the base after
// building.
Status AssignHarvesters::MineIdleWorkers(const Unit* worker, AbilityID worker_gather_command, UnitTypeID vespene_building_type) {
    const ObservationInterface* observation = bot.Observation();
    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    Units geysers = observation->GetUnits(Unit::Alliance::Self, IsUnit(vespene_building_type));

    const Unit* valid_mineral_patch = nullptr;

    if (bases.empty()) {
        return Status::Failure;
    }

    for (const auto& geyser : geysers) {
        if (geyser->assigned_harvesters < geyser->ideal_harvesters) {
            bot.Actions()->UnitCommand(worker, worker_gather_command, geyser);
            return Status::Success;
        }
    }
    // Search for a base that is missing workers.
    for (const auto& base : bases) {
        // If we have already mined out here skip the base.
        if (base->ideal_harvesters == 0 || base->build_progress != 1) {
            continue;
        }
        if (base->assigned_harvesters < base->ideal_harvesters) {
            valid_mineral_patch = FindNearestMineralPatch(base->pos);
            bot.Actions()->UnitCommand(worker, worker_gather_command, valid_mineral_patch);
            return Status::Success;
        }
    }

    if (!worker->orders.empty()) {
        return Status::Success;
    }

    // If all workers are spots are filled just go to any base.
    const Unit* random_base = GetRandomEntry(bases);
    valid_mineral_patch = FindNearestMineralPatch(random_base->pos);
    bot.Actions()->UnitCommand(worker, worker_gather_command, valid_mineral_patch);
    return Status::Success;
}

const Unit* AssignHarvesters::FindNearestMineralPatch(const Point2D& start) {
    Units units = bot.Observation()->GetUnits(Unit::Alliance::Neutral);
    float distance = std::numeric_limits<float>::max();
    const Unit* target = nullptr;
    for (const auto& u : units) {
        if (u->unit_type == UNIT_TYPEID::NEUTRAL_MINERALFIELD) {
            float d = DistanceSquared2D(u->pos, start);
            if (d < distance) {
                distance = d;
                target = u;
            }
        }
    }
    // If we never found one return false;
    if (distance == std::numeric_limits<float>::max()) {
        return target;
    }
    return target;
}
