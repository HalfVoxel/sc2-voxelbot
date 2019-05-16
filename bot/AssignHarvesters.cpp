#include <libvoxelbot/utilities/predicates.h>
#include "behaviortree/StrategicNodes.h"
#include "Bot.h"

using namespace BOT;
using namespace std;
using namespace sc2;

BOT::Status AssignHarvesters::OnTick() {
    Units workers = agent->Observation()->GetUnits(Unit::Alliance::Self, IsUnit(workerUnitType));
    for (const auto& worker : workers) {
        if (worker->orders.empty()) {
            MineIdleWorkers(worker, ABILITY_ID::HARVEST_GATHER, gasBuildingType, nullptr);
        }
    }
    return ManageWorkers(workerUnitType, abilityType, gasBuildingType);
}

// To ensure that we do not over or under saturate any base.
Status AssignHarvesters::ManageWorkers(UNIT_TYPEID worker_type, AbilityID worker_gather_command, UNIT_TYPEID vespene_building_type) {
    const ObservationInterface* observation = agent->Observation();
    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    Units geysers = observation->GetUnits(Unit::Alliance::Self, IsUnit(vespene_building_type));

    // Only allow worker management to run every 3 ticks
    // Every 2 ticks is reasonable as orders sometimes take up to 2 ticks to show up in the API.
    // Go for 3 ticks... because I dunno
    if (observation->GetGameLoop() % 3 != 0) {
        return Status::Success;
    }

    if (bases.empty()) {
        return Status::Failure;
    }

    // Ensure at least 50% of the harvesters mine minerals
    int maxAllowedGasHarvesters = 0;

    int highestNeededHarvesters = -1000;
    int lowestNeededHarvesters = 1000;
    for (const auto* base : bases) {
        // If we have already mined out or still building here skip the base.
        if (base->ideal_harvesters == 0 || base->build_progress != 1) {
            continue;
        }

        maxAllowedGasHarvesters += base->assigned_harvesters;
        highestNeededHarvesters = max(highestNeededHarvesters, base->ideal_harvesters - base->assigned_harvesters);
        lowestNeededHarvesters = min(lowestNeededHarvesters, base->ideal_harvesters - base->assigned_harvesters);
    }
    for (const auto* geyser : geysers) {
        highestNeededHarvesters = max(highestNeededHarvesters, geyser->ideal_harvesters - geyser->assigned_harvesters);
    }

    // If we want to get rid of harvesters in a base AND there is another base/geyser that wants harvesters more than the original wants them
    // (note: the other base might still want to get rid of them, just less than we do)
    // Find the first building which wants to get rid of the most harvesters (i.e. lowest need of harvesters, it will be negative)
    if (highestNeededHarvesters > lowestNeededHarvesters && lowestNeededHarvesters < 0) {
        for (const auto* base : bases) {
            // If we have already mined out or still building here skip the base.
            if (base->ideal_harvesters == 0 || base->build_progress != 1) {
                continue;
            }

            int wantedHarvesters = base->ideal_harvesters - base->assigned_harvesters;
            
            if (wantedHarvesters == lowestNeededHarvesters) {
                Units workers = observation->GetUnits(Unit::Alliance::Self, IsUnit(worker_type));

                for (const auto* worker : workers) {
                    // Find a worker assigned to this base
                    if (!worker->orders.empty() && worker->orders.front().target_unit_tag == base->tag) {
                        // This should allow them to be picked up by mineidleworkers()
                        return MineIdleWorkers(worker, worker_gather_command, vespene_building_type, base);
                    }
                }
            }
        }
    }

    // Fixes case with exactly 1 harvester
    maxAllowedGasHarvesters--;

    const float GeyserBuildThreshold = 0.9f;

    Units workers = observation->GetUnits(Unit::Alliance::Self, IsUnit(worker_type));
    
    for (const auto* geyser : geysers) {
        maxAllowedGasHarvesters -= geyser->assigned_harvesters;
    }

    for (const auto* geyser : geysers) {
        if (geyser->assigned_harvesters > geyser->ideal_harvesters || geyser->build_progress <= GeyserBuildThreshold) {
            for (const auto* worker : workers) {
                if (!worker->orders.empty()) {
                    if (worker->orders.front().target_unit_tag == geyser->tag && worker->orders.front().ability_id != ABILITY_ID::BUILD_REFINERY) {
                        // This should allow them to be picked up by mineidleworkers()
                        return MineIdleWorkers(worker, worker_gather_command, vespene_building_type, nullptr);
                    }
                }
            }
        } else if (geyser->assigned_harvesters < geyser->ideal_harvesters && geyser->build_progress > GeyserBuildThreshold && maxAllowedGasHarvesters > 0) {
            float bestScore = -10000;
            const Unit* bestWorker = nullptr;
            for (const auto* worker : workers) {
                if (!worker->orders.empty()) {
                    // This should move a worker that isn't mining gas to gas
                    const Unit* target = observation->GetUnit(worker->orders.front().target_unit_tag);
                    if (target == nullptr || target->unit_type == vespene_building_type) {
                        continue;
                    }
                }
                
                float score = 0;
                if (worker->orders.empty()) score += 1;
                if (!carriesResources(worker)) score += 1;
                
                score -= Distance2D(worker->pos, geyser->pos) * 0.1f;

                if (score > bestScore) {
                    bestScore = score;
                    bestWorker = worker;
                }
            }

            if (bestWorker != nullptr) {
                // This should allow them to be picked up by mineidleworkers()
                maxAllowedGasHarvesters--;
                return MineIdleWorkers(bestWorker, worker_gather_command, vespene_building_type, nullptr);
            }
        }
    }
    return Status::Success;
}

// Mine the nearest mineral to Town hall.
// If we don't do this, probes may mine from other patches if they stray too far from the base after
// building.
Status AssignHarvesters::MineIdleWorkers(const Unit* worker, AbilityID worker_gather_command, UnitTypeID vespene_building_type, const Unit* currentBase) {
    const ObservationInterface* observation = agent->Observation();
    Units bases = observation->GetUnits(Unit::Alliance::Self, IsTownHall());
    Units geysers = observation->GetUnits(Unit::Alliance::Self, IsUnit(vespene_building_type));

    const Unit* valid_mineral_patch = nullptr;

    if (bases.empty()) {
        return Status::Failure;
    }

    for (const auto& geyser : geysers) {
        if (geyser->assigned_harvesters < geyser->ideal_harvesters) {
            agent->Actions()->UnitCommand(worker, worker_gather_command, geyser);
            return Status::Success;
        }
    }
    // Search for the closest base that is missing workers.
    const Unit* closestBase = nullptr;
    float baseScore = 0;
    for (const auto* base : bases) {
        if (base->ideal_harvesters == 0) continue;

        float score = 0;
        if (base->assigned_harvesters < base->ideal_harvesters) score += 1;
        if (base->assigned_harvesters < (base->ideal_harvesters * 3)/2) score += 1;
        score += 0.001f * (base->ideal_harvesters - base->assigned_harvesters);
        score -= 0.02f * Distance2D(worker->pos, base->pos);
        if (score > baseScore) {
            baseScore = score;
            closestBase = base;
        }
    }

    if (closestBase != nullptr) {
        if (closestBase == currentBase) return Status::Success;

        // Find the closest mineral patch by interpolating slightly towards the worker position to break ties between
        // patches that are equally close to the base
        valid_mineral_patch = FindNearestMineralPatch(closestBase->pos, worker->pos);
        if (valid_mineral_patch != nullptr) {
            agent->Actions()->UnitCommand(worker, worker_gather_command, valid_mineral_patch);
            return Status::Success;
        }
    }

    if (!worker->orders.empty()) {
        return Status::Success;
    }

    // If all workers are spots are filled just go to any base.
    const Unit* random_base = GetRandomEntry(bases);
    valid_mineral_patch = FindNearestMineralPatch(random_base->pos, worker->pos);
    if (valid_mineral_patch != nullptr) {
        agent->Actions()->UnitCommand(worker, worker_gather_command, valid_mineral_patch);
        return Status::Success;
    } else {
        return Status::Failure;
    }
}

const Unit* AssignHarvesters::FindNearestMineralPatch(Point2D start, const Point2D workerPosition) {
    Units units = agent->Observation()->GetUnits(Unit::Alliance::Neutral);
    float distance = std::numeric_limits<float>::max();
    const Unit* target = nullptr;

    // Never search more than this distance away from bases
    float thresholdDistance = 15;
    for (const auto& u : units) {
        if (isMineralField(u->unit_type)) {
            float d = DistanceSquared2D(u->pos, start);
            float d2 = DistanceSquared2D(u->pos, workerPosition);

            // Avoid searching further away than the threshold... but if there are no alternatives find the closest one to the base
            if (d > thresholdDistance*thresholdDistance) d2 += d * 100;

            if (d2 < distance) {
                distance = d2;
                target = u;
            }
        }
    }
    return target;
}
