#include "BuildingPlacement.h"
#include <vector>
#include <libvoxelbot/utilities/predicates.h>
#include "Bot.h"
#include <random>

using namespace std;
using namespace sc2;

Point2D Rotate(Point2D p, float degrees) {
    degrees = degrees * 3.14159265358979323846 / 180;
    return Point2D(p.x * cos(degrees) - p.y * sin(degrees),
                   p.x * sin(degrees) + p.y * cos(degrees));
}

void BuildingPlacement::OnGameStart() {
    lastKnownGoodPlacements.clear();
    locationQueues.clear();
    FindWallPlacements(bot->startLocation_);
}

void BuildingPlacement::OnStep() {
}
void BuildingPlacement::FindWallPlacements(Point3D startLocation_) {
    auto& game_info_ = bot->game_info_;
    size_t size = game_info_.placement_grid.data.size();
    vector<int> diff(size);

    for (int i = 0; i < size; ++i) {
        if (game_info_.placement_grid.data[i] == 0 && game_info_.pathing_grid.data[i] == 0) {
            diff[i] = 1;
            Point2D p = bot->GetMapCoordinate(i);
            bot->Debug()->DebugSphereOut(Point3D(p.x + 0.5, p.y - 0.5, startLocation_.z), 0.5, Colors::Red);
        }
    }

    int mapHeuristic = game_info_.height;
    int start_index = bot->GetPositionIndex(startLocation_.x, startLocation_.y);
    Point2D start2D = Point2D(startLocation_.x, startLocation_.y);
    for (int i = 0; i < size; ++i) {
        if (diff[i] == 1) {
            for (int j = 0; j < size; ++j) {
                Point2D p = bot->GetMapCoordinate(j);
                if (Distance2D(bot->GetMapCoordinate(i), p) <= 2 && game_info_.pathing_grid.data[j] == 0 && diff[j] == 0) {
                    if (abs(game_info_.terrain_height.data[j] - game_info_.terrain_height.data[start_index]) < 2 && Distance2D(p, start2D) < mapHeuristic / 4) {  //Height filter messes up on some maps:
                        bot->Debug()->DebugSphereOut(Point3D(p.x + 0.5, p.y - 0.5, startLocation_.z), 0.5, Colors::Blue);
                        diff[j] = 2;
                    }
                }
            }
        }
    }

    auto& supplyDepotPositions = locationQueues[UNIT_TYPEID::TERRAN_SUPPLYDEPOT];
    for (int i = 0; i < size; ++i) {
        if (diff[i] == 2) {
            Point2D p = bot->GetMapCoordinate(i);
            if (diff[bot->GetPositionIndex(p.x - 1, p.y)] == 2 && diff[bot->GetPositionIndex(p.x - 1, p.y + 1)] == 2 && diff[bot->GetPositionIndex(p.x, p.y + 1)] == 2) {
                bot->Debug()->DebugSphereOut(Point3D(p.x, p.y, startLocation_.z), 0.5, Colors::Green);

                supplyDepotPositions.push_back(p);
            }
        }
    }

    if (supplyDepotPositions.size() == 2) {
        Point2D vec = (supplyDepotPositions.at(1) - supplyDepotPositions.at(0));
        Point2D point2_d = Rotate(vec, 90);
        Normalize2D(point2_d);
        point2_d *= 1.5;
        Point2D newPoint = supplyDepotPositions.at(0) + vec / 2 + point2_d;
        if (!bot->Query()->Placement(ABILITY_ID::BUILD_BARRACKS, newPoint)) {
            newPoint -= point2_d * 2;
        }
        Point2D p(newPoint.x, newPoint.y);
        bot->Debug()->DebugSphereOut(Point3D(p.x, p.y, startLocation_.z), 0.5, Colors::Green);
        wallPlacement = p;
        locationQueues[UNIT_TYPEID::TERRAN_BARRACKS].push_back(p);
    }
}

bool requiresPower (sc2::UNIT_TYPEID unit) {
    if(!isStructure(unit)) return false;

    switch(unit) {
        case UNIT_TYPEID::PROTOSS_NEXUS:
        case UNIT_TYPEID::PROTOSS_ASSIMILATOR:
        case UNIT_TYPEID::PROTOSS_PYLON:
            return false;
        default:
            return true;
    }
}

void BuildingPlacement::clearLastKnownGoodPlacements() {
    lastKnownGoodPlacements.clear();
}

default_random_engine rnd(time(0));

Point2D BuildingPlacement::GetReasonablePlacement(sc2::UnitTypeID unitType, sc2::ABILITY_ID abilityType, bool isWarping) {
    auto observation = bot->Observation();
    auto query = bot->Query();

    if (abilityType == sc2::ABILITY_ID::INVALID) abilityType = getUnitData(unitType).ability_id;

    auto& queue = locationQueues[unitType];
    for (auto p : queue) {
        // TOOD: Perhaps add timeout to avoid
        // 1. Two SCVs trying to build in the same location
        // 2. SCVs repeatedly trying, but failing to build in some location
        if (query->Placement(abilityType, p)) {
            return p;
        }
    }

    // Note: Potentially slow as it is O(n)
    auto units = observation->GetUnits(Unit::Alliance::Self, IsStructure(observation));

    // Change seed roughly every 5 seconds
    // default_random_engine rnd(observation->GetGameLoop() / (22*5));
    uniform_real_distribution<double> random01(-0.5, 0.5);

    float rx, ry;

    rx = random01(rnd);
    ry = random01(rnd);
    auto bestPos = Point2D(bot->startLocation_.x + rx * 15.0f, bot->startLocation_.y + ry * 15.0f);
    // Try to find a valid position close to another building

    bool needsNearbyPylon = isWarping || requiresPower(unitType);
    // Pick the potential position with the highest score
    double bestScore = -10000;
    if (units.size() > 0) {
        for (int i = 0; i < 10; i++) {
            uniform_int_distribution<int> randomUnitDist(0, units.size() - 1);
            auto randomUnit = units[randomUnitDist(rnd)];

            // Such random placement
            rx = random01(rnd);
            ry = random01(rnd);

            auto p = Point2D(randomUnit->pos.x + rx * 15.0f, randomUnit->pos.y + ry * 15.0f);
            if (!isWarping && i < lastKnownGoodPlacements.size()) p = lastKnownGoodPlacements[i];

            double score = bot->influenceManager.safeBuildingMap(p);

            // The score is infinity on tiles which the enemy cannot reach at all... but we don't like that because it messes up math
            if (!isfinite(score)) {
                // Check if positive or negative infinity
                if (score > 0) score = 500;
                else continue;
            }

            if (score > bestScore) {
                if (needsNearbyPylon) {
                    int anyNearby = 0;
                    for (auto u : bot->ourUnits()) {
                        if (u->unit_type == UNIT_TYPEID::PROTOSS_PYLON && DistanceSquared2D(u->pos, p) < 6.5f*6.5f) {
                            anyNearby = max(anyNearby, u->build_progress == 1 ? 2 : 1);
                        }
                    }

                    // No nearby pylons
                    if (anyNearby == 0) score -= 1000;

                    // Nearby but not finished pylons
                    if (anyNearby == 1) score -= 100;
                }

                if (score > bestScore && !query->Placement(abilityType, p)) score -= 2000;

                if (score > bestScore) {
                    bestPos = p;
                    bestScore = score;
                }
            }
        }
    }

    lastKnownGoodPlacements.push_back(bestPos);
    if (lastKnownGoodPlacements.size() > 4) lastKnownGoodPlacements.erase(lastKnownGoodPlacements.begin());
    agent->Debug()->DebugSphereOut(Point3D(bestPos.x, bestPos.y, bot->startLocation_.z), 1, query->Placement(abilityType, bestPos) ? Colors::Green : Colors::Red);
    return bestPos;
}