#include "BuildingPlacement.h"
#include <vector>
#include "Predicates.h"

using namespace std;
using namespace sc2;

InfluenceMap safeBuildingMap;

Point2D Rotate(Point2D p, float degrees) {
    degrees = degrees * 3.14159265358979323846 / 180;
    return Point2D(p.x * cos(degrees) - p.y * sin(degrees),
                       p.x * sin(degrees) + p.y * cos(degrees));
}

void BuildingPlacement::OnGameStart() {
    FindWallPlacements(bot.startLocation_);
}

void BuildingPlacement::OnStep() {

}
void BuildingPlacement::FindWallPlacements(Point3D startLocation_) {
    auto& game_info_ = bot.game_info_;
    size_t size = game_info_.placement_grid.data.size();
    vector<int> diff(size);

    for (int i = 0; i < size; ++i) {
        if (game_info_.placement_grid.data[i] == 0 && game_info_.pathing_grid.data[i] == 0) {
            diff[i] = 1;
            Point2D p = bot.GetMapCoordinate(i);
            bot.Debug()->DebugSphereOut(Point3D(p.x + 0.5, p.y - 0.5, startLocation_.z), 0.5, Colors::Red);
        }
    }

    int mapHeuristic = game_info_.height;
    int start_index = bot.GetPositionIndex(startLocation_.x, startLocation_.y);
    Point2D start2D = Point2D(startLocation_.x, startLocation_.y);
    for (int i = 0; i < size; ++i) {
        if (diff[i] == 1) {
            for (int j = 0; j < size; ++j) {
                Point2D p = bot.GetMapCoordinate(j);
                if (Distance2D(bot.GetMapCoordinate(i), p) <= 2 && game_info_.pathing_grid.data[j] ==0 && diff[j] == 0) {
                    if (abs(game_info_.terrain_height.data[j] - game_info_.terrain_height.data[start_index]) < 2 && Distance2D(p, start2D) < mapHeuristic / 4) { //Height filter messes up on some maps: 
                        bot.Debug()->DebugSphereOut(Point3D(p.x + 0.5, p.y - 0.5, startLocation_.z),0.5, Colors::Blue);
                        diff[j] = 2;
                    }
                }
            }
        }
    }

    auto& supplyDepotPositions = locationQueues[UNIT_TYPEID::TERRAN_SUPPLYDEPOT];
    for (int i = 0; i < size; ++i) {
        if (diff[i] == 2) {
            Point2D p = bot.GetMapCoordinate(i);
            if (diff[bot.GetPositionIndex(p.x - 1, p.y)] == 2 && diff[bot.GetPositionIndex(p.x - 1, p.y + 1)
                ] == 2 && diff[bot.GetPositionIndex(p.x, p.y + 1)] == 2) {
                bot.Debug()->DebugSphereOut(Point3D(p.x + 0.5, p.y - 0.5, startLocation_.z), 0.5, Colors::Green);

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
        if (!bot.Query()->Placement(ABILITY_ID::BUILD_BARRACKS, newPoint)) {
            newPoint -= point2_d * 2;
        }
        Point2D p(newPoint.x, newPoint.y);
        bot.Debug()->DebugSphereOut(Point3D(p.x, p.y, startLocation_.z), 0.5, Colors::Green);
        wallPlacement = p;
        locationQueues[UNIT_TYPEID::TERRAN_BARRACKS].push_back(p);
    }
    bot.Debug()->SendDebug();
}

Point2D BuildingPlacement::GetReasonablePlacement(sc2::UnitTypeID unitType) {
    auto observation = bot.Observation();
    auto query = bot.Query();

    auto abilityType = observation->GetUnitTypeData(false)[unitType].ability_id;

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

    float rx, ry;

    rx = GetRandomScalar();
    ry = GetRandomScalar();
    auto bestPos = Point2D(bot.startLocation_.x + rx * 15.0f, bot.startLocation_.y + ry * 15.0f);
    // Try to find a valid position close to another building
    if (units.size() > 0) {
        // Pick the potential position with the highest score
        double bestScore = -1000;
        for (int i = 0; i < 10; i++) {
            auto randomUnit = units[rand() % units.size()];

            // Such random placement
            rx = GetRandomScalar();
            ry = GetRandomScalar();

            auto p = Point2D(randomUnit->pos.x + rx * 15.0f, randomUnit->pos.y + ry * 15.0f);

            double score = safeBuildingMap(p);
            if (score > bestScore && query->Placement(abilityType, p)) {
                bestPos = p;
                bestScore = score;
            }
        }
    }

    return bestPos;
}