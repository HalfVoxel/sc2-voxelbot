#include <libvoxelbot/utilities/influence.h>
#include <libvoxelbot/utilities/pathfinding.h>
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/utilities/renderer.h>
#include "sc2api/sc2_proto_to_pods.h"
#include "Bot.h"

using namespace std;
using namespace sc2;

InfluenceMap distanceCache;

void InfluenceManager::Init(MapRenderer* renderer) {
    this->renderer = renderer;

    // 1 at traversable cells, 0 at walls
    pathing_grid = InfluenceMap(bot->game_info_.pathing_grid);

    // Remove the original command center/nexus/hatchery from the pathfinding map
    // (it is marked as blocked at the start of the game, we don't want that)
    vector<vector<double>> square5x5 = { { 0, 1, 1, 1, 0 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 0, 1, 1, 1, 0 } };
    pathing_grid.addInfluence(square5x5, bot->startLocation_ - Point2D(0.5, 0.5));
    for (auto p : bot->game_info_.enemy_start_locations) {
        pathing_grid.addInfluence(square5x5, p - Point2D(0.5, 0.5));
    }

    // 1 vs infinity in cost
    pathing_cost = (pathing_grid - 1).replace_nonzero(numeric_limits<double>::infinity()) + 1;
    // 1 vs 100 in cost
    pathing_cost_finite = (pathing_grid - 1).replace_nonzero(99) + 1;

    placement_grid = InfluenceMap(bot->game_info_.placement_grid);
    enemyDensity = InfluenceMap(pathing_grid.w, pathing_grid.h);
    valueMap = InfluenceMap(pathing_grid.w, pathing_grid.h);
    safeBuildingMap = InfluenceMap(pathing_grid.w, pathing_grid.h);
    scoutingMap = InfluenceMap(pathing_grid.w, pathing_grid.h);
    scanningMap = InfluenceMap(pathing_grid.w, pathing_grid.h);
    distanceCache = InfluenceMap(pathing_grid.w, pathing_grid.h);
    lastSeenMap = InfluenceMap(pathing_grid.w, pathing_grid.h);
}

double millis() {
    return (1000 * clock()) / CLOCKS_PER_SEC;
}

void InfluenceManager::OnStep() {
    const int InfluenceFrameInterval = 10;
    const int DistanceFrameInterval = 50;
    if ((ticks % InfluenceFrameInterval) == 0) {
        double scoutingUncertainty = 0.005;
        double spread = 5;
        auto observation = bot->Observation();
        for (auto unit : observation->GetUnits(Unit::Alliance::Self)) {
            if (isStructure(unit->unit_type)) {
                valueMap.addInfluence(1.0 / spread, unit->pos);
            } else if (!IsArmy(observation)(*unit)) {
                valueMap.addInfluence(0.2 / spread, unit->pos);
            }
        }
        valueMap.propagateSum(0.0, 1.0, pathing_grid);

        // Normalize
        valueMap *= 1.0 / (0.0001 + valueMap.maxFinite());

        /*for (auto unit : bot->enemyUnits()) {
            if (isStructure(unit->unit_type)) {
                enemyDensity.addInfluence(1, unit->pos);
                scoutingMap.addInfluence(scoutingUncertainty, unit->pos);
            } else if (isArmy(unit->unit_type)) {
                enemyDensity.addInfluence(0.8, unit->pos);
            } else {
                enemyDensity.addInfluence(0.2, unit->pos);
            }
        }*/
        for (auto unit : bot->deductionManager.SampleUnitPositions(1)) {
            if (isStructure(unit.first.type)) {
                enemyDensity.addInfluence(1, unit.second);
                scoutingMap.addInfluence(scoutingUncertainty, unit.second);
            } else if (isArmy(unit.first.type)) {
                enemyDensity.addInfluence(0.8, unit.second);
            } else {
                enemyDensity.addInfluence(0.2, unit.second);
            }
        }

        for (auto p : bot->game_info_.enemy_start_locations) {
            // enemyDensity.addInfluence(1, p);
            scoutingMap.addInfluence(scoutingUncertainty * 1.5, p);
        }

        for (auto p : bot->expansions_) {
            scoutingMap.addInfluence(scoutingUncertainty, p);
        }

        // 0 = invisible, 1 = partial, 2 = full visibility
        auto visibilityMap = InfluenceMap(bot->Observation()->GetVisibilityMap());

        for (int y = 0; y < scoutingMap.h; y++) {
            for (int x = 0; x < scoutingMap.w; x++) {
                Point2D p = Point2D(x, y);
                if (visibilityMap(x, y) != 2 && pathing_grid(p) != 0) {
                    scoutingMap.addInfluence(scoutingUncertainty, p);
                    lastSeenMap.addInfluence(ticksToSeconds(InfluenceFrameInterval), p);
                } else {
                    scoutingMap.setInfluence(0, p);
                    scanningMap.setInfluence(0, p);
                    lastSeenMap.setInfluence(0, p);
                }
            }
        }

        enemyDensity.propagateSum(exp(-4), 0.2, pathing_grid);

        if ((ticks % DistanceFrameInterval) == 0) {
            // Binary map of where the enemy is
            auto flood = enemyDensity;
            flood.threshold(flood.max() * 0.2);

            // Normalized distances to the enemy between 0 and 1 where 1 the size of the whole map
            distanceCache = getDistances(flood, pathing_cost + valueMap * 2000);
            auto& distances = distanceCache;
            distances *= 1.0 / max(distances.w, distances.h);

            // How useful it is to defend a particular point
            // focuses on the perimeter of the base
            auto defensivePotential = valueMap / (distances + 20);
            defensivePotential *= 1 / (0.001 + defensivePotential.maxFinite());

            // Find the best spot to defend
            // TODO: Send different squads to different positions
            auto best = defensivePotential.argmax();
            if (bot->tacticalManager != nullptr) bot->tacticalManager->preferredArmyPosition = Point2D(best.x, best.y);

            // Make it good to build buildings as far away from the enemy as possible
            // preferably behind other buildings and defences.
            // Also avoid including infinities
            // safeBuildingMap = distances.replace(numeric_limits<double>::infinity(), 1000);
            safeBuildingMap = distances;

            defensivePotential *= 1.0 / defensivePotential.max();

            auto g = placement_grid;
            g.threshold(1);
            auto g2 = (g * -1) + 1;
            g.propagateMax(0.0, 1.0, g2);
            g.propagateMax(0.0, 1.0, g2);

            // Invert
            g.max(valueMap);
            g.propagateMax(0.0, 0.7, g2);
            g.propagateMax(0.0, 0.7, g2);
            g.propagateMax(0.0, 0.7, g2);
            renderer->renderInfluenceMap(g, 1, 2);

            g *= -1;
            g += 1;

            safeBuildingMap *= g;

            // Normalize
            safeBuildingMap *= 1.0 / safeBuildingMap.maxFinite();
            safeBuildingMap = safeBuildingMap.replace_nan(numeric_limits<double>::infinity());
            
            // defensivePotential.renderNormalized(1, 2);
        }

        // Render all maps for debugging
        // Coordinates in a tile layout with 0,0 being the top-left corner of the debugging window.
        auto d2 = distanceCache;
        d2 *= 1.0 / d2.maxFinite();

        renderer->renderInfluenceMapNormalized(d2, 0, 0);
        // enemyDensity.render(0, 1);
        // valueMap.renderNormalized(0, 2);
        renderer->renderInfluenceMap(scoutingMap, 1, 1);
        // flood.render(1, 1);
        // scanningMap.render(1, 1);
        renderer->renderInfluenceMap(placement_grid, 0, 1);
        renderer->renderInfluenceMapNormalized(safeBuildingMap, 0, 2);
        renderer->renderInfluenceMap(visibilityMap, 1, 0);
    }
}