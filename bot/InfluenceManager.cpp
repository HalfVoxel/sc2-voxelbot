#include "Influence.h"
#include "Bot.h"
#include "Predicates.h"
#include "Pathfinding.h"
#include "Renderer.h"

using namespace std;
using namespace sc2;

InfluenceMap pathing_grid;
InfluenceMap pathing_cost;
InfluenceMap placement_grid;
InfluenceMap enemyDensity;
InfluenceMap valueMap;
InfluenceMap scoutingMap;

void InfluenceManager::Init() {
    // 1 at traversable cells, 0 at walls
    pathing_grid = InfluenceMap(bot.game_info_.pathing_grid);

    // Remove the original command center/nexus/hatchery from the pathfinding map
    // (it is marked as blocked at the start of the game, we don't want that)
    vector<vector<double>> square5x5 = { { 0, 1, 1, 1, 0 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 0, 1, 1, 1, 0 } };
    pathing_grid.addInfluence(square5x5, bot.startLocation_ - Point2D(0.5, 0.5));
    for (auto p : bot.game_info_.enemy_start_locations) {
        pathing_grid.addInfluence(square5x5, p - Point2D(0.5, 0.5));
    }

    pathing_cost = (pathing_grid - 1).replace_nonzero(numeric_limits<double>::infinity()) + 1;

    placement_grid = InfluenceMap(bot.game_info_.placement_grid);
    enemyDensity = InfluenceMap(pathing_grid.w, pathing_grid.h);
    valueMap = InfluenceMap(pathing_grid.w, pathing_grid.h);
    safeBuildingMap = InfluenceMap(pathing_grid.w, pathing_grid.h);
    scoutingMap = InfluenceMap(pathing_grid.w, pathing_grid.h);
}

void InfluenceManager::OnStep() {
    const int InfluenceFrameInterval = 10;
    if ((ticks % InfluenceFrameInterval) == 0) {

        double scoutingUncertainty = 0.005;
        double spread = 5;
        auto observation = bot.Observation();
        for (auto unit : observation->GetUnits(Unit::Alliance::Self)) {
            if (IsStructure(observation)(*unit)) {
                valueMap.addInfluence(1.0 / spread, unit->pos);
            } else if (!IsArmy(observation)(*unit)) {
                valueMap.addInfluence(0.2 / spread, unit->pos);
            }
        }
        valueMap.propagateSum(0.0, 1.0, pathing_grid);

        // Normalize
        valueMap *= 1.0/(0.0001 + valueMap.maxFinite());

        for (auto unit : observation->GetUnits(Unit::Alliance::Enemy)) {
            if (IsStructure(observation)(*unit)) {
                enemyDensity.addInfluence(1, unit->pos);
                scoutingMap.addInfluence(scoutingUncertainty, unit->pos);
            } else if (IsArmy(observation)(*unit)) {
                enemyDensity.addInfluence(0.8, unit->pos);
            } else {
                enemyDensity.addInfluence(0.2, unit->pos);
            }
        }

        for (auto p : bot.game_info_.enemy_start_locations) {
            enemyDensity.addInfluence(1, p);
            scoutingMap.addInfluence(scoutingUncertainty, p);
        }

        for(int i = 0; i < scoutingMap.w; i++){
            for (int j = 0; j < scoutingMap.h; j++) {
                Point2D p = Point2D(i, j);
                if(observation->GetVisibility(p) != Visibility::Visible){
                    scoutingMap.addInfluence(scoutingUncertainty, p);
                } else {
                    scoutingMap.setInfluence(0, p);
                }
            }
        }

        enemyDensity.propagateSum(exp(-4), 0.2, pathing_grid);

        // Binary map of where the enemy is
        auto flood = enemyDensity;
        flood.threshold(flood.max() * 0.2);

        // Normalized distances to the enemy between 0 and 1 where 1 the size of the whole map
        auto distances = getDistances(flood, pathing_cost + valueMap*2000);
        distances *= 1.0/max(distances.w, distances.h);

        // How useful it is to defend a particular point
        // focuses on the perimeter of the base
        auto defensivePotential = valueMap / (distances + 20);
        defensivePotential *= 1 / (0.001 + defensivePotential.maxFinite());

        // Find the best spot to defend
        // TODO: Send different squads to different positions
        auto best = defensivePotential.argmax();
        bot.tactical_manager->preferredArmyPosition = Point2D(best.x, best.y);

        // Make it good to build buildings as far away from the enemy as possible
        // preferably behind other buildings and defences.
        safeBuildingMap = distances;

        // Render all maps for debugging
        // Coordinates in a tile layout with 0,0 being the top-left corner of the debugging window.
        distances.render(0, 0);
        enemyDensity.render(0, 1);
        valueMap.renderNormalized(0, 2);
        scoutingMap.render(1, 0);
        flood.render(1, 1);
        defensivePotential.render(1, 2);
        Render();
    }
}