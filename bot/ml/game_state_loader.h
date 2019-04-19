#pragma once
#include "sc2api/sc2_api.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "replay.h"
#include "simulator.h"


struct CoordinateRemapper {
    sc2::Point2D mn;
    sc2::Point2D mx;
    bool flipX;
    bool flipY;

    CoordinateRemapper(sc2::Point2D mn, sc2::Point2D mx, bool flipX, bool flipY) : mn(mn), mx(mx), flipX(flipX), flipY(flipY) {}

    sc2::Point2D transform(sc2::Point2D coord) const;
    sc2::Point2DI transformCell(sc2::Point2D coord, int scale) const;
    sc2::Point2D inverseTransform(sc2::Point2D coord) const;
};

ReplaySession loadReplayBinary(std::string data);

bool filterSession(const ReplaySession& session);

SimulatorState loadPlanningEnvSession(const ReplaySession& session);
// struct GameStateLoader {
//     const AvailableUnitTypes& unitMappings;
//     const ObserverSession& session;

//     GameStateLoader(const ObserverSession& session, const AvailableUnitTypes& unitMappings) : session(session), unitMappings(unitMappings) {}

//     // void loadMovementTarget();
// };

// void loadSessionMovementTarget(ObserverSession& session, UnitMappings unitMappings) {
//     if (!filterSession(session, unitMappings))
// }

// def loadSessionMovementTarget(session, loader: BuildOrderLoader, store_fn, statistics):
//     if not filterSession(session, loader):
//         return

//     map_name = session["gameInfo"]["map_name"]
//     if map_name == "Stasis LE":
//         print(f"Skipping blacklisted map {map_name}")
//         return

//     # winner = 0 | 1
//     winner = calculateWinner(session)
//     if winner is None:
//         return

//     # Map unit -> last known position
//     # For every time
//     #   If unit.pos is far from its last known position. Mark the unit as moved

//     playerID = winner + 1
//     observationSession = {
//         "observations": session["observations"][playerID-1],
//         "gameInfo": session["gameInfo"],
//         "replayInfo": session["replayInfo"]
//     }
//     res = loadSessionMovementTarget2(observationSession, playerID, loader, 'random', session["data_path"])
//     if res is not None:
//         store_fn(res)


// def loadSessionMovementTarget2(observationSession, playerID, loader: BuildOrderLoader, unit_tag_mask, data_path):
//     ''' unit_tag_mask is either 'random' or a set of unit tags'''

//     replay_path = observationSession["replayInfo"]["replay_path"]
//     playerIndex = playerID - 1
//     opponentPlayerID = 3 - playerID

//     map_size = find_map_size(observationSession, playerID)

//     # Coordinates are normalized so that player [playerID] is always in the lower left corner.
//     mirror = False

//     selfStates = observationSession["observations"]["selfStates"]
//     rawUnits = observationSession["observations"]["rawUnits"]
//     minimap_size = 14

//     # In timesteps, so Nx5 seconds
//     # Note: if unit_tag_mask is provided then we don't use lookahead for anything, so it doesn't constrain the times
//     lookaheadTime = 4 if unit_tag_mask == 'random' else 0
//     max_time = len(rawUnits) - lookaheadTime

//     if max_time <= 0:
//         print("Skipping game with too few samples")
//         return

//     player1obs2 = [
//         playerObservationTensor2(loader, s, r, playerID, map_size, minimap_size, mirror)
//         for (s, r) in zip(selfStates[:max_time], rawUnits[:max_time])
//     ]

//     player1minimap1 = [minimapLayers(loader, r, playerID, map_size, minimap_size, mirror) for r in rawUnits[:max_time]]
//     player1minimap2 = [minimapLayers(loader, r, opponentPlayerID, map_size, minimap_size, mirror) for r in rawUnits[:max_time]]
//     player1minimap = torch.stack([torch.cat((m1, m2), dim=0) for (m1, m2) in zip(player1minimap1, player1minimap2)])

//     if unit_tag_mask == "random":
//         player1movementMinimap, unit_type_counts, target_positions, fractionSimilarOrders, attack_order = sampleMovementGroupOrder(rawUnits, playerID, map_size, loader, minimap_size, mirror, lookaheadTime)
//     else:
//         target_positions = None
//         assert len(rawUnits) == 1
//         units = [u for u in rawUnits[0]["units"] if u["tag"] in unit_tag_mask]
//         unitsComplement = [u for u in rawUnits[0]["units"] if u["owner"] == playerID and isMovableUnit(u, loader) and u["tag"] not in unit_tag_mask]
//         player1movementMinimap, unit_type_counts, fractionSimilarOrders, attack_order = movementGroupFeatures(units, None, unitsComplement, rawUnits[0]["units"], None, loader, map_size, minimap_size, mirror)
//         player1movementMinimap = torch.tensor(player1movementMinimap).unsqueeze(0)
//         unit_type_counts = torch.tensor(unit_type_counts).unsqueeze(0)
//         fractionSimilarOrders = torch.tensor(fractionSimilarOrders, dtype=torch.float32).unsqueeze(0)

//     player1minimap = torch.cat([player1minimap, player1movementMinimap], dim=1)

//     return MovementTargetTrace(
//         states=torch.stack([x[0] for x in player1obs2]),
//         minimap_states=player1minimap,
//         replay_path=replay_path,
//         target_positions=target_positions,
//         unit_type_counts=unit_type_counts,
//         fraction_similar_orders=fractionSimilarOrders,
//         attack_order=attack_order,
//         data_path=data_path,
//         playerID=playerID,
//         pathfinding_minimap=loadPathfindingMinimap(observationSession, map_size, mirror)
//     )