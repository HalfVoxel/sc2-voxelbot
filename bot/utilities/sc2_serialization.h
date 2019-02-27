#pragma once
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace sc2 {
    template <class Archive>
    void serialize(Archive& archive, PlayerInfo& playerInfo) {
        archive(
            cereal::make_nvp("player_id", playerInfo.player_id),
            cereal::make_nvp("player_type", playerInfo.player_type),
            cereal::make_nvp("race_requested", playerInfo.race_requested),
            cereal::make_nvp("race_actual", playerInfo.race_actual),
            cereal::make_nvp("difficulty", playerInfo.difficulty)
        );
    }

    template <class Archive>
    void serialize(Archive& archive, GameInfo& gameInfo) {
        archive(
            cereal::make_nvp("player_info", gameInfo.player_info),
            cereal::make_nvp("map_name", gameInfo.map_name),
            cereal::make_nvp("local_map_path", gameInfo.local_map_path),
            cereal::make_nvp("width", gameInfo.width),
            cereal::make_nvp("height", gameInfo.height),
            cereal::make_nvp("playable_min", gameInfo.playable_min),
            cereal::make_nvp("playable_max", gameInfo.playable_max)
        );
    }

    template <class Archive>
    void serialize(Archive& archive, ReplayInfo& replayInfo) {
        archive(
            cereal::make_nvp("duration_gameloops", replayInfo.duration_gameloops),
            cereal::make_nvp("replay_path", replayInfo.replay_path),
            cereal::make_nvp("version", replayInfo.version),
            cereal::make_nvp("num_players", replayInfo.num_players),
            cereal::make_nvp("map_name", replayInfo.map_name),
            cereal::make_nvp("map_path", replayInfo.map_path)
        );
    }

    template <class Archive>
    void serialize(Archive& archive, Point2D& p) {
        archive(
            cereal::make_nvp("x", p.x),
            cereal::make_nvp("y", p.y)
        );
    }

    template <class Archive>
    void serialize(Archive& archive, Point3D& p) {
        archive(
            cereal::make_nvp("x", p.x),
            cereal::make_nvp("y", p.y),
            cereal::make_nvp("z", p.z)
        );
    }

    template <class Archive>
    void serialize(Archive& archive, Unit& unit) {
        UNIT_TYPEID unit_type = unit.unit_type;
        archive(
            cereal::make_nvp("display_type", unit.display_type),
            cereal::make_nvp("tag", unit.tag),
            cereal::make_nvp("unit_type", unit_type),
            cereal::make_nvp("owner", unit.owner),
            cereal::make_nvp("pos", unit.pos),
            cereal::make_nvp("facing", unit.facing),
            cereal::make_nvp("radius", unit.radius),
            cereal::make_nvp("build_progress", unit.build_progress),
            cereal::make_nvp("cloak", unit.cloak),
            cereal::make_nvp("detect_range", unit.detect_range),
            cereal::make_nvp("is_blip", unit.is_blip),
            cereal::make_nvp("health", unit.health),
            cereal::make_nvp("health_max", unit.health_max),
            cereal::make_nvp("shield", unit.shield),
            cereal::make_nvp("shield_max", unit.shield_max),
            cereal::make_nvp("energy", unit.energy),
            cereal::make_nvp("energy_max", unit.energy_max),
            cereal::make_nvp("mineral_contents", unit.mineral_contents),
            cereal::make_nvp("vespene_contents", unit.vespene_contents),
            cereal::make_nvp("is_flying", unit.is_flying),
            cereal::make_nvp("is_burrowed", unit.is_burrowed),
            cereal::make_nvp("weapon_cooldown", unit.weapon_cooldown),
            cereal::make_nvp("cargo_space_taken", unit.cargo_space_taken),
            cereal::make_nvp("cargo_space_max", unit.cargo_space_max),
            cereal::make_nvp("engaged_target_tag", unit.engaged_target_tag),
            cereal::make_nvp("is_powered", unit.is_powered),
            cereal::make_nvp("is_alive", unit.is_alive)
        );
    }
};
