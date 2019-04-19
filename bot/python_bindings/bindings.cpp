#include <pybind11/stl_bind.h>
#include <pybind11/pybind11.h>
#include "bindings.h"
#include "../ml/replay.h"
#include "../build_order_train/serialized_state.h"
#include "../utilities/sc2_serialization.h"
#include <fstream>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include "../utilities/profiler.h"
#include "../ml/rl_planning_env.h"

using namespace std;
using namespace sc2;

PYBIND11_MAKE_OPAQUE(std::vector<Meh>);
// PYBIND11_MAKE_OPAQUE(std::vector<SerializedState>);
PYBIND11_MAKE_OPAQUE(std::vector<PlayerObservations>);
PYBIND11_MAKE_OPAQUE(std::vector<RawState>);
PYBIND11_MAKE_OPAQUE(std::vector<Unit>);


ReplaySession load_replay(string jsonData, string filepath) {
    Stopwatch w1;
    stringstream json(jsonData);
    cereal::JSONInputArchive archive(json);
    ReplaySession session;
    session.serialize(archive);
    w1.stop();

    Stopwatch w2;
    ofstream json2(filepath);
    {
        cereal::BinaryOutputArchive archive2(json2);
        session.serialize(archive2);
    }
    json2.close();
    w2.stop();

    Stopwatch w3;
    // stringstream json3(json2.str());
    // cereal::BinaryInputArchive archive3(json3);
    // session.serialize(archive3);
    w3.stop();

    cout << w1.millis() << " " << w2.millis() << " " << w3.millis() << endl;
    return session;
}

ReplaySession load_replay2(string jsonData, pybind11::object output) {
    Stopwatch w1;
    stringstream json(jsonData);
    cereal::JSONInputArchive archive(json);
    ReplaySession session;
    session.serialize(archive);
    w1.stop();

    Stopwatch w2;
    stringstream json2;
    {
        cereal::BinaryOutputArchive archive2(json2);
        session.serialize(archive2);
    }
    output.attr("write")(pybind11::bytes(json2.str()));
    w2.stop();

    Stopwatch w3;
    // stringstream json3(json2.str());
    // cereal::BinaryInputArchive archive3(json3);
    // session.serialize(archive3);
    w3.stop();

    cout << w1.millis() << " " << w2.millis() << " " << w3.millis() << endl;
    return session;
}

ReplaySession load_binary2(std::string f) {
    Stopwatch w;
    ReplaySession session;
    {
        stringstream json3(f);
        cereal::BinaryInputArchive archive3(json3);
        session.serialize(archive3);
    }
    w.stop();
    cout << w.millis() << endl;
    return session;
}

PYBIND11_MODULE(botlib_bindings, m) {
	pybind11::bind_vector<std::vector<Meh>>(m, "VectorMeh");
    // pybind11::bind_vector<std::vector<SerializedState>>(m, "VectorSerializedState");
    pybind11::bind_vector<vector<PlayerObservations>>(m, "VectorPlayerObservations");
    pybind11::bind_vector<vector<RawState>>(m, "VectorRawState");
    pybind11::bind_vector<vector<Unit>>(m, "VectorUnit");

    m.def("load", &load_replay2);
    // m.def("load_binary", &loadReplayBinary);

    pybind11::class_<Example>(m, "Example")
        .def(pybind11::init())
        .def_readwrite("meh", &Example::meh)
        .def_readwrite("blah", &Example::blah)
    ;

    pybind11::class_<Meh>(m, "Meh")
        .def(pybind11::init())
        .def_readwrite("what", &Meh::what)
    ;

    pybind11::class_<PlayerInfo>(m, "PlayerInfo")
        .def_readwrite("player_id", &PlayerInfo::player_id)
        .def_readwrite("player_type", &PlayerInfo::player_type)
        .def_readwrite("race_requested", &PlayerInfo::race_requested)
        .def_readwrite("race_actual", &PlayerInfo::race_actual)
        .def_readwrite("difficulty", &PlayerInfo::difficulty)
    ;

    pybind11::class_<GameInfo>(m, "GameInfo")
        .def_readwrite("player_info", &GameInfo::player_info)
        .def_readwrite("map_name", &GameInfo::map_name)
        .def_readwrite("local_map_path", &GameInfo::local_map_path)
        .def_readwrite("width", &GameInfo::width)
        .def_readwrite("height", &GameInfo::height)
        .def_readwrite("playable_min", &GameInfo::playable_min)
        .def_readwrite("playable_max", &GameInfo::playable_max)
    ;

    pybind11::class_<ReplayInfo>(m, "ReplayInfo")
        .def_readwrite("duration_gameloops", &ReplayInfo::duration_gameloops)
        .def_readwrite("replay_path", &ReplayInfo::replay_path)
        .def_readwrite("version", &ReplayInfo::version)
        .def_readwrite("num_players", &ReplayInfo::num_players)
        .def_readwrite("map_name", &ReplayInfo::map_name)
        .def_readwrite("map_path", &ReplayInfo::map_path)
    ;

    pybind11::class_<Point2D>(m, "Point2D")
        .def_readwrite("x", &Point2D::x)
        .def_readwrite("y", &Point2D::y)
    ;

    pybind11::class_<Point3D>(m, "Point3D")
        .def_readwrite("x", &Point3D::x)
        .def_readwrite("y", &Point3D::y)
        .def_readwrite("z", &Point3D::z)
    ;

    
    pybind11::class_<UnitOrder>(m, "UnitOrder")
        .def_readwrite("ability_id", &UnitOrder::ability_id)
        .def_readwrite("target_unit_tag", &UnitOrder::target_unit_tag)
        .def_readwrite("target_pos", &UnitOrder::target_pos)
        .def_readwrite("progress", &UnitOrder::progress);

    pybind11::class_<PassengerUnit>(m, "PassengerUnit")
        .def_readwrite("tag", &PassengerUnit::tag)
        .def_readwrite("health", &PassengerUnit::health)
        .def_readwrite("health_max", &PassengerUnit::health_max)
        .def_readwrite("shield", &PassengerUnit::shield)
        .def_readwrite("shield_max", &PassengerUnit::shield_max)
        .def_readwrite("energy", &PassengerUnit::energy)
        .def_readwrite("energy_max", &PassengerUnit::energy_max)
        .def_readwrite("unit_type", &PassengerUnit::unit_type);

    pybind11::class_<Unit>(m, "Unit")
        .def_readwrite("display_type", &Unit::display_type)
        .def_readwrite("tag", &Unit::tag)
        .def_property("unit_type", [](Unit& u) { return (int)u.unit_type; }, [](Unit& u, int val) { u.unit_type = val; })
        .def_readwrite("owner", &Unit::owner)
        .def_readwrite("pos", &Unit::pos)
        .def_readwrite("facing", &Unit::facing)
        .def_readwrite("radius", &Unit::radius)
        .def_readwrite("build_progress", &Unit::build_progress)
        .def_readwrite("cloak", &Unit::cloak)
        .def_readwrite("detect_range", &Unit::detect_range)
        .def_readwrite("is_blip", &Unit::is_blip)
        .def_readwrite("health", &Unit::health)
        .def_readwrite("health_max", &Unit::health_max)
        .def_readwrite("shield", &Unit::shield)
        .def_readwrite("shield_max", &Unit::shield_max)
        .def_readwrite("energy", &Unit::energy)
        .def_readwrite("energy_max", &Unit::energy_max)
        .def_readwrite("mineral_contents", &Unit::mineral_contents)
        .def_readwrite("vespene_contents", &Unit::vespene_contents)
        .def_readwrite("is_flying", &Unit::is_flying)
        .def_readwrite("is_burrowed", &Unit::is_burrowed)
        .def_readwrite("weapon_cooldown", &Unit::weapon_cooldown)
        .def_readwrite("orders", &Unit::orders)
        .def_readwrite("add_on_tag", &Unit::add_on_tag)
        .def_readwrite("passengers", &Unit::passengers)
        .def_readwrite("cargo_space_taken", &Unit::cargo_space_taken)
        .def_readwrite("cargo_space_max", &Unit::cargo_space_max)
        .def_readwrite("engaged_target_tag", &Unit::engaged_target_tag)
        .def_readwrite("buffs", &Unit::buffs)
        .def_readwrite("is_powered", &Unit::is_powered)
        .def_readwrite("is_alive", &Unit::is_alive)
        .def_readwrite("last_seen_game_loop", &Unit::last_seen_game_loop);

    pybind11::class_<RawState>(m, "RawState")
        .def_readwrite("units", &RawState::units, pybind11::return_value_policy::reference);

    pybind11::class_<PlayerObservations>(m, "PlayerObservations")
        .def_readwrite("selfStates", &PlayerObservations::selfStates, pybind11::return_value_policy::reference)
        .def_readwrite("enemyStates", &PlayerObservations::enemyStates, pybind11::return_value_policy::reference)
        .def_readwrite("rawUnits", &PlayerObservations::rawUnits, pybind11::return_value_policy::reference)
    ;

    pybind11::class_<ObserverSession>(m, "ObserverSession")
        .def_readwrite("observations", &ObserverSession::observations, pybind11::return_value_policy::reference)
        .def_readwrite("gameInfo", &ObserverSession::gameInfo, pybind11::return_value_policy::reference)
        .def_readwrite("replayInfo", &ObserverSession::replayInfo, pybind11::return_value_policy::reference)
        .def_readwrite("winner", &ObserverSession::winner)
    ;

    pybind11::class_<ReplaySession>(m, "ReplaySession")
        .def_readwrite("observations", &ReplaySession::observations, pybind11::return_value_policy::reference)
        .def_readwrite("gameInfo", &ReplaySession::gameInfo, pybind11::return_value_policy::reference)
        .def_readwrite("replayInfo", &ReplaySession::replayInfo, pybind11::return_value_policy::reference)
        .def_readwrite("winner", &ReplaySession::winner)
        .def_readwrite("mmrs", &ReplaySession::mmrs)
    ;

    pybind11::class_<SerializedState>(m, "SerializedState")
        .def_readwrite("time", &SerializedState::time)
        .def_readwrite("units", &SerializedState::units)
        .def_readwrite("unitsInProgress", &SerializedState::unitsInProgress)
        .def_readwrite("minerals", &SerializedState::minerals)
        .def_readwrite("vespene", &SerializedState::vespene)
        .def_readwrite("mineralsPerSecond", &SerializedState::mineralsPerSecond)
        .def_readwrite("vespenePerSecond", &SerializedState::vespenePerSecond)
        .def_readwrite("foodAvailable", &SerializedState::foodAvailable)
        .def_readwrite("race", &SerializedState::race)
        .def_readwrite("highYieldMineralSlots", &SerializedState::highYieldMineralSlots)
        .def_readwrite("lowYieldMineralSlots", &SerializedState::lowYieldMineralSlots)
        .def_readwrite("version", &SerializedState::version)
        .def_readwrite("upgrades", &SerializedState::upgrades)
    ;

    pybind11::class_<RLPlanningEnv>(m, "RLPlanningEnv")
        .def("step", &RLPlanningEnv::step)
        .def("observe", &RLPlanningEnv::observe, pybind11::return_value_policy::move)
        .def("print", &RLPlanningEnv::print)
        .def("actionName", &RLPlanningEnv::actionName)
        .def("visualizationInfo", &RLPlanningEnv::visualizationInfo)
    ;

    pybind11::class_<RLEnvManager>(m, "RLEnvManager")
        .def(pybind11::init<pybind11::object, pybind11::object, vector<string>>())
        .def("getEnv", &RLEnvManager::getEnv)
    ;
}
