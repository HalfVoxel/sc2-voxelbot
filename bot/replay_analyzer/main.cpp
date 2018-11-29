#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_api.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"

#include <iostream>
#include "../DependencyAnalyzer.h"
#include "sc2utils/sc2_manage_process.h"

const char* kReplayFolder = "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays";
const char* kReplayListProtoss = "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays_protoss.txt";
// Protoss:
// e5242d0a121db241ccfca68150feea57deeb82b9d7000e7d00c84b5cba4e511e.SC2Replay
using namespace std;
using namespace sc2;

class Replay : public sc2::ReplayObserver {
   public:
    std::vector<uint32_t> count_units_built_;

    vector<vector<int>> unit_implies_has_had_unit;
    vector<vector<int>> unit_implies_has_had_unit_total;

    Replay()
        : sc2::ReplayObserver() {
    }

    void OnGameStart() final {
        DependencyAnalyzer deps;
        deps.analyze();
        exit(0);

        cout << "Started game..." << endl;
        const sc2::ObservationInterface* obs = Observation();
        assert(obs->GetUnitTypeData().size() > 0);
        int numUnits = obs->GetUnitTypeData().size();
        count_units_built_ = vector<uint32_t>(numUnits, 0);
        if (unit_implies_has_had_unit.size() == 0) {
            unit_implies_has_had_unit = vector<vector<int>>(numUnits, vector<int>(numUnits, 0));
            unit_implies_has_had_unit_total = vector<vector<int>>(numUnits, vector<int>(numUnits, 0));
        }
    }

    void OnUnitCreated(const sc2::Unit* unit) final {
        return;
        assert(uint32_t(unit->unit_type) < count_units_built_.size());
        ++count_units_built_[unit->unit_type];

        auto ourUnits = Observation()->GetUnits(Unit::Alliance::Self);
        // auto enemyUnits = Observation()->GetUnits(Unit::Alliance::Enemy);
        for (int i = 0; i < count_units_built_.size(); i++) {
            if (count_units_built_[i] > 0) {
                unit_implies_has_had_unit[unit->unit_type][i]++;
            }
            unit_implies_has_had_unit_total[unit->unit_type][i]++;
        }
    }

    void OnStep() final {
    }

    void OnGameEnd() final {
        std::cout << "Units created:" << std::endl;
        const sc2::ObservationInterface* obs = Observation();
        const sc2::UnitTypes& unit_types = obs->GetUnitTypeData();
        for (uint32_t i = 0; i < count_units_built_.size(); ++i) {
            if (count_units_built_[i] == 0) {
                continue;
            }

            std::cout << unit_types[i].name << ": " << std::to_string(count_units_built_[i]) << std::endl;
        }

        for (uint32_t i = 0; i < count_units_built_.size(); ++i) {
            for (uint32_t j = 0; j < count_units_built_.size(); ++j) {
                if (unit_implies_has_had_unit[i][j] == unit_implies_has_had_unit_total[i][j] && unit_implies_has_had_unit_total[i][j] > 2) {
                    cout << unit_types[i].name << " implies " << unit_types[j].name << endl;
                } else if (unit_implies_has_had_unit[i][j] > unit_implies_has_had_unit_total[i][j] * 0.9f) {
                    cout << unit_types[i].name << " softly implies " << unit_types[j].name << " (" << (unit_implies_has_had_unit[i][j] / (double)unit_implies_has_had_unit_total[i][j]) << endl;
                }
            }
        }

        /*int numUnits = unit_implies_has_had_unit.size();
        unit_direct_implications = vector<vector<int>> (numUnits, vector<int>(numUnits, 0));
        for (uint32_t i = 0; i < count_units_built_.size(); ++i) {
            for (uint32_t j = 0; j < count_units_built_.size(); ++j) {
                if (unit_implies_has_had_unit_total[i][j] <= 2) continue;

                float implies = unit_implies_has_had_unit[i][j] / (float)unit_implies_has_had_unit_total[i][j];
                for (uint32_t i2 = 0; i2 < count_units_built_.size(); ++i2) {
                    if (unit_implies_has_had_unit_total[i][i2] <= 2) continue;

            }
        }*/

        std::cout << "Finished" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    // if (!coordinator.SetReplayPath(kReplayFolder)) {
    //     std::cout << "Unable to find replays." << std::endl;
    //     return 1;
    // }
    if (!coordinator.LoadReplayList(kReplayListProtoss)) {
        std::cout << "Unable to find replays." << std::endl;
        return 1;
    }

    Replay replay_observer1;

    coordinator.AddReplayObserver(&replay_observer1);

    while (coordinator.Update())
        ;
    while (!sc2::PollKeyPress())
        ;
}