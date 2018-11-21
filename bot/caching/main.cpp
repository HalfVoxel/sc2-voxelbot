#include <fstream>
#include <iostream>
#include <queue>
#include "../Bot.h"
#include "../DependencyAnalyzer.h"
#include "../Mappings.h"
#include "../generated/abilities.h"
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"

using namespace sc2;
using namespace std;

static const char* EmptyMap = "Test/Empty.SC2Map";

// Model
// Σ Pi + Σ PiSij
class CachingBot : public sc2::Agent {
    ofstream results;
    int tick = 0;

   public:
    void OnGameLoading() {
    }

    void OnGameStart() override {
        results = ofstream("bot/generated/abilities.h");
        results << "#pragma once" << endl;
        results << "#include<vector>" << endl;
        results << "extern std::vector<std::pair<int,int>> unit_type_has_ability;" << endl;
        results.close();

        results = ofstream("bot/generated/abilities.cpp");
        results << "#include \"abilities.h\"" << endl;
        Debug()->DebugEnemyControl();
        Debug()->DebugShowMap();
        Debug()->DebugIgnoreFood();
        Debug()->DebugIgnoreResourceCost();
        Debug()->DebugGiveAllTech();
        initMappings(Observation());

        const sc2::UnitTypes& unit_types = Observation()->GetUnitTypeData();

        Point2D mn = Observation()->GetGameInfo().playable_min;
        Point2D mx = Observation()->GetGameInfo().playable_max;

        DependencyAnalyzer deps;
        deps.analyze(Observation());

        // Note: this depends on output from this program, so it really has to be run twice to get the correct output.
        ofstream unit_lookup = ofstream("bot/generated/units.txt");
        for (const UnitTypeData& type : unit_types) {
            if (!type.available)
                continue;
            unit_lookup << type.name << " " << UnitTypeToName(type.unit_type_id) << endl;
            unit_lookup << "\tID: " << (int)type.unit_type_id << endl;
            unit_lookup << "\tMinerals: " << type.mineral_cost << endl;
            unit_lookup << "\tVespene: " << type.vespene_cost << endl;
            unit_lookup << "\tSupply Required: " << type.food_required << endl;
            unit_lookup << "\tSupply Provided: " << type.food_provided << endl;
            unit_lookup << "\tBuild time: " << type.build_time << endl;
            unit_lookup << "\tArmor: " << type.armor << endl;
            unit_lookup << "\tMovement speed: " << type.movement_speed << endl;
            for (auto t : type.tech_alias) {
                unit_lookup << "\tTech Alias: " << UnitTypeToName(t) << endl;
            }
            unit_lookup << "\tUnit Alias: " << UnitTypeToName(type.unit_alias) << endl;
            unit_lookup << "\tAbility: " << AbilityTypeToName(type.ability_id) << endl;
            unit_lookup << "\tHas been: ";
            for (auto u : hasBeen(type.unit_type_id)) {
                unit_lookup << UnitTypeToName(u) << ", ";
            }
            unit_lookup << endl;
            unit_lookup << "\tCan become: ";
            for (auto u : canBecome(type.unit_type_id)) {
                unit_lookup << UnitTypeToName(u) << ", ";
            }
            unit_lookup << endl;
            unit_lookup << "\tDependencies: ";
            for (auto u : deps.allUnitDependencies[(int)type.unit_type_id]) {
                unit_lookup << UnitTypeToName(u) << ", ";
            }
            unit_lookup << endl;

            unit_lookup << endl;
        }

        int i = 0;
        for (const UnitTypeData& type : unit_types) {
            if (!type.available)
                continue;
            float x = mn.x + ((rand() % 1000) / 1000.0f) * (mx.x - mn.x);
            float y = mn.y + ((rand() % 1000) / 1000.0f) * (mx.y - mn.y);
            Debug()->DebugCreateUnit(type.unit_type_id, Point2D(x, y));
            i++;
        }

        Debug()->SendDebug();
    }

    void OnStep() override {
        if (tick == 2) {
            results << "std::vector<std::pair<int,int>> unit_type_has_ability = {" << endl;
            auto ourUnits = Observation()->GetUnits(Unit::Alliance::Self);
            auto abilities = Query()->GetAbilitiesForUnits(ourUnits, true);
            const sc2::UnitTypes& unitTypes = Observation()->GetUnitTypeData();
            for (int i = 0; i < ourUnits.size(); i++) {
                AvailableAbilities& abilitiesForUnit = abilities[i];
                for (auto availableAbility : abilitiesForUnit.abilities) {
                    results << "\t{ " << ourUnits[i]->unit_type << ", " << availableAbility.ability_id << " }," << endl;
                    cout << unitTypes[ourUnits[i]->unit_type].name << " has " << AbilityTypeToName(availableAbility.ability_id) << endl;
                }
            }

            results << "};" << endl;

            Debug()->DebugEndGame(false);
        }

        Actions()->SendActions();
        Debug()->SendDebug();
        tick++;
    }
};

int main(int argc, char* argv[]) {
    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetMultithreaded(true);

    CachingBot bot;
    agent = bot;
    coordinator.SetParticipants({ CreateParticipant(Race::Terran, &bot) });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    bot.OnGameLoading();
    coordinator.StartGame(EmptyMap);

    while (coordinator.Update() && !do_break) {
        if (PollKeyPress()) {
            do_break = true;
        }
    }
    return 0;
}
