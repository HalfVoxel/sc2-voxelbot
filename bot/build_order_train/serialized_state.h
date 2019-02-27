#pragma once
#include "../BuildOptimizerGenetic.h"
#include "../utilities/mappings.h"
#include "../utilities/predicates.h"
#include "../utilities/stdutils.h"
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <vector>
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"

struct SerializedUnitInProgress {
    sc2::UNIT_TYPEID type;
    float remainingTime; // seconds

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(type),
            CEREAL_NVP(remainingTime)
        );
    }
};

struct UnitCount {
    sc2::UNIT_TYPEID type;
    int count;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(type), CEREAL_NVP(count));
    }
};

struct SerializedUnit {
    sc2::UNIT_TYPEID type;
    sc2::UNIT_TYPEID addon;
    int totalCount;
    int availableCount;

    SerializedUnit(const BuildUnitInfo& unit)
        : type(unit.type), addon(unit.addon), totalCount(unit.units), availableCount(unit.availableUnits()) {
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(type),
            CEREAL_NVP(addon),
            CEREAL_NVP(totalCount),
            CEREAL_NVP(availableCount)
        );
    }
};

struct SerializedState {
    float time;
    std::vector<SerializedUnit> units;
    std::vector<SerializedUnitInProgress> unitsInProgress;
    float minerals;
    float vespene;
    float mineralsPerSecond;
    float vespenePerSecond;
    float foodAvailable;
    sc2::Race race;
    int highYieldMineralSlots;
    int lowYieldMineralSlots;
    int version;
    std::vector<BaseInfo> baseInfos;
    std::vector<sc2::UPGRADE_ID> upgrades;

    SerializedState(const BuildState& state) {
        for (auto& u : state.units) {
            units.push_back(SerializedUnit(u));
        }

        for (auto& ev : state.events) {
            if (ev.type == BuildEventType::FinishedUnit) {
                auto createdUnit = abilityToUnit(ev.ability);
                if (createdUnit != sc2::UNIT_TYPEID::INVALID) {
                    auto remainingTime = ev.time - state.time;
                    assert(remainingTime >= 0);
                    unitsInProgress.push_back({
                        createdUnit,
                        remainingTime,
                    });
                }
            }
        }

        time = state.time;
        minerals = state.resources.minerals;
        vespene = state.resources.vespene;
        auto miningSpeed = state.miningSpeed();
        mineralsPerSecond = miningSpeed.mineralsPerSecond;
        vespenePerSecond = miningSpeed.vespenePerSecond;
        foodAvailable = state.foodAvailable();
        race = state.race;
        version = 3;

        highYieldMineralSlots = 0;
        lowYieldMineralSlots = 0;
        for (auto b : state.baseInfos) {
            auto slots = b.mineralSlots();
            highYieldMineralSlots += slots.first;
            lowYieldMineralSlots += slots.second;
        }
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(units),
            CEREAL_NVP(unitsInProgress),
            CEREAL_NVP(upgrades),
            CEREAL_NVP(time),
            CEREAL_NVP(minerals),
            CEREAL_NVP(vespene),
            CEREAL_NVP(mineralsPerSecond),
            CEREAL_NVP(vespenePerSecond),
            CEREAL_NVP(highYieldMineralSlots),
            CEREAL_NVP(lowYieldMineralSlots),
            CEREAL_NVP(foodAvailable)
        );
    }
};

struct Session {
    std::vector<SerializedState> states;
    std::vector<sc2::UNIT_TYPEID> actions;
    std::vector<UnitCount> goal;
    bool failed = false;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(failed), CEREAL_NVP(states), CEREAL_NVP(actions), CEREAL_NVP(goal));
    }
};
