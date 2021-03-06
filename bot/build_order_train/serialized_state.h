#pragma once
#include <libvoxelbot/utilities/cereal_json.h>


#include <libvoxelbot/buildorder/optimizer.h>
#include <libvoxelbot/utilities/mappings.h>
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/utilities/stdutils.h>
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
    sc2::UNIT_TYPEID type = sc2::UNIT_TYPEID::INVALID;
    sc2::UNIT_TYPEID addon = sc2::UNIT_TYPEID::INVALID;
    int totalCount = 0;
    int availableCount = 0;

    SerializedUnit() {}

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
    float time = 0;
    std::vector<SerializedUnit> units;
    std::vector<SerializedUnitInProgress> unitsInProgress;
    float minerals = 0;
    float vespene = 0;
    float mineralsPerSecond = 0;
    float vespenePerSecond = 0;
    float foodAvailable = 0;
    sc2::Race race = sc2::Race::Random;
    int highYieldMineralSlots = 0;
    int lowYieldMineralSlots = 0;
    int version = 0;
    std::vector<sc2::UPGRADE_ID> upgrades;

    SerializedState() {}

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

