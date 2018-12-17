#pragma once
#include <vector>
#include "sc2api/sc2_interfaces.h"

struct BuildResources {
    float minerals;
    float vespene;

    BuildResources(float minerals, float vespene)
        : minerals(minerals), vespene(vespene) {}

    inline void simulateMining(std::pair<float, float> miningSpeed, float dt) {
        minerals += miningSpeed.first * dt;
        vespene += miningSpeed.second * dt;
    }
};

struct BuildState;

struct BuildUnitInfo {
    sc2::UNIT_TYPEID type;
    sc2::UNIT_TYPEID addon;
    int units;
    // E.g. constructing a building, training a unit, etc.
    int busyUnits;

    BuildUnitInfo()
        : type(sc2::UNIT_TYPEID::INVALID), addon(sc2::UNIT_TYPEID::INVALID), units(0), busyUnits(0) {}
    BuildUnitInfo(sc2::UNIT_TYPEID type, sc2::UNIT_TYPEID addon, int units)
        : type(type), addon(addon), units(units), busyUnits(0) {}

    inline int availableUnits() const {
        if (addon == sc2::UNIT_TYPEID::TERRAN_REACTOR) {
            return units - busyUnits / 2;
        } else {
            return units - busyUnits;
        }
    }
};

enum BuildEventType {
    FinishedUnit,
    SpawnLarva,
    MuleTimeout,
};

struct BuildEvent {
    BuildEventType type;
    sc2::ABILITY_ID ability;
    sc2::UNIT_TYPEID caster;
    sc2::UNIT_TYPEID casterAddon;
    float time;

    BuildEvent(BuildEventType type, float time, sc2::UNIT_TYPEID caster, sc2::ABILITY_ID ability)
        : type(type), ability(ability), caster(caster), casterAddon(sc2::UNIT_TYPEID::INVALID), time(time) {}

    bool impactsEconomy() const;
    void apply(BuildState& state);

    bool operator<(const BuildEvent& other) const {
        return time < other.time;
    }
};

struct BuildState {
    float time;
    sc2::Race race;

    std::vector<BuildUnitInfo> units;
    std::vector<BuildEvent> events;
    BuildResources resources;

    BuildState()
        : time(0), race(sc2::Race::Terran), units(), events(), resources(0, 0) {}
    BuildState(std::vector<std::pair<sc2::UNIT_TYPEID, int>> unitCounts)
        : time(0), race(sc2::Race::Terran), units(), events(), resources(0, 0) {
        for (auto u : unitCounts)
            addUnits(u.first, u.second);
    }

    void makeUnitsBusy(sc2::UNIT_TYPEID type, sc2::UNIT_TYPEID addon, int delta);

    void addUnits(sc2::UNIT_TYPEID type, int delta);

    void addUnits(sc2::UNIT_TYPEID type, sc2::UNIT_TYPEID addon, int delta);

    std::pair<float, float> miningSpeed() const;

    float timeToGetResources(std::pair<float, float> miningSpeed, float mineralCost, float vespeneCost) const;

    void addEvent(BuildEvent event);

    // All actions up to and including the end time will have been completed
    void simulate(float endTime);

    bool simulateBuildOrder(std::vector<sc2::UNIT_TYPEID> buildOrder, std::function<void(int)> = nullptr);

    // Note that food is a floating point number, zerglings in particular use 0.5 food.
    // It is still safe to work with floating point numbers because they can exactly represent whole numbers and whole numbers + 0.5 exactly up to very large values.
    float foodAvailable() const;

    bool hasEquivalentTech(sc2::UNIT_TYPEID type) const;
};

std::vector<sc2::UNIT_TYPEID> findBestBuildOrderGenetic(const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& startingUnits, const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& target);
std::vector<sc2::UNIT_TYPEID> findBestBuildOrderGenetic(const BuildState& startState, const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& target, const std::vector<sc2::UNIT_TYPEID>* seed = nullptr);
void unitTestBuildOptimizer();
void printBuildOrderDetailed(const BuildState& startState, std::vector<sc2::UNIT_TYPEID> buildOrder);