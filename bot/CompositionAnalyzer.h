#pragma once
#include <fstream>
#include <queue>
#include <vector>
#include "Bot.h"
#include "cereal/cereal.hpp"
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

struct UnitItem {
    sc2::UNIT_TYPEID type;
    int count;

    template<class Archive>
    void serialize(Archive & archive) {
        archive(cereal::make_nvp("type", type), cereal::make_nvp("count", count));
    }
};

struct SideResult {
    std::vector<UnitItem> unitCounts;
    float remainingLifeFraction;
    float damageTaken;

    template<class Archive>
    inline void serialize(Archive & archive) {
        archive(CEREAL_NVP(unitCounts), CEREAL_NVP(remainingLifeFraction), CEREAL_NVP(damageTaken));
    }
};

struct Result {
    SideResult side1;
    SideResult side2;

    template<class Archive>
    inline void serialize(Archive & archive) {
        archive(CEREAL_NVP(side1), CEREAL_NVP(side2));
    }
};

class CompositionAnalyzer;

class CompositionAnalyzer : public sc2::Agent {
	struct Site {
	    CompositionAnalyzer& simulator;
	    sc2::Point2D tileMn;
	    sc2::Point2D tileMx;
	    std::vector<sc2::Point2D> points;
	    int state = 0;
	    std::vector<const sc2::Unit*> units;
	    int notInCombat = 0;
	    std::pair<std::vector<std::pair<sc2::UNIT_TYPEID, int>>, std::vector<std::pair<sc2::UNIT_TYPEID, int>>> queItem;

	    bool IsDone();
	    void Attack();
	    void writeUnits(std::vector<std::pair<sc2::UNIT_TYPEID, int>> u);
	    void writeResult();
	    void kill();
	    void OnStep();
	    Site(CompositionAnalyzer& simulator, sc2::Point2D tileMn, sc2::Point2D tileMx)
        : simulator(simulator), tileMn(tileMn), tileMx(tileMx) {}
	};

	int localSimulations = 0;
    std::ofstream results;
    std::queue<std::pair<std::vector<std::pair<sc2::UNIT_TYPEID, int>>, std::vector<std::pair<sc2::UNIT_TYPEID, int>>>> que;

    int tick = 0;
    int simulation = 0;

    std::vector<Site> sites;

public:
    void OnGameLoading();
    bool ShouldReload();
    virtual void OnGameStart() override;
    virtual void OnStep() override;
    CompositionAnalyzer();
};
