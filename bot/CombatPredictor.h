#pragma once
#include "sc2api/sc2_interfaces.h"
#include "build_optimizer_nn.h"
#include "BuildOptimizer.h"

struct CombatUnit {
	int owner;
	sc2::UNIT_TYPEID type;
	float health;
	float health_max;
	float shield;
	float shield_max;
	float energy;
	bool is_flying;
	void modifyHealth(float delta);

	CombatUnit() {}
	CombatUnit (int owner, sc2::UNIT_TYPEID type, int health, bool flying) : owner(owner), type(type), health(health), health_max(health), shield(0), shield_max(0), energy(50), is_flying(flying) {}
};

struct CombatState {
	std::vector<CombatUnit> units;
	// Owner with the highest total health summed over all units
	int owner_with_best_outcome() const;
};

struct CombatResult {
	float time;
	CombatState state;
};

struct CombatRecording;

struct CombatPredictor {
	void init();
	CombatResult predict_engage(const CombatState& state, bool debug=false, bool badMicro=false, CombatRecording* recording=nullptr) const;
	void unitTest(const BuildOptimizerNN& buildTimePredictor) const;
};

CombatUnit makeUnit(int owner, sc2::UNIT_TYPEID type);

extern std::vector<sc2::UNIT_TYPEID> availableUnitTypesTerran;

std::vector<std::pair<sc2::UNIT_TYPEID,int>> findBestCompositionGenetic(const CombatPredictor& predictor, const std::vector<sc2::UNIT_TYPEID>& availableUnitTypes, const CombatState& opponent, const BuildOptimizerNN* buildTimePredictor = nullptr, const BuildState* startingBuildState = nullptr, std::vector<std::pair<sc2::UNIT_TYPEID,int>>* seedComposition = nullptr);

struct CombatRecorder {
private:
	std::vector<std::pair<float, std::vector<sc2::Unit>>> frames;
public:
	void tick();
	void finalize();
};