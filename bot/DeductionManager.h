#pragma once
#include "sc2api/sc2_api.h"
#include "CombatPredictor.h"
#include <set>
#include <vector>

struct Spending {
	int spentMinerals;
	int spentGas;

	inline Spending () : spentMinerals(0), spentGas(0) {}
	inline Spending (int minerals, int gas) : spentMinerals(minerals), spentGas(gas) {}
};

struct UnitTypeInfo {
	int total;
	int alive;
	int dead;
};

struct DeductionManager {
	Spending spending;
	Spending freeResources;
	Spending startingResources;
	int playerID;
	sc2::Race race;
private:
	std::vector<const sc2::Unit*> observedUnitInstances;
	std::map<sc2::Tag, sc2::UNIT_TYPEID> observedUnits;
	std::set<sc2::UNIT_TYPEID> observedUnitTypes;
	std::vector<int> expectedObservations; // Indexed by unit type
	std::vector<int> aliveAdjustment;
public:
	void OnGameStart(int playerID);
	/** Note that the player has some units, but has received them without paying in any way (e.g. starting units) */
	void ExpectObservation(sc2::UNIT_TYPEID unitType, int count);
	void Observe(std::vector<const sc2::Unit*>& units);
	std::vector<std::pair<sc2::UNIT_TYPEID, int>> GetKnownUnits();
	std::vector<std::pair<sc2::UNIT_TYPEID, int>> ApproximateArmy(float scale);
	std::vector<std::pair<CombatUnit, sc2::Point2D>> SampleUnitPositions(float scale);
private:
	std::vector<UnitTypeInfo> Summary();
	void Observe(const sc2::Unit* unit);
	void Observe(sc2::UNIT_TYPEID unitType);
};
