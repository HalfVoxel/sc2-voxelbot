#pragma once
#include "sc2api/sc2_interfaces.h"
#include "build_optimizer_nn.h"
#include "BuildOptimizer.h"
#include <limits>
#include "utilities/stdutils.h"

struct AvailableUnitTypes;

inline bool canBeAttackedByAirWeapons(sc2::UNIT_TYPEID type) {
    return isFlying(type) || type == sc2::UNIT_TYPEID::PROTOSS_COLOSSUS;
}

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
	CombatUnit(const sc2::Unit& unit) : owner(unit.owner), type(unit.unit_type), health(unit.health), health_max(unit.health_max), shield(unit.shield), shield_max(unit.shield_max), energy(unit.energy), is_flying(unit.is_flying) {}
};

struct CombatState {
	std::vector<CombatUnit> units;
	// Owner with the highest total health summed over all units
	int owner_with_best_outcome() const;
	std::string str();
};

struct CombatResult {
	float time = 0;
	CombatState state;
};

struct CombatRecordingFrame {
    int tick;
    std::vector<std::tuple<sc2::UNIT_TYPEID, int, float>> healths;
    void add(sc2::UNIT_TYPEID type, int owner, float health, float shield);
};

struct CombatRecording {
	std::vector<CombatRecordingFrame> frames;
	void writeCSV(std::string filename);
};

struct CombatUpgrades;

struct WeaponInfo {
   private:
    mutable std::vector<float> dpsCache;
    float baseDPS;

   public:
    bool available;
    float splash;
    const sc2::Weapon* weapon;

    float getDPS() const {
        return baseDPS;
    }

    float getDPS(sc2::UNIT_TYPEID target) const;

    float range() const;

    WeaponInfo()
        : baseDPS(0), available(false), splash(0), weapon(nullptr) {
    }

    WeaponInfo(const sc2::Weapon* weapon, sc2::UNIT_TYPEID type, const CombatUpgrades& upgrades, const CombatUpgrades& targetUpgrades);
};

struct CombatSettings {
    bool badMicro = false;
	bool debug = false;
    bool enableSplash = true;
    bool enableTimingAdjustment = true;
    bool enableSurroundLimits = true;
    bool enableMeleeBlocking = true;
	bool assumeReasonablePositioning = true;
	float maxTime = std::numeric_limits<float>::infinity();
};

struct CombatUpgrades {
	std::vector<sc2::UPGRADE_ID> upgrades;
	bool hasUpgrade(sc2::UPGRADE_ID upgrade) const {
		return contains(upgrades, upgrade);
	}

	uint64_t hash() const {
		uint64_t h = 0;
		for (auto u : upgrades) {
			h = h * 31 ^ (uint64_t)u;
		}
		return h;
	}
};

struct UnitCombatInfo {
    WeaponInfo groundWeapon;
    WeaponInfo airWeapon;

    UnitCombatInfo(sc2::UNIT_TYPEID type, const CombatUpgrades& upgrades, const CombatUpgrades& targetUpgrades);
};

struct CombatEnvironment {
	std::vector<UnitCombatInfo> combatInfo;

	CombatEnvironment(const CombatUpgrades& upgrades, const CombatUpgrades& targetUpgrades);

	// TODO: Air?
	float attackRange(sc2::UNIT_TYPEID type) const;
	bool isMelee(sc2::UNIT_TYPEID type) const;
	float calculateDPS(sc2::UNIT_TYPEID type, bool air) const;
	float calculateDPS(const std::vector<CombatUnit>& units, bool air) const;
	float calculateDPS(CombatUnit& unit1, CombatUnit& unit2) const;
};

struct CombatPredictor {
	std::map<uint64_t, CombatEnvironment> combatEnvironments;
	CombatEnvironment defaultCombatEnvironment;
	CombatPredictor();
	void init();
	CombatResult predict_engage(const CombatState& state, bool debug=false, bool badMicro=false, CombatRecording* recording=nullptr, int defenderPlayer = 1) const;
	CombatResult predict_engage(const CombatState& state, CombatSettings settings, CombatRecording* recording=nullptr, int defenderPlayer = 1) const;
	void unitTest(const BuildOptimizerNN& buildTimePredictor) const;

	const CombatEnvironment& getCombatEnvironment(const CombatUpgrades& upgrades, const CombatUpgrades& targetUpgrades);
	float targetScore(const CombatUnit& unit, bool hasGround, bool hasAir) const;
};

CombatUnit makeUnit(int owner, sc2::UNIT_TYPEID type);

extern const std::vector<sc2::UNIT_TYPEID> availableUnitTypesTerran;
extern const std::vector<sc2::UNIT_TYPEID> availableUnitTypesProtoss;

std::vector<std::pair<sc2::UNIT_TYPEID,int>> findBestCompositionGenetic(const CombatPredictor& predictor, const AvailableUnitTypes& availableUnitTypes, const CombatState& opponent, const BuildOptimizerNN* buildTimePredictor = nullptr, const BuildState* startingBuildState = nullptr, std::vector<std::pair<sc2::UNIT_TYPEID,int>>* seedComposition = nullptr);

struct CombatRecorder {
private:
	std::vector<std::pair<float, std::vector<sc2::Unit>>> frames;
public:
	void tick();
	void finalize(std::string filename="recording.csv");
};

