#include "CombatPredictor.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <random>
#include <iostream>
#include "Bot.h"
#include "utilities/mappings.h"
#include "utilities/profiler.h"
#include "utilities/stdutils.h"
#include "utilities/predicates.h"
#include "unit_lists.h"
#include <sstream>
#include <chrono>

using namespace std;
using namespace sc2;

string CombatState::str() {
    stringstream ss;
    for (auto& u : units) {
        ss << u.owner << ": " << getUnitData(u.type).name << " " << u.health << "+" << u.shield << "/" << u.health_max << "+" << u.shield_max << endl;
    }
    return ss.str();
}

int getDamageBonus(UNIT_TYPEID unit, const CombatUpgrades& upgrades) {
    if (isStructure(unit)) return 0;

    int bonus = 0;
    switch(getUnitData(unit).race) {
    case Race::Protoss:
        if (isFlying(unit)) {
            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSAIRWEAPONSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSAIRWEAPONSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSAIRWEAPONSLEVEL3)) bonus += 1;
        } else {
            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSGROUNDWEAPONSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSGROUNDWEAPONSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSGROUNDWEAPONSLEVEL3)) bonus += 1;
        }
        break;
    case Race::Zerg:
        if (isFlying(unit)) {
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGFLYERWEAPONSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGFLYERWEAPONSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGFLYERWEAPONSLEVEL3)) bonus += 1;
        } else if (isMelee(unit)) {
            // TODO: Might need to check which units are actually melee
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGMELEEWEAPONSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGMELEEWEAPONSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGMELEEWEAPONSLEVEL3)) bonus += 1;
        } else {
            // Ranged
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGMISSILEWEAPONSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGMISSILEWEAPONSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::ZERGMISSILEWEAPONSLEVEL3)) bonus += 1;
        }
        break;
    case Race::Terran: {
        auto canonical = canonicalize(unit);
        auto& casters = abilityToCasterUnit(getUnitData(canonical).ability_id);
        if (casters.size() > 0 && casters[0] == UNIT_TYPEID::TERRAN_BARRACKS) {
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANINFANTRYWEAPONSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANINFANTRYWEAPONSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANINFANTRYWEAPONSLEVEL3)) bonus += 1;
        } else if (casters.size() > 0 && casters[0] == UNIT_TYPEID::TERRAN_FACTORY) {
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANVEHICLEWEAPONSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANVEHICLEWEAPONSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANVEHICLEWEAPONSLEVEL3)) bonus += 1;
        } else if (casters.size() > 0 && casters[0] == UNIT_TYPEID::TERRAN_STARPORT) {
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANSHIPWEAPONSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANSHIPWEAPONSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::TERRANSHIPWEAPONSLEVEL3)) bonus += 1;
        }
        break;
    }
    default:
        break;
    }
    return bonus;
}

float getArmorBonus(UNIT_TYPEID unit, const CombatUpgrades& upgrades) {
    if (isStructure(unit)) {
        if (getUnitData(unit).race == Race::Terran && upgrades.hasUpgrade(UPGRADE_ID::TERRANBUILDINGARMOR)) return 2;
        return 0;
    }

    float bonus = 0;
    switch(getUnitData(unit).race) {
        case Race::Protoss:
            if (isFlying(unit)) {
                if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSAIRARMORSLEVEL1)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSAIRARMORSLEVEL2)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSAIRARMORSLEVEL3)) bonus += 1;
            } else {
                if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSGROUNDARMORSLEVEL1)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSGROUNDARMORSLEVEL2)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSGROUNDARMORSLEVEL3)) bonus += 1;
            }

            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSSHIELDSLEVEL1)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSSHIELDSLEVEL2)) bonus += 1;
            if (upgrades.hasUpgrade(UPGRADE_ID::PROTOSSSHIELDSLEVEL3)) bonus += 1;
            break;
        case Race::Zerg:
            if (isFlying(unit)) {
                if (upgrades.hasUpgrade(UPGRADE_ID::ZERGFLYERARMORSLEVEL1)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::ZERGFLYERARMORSLEVEL2)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::ZERGFLYERARMORSLEVEL3)) bonus += 1;
            } else {
                if (upgrades.hasUpgrade(UPGRADE_ID::ZERGGROUNDARMORSLEVEL1)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::ZERGGROUNDARMORSLEVEL2)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::ZERGGROUNDARMORSLEVEL3)) bonus += 1;
            }
        case Race::Terran: {
            auto canonical = canonicalize(unit);
            auto& casters = abilityToCasterUnit(getUnitData(canonical).ability_id);
            if (casters.size() > 0 && casters[0] == UNIT_TYPEID::TERRAN_BARRACKS) {
                if (upgrades.hasUpgrade(UPGRADE_ID::TERRANINFANTRYARMORSLEVEL1)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::TERRANINFANTRYARMORSLEVEL2)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::TERRANINFANTRYARMORSLEVEL3)) bonus += 1;
            } else if (casters.size() > 0 && (casters[0] == UNIT_TYPEID::TERRAN_FACTORY || casters[0] == UNIT_TYPEID::TERRAN_STARPORT)) {
                if (upgrades.hasUpgrade(UPGRADE_ID::TERRANVEHICLEANDSHIPARMORSLEVEL1)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::TERRANVEHICLEANDSHIPARMORSLEVEL2)) bonus += 1;
                if (upgrades.hasUpgrade(UPGRADE_ID::TERRANVEHICLEANDSHIPARMORSLEVEL3)) bonus += 1;
            }
            break;
        }
        default:
            break;
    }
    return bonus;
}

float calculateDPS(UNIT_TYPEID attacker, UNIT_TYPEID target, const Weapon& weapon, const CombatUpgrades& attackerUpgrades, const CombatUpgrades& targetUpgrades) {
    // canBeAttackedByAirWeapons is primarily for coloussus.
    if (weapon.type == Weapon::TargetType::Any || (weapon.type == Weapon::TargetType::Air ? canBeAttackedByAirWeapons(target) : !isFlying(target))) {
        float dmg = weapon.damage_;
        for (auto& b : weapon.damage_bonus) {
            if (contains(getUnitData(target).attributes, b.attribute)) {
                dmg += b.bonus;
            }
        }

        dmg += getDamageBonus(attacker, attackerUpgrades);

        float armor = getUnitData(target).armor + getArmorBonus(target, targetUpgrades);

        // Note: cannot distinguish between damage to shields and damage to health yet, so average out so that the armor is about 0.5 over the whole shield+health of the unit
        // Important only for protoss
        armor *= maxHealth(target) / (maxHealth(target) + maxShield(target));

        float timeBetweenAttacks = weapon.speed;

        if (attacker == UNIT_TYPEID::PROTOSS_ADEPT && attackerUpgrades.hasUpgrade(UPGRADE_ID::ADEPTPIERCINGATTACK)) timeBetweenAttacks /= 1.45f;

        return max(0.0f, dmg - armor) * weapon.attacks / timeBetweenAttacks;
    }

    return 0;
}

float WeaponInfo::getDPS(UNIT_TYPEID target, float modifier) const {
    if (!available)
        return 0;
    
    if ((int)target >= (int)dpsCache.size()) {
        throw std::out_of_range(UnitTypeToName(target));    
    }

    // TODO: Modifier ignores speed upgrades
    return max(0.0f, dpsCache[(int)target] + modifier*weapon->attacks/weapon->speed);
}

float WeaponInfo::range() const {
    return weapon != nullptr ? weapon->range : 0;
}

float UnitCombatInfo::attackInterval() const {
    float v = numeric_limits<float>::infinity();
    if (airWeapon.weapon != nullptr) v = airWeapon.weapon->speed;
    if (groundWeapon.weapon != nullptr) v = min(v, groundWeapon.weapon->speed);
    return v;
}

WeaponInfo::WeaponInfo(const Weapon* weapon, UNIT_TYPEID type, const CombatUpgrades& upgrades, const CombatUpgrades& targetUpgrades) {
    available = true;
    splash = 0;
    this->weapon = weapon;
    // TODO: Range upgrades!

    baseDPS = (weapon->damage_ + getDamageBonus(type, upgrades)) * weapon->attacks / weapon->speed;

    auto& unitTypes = getUnitTypes();
    for (size_t i = 0; i < unitTypes.size(); i++) {
        dpsCache.push_back(calculateDPS(type, (UNIT_TYPEID)i, *weapon, upgrades, targetUpgrades));
    }
}


UnitCombatInfo::UnitCombatInfo(UNIT_TYPEID type, const CombatUpgrades& upgrades, const CombatUpgrades& targetUpgrades) {
    auto& data = getUnitData(type);

    for (const Weapon& weapon : data.weapons) {
        if (weapon.type == Weapon::TargetType::Any || weapon.type == Weapon::TargetType::Air) {
            if (airWeapon.available) {
                cerr << "For unit type " << UnitTypeToName(type) << endl;
                cerr << "Weapon slot is already used";
                assert(false);
            }
            airWeapon = WeaponInfo(&weapon, type, upgrades, targetUpgrades);
        }
        if (weapon.type == Weapon::TargetType::Any || weapon.type == Weapon::TargetType::Ground) {
            if (groundWeapon.available) {
                cerr << "For unit type " << UnitTypeToName(type) << endl;
                cerr << "Weapon slot is already used";
                assert(false);
            }
            groundWeapon = WeaponInfo(&weapon, type, upgrades, targetUpgrades);
        }
    }
}

void CombatRecordingFrame::add(UNIT_TYPEID type, int owner, float health, float shield) {
    for (size_t i = 0; i < healths.size(); i++) {
        if (get<0>(healths[i]) == type && get<1>(healths[i]) == owner) {
            healths[i] = make_tuple(type, owner, get<2>(healths[i]) + health + shield);
            return;
        }
    }
    healths.push_back(make_tuple(type, owner, health + shield));
}

void CombatRecording::writeCSV(string filename) {
    ofstream output(filename);

    set<pair<UNIT_TYPEID, int>> types;
    for (auto& f : frames) {
        for (auto p : f.healths) {
            types.insert(make_pair(get<0>(p), get<1>(p)));
        }
    }

    // Make sure all frames contain the same types
    for (auto& f : frames) {
        for (auto t : types) {
            f.add(get<0>(t), get<1>(t), 0, 0);
        }
    }

    // Sort the types
    for (auto& f : frames) {
        sortByValueDescending<tuple<UNIT_TYPEID, int, float>, float>(f.healths, [](const auto& p) { return (float)get<0>(p) + 100000 * get<1>(p); });
    }

    for (int owner = 1; owner <= 2; owner++) {
        output << "Time\t";
        for (auto p : frames[0].healths) {
            if (get<1>(p) == owner) {
                output << getUnitData(get<0>(p)).name << "\t";
            }
        }
        output << endl;
        for (auto& f : frames) {
            output << (ticksToSeconds(f.tick - frames[0].tick)) << "\t";
            for (auto p : f.healths) {
                if (get<1>(p) == owner) {
                    output << get<2>(p) << "\t";
                }
            }
            output << endl;
        }
        output << "-------------" << endl;
    }
    output.close();
}

void CombatRecorder::tick() {
    vector<Unit> units;
    for (auto u : agent->Observation()->GetUnits()) {
        units.push_back(*u);
    }
    frames.push_back(make_pair(agent->Observation()->GetGameLoop(), units));
}

void CombatRecorder::finalize(string filename) {
    map<Tag, float> totalHealths;
    set<Tag> inCombat;
    assert(frames.size() > 1);
    size_t firstFrame = frames.size() - 1;
    size_t lastFrame = 0;
    for (size_t i = 0; i < frames.size(); i++) {
        auto& p = frames[i];
        for (auto& u : p.second) {
            float h = u.is_alive ? u.health + u.shield : 0;
            if (totalHealths.find(u.tag) != totalHealths.end()) {
                float minDist = 10000;
                if (!inCombat.count(u.tag)) {
                    for (auto& u2 : p.second) {
                        if (inCombat.count(u2.tag)) {
                            minDist = min(minDist, DistanceSquared2D(u2.pos, u.pos));
                        }
                    }
                }
                if (h != totalHealths[u.tag] || u.engaged_target_tag != NullTag || minDist < 5 * 5) {
                    inCombat.insert(u.tag);
                    firstFrame = min(firstFrame, i - 1);
                    lastFrame = max(lastFrame, i);
                }
            }

            totalHealths[u.tag] = h;
        }
    }

    CombatRecording recording;
    for (auto i = firstFrame; i <= lastFrame; i++) {
        auto& p = frames[i];
        CombatRecordingFrame frame;
        frame.tick = p.first;
        for (auto& u : p.second) {
            if (inCombat.count(u.tag)) {
                if (u.is_alive) frame.add(u.unit_type, u.owner, u.health, u.shield);
                else frame.add(u.unit_type, u.owner, 0, 0);
            }
        }
        recording.frames.push_back(frame);
    }

    recording.writeCSV(filename);
};

CombatPredictor::CombatPredictor() : defaultCombatEnvironment({}, {}) {    
}

void CombatPredictor::init() {
    

};

CombatUpgrades::CombatUpgrades() {
    // It's a bit tricky to set the template argument to the upgrades array based on the availableUpgrades list
    // so instead we just validate it at runtime here.
    // If new upgrades are introduced the upgrades bitset needs to be expanded.
    assert(upgrades.size() >= availableUpgrades.size());
}

sc2::UPGRADE_ID CombatUpgrades::iterator::operator*() const { return availableUpgrades.getBuildOrderItem(index).upgradeID(); }

bool CombatUpgrades::hasUpgrade(sc2::UPGRADE_ID upgrade) const {
    return upgrades[availableUpgrades.getIndex(upgrade)];
}

void CombatUpgrades::add(sc2::UPGRADE_ID upgrade) {
    upgrades[availableUpgrades.getIndex(upgrade)] = true;
}

CombatEnvironment::CombatEnvironment(const CombatUpgrades& upgrades, const CombatUpgrades& targetUpgrades) : upgrades({{ upgrades, targetUpgrades }}) {
    assertMappingsInitialized();
    auto& unitTypes = getUnitTypes();
    for (size_t i = 0; i < unitTypes.size(); i++) {
        combatInfo[0].push_back(UnitCombatInfo((UNIT_TYPEID)i, upgrades, targetUpgrades));
    }
    for (size_t i = 0; i < unitTypes.size(); i++) {
        combatInfo[1].push_back(UnitCombatInfo((UNIT_TYPEID)i, targetUpgrades, upgrades));
    }

    for (int owner = 0; owner < 2; owner++) {
        combatInfo[owner][(int)UNIT_TYPEID::TERRAN_LIBERATOR].airWeapon.splash = 3.0;
        combatInfo[owner][(int)UNIT_TYPEID::TERRAN_MISSILETURRET].airWeapon.splash = 3.0;
        combatInfo[owner][(int)UNIT_TYPEID::TERRAN_SIEGETANKSIEGED].groundWeapon.splash = 4.0;
        combatInfo[owner][(int)UNIT_TYPEID::TERRAN_HELLION].groundWeapon.splash = 2;
        combatInfo[owner][(int)UNIT_TYPEID::TERRAN_HELLIONTANK].groundWeapon.splash = 3;
        combatInfo[owner][(int)UNIT_TYPEID::ZERG_MUTALISK].groundWeapon.splash = 1.44f;
        combatInfo[owner][(int)UNIT_TYPEID::ZERG_MUTALISK].airWeapon.splash = 1.44f;
        combatInfo[owner][(int)UNIT_TYPEID::TERRAN_THOR].airWeapon.splash = 3;
        combatInfo[owner][(int)UNIT_TYPEID::PROTOSS_ARCHON].groundWeapon.splash = 3;
        combatInfo[owner][(int)UNIT_TYPEID::PROTOSS_ARCHON].airWeapon.splash = 3;
        combatInfo[owner][(int)UNIT_TYPEID::PROTOSS_COLOSSUS].groundWeapon.splash = 3;
    }
}

const CombatEnvironment& CombatPredictor::combineCombatEnvironment(const CombatEnvironment* env, const CombatUpgrades& upgrades, int upgradesOwner) const {
    assert(upgradesOwner == 1 || upgradesOwner == 2);

    array<CombatUpgrades, 2> newUpgrades = env != nullptr ? env->upgrades : array<CombatUpgrades, 2>{{ {}, {} }};
    newUpgrades[upgradesOwner - 1].combine(upgrades);

    return getCombatEnvironment(newUpgrades[0], newUpgrades[1]);
}

const CombatEnvironment& CombatPredictor::getCombatEnvironment(const CombatUpgrades& upgrades, const CombatUpgrades& targetUpgrades) const {
    uint64_t hash = (upgrades.hash() * 5123143) ^ targetUpgrades.hash();
    auto it = combatEnvironments.find(hash);
    if (it != combatEnvironments.end()) {
        return (*it).second;
    }

    // Note: references to map values are guaranteed to remain valid even after additional insertion into the map
    return (*combatEnvironments.emplace(make_pair(hash, CombatEnvironment(upgrades, targetUpgrades))).first).second;
}

// TODO: Air?
float CombatEnvironment::attackRange(int owner, UNIT_TYPEID type) const {
    auto& info = combatInfo[owner - 1][(int)type];
    return max(info.airWeapon.range(), info.groundWeapon.range());
}

float CombatEnvironment::attackRange(const CombatUnit& unit) const {
    auto& info = combatInfo[unit.owner - 1][(int)unit.type];
    float range = max(info.airWeapon.range(), info.groundWeapon.range());

    if (unit.type == UNIT_TYPEID::PROTOSS_COLOSSUS && upgrades[unit.owner-1].hasUpgrade(UPGRADE_ID::EXTENDEDTHERMALLANCE)) range += 2;

    return range;
}

const UnitCombatInfo& CombatEnvironment::getCombatInfo(const CombatUnit& unit) const {
    return combatInfo[unit.owner - 1][(int)unit.type];
}

float CombatEnvironment::calculateDPS(int owner, UNIT_TYPEID type, bool air) const {
    auto& info = combatInfo[owner - 1][(int)type];
    return air ? info.airWeapon.getDPS() : info.groundWeapon.getDPS();
}

float CombatEnvironment::calculateDPS(const vector<CombatUnit>& units, bool air) const {
    float dps = 0;
    for (auto& u : units)
        dps += calculateDPS(u.owner, u.type, air);
    return dps;
}

float CombatEnvironment::calculateDPS(const CombatUnit& unit, bool air) const {
    auto& info = combatInfo[unit.owner - 1][(int)unit.type];
    return air ? info.airWeapon.getDPS() : info.groundWeapon.getDPS();
}

float CombatEnvironment::calculateDPS(const CombatUnit& unit1, const CombatUnit& unit2) const {
    auto& info = combatInfo[unit1.owner - 1][(int)unit1.type];
    return max(info.groundWeapon.getDPS(unit2.type), info.airWeapon.getDPS(unit2.type));
}

vector<CombatUnit*> filterByOwner(vector<CombatUnit>& units, int owner) {
    vector<CombatUnit*> result;
    for (auto& u : units) {
        if (u.owner == owner) {
            result.push_back(&u);
        }
    }
    return result;
}

float CombatPredictor::targetScore(const CombatUnit& unit, bool hasGround, bool hasAir) const {
    const float VESPENE_MULTIPLIER = 1.5f;
    float cost = getUnitData(unit.type).mineral_cost + VESPENE_MULTIPLIER * getUnitData(unit.type).vespene_cost;

    float score = 0;

    float airDPS = defaultCombatEnvironment.calculateDPS(1, unit.type, true);
    float groundDPS = defaultCombatEnvironment.calculateDPS(1, unit.type, false);

    score += 0.01 * cost;
    score += 1000 * max(groundDPS, airDPS);
    // cout << "Score for " << UnitTypeToName(unit.type) << " " << unit.health << " " << unit.shield << " " << score << " " << groundDPS << " " << airDPS << " " << cost << endl;

    // If we don't have any ground units (therefore we only have air units) and the attacker cannot attack air, then give it a low score
    if (!hasGround && airDPS == 0)
        score *= 0.01f;
    else if (!hasAir && groundDPS == 0)
        score *= 0.01f;
    // If the unit cannot attack, give it a really low score (todo: carriers??)
    else if (airDPS == 0 && groundDPS == 0)
        score *= 0.01f;

    return score;
}

void CombatUnit::modifyHealth(float delta) {
    if (delta < 0) {
        delta = -delta;
        shield -= delta;
        if (shield < 0) {
            delta = -shield;
            shield = 0;
            health = max(0.0f, health - delta);
        }
    } else {
        health += delta;
        health = min(health, health_max);
    }
}

struct SurroundInfo {
    int maxAttackersPerDefender;
    int maxMeleeAttackers;
};

/** Calculates how many melee units can be in combat at the same time and how many melee units can attack a single enemy.
 * For example in the case of 1 marine defender, then several attackers can surround that unit.
 * However in the case of many marines and many attackers, only some attackers will be able to get in
 * range of the defenders (assuming they are clumped up) and a given defender can only be attacked
 * by a small number of attackers.
 *
 * This assumes that the enemy is clumped up in a single blob and that the attackers are roughly the size of zealots.
 */
SurroundInfo maxSurround(float enemyGroundUnitArea, int enemyGroundUnits) {
    // Assume a packing density of about 60% (circles have a packing density of about 90%)
    if (enemyGroundUnits > 1)
        enemyGroundUnitArea /= 0.6;
    float radius = sqrt(enemyGroundUnitArea / M_PI);

    float representativeMeleeUnitRadius = unitRadius(UNIT_TYPEID::PROTOSS_ZEALOT);

    float circumferenceDefenders = radius * (2 * M_PI);
    float circumferenceAttackers = (radius + representativeMeleeUnitRadius) * (2 * M_PI);

    float approximateDefendersInMeleeRange = min((float)enemyGroundUnits, circumferenceDefenders / (2 * representativeMeleeUnitRadius));
    float approximateAttackersInMeleeRange = circumferenceAttackers / (2 * representativeMeleeUnitRadius);

    int maxAttackersPerDefender = approximateDefendersInMeleeRange > 0 ? (int)ceil(approximateAttackersInMeleeRange / approximateDefendersInMeleeRange) : 1;
    int maxMeleeAttackers = (int)ceil(approximateAttackersInMeleeRange);
    return { maxAttackersPerDefender, maxMeleeAttackers };
}

// Hash for combat input
unsigned long long combatHash(const CombatState& state, bool badMicro, int defenderPlayer, float maxTime) {
    unsigned long long h = 0L;
    h = h ^ defenderPlayer;
    h = h*31 ^ (int)badMicro;
    h = h*31 ^ (int)round(maxTime);
    for (auto& u : state.units) {
        h = (h * 31) ^ (unsigned long long)u.energy;
        h = (h * 31) ^ (unsigned long long)u.health;
        h = (h * 31) ^ (unsigned long long)u.shield;
        h = (h * 31) ^ (unsigned long long)u.type;
        h = (h * 31) ^ (unsigned long long)u.owner;
    }
    return h;
}

map<unsigned long long, CombatResult> seen_combats;

int counter = 0;

float timeToBeAbleToAttack (const CombatEnvironment& env, CombatUnit& unit, float distanceToEnemy) {
    auto& unitTypeData = getUnitData(unit.type);
    return unitTypeData.movement_speed > 0 ? max(0.0f, distanceToEnemy - env.attackRange(unit)) / unitTypeData.movement_speed : 100000;
}

// Owner = 1 is the defender, Owner != 1 is an attacker
CombatResult CombatPredictor::predict_engage(const CombatState& inputState, bool debug, bool badMicro, CombatRecording* recording, int defenderPlayer) const {
    CombatSettings settings;
    settings.badMicro = badMicro;
    settings.debug = debug;
    return predict_engage(inputState, settings, recording, defenderPlayer);
}

CombatResult CombatPredictor::predict_engage(const CombatState& inputState, CombatSettings settings, CombatRecording* recording, int defenderPlayer) const {
#if CACHE_COMBAT
    auto h = combatHash(inputState, badMicro, defenderPlayer);
    counter++;
    
    // Determine if we have already seen this combat before, and if so, just return the previous outcome
    if (seen_combats.find(h) != seen_combats.end()) {
        auto& cachedResult = seen_combats[h];
        assert(cachedResult.state.units.size() == inputState.units.size());
        for (int i = 0; i < cachedResult.state.units.size(); i++) {
            if (cachedResult.state.units[i].type != inputState.units[i].type) {
                cerr << "Hash collision???" << endl;
                cerr << UnitTypeToName(cachedResult.state.units[i].type) << " " << UnitTypeToName(inputState.units[i].type) << endl;
                assert(false);
            }
        }
        return seen_combats[h];
    }
#endif
    const auto& env = inputState.environment != nullptr ? *inputState.environment : defaultCombatEnvironment;
    bool debug = settings.debug;
    // Copy state
    CombatResult result;
    result.state = inputState;
    CombatState& state = result.state;

    vector<shared_ptr<CombatUnit>> temporaryUnits;
    // TODO: Is it 1 and 2?
    auto units1 = filterByOwner(state.units, 1);
    auto units2 = filterByOwner(state.units, 2);

    // TODO: Might always initialize to seed 0, check this
    auto rng = std::default_random_engine{};
    shuffle(begin(units1), end(units1), rng);
    shuffle(begin(units2), end(units2), rng);

    // sortByValueDescending<CombatUnit*>(units1, [=] (auto u) { return targetScore(*u, true, true); });
    // sortByValueDescending<CombatUnit*>(units2, [=] (auto u) { return targetScore(*u, true, true); });

    array<float, 2> averageHealthByTime = {{ 0, 0 }};
    array<float, 2> averageHealthByTimeWeight = {{ 0, 0 }};

    float maxRangeDefender = 0;
    float fastestAttackerSpeed = 0;
    if (defenderPlayer == 1 || defenderPlayer == 2) {
        // One player is the attacker and one is the defender
        for (auto& u : (defenderPlayer == 1 ? units1 : units2)) {
            maxRangeDefender = max(maxRangeDefender, env.attackRange(*u));
        }
        for (auto& u : (defenderPlayer == 1 ? units2 : units1)) {
            fastestAttackerSpeed = max(fastestAttackerSpeed, getUnitData(u->type).movement_speed);
        }
    } else {
        // Both players are attackers
        for (auto& u : state.units) {
            maxRangeDefender = max(maxRangeDefender, env.attackRange(u));
        }
        for (auto& u : state.units) {
            fastestAttackerSpeed = max(fastestAttackerSpeed, getUnitData(u.type).movement_speed);
        }
    }

    float time = settings.startTime;
    bool changed = true;
    // Note: required in case of healers on both sides to avoid inf loop
    const int MAX_ITERATIONS = 100;
    int recordingStartTick = 0;
    if (recording != nullptr && !recording->frames.empty()) recordingStartTick = ticksToSeconds(recording->frames.rbegin()->tick) + 1 - time;

    // If the combat starts from scratch, assume all buffs are gone
    if (settings.startTime == 0) {
        for (auto& u : state.units) {
            u.buffTimer = 0;
        }
    }

    for (int it = 0; it < MAX_ITERATIONS && changed; it++) {
        int hasAir1 = 0;
        int hasAir2 = 0;
        int hasGround1 = 0;
        int hasGround2 = 0;
        float groundArea1 = 0;
        float groundArea2 = 0;
        for (auto u : units1) {
            if (u->health > 0) {
                hasAir1 += canBeAttackedByAirWeapons(u->type);
                hasGround1 += !u->is_flying;
                float r = unitRadius(u->type);
                groundArea1 += r * r;

                averageHealthByTime[0] += time * (u->health + u->shield);
                averageHealthByTimeWeight[0] += u->health + u->shield;
            }
        }
        for (auto u : units2) {
            if (u->health > 0) {
                hasAir2 += canBeAttackedByAirWeapons(u->type);
                hasGround2 += !u->is_flying;
                float r = unitRadius(u->type);
                groundArea2 += r * r;

                averageHealthByTime[1] += time * (u->health + u->shield);
                averageHealthByTimeWeight[1] += u->health + u->shield;
            }
        }

        if (recording != nullptr) {
            CombatRecordingFrame frame;
            frame.tick = (int)round((recordingStartTick + time) * 22.4f);
            for (auto u : units1) {
                frame.add(u->type, u->owner, u->health, u->shield);
            }
            for (auto u : units2) {
                frame.add(u->type, u->owner, u->health, u->shield);
            }
            recording->frames.push_back(frame);
        }

        // Calculate info about how many melee units we can use right now
        // and how many melee units that can attack a single unit.
        // For example in the case of 1 marine defender, then several attackers can surround that unit.
        // However in the case of many marines and many attackers, only some attackers will be able to get in
        // range of the defenders (assuming they are clumped up) and a given defender can only be attacked
        // by a small number of attackers.
        // Note that the parameters are for the enemy data.
        SurroundInfo surroundInfo1 = maxSurround(groundArea2 * M_PI, hasGround2);
        SurroundInfo surroundInfo2 = maxSurround(groundArea1 * M_PI, hasGround1);

        // Use a finer timestep for earlier times in the simulation
        // and make it coarser over time. This ensures that even very long simulations can be evaluated in a reasonable time.
        float dt = min(5, 1 + (it / 10));
        if (debug)
            cout << "Iteration " << it << " Time: " << time << endl;
        changed = false;

        // Check guardian shields.
        // Guardian shield is approximated as each shield protecting a fixed area of units as long
        // as the shield is active. The first N units in each army, such that the total area of all units up to unit N, are assumed to be protected
        const float GuardianShieldUnits = 4.5f*4.5f*M_PI * 0.4f;
        // Number of units in each army that are shielded by the guardian shield
        array<float, 2> guardianShieldedUnitFraction = {{ 0, 0 }};
        array<bool, 2> guardianShieldCoversAllUnits = {{ false, false }};

        for (int group = 0; group < 2; group++) {
            float guardianShieldedArea = 0;
            auto& g = group == 0 ? units1 : units2;
            for (auto* u : g) {
                if (u->type == UNIT_TYPEID::PROTOSS_SENTRY && u->buffTimer > 0) {
                    u->buffTimer -= dt;
                    guardianShieldedArea += GuardianShieldUnits;
                }
            }

            float totalArea = 0;
            for (size_t i = 0; i < g.size(); i++) {
                float r = unitRadius(g[i]->type);
                totalArea += r*r*M_PI;
            }

            guardianShieldCoversAllUnits[group] = guardianShieldedArea > totalArea;
            guardianShieldedUnitFraction[group] = min(0.8f, guardianShieldedArea / (0.001f+ totalArea));
        }


        for (int group = 0; group < 2; group++) {
            // TODO: Group 1 always attacks first
            auto& g1 = group == 0 ? units1 : units2;
            auto& g2 = group == 0 ? units2 : units1;
            SurroundInfo surround = group == 0 ? surroundInfo1 : surroundInfo2;
            // The melee units furthest back in the group have to move around the whole group and around the whole enemy group
            // (very very approximative)
            float maxExtraMeleeDistance = sqrt(groundArea1 / M_PI) * M_PI + sqrt(groundArea2 / M_PI) * M_PI;

            int numMeleeUnitsUsed = 0;
            bool didActivateGuardianShield = false;

            // Fraction of opponent units that are melee units (and are still alive)
            float opponentFractionMeleeUnits = 0;
            for (auto* u : g2) {
                if (isMelee(u->type) && u->health > 0) opponentFractionMeleeUnits += 1;
            }
            if (g2.size() > 0) opponentFractionMeleeUnits /= g2.size();

            // Only a single healer can heal a given unit at a time
            // (holds for medivacs and shield batteries at least)
            vector<bool> hasBeenHealed(g1.size());
            // How many melee units that have attacked a particular enemy so far
            vector<int> meleeUnitAttackCount(g2.size());

            if (debug) {
                cout << "Max meleee attackers: " << surround.maxMeleeAttackers << " " << surround.maxAttackersPerDefender << " num units: " << g1.size() << endl;
            }

            for (size_t i = 0; i < g1.size(); i++) {
                auto& unit = *g1[i];

                if (unit.health == 0)
                    continue;
                
                auto& unitTypeData = getUnitData(unit.type);
                float airDPS = env.calculateDPS(unit, true);
                float groundDPS = env.calculateDPS(unit, false);

                if (debug)
                    cout << "Processing " << UnitTypeToName(unit.type) << " " << unit.health << "+" << unit.shield << " "
                         << "e=" << unit.energy << endl;

                if (unit.type == UNIT_TYPEID::TERRAN_MEDIVAC) {
                    if (unit.energy > 0) {
                        // Pick a random target
                        size_t offset = (size_t)rand() % g1.size();
                        const float HEALING_PER_NORMAL_SPEED_SECOND = 12.6 / 1.4f;
                        for (size_t j = 0; j < g1.size(); j++) {
                            size_t index = (j + offset) % g1.size();
                            auto& other = *g1[index];
                            if (index != i && !hasBeenHealed[index] && other.health > 0 && other.health < other.health_max && contains(getUnitData(other.type).attributes, Attribute::Biological)) {
                                other.modifyHealth(HEALING_PER_NORMAL_SPEED_SECOND * dt);
                                hasBeenHealed[index] = true;
                                changed = true;
                                break;
                            }
                        }
                    }
                    continue;
                }

                if (unit.type == UNIT_TYPEID::PROTOSS_SHIELDBATTERY) {
                    if (unit.energy > 0) {
                        // Pick a random target
                        size_t offset = (size_t)rand() % g1.size();
                        const float SHIELDS_PER_NORMAL_SPEED_SECOND = 50.4 / 1.4f;
                        const float ENERGY_USE_PER_SHIELD = 1.0f / 3.0f;
                        for (size_t j = 0; j < g1.size(); j++) {
                            size_t index = (j + offset) % g1.size();
                            auto& other = *g1[index];
                            if (index != i && !hasBeenHealed[index] && other.health > 0 && other.shield < other.shield_max) {
                                float delta = min(min(other.shield_max - other.shield, SHIELDS_PER_NORMAL_SPEED_SECOND * dt), unit.energy / ENERGY_USE_PER_SHIELD);
                                assert(delta >= 0);
                                other.shield += delta;
                                assert(other.shield >= 0 && other.shield <= other.shield_max + 0.001f);
                                unit.energy -= delta * ENERGY_USE_PER_SHIELD;
                                hasBeenHealed[index] = true;
                                changed = true;
                                // cout << "Restoring " << delta << " shields. New health is " << other.health << "+" << other.shield << endl;
                                break;
                            }
                        }
                    }
                    continue;
                }

                if (unit.type == UNIT_TYPEID::ZERG_INFESTOR) {
                    if (unit.energy > 25) {
                        // Spawn an infested terran
                        unit.energy -= 25;
                        auto u = makeUnit(unit.owner, UNIT_TYPEID::ZERG_INFESTORTERRAN);
                        // Uses energy as timeout in seconds
                        u.energy = 21 * 1.4f;
                        auto up = make_shared<CombatUnit>();
                        *up = u;
                        temporaryUnits.push_back(up);
                        // Note: a bit ugly, extracting a raw pointer from a shared one
                        g1.push_back(&**temporaryUnits.rbegin());
                        changed = true;
                    }
                    continue;
                }
                
                // Uses energy as timeout
                if (unit.type == UNIT_TYPEID::ZERG_INFESTORTERRAN) {
                    unit.energy -= dt;
                    if (unit.energy <= 0) {
                        unit.modifyHealth(-100000);
                        // TODO: Remove from group
                        changed = true;
                        continue;
                    }
                }

                if (unit.type == UNIT_TYPEID::PROTOSS_SENTRY && unit.energy >= 75 && !didActivateGuardianShield) {
                    if (!guardianShieldCoversAllUnits[group]) {
                        unit.energy -= 75;
                        unit.buffTimer = 11.0f;
                        // Make sure only one guardian shield per team gets activated per tick
                        // If we didn't do this check then all guardian shields would go up in tick 1 even though fewer might be enough
                        didActivateGuardianShield = true;
                    }
                }

                if (airDPS == 0 && groundDPS == 0)
                    continue;
                
                if (settings.workersDoNoDamage && isBasicHarvester(unit.type))
                    continue;

                bool isUnitMelee = isMelee(unit.type);
                if (isUnitMelee && numMeleeUnitsUsed >= surround.maxMeleeAttackers && settings.enableSurroundLimits)
                    continue;
                
                if (settings.enableTimingAdjustment) {
                    if (group + 1 != defenderPlayer) {
                        // Attacker (move until we are within range of enemy)
                        float distanceToEnemy = maxRangeDefender;
                        if (isUnitMelee) {
                            // Make later melee units take longer to reach the enemy.
                            distanceToEnemy += maxExtraMeleeDistance * (i / (float)g1.size());
                        }
                        float timeToReachEnemy = timeToBeAbleToAttack(env, unit, distanceToEnemy);
                        if (time < timeToReachEnemy) {
                            changed = true;
                            continue;
                        }
                    } else {
                        // Defender (stay put until attacker comes within range)
                        float timeToReachEnemy = fastestAttackerSpeed > 0 ? (maxRangeDefender - env.attackRange(unit)) / fastestAttackerSpeed : 100000;
                        if (time < timeToReachEnemy) {
                            changed = true;
                            continue;
                        }
                    }
                }

                // if (debug) cout << "DPS: " << groundDPS << " " << airDPS << endl;

                CombatUnit* bestTarget = nullptr;
                int bestTargetIndex = -1;
                float bestScore = 0;
                const WeaponInfo* bestWeapon = nullptr;

                for (size_t j = 0; j < g2.size(); j++) {
                    auto& other = *g2[j];
                    if (other.health == 0)
                        continue;

                    if (((canBeAttackedByAirWeapons(other.type) && airDPS > 0) || (!other.is_flying && groundDPS > 0))) {
                        const auto& info = env.getCombatInfo(unit);
                        auto& otherData = getUnitData(other.type);

                        float airDPS2 = info.airWeapon.getDPS(other.type);
                        float groundDPS2 = info.groundWeapon.getDPS(other.type);

                        auto dps = max(groundDPS2, airDPS2);
                        float score = dps * targetScore(other, group == 0 ? hasGround1 : hasGround2, group == 0 ? hasAir1 : hasAir2) * 0.001f;
                        if (group == 1 && settings.badMicro)
                            score = -score;

                        if (isUnitMelee) {
                            if (settings.enableSurroundLimits && meleeUnitAttackCount[j] >= surround.maxAttackersPerDefender) {
                                // Can't attack this unit, too many melee units are attacking it already
                                continue;
                            }

                            // Assume the enemy has positioned its units in the most annoying way possible
                            // so that we have to attack the ones we really do not want to attack first.
                            if (!settings.badMicro && settings.assumeReasonablePositioning)
                                score = -score;

                            // Melee units should attack other melee units first, and have no splash against ranged units (assume reasonable micro)
                            if (settings.enableMeleeBlocking && isMelee(other.type))
                                score += 1000;
                            // Check for kiting, hard to attack units with a higher movement speed
                            else if (settings.enableMeleeBlocking && unitTypeData.movement_speed < 1.05f * otherData.movement_speed)
                                score -= 500;
                        } else {
                            if (!isFlying(unit.type)) {
                                // If the other unit has a significantly higher range than this unit, then assume we cannot attack it until most melee units are dead
                                float rangeDiff = env.attackRange(other) - env.attackRange(unit);
                                if (opponentFractionMeleeUnits > 0.5f && rangeDiff > 0.5f) {
                                    score -= 1000;
                                } else if (opponentFractionMeleeUnits > 0.3f && rangeDiff > 1.0f) {
                                    score -= 1000;
                                }
                            }
                        }

                        // if (debug) cout << "Potential target: " << UnitTypeToName(other.type) << " score: " << score << endl;

                        if (bestTarget == nullptr || score > bestScore || (score == bestScore && unit.health + unit.shield < bestTarget->health + bestTarget->shield)) {
                            bestScore = score;
                            bestTarget = g2[j];
                            bestTargetIndex = j;
                            bestWeapon = groundDPS2 > airDPS2 ? &info.groundWeapon : &info.airWeapon;
                        }
                    }
                }

                if (bestTarget != nullptr) {
                    if (isUnitMelee) {
                        numMeleeUnitsUsed += 1;
                    }
                    meleeUnitAttackCount[bestTargetIndex]++;

                    // Model splash as if the unit can fire at multiple targets.
                    // TODO: Maybe change the secondary targets to random ones instead of the best targets (can cause a too big advantage of splash, not enough damage spread)
                    float remainingSplash = max(1.0f, bestWeapon->splash);
                    // if (weaponSplash.find(unit.type) != weaponSplash.end()) remainingSplash = weaponSplash[unit.type];

                    // cout << UnitTypeToName(unit.type) << " melee: " << isUnitMelee << " can attack? " << (bestTarget != nullptr) << endl;

                    auto& other = *bestTarget;
                    changed = true;
                    // Pick
                    bool shielded = !isUnitMelee && uniform_real_distribution<float>()(rng) < guardianShieldedUnitFraction[1 - group];
                    auto dps = bestWeapon->getDPS(other.type, shielded ? -2 : 0) * min(1.0f, remainingSplash);
                    float damageMultiplier = 1;

                    if (unit.type == UNIT_TYPEID::PROTOSS_CARRIER) {
                        // Simulate interceptors being destroyed by reducing the DPS as the carrier loses health
                        damageMultiplier = (unit.health + unit.shield) / (unit.health_max + unit.shield_max);

                        // Interceptors take some time to launch. It takes about 4 seconds to launch all interceptors.
                        // (todo: scale with 1.4?)
                        damageMultiplier *= min(1.0f, time / 4.0f);
                    }

                    other.modifyHealth(-dps * damageMultiplier * dt);

                    if (other.health == 0) {
                        // Remove the unit from the group to avoid spending CPU cycles on it
                        // Note that this invalidates the 'bestTarget' and 'other' variables
                        g2[bestTargetIndex] = *g2.rbegin();
                        meleeUnitAttackCount[bestTargetIndex] = *meleeUnitAttackCount.rbegin();
                        g2.pop_back();
                        meleeUnitAttackCount.pop_back();
                        bestTarget = nullptr;
                    }

                    // cout << "Picked target " << (-dps*dt) << " " << other.health << endl;

                    remainingSplash -= 1;
                    // Splash is only applied when a melee unit attacks another melee unit
                    // or a non-melee unit does splash damage.
                    // If it is a melee unit, then splash is only applied to other melee units.
                    // TODO: Better rule: units only apply splash to other units that have a shorter range than themselves, or this unit has a higher movement speed than the other one
                    if (settings.enableSplash && remainingSplash > 0.001f && (!isUnitMelee || isMelee(other.type)) && g2.size() > 0) {
                        // Apply remaining splash to other random melee units
                        size_t offset = (size_t)rand() % g2.size();
                        for (size_t j = 0; j < g2.size() && remainingSplash > 0.001f; j++) {
                            size_t splashIndex = (j + offset) % g2.size();
                            auto* splashOther = g2[splashIndex];
                            if (splashOther != bestTarget && splashOther->health > 0 && (!isUnitMelee || isMelee(splashOther->type))) {
                                bool shieldedOther = !isUnitMelee && uniform_real_distribution<float>()(rng) < guardianShieldedUnitFraction[1 - group];
                                auto dps = bestWeapon->getDPS(splashOther->type, shieldedOther ? -2 : 0) * min(1.0f, remainingSplash);
                                if (dps > 0) {
                                    splashOther->modifyHealth(-dps * damageMultiplier * dt);
                                    remainingSplash -= 1.0f;

                                    if (splashOther->health == 0) {
                                        // Remove the unit from the group to avoid spending CPU cycles on it
                                        g2[splashIndex] = *g2.rbegin();
                                        meleeUnitAttackCount[splashIndex] = *meleeUnitAttackCount.rbegin();
                                        g2.pop_back();
                                        meleeUnitAttackCount.pop_back();
                                        j--;
                                        if (g2.size() == 0) break;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (debug)
                cout << "Meleee attackers used: " << numMeleeUnitsUsed << " did change during last iteration: " << changed << endl;
        }

        time += dt;
        if (time >= settings.maxTime) break;
    }

    result.time = time;

    averageHealthByTime[0] /= max(0.01f, averageHealthByTimeWeight[0]);
    averageHealthByTime[1] /= max(0.01f, averageHealthByTimeWeight[1]);

    result.averageHealthTime = averageHealthByTime;

    // Remove all temporary units
    assert(state.units.size() == inputState.units.size());

#if CACHE_COMBAT
    seen_combats[h] = result;
#endif
    return result;
}

int CombatState::owner_with_best_outcome() const {
    int maxIndex = 0;
    for (auto& u : units)
        maxIndex = max(maxIndex, u.owner);
    vector<float> totalHealth(maxIndex + 1);
    for (auto& u : units)
        totalHealth[u.owner] += u.health + u.shield;

    // cout << totalHealth[0] << " " << totalHealth[1] << " " << totalHealth[2] << endl;
    int winner = 0;
    for (int i = 0; i <= maxIndex; i++)
        if (totalHealth[i] > totalHealth[winner])
            winner = i;
    return winner;
}

int testCombat(const CombatPredictor& predictor, const CombatState& state) {
    return predictor.predict_engage(state).state.owner_with_best_outcome();
}

const vector<UNIT_TYPEID> availableUnitTypesTerran = {
    UNIT_TYPEID::TERRAN_LIBERATOR,
    // UNIT_TYPEID::TERRAN_BATTLECRUISER,
    UNIT_TYPEID::TERRAN_BANSHEE,
    UNIT_TYPEID::TERRAN_CYCLONE,
    UNIT_TYPEID::TERRAN_GHOST,
    UNIT_TYPEID::TERRAN_HELLION,
    UNIT_TYPEID::TERRAN_HELLIONTANK,
    UNIT_TYPEID::TERRAN_MARAUDER,
    UNIT_TYPEID::TERRAN_MARINE,
    UNIT_TYPEID::TERRAN_MEDIVAC,
    // UNIT_TYPEID::TERRAN_MISSILETURRET,
    // UNIT_TYPEID::TERRAN_MULE,
    UNIT_TYPEID::TERRAN_RAVEN,
    UNIT_TYPEID::TERRAN_REAPER,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SIEGETANK,
    UNIT_TYPEID::TERRAN_SIEGETANKSIEGED,
    UNIT_TYPEID::TERRAN_THOR,
    UNIT_TYPEID::TERRAN_THORAP,
    UNIT_TYPEID::TERRAN_VIKINGASSAULT,
    UNIT_TYPEID::TERRAN_VIKINGFIGHTER,
    // UNIT_TYPEID::TERRAN_WIDOWMINE,
    // UNIT_TYPEID::TERRAN_WIDOWMINEBURROWED,
};

const vector<UNIT_TYPEID> availableUnitTypesProtoss = {
    UNIT_TYPEID::PROTOSS_ADEPT,
    // UNIT_TYPEID::PROTOSS_ARCHON, // Archon needs special rules before it is supported by the build optimizer
    UNIT_TYPEID::PROTOSS_CARRIER,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    UNIT_TYPEID::PROTOSS_DARKTEMPLAR,
    UNIT_TYPEID::PROTOSS_DISRUPTOR,
    UNIT_TYPEID::PROTOSS_HIGHTEMPLAR,
    UNIT_TYPEID::PROTOSS_IMMORTAL,
    // UNIT_TYPEID::PROTOSS_MOTHERSHIP,
    UNIT_TYPEID::PROTOSS_OBSERVER,
    UNIT_TYPEID::PROTOSS_ORACLE,
    UNIT_TYPEID::PROTOSS_PHOENIX,
    // UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_SENTRY,
    UNIT_TYPEID::PROTOSS_STALKER,
    UNIT_TYPEID::PROTOSS_TEMPEST,
    UNIT_TYPEID::PROTOSS_VOIDRAY,
    UNIT_TYPEID::PROTOSS_WARPPRISM,
    UNIT_TYPEID::PROTOSS_ZEALOT,
};

const vector<UNIT_TYPEID> availableUnitTypesZerg = {
    UNIT_TYPEID::ZERG_BANELING,
    UNIT_TYPEID::ZERG_BROODLORD,
    UNIT_TYPEID::ZERG_CORRUPTOR,
    UNIT_TYPEID::ZERG_DRONE,
    UNIT_TYPEID::ZERG_HYDRALISK,
    UNIT_TYPEID::ZERG_INFESTOR,
    // UNIT_TYPEID::ZERG_INFESTORTERRAN,
    // UNIT_TYPEID::ZERG_LOCUSTMP,
    // UNIT_TYPEID::ZERG_LOCUSTMPFLYING,
    UNIT_TYPEID::ZERG_LURKERMP,
    UNIT_TYPEID::ZERG_MUTALISK,
    UNIT_TYPEID::ZERG_OVERSEER,
    UNIT_TYPEID::ZERG_QUEEN,
    UNIT_TYPEID::ZERG_RAVAGER,
    UNIT_TYPEID::ZERG_ROACH,
    // UNIT_TYPEID::ZERG_SPINECRAWLER,
    // UNIT_TYPEID::ZERG_SPORECRAWLER,
    UNIT_TYPEID::ZERG_SWARMHOSTMP,
    UNIT_TYPEID::ZERG_ULTRALISK,
    UNIT_TYPEID::ZERG_VIPER,
    UNIT_TYPEID::ZERG_ZERGLING,
};

float CombatPredictor::mineralScore(const CombatState& initialState, const CombatResult& combatResult, int player, const vector<float>& timeToProduceUnits, const CombatUpgrades upgrades) const {
    assert(timeToProduceUnits.size() == 3);
    // The combat result may contain more units due to temporary units spawning (e.g. infested terran, etc.) however never fewer.
    // The first N units correspond to all the N units in the initial state.
    assert(combatResult.state.units.size() >= initialState.units.size());
    float totalScore = 0;
    float ourScore = 0;
    float enemyScore = 0;
    float lossScore = 0;
    float ourSupply = 0;
    for (size_t i = 0; i < initialState.units.size(); i++) {
        auto& unit1 = initialState.units[i];
        auto& unit2 = combatResult.state.units[i];
        assert(unit1.type == unit2.type);

        float healthDiff = (unit2.health - unit1.health) + (unit2.shield - unit1.shield);
        float damageTakenFraction = -healthDiff / (unit1.health_max + unit1.shield_max);
        auto& unitTypeData = getUnitData(unit1.type);

        float cost = unitTypeData.mineral_cost + 1.2f * unitTypeData.vespene_cost;
        if (unit1.owner == player) {
            ourScore += cost * -(1 + damageTakenFraction);
            // if (unit1.type == UNIT_TYPEID::PROTOSS_COLOSSUS) ourScore += 500;
        } else {
            if (defaultCombatEnvironment.calculateDPS(unit2, false) > 0) {
                lossScore += cost * (-10 * (1 - damageTakenFraction));
                ourSupply += unitTypeData.food_required;
            } else {
                lossScore += cost * (-1 * (1 - damageTakenFraction));
            }
            // lossScore += cost * (-100 * (1 - damageTakenFraction));
            enemyScore += cost * (1 + damageTakenFraction);
        }
        // totalScore += score;
    }

    for (auto u : upgrades) {
        auto& data = getUpgradeData(u);
        ourScore -= data.mineral_cost + 1.2f * data.vespene_cost;
    }

    // Handle added units (usually temporary units)
    for (size_t i = initialState.units.size(); i < combatResult.state.units.size(); i++) {
        auto& unit2 = combatResult.state.units[i];

        float healthDiff = unit2.health + unit2.shield;
        float damageTakenFraction = -healthDiff / (unit2.health_max + unit2.shield_max);

        // Unit is likely temporary, use a small cost
        float cost = 5;
        if (unit2.owner != player) {
            lossScore += cost * (-100 * (1 - damageTakenFraction));
            enemyScore += cost * (1 + damageTakenFraction);
        }
    }

    // cout << "Extra cost " << -ourScore << " " << (timeToProduceUnits[1] + 1.2f * timeToProduceUnits[1]) << endl;
    // TODO: Add pylon costs
    ourScore -= timeToProduceUnits[1] + 1.2f * timeToProduceUnits[2];
    // Add pylon/overlord/supply depot cost
    ourSupply -= (ourSupply / 8.0f) * 100;

    // float timeScore = -100 * (pow(timeToProduceUnits/(1*60), 2));
    // float timeMult = 2 * 30/(30 + timeToProduceUnits);
    float timeMult = min(1.0f, 2 * 30/(30 + timeToProduceUnits[0]));
    // float timeMult2 = min(1.0f, 2 * 10/(10 + combatResult.time));
    totalScore = ourScore + enemyScore*timeMult + lossScore;
    // cout << ourScore << " " << enemyScore << " " << lossScore << endl;

    if (combatResult.time > 20) {
        bool hasAnyGroundUnits = false;
        for (size_t i = 0; i < initialState.units.size(); i++) {
            if (combatResult.state.units[i].owner == player && combatResult.state.units[i].health > 0 && !combatResult.state.units[i].is_flying) {
                hasAnyGroundUnits = true;
            }
        }

        // cout << "Estimated time to produce units: " << timeToProduceUnits[0] << " and to beat enemy " << combatResult.time << " " << hasAnyGroundUnits << endl;
        // for (auto u : initialState.units) {
        //     if (u.owner == player) cout << getUnitData(u.type).name << ", ";
        // }
        // cout << endl;

        if (!hasAnyGroundUnits) {
            totalScore -= 1000 * (combatResult.time - 20)/20.0f;
        }
    }

    // totalScore += -1000 * max(0.0f, 1 - ourScore/1000.0f);
    // totalScore = -timeToProduceUnits[0] + lossScore;

    // f(a,ta) < f(b,tb) => f(a,ta - x) < f(b, tb - x)
    // g(t-x) > g(t) : x > 0
    // Ka + Ea*g(ta) < Kb + Eb*g(tb)
    // Ka + Ea*g(ta-x) < Kb + Eb*g(tb-x)

    // Ka + Ea*g(ta-x) < Kb + Eb*g(tb-x)

    // Ea*g(ta) + Ea*g(ta-x) < Eb*g(tb) + Eb*g(tb-x)
    // Ea*g(ta) + Ea*g(ta-x) < Eb*g(tb) + Eb*g(tb-x)
    // cout << "Score " << totalScore << endl;
    return totalScore;
}

float CombatPredictor::mineralScoreFixedTime(const CombatState& initialState, const CombatResult& combatResult, int player, const vector<float>& timeToProduceUnits, const CombatUpgrades upgrades) const {
    if (combatResult.state.owner_with_best_outcome() != player) {
        return -10000 + mineralScore(initialState, combatResult, player, timeToProduceUnits, upgrades);
    }

    assert(timeToProduceUnits.size() == 3);
    // The combat result may contain more units due to temporary units spawning (e.g. infested terran, etc.) however never fewer.
    // The first N units correspond to all the N units in the initial state.
    assert(combatResult.state.units.size() >= initialState.units.size());
    float totalScore = 0;
    float ourScore = 0;
    float enemyScore = 0;
    float lossScore = 0;
    float totalCost = 0;
    float ourDamageTakenCost = 0;
    for (size_t i = 0; i < initialState.units.size(); i++) {
        auto& unit1 = initialState.units[i];
        auto& unit2 = combatResult.state.units[i];
        assert(unit1.type == unit2.type);

        float healthDiff = (unit2.health - unit1.health) + (unit2.shield - unit1.shield);
        float damageTakenFraction = -healthDiff / (unit1.health_max + unit1.shield_max);
        auto& unitTypeData = getUnitData(unit1.type);

        float cost = unitTypeData.mineral_cost + 1.2f * unitTypeData.vespene_cost;
        if (unit1.owner == player) {
            totalCost += cost;
            ourScore += cost * -(1 + damageTakenFraction);
            ourDamageTakenCost += cost * damageTakenFraction;
            // if (unit1.type == UNIT_TYPEID::PROTOSS_COLOSSUS) ourScore += 500;
        } else {
            if (defaultCombatEnvironment.calculateDPS(unit2, false) > 0) {
                lossScore += cost * (-10 * (1 - damageTakenFraction));
            } else {
                lossScore += cost * (-1 * (1 - damageTakenFraction));
            }
            // lossScore += cost * (-100 * (1 - damageTakenFraction));
            enemyScore += cost * (1 + damageTakenFraction);
        }
        // totalScore += score;
    }

    float lostResourcesFraction = ourDamageTakenCost / (0.001f + ourScore);

    for (auto u : upgrades) {
        auto& data = getUpgradeData(u);
        ourScore -= data.mineral_cost + 1.2f * data.vespene_cost;
    }

    float timeToDefeatEnemy = combatResult.averageHealthTime[0];
    if (combatResult.state.owner_with_best_outcome() != player) {
        timeToDefeatEnemy = 20;
    }
    timeToDefeatEnemy = std::min(20.0f, timeToDefeatEnemy);

    // cout << "Extra cost " << -ourScore << " " << (timeToProduceUnits[1] + 1.2f * timeToProduceUnits[1]) << endl;
    // TODO: Add pylon costs
    ourScore -= timeToProduceUnits[1] + 1.2f * timeToProduceUnits[2];

    // float timeScore = -100 * (pow(timeToProduceUnits/(1*60), 2));
    // float timeMult = 2 * 30/(30 + timeToProduceUnits);

    enemyScore -= ourDamageTakenCost;
    float timeMult = 1.0f; // 10/(10 + pow(timeToDefeatEnemy, 2)/5.0f);
    totalScore = enemyScore*timeMult;

    // Tie breaker
    totalScore += ourScore*0.001f;

    // cout << ourScore << " " << enemyScore << " " << lossScore << endl;

    if (combatResult.time > 20) {
        bool hasAnyGroundUnits = false;
        for (size_t i = 0; i < initialState.units.size(); i++) {
            if (combatResult.state.units[i].owner == player && combatResult.state.units[i].health > 0 && !combatResult.state.units[i].is_flying) {
                hasAnyGroundUnits = true;
            }
        }

        if (!hasAnyGroundUnits) {
            totalScore -= 1000 * (combatResult.time - 20)/20.0f;
        }
    }

    cout << "Estimated time to produce units: " << timeToProduceUnits[0] << " total cost " << totalCost << endl;
    // cout << "Estimated time to produce units: " << timeToProduceUnits[0] << " and to beat enemy " << combatResult.time << " (" << combatResult.averageHealthTime[0] << ", " << combatResult.averageHealthTime[1] << ")" << " " << timeToDefeatEnemy << " " << enemyScore << endl;

    return totalScore;
}

CombatUnit makeUnit(int owner, UNIT_TYPEID type) {
    CombatUnit unit;
    unit.health_max = unit.health = maxHealth(type);
    unit.shield_max = unit.shield = maxShield(type);

    // Immortals absorb up to 100 damage over the first two seconds that it is attacked (barrier ability).
    // Approximate this by adding 50 extra shields.
    // Note: shield > max_shield prevents the shield battery from healing it during this time.
    if (type == UNIT_TYPEID::PROTOSS_IMMORTAL) {
        unit.shield += 50;
    }
    unit.energy = 100;
    unit.owner = owner;
    unit.is_flying = isFlying(type);
    unit.type = type;
    return unit;
}

void createState(const CombatState& state, float offset = 0) {
    if (!agent) return;
    for (auto& u : state.units) {
        agent->Debug()->DebugCreateUnit(u.type, Point2D(40 + 20 * (u.owner == 1 ? -1 : 1), 20 + offset), u.owner);
    }
}

struct CompositionGene {
//    private:
    // Indices are into the availableUnitTypes list
    vector<int> unitCounts;

   public:
    vector<pair<UNIT_TYPEID, int>> getUnits(const AvailableUnitTypes& availableUnitTypes) const {
        assert(unitCounts.size() == availableUnitTypes.size());
        vector<pair<UNIT_TYPEID, int>> result;
        for (size_t i = 0; i < unitCounts.size(); i++) {
            if (unitCounts[i] > 0) {
                auto type = availableUnitTypes.getUnitType(i);

                // Make sure this is a unit and not an upgrade
                if (type != UNIT_TYPEID::INVALID) {
                    result.emplace_back(type, unitCounts[i]);
                }
            }
        }
        return result;
    }

    CombatUpgrades getUpgrades(const AvailableUnitTypes& availableUnitTypes) const {
        CombatUpgrades result;
        for (size_t i = 0; i < unitCounts.size(); i++) {
            if (unitCounts[i] > 0) {
                auto item = availableUnitTypes.getBuildOrderItem(i);
                if (!item.isUnitType()) {
                    // An upgrade
                    auto upgrade = item.upgradeID();

                    // If this is an enumerated upgrade then add LEVEL1, LEVEL2 up to LEVEL3 depending on the upgrade count.
                    // If it is a normal upgrade, then just add it regardless of the unit count
                    if(isUpgradeWithLevels(upgrade)) {
                        int finalUpgrade = (int)upgrade + min(unitCounts[i] - 1, 2);
                        for (int upgradeIndex = (int)upgrade; upgradeIndex <= finalUpgrade; upgradeIndex++) {
                            result.add((UPGRADE_ID)upgradeIndex);
                        }
                        break;
                    } else {
                        result.add(upgrade);
                    }
                }
            }
        }

        return result;
    }

    vector<pair<int, int>> getUnitsUntyped(const AvailableUnitTypes& availableUnitTypes) const {
        assert(unitCounts.size() == availableUnitTypes.size());
        vector<pair<int, int>> result;
        for (size_t i = 0; i < unitCounts.size(); i++) {
            if (unitCounts[i] > 0) {
                auto type = availableUnitTypes.getUnitType(i);

                // Make sure this is a unit and not an upgrade
                if (type != UNIT_TYPEID::INVALID) {
                    result.emplace_back((int)type, unitCounts[i]);
                }
            }
        }
        return result;
    }

    void addToState(const CombatPredictor& predictor, CombatState& state, const AvailableUnitTypes& availableUnitTypes, int owner) const {
        assert(unitCounts.size() == availableUnitTypes.size());
        for (size_t i = 0; i < unitCounts.size(); i++) {
            if (unitCounts[i] > 0) {
                auto type = availableUnitTypes.getUnitType(i);

                // Make sure this is a unit and not an upgrade
                if (type != UNIT_TYPEID::INVALID) {
                    for (int j = 0; j < unitCounts[i]; j++) {
                        state.units.push_back(makeUnit(owner, availableUnitTypes.getUnitType(i)));
                    }
                }
            }
        }
        state.environment = &predictor.combineCombatEnvironment(state.environment, getUpgrades(availableUnitTypes), owner);
    }

    void mutate(float amount, default_random_engine& seed, const AvailableUnitTypes& availableUnitTypes) {
        bernoulli_distribution shouldMutateFromNone(amount);
        bernoulli_distribution shouldMutate(amount * 2);
        bernoulli_distribution shouldSwapMutate(0.2);
        for (size_t i = 0; i < unitCounts.size(); i++) {
            if (unitCounts[i] > 0 ? shouldMutate(seed) : shouldMutateFromNone(seed)) {
                int mx = availableUnitTypes.armyCompositionMaximum(i);
                if (mx == 1) {
                    bernoulli_distribution dist(0.2f);
                    unitCounts[i] = (int)dist(seed);
                } else {
                    if (unitCounts[i] > 0 && shouldSwapMutate(seed)) {
                        uniform_int_distribution<int> swapIndex(0, unitCounts.size() - 1);
                        swap(unitCounts[i], unitCounts[swapIndex(seed)]);
                    } else {
                        geometric_distribution<int> dist(1.0f / (2+unitCounts[i]));
                        unitCounts[i] = min(dist(seed), mx);
                    }
                }
            }
        }
    }

    CompositionGene()
        : CompositionGene(0) {}

    CompositionGene(int units)
        : unitCounts(units) {
    }

    void scale(float scale, const AvailableUnitTypes& availableUnitTypes) {
        float offset = 0;
        for (size_t i = 0; i < unitCounts.size(); i++) {
            int mx = availableUnitTypes.armyCompositionMaximum(i);
            float nextPoint = offset + unitCounts[i] * scale;
            unitCounts[i] = (int)round(nextPoint) - (int)round(offset);
            unitCounts[i] = min(unitCounts[i], mx);
            offset = nextPoint;
        }
    }

    static CompositionGene crossover(const CompositionGene& parent1, const CompositionGene& parent2, default_random_engine& seed) {
        assert(parent1.unitCounts.size() == parent2.unitCounts.size());
        bernoulli_distribution dist(0.5);
        CompositionGene gene(parent1.unitCounts.size());
        for (size_t i = 0; i < gene.unitCounts.size(); i++) {
            gene.unitCounts[i] = dist(seed) ? parent1.unitCounts[i] : parent2.unitCounts[i];
        }
        return gene;
    }

    CompositionGene(const AvailableUnitTypes& availableUnitTypes, int meanTotalCount, default_random_engine& seed)
        : unitCounts(availableUnitTypes.size()) {
        int totalUnits = (int)round(exponential_distribution<float>(1.0f / meanTotalCount)(seed));
        vector<int> splitPoints(unitCounts.size()+1);
        auto splitDist = uniform_int_distribution<int>(0, totalUnits);
        for (size_t i = 1; i < unitCounts.size(); i++) {
            splitPoints[i] = splitDist(seed);
        }
        splitPoints[0] = 0;
        splitPoints[splitPoints.size() - 1] = totalUnits;
        sort(splitPoints.begin(), splitPoints.end());

        for (size_t i = 0; i < unitCounts.size(); i++) {
            unitCounts[i] = splitPoints[i+1] - splitPoints[i];
            unitCounts[i] = min(availableUnitTypes.armyCompositionMaximum(i), unitCounts[i]);
        }
    }

    CompositionGene(const AvailableUnitTypes& availableUnitTypes, const vector<pair<UNIT_TYPEID, int>>& units)
        : unitCounts(availableUnitTypes.size()) {
        for (auto u : units) {
            unitCounts[availableUnitTypes.getIndex(u.first)] += u.second;
        }
    }
};

void scaleUntilWinning(const CombatPredictor& predictor, const CombatState& opponent, const AvailableUnitTypes& availableUnitTypes, CompositionGene& gene) {
    for (int its = 0; its < 5; its++) {
        CombatState state = opponent;
        gene.addToState(predictor, state, availableUnitTypes, 2);
        if (predictor.predict_engage(state, false, false).state.owner_with_best_outcome() == 2) break;

        gene.scale(1.5f, availableUnitTypes);
    }
}

float calculateFitness(const CombatPredictor& predictor, const CombatState& opponent, const AvailableUnitTypes& availableUnitTypes, CompositionGene& gene, const vector<float>& timeToProduceUnits) {
    CombatState state = opponent;
    
    // Upgrades except the ones we already have
    auto upgrades = gene.getUpgrades(availableUnitTypes);
    if (state.environment != nullptr) upgrades.remove(state.environment->upgrades[1]);

    gene.addToState(predictor, state, availableUnitTypes, 2);
    // TODO: Ignore current upgrades in costs
    return predictor.mineralScore(state, predictor.predict_engage(state, false, false), 2, timeToProduceUnits, upgrades);  // + mineralScore(state, predictor.predict_engage(state, false, true), 2)) * 0.5f;
}

float calculateFitnessFixedTime(const CombatPredictor& predictor, const CombatState& opponent, const AvailableUnitTypes& availableUnitTypes, CompositionGene& gene, const vector<float>& timeToProduceUnits) {
    CombatState state = opponent;

    // Upgrades except the ones we already have
    auto upgrades = gene.getUpgrades(availableUnitTypes);
    if (state.environment != nullptr) upgrades.remove(state.environment->upgrades[1]);

    gene.addToState(predictor, state, availableUnitTypes, 2);
    // TODO: Ignore current upgrades in costs
    return predictor.mineralScoreFixedTime(state, predictor.predict_engage(state, false, false), 2, timeToProduceUnits, gene.getUpgrades(availableUnitTypes));  // + mineralScore(state, predictor.predict_engage(state, false, true), 2)) * 0.5f;
    
    // return predictor.mineralScore(state, predictor.predict_engage(state, false, false), 2, timeToProduceUnits, upgrades);
}

void logRecordings(CombatState& state, const CombatPredictor& predictor, float spawnOffset, string msg) {
    createState(state, spawnOffset);
    // cout << "Duration " << watch.millis() << " ms" << endl;
    vector<int> mineralCosts(3);
    vector<int> vespeneCosts(3);
    for (auto u : state.units) {
        mineralCosts[u.owner] += getUnitData(u.type).mineral_cost;
        vespeneCosts[u.owner] += getUnitData(u.type).vespene_cost;
    }

    cout << "Team 1 costs: " << mineralCosts[1] << "+" << vespeneCosts[1] << endl;
    cout << "Team 2 costs: " << mineralCosts[2] << "+" << vespeneCosts[2] << endl;
    // createState(state);
    
    CombatRecording recording;
    CombatResult newResult = predictor.predict_engage(state, false, false, &recording);
    recording.writeCSV(msg + "2.csv");

    CombatRecording recording2;
    CombatResult newResult2 = predictor.predict_engage(state, false, true, &recording2);
    recording2.writeCSV(msg + "3.csv");

    CombatRecording recording3;
    CombatSettings settings;
    settings.maxTime = 3;
    settings.startTime = 0;
    for (int i = 0; i < 13; i++) {
        CombatResult newResult2 = predictor.predict_engage(state, settings, &recording3);
        state = newResult2.state;
        settings.startTime = newResult2.time;
        settings.maxTime = newResult2.time + 3;
    }
    recording3.writeCSV(msg + "4.csv");
}

int64_t micros() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

ArmyComposition findBestCompositionGenetic(const CombatPredictor& predictor, const AvailableUnitTypes& availableUnitTypes, const CombatState& opponent, const BuildOptimizerNN* buildTimePredictor, const BuildState* startingBuildState, vector<pair<UNIT_TYPEID,int>>* seedComposition) {
    CompositionSearchSettings settings(predictor, availableUnitTypes, buildTimePredictor);
    return findBestCompositionGenetic(opponent, settings, startingBuildState, seedComposition);
}

ArmyComposition findBestCompositionGenetic(const CombatState& opponent, CompositionSearchSettings settings, const BuildState* startingBuildState, std::vector<std::pair<sc2::UNIT_TYPEID,int>>* seedComposition) {
    auto& predictor = settings.combatPredictor;
    auto* buildTimePredictor = settings.buildTimePredictor;
    auto& availableUnitTypes = settings.availableUnitTypes;

    Stopwatch watch;

    const int POOL_SIZE = 20;
    const float mutationRate = 0.2f;
    vector<CompositionGene> generation(POOL_SIZE);
    default_random_engine rnd(micros());
    for (auto& gene : generation) {
        gene = CompositionGene(availableUnitTypes, 10, rnd);

        // cout << "Seed ";
        // for (auto u : gene.getUnits(availableUnitTypes)) {
        //     cout << getUnitData(u.first).name << " " << u.second << ", ";
        // }
        // cout << endl;
    }

    vector<pair<int,int>> startingUnitsNN;
    if (startingBuildState != nullptr && buildTimePredictor != nullptr) {
        for (auto u : startingBuildState->units) startingUnitsNN.push_back({(int)u.type, u.units});
        for (auto u : startingBuildState->upgrades) startingUnitsNN.push_back({ (int)u + UPGRADE_ID_OFFSET, 1 });
    }
    
    for (int i = 0; i < 50; i++) {
        assert(generation.size() == POOL_SIZE);
        if (i == 20 && seedComposition != nullptr) {
            generation[generation.size()-1] = CompositionGene(availableUnitTypes, *seedComposition);
        }

        vector<float> fitness(generation.size());
        vector<int> indices(generation.size());

        if (false) {
            vector<vector<pair<int,int>>> targetUnitsNN(generation.size());
            for (size_t j = 0; j < generation.size(); j++) {
                assert(generation[j].unitCounts.size() == availableUnitTypes.size());
                scaleUntilWinning(predictor, opponent, availableUnitTypes, generation[j]);
                targetUnitsNN[j] = generation[j].getUnitsUntyped(availableUnitTypes);
                auto upgrades = generation[j].getUpgrades(availableUnitTypes);
                upgrades.remove(startingBuildState->upgrades);
                for (auto u : upgrades) targetUnitsNN[j].push_back({ (int)u + UPGRADE_ID_OFFSET, 1 });
            }

            vector<vector<float>> timesToProduceUnits = startingBuildState != nullptr && buildTimePredictor != nullptr ? buildTimePredictor->predictTimeToBuild(startingUnitsNN, startingBuildState->resources, targetUnitsNN) : vector<vector<float>>(generation.size(), vector<float>(3));

            for (size_t j = 0; j < generation.size(); j++) {
                indices[j] = j;
                fitness[j] = calculateFitness(predictor, opponent, availableUnitTypes, generation[j], timesToProduceUnits[j]);
            }
        } else {
            for (int i = 0; i < 4; i++) {
                vector<vector<pair<int,int>>> targetUnitsNN(generation.size());
                for (size_t j = 0; j < generation.size(); j++) {
                    assert(generation[j].unitCounts.size() == availableUnitTypes.size());
                    targetUnitsNN[j] = generation[j].getUnitsUntyped(availableUnitTypes);
                    auto upgrades = generation[j].getUpgrades(availableUnitTypes);
                    upgrades.remove(startingBuildState->upgrades);
                    for (auto u : upgrades) targetUnitsNN[j].push_back({ (int)u + UPGRADE_ID_OFFSET, 1 });
                }

                vector<vector<float>> timesToProduceUnits = startingBuildState != nullptr && buildTimePredictor != nullptr ? buildTimePredictor->predictTimeToBuild(startingUnitsNN, startingBuildState->resources, targetUnitsNN) : vector<vector<float>>(generation.size(), vector<float>(3));
                for (size_t j = 0; j < generation.size(); j++) {
                    float factor = settings.availableTime / max(0.001f, timesToProduceUnits[j][0]);
                    factor = max(0.5f, min(1.5f, factor));
                    if (abs(factor - 1.0) > 0.01f) {
                        // for (auto u : generation[j].getUnits(availableUnitTypes)) {
                        //     cout << getUnitData(u.first).name << " " << u.second << ", ";
                        // }
                        // cout << endl;
                        // cout << "Scaling " << factor << endl;
                        generation[j].scale(factor, availableUnitTypes);
                    }
                }
            }

            {
                vector<vector<pair<int,int>>> targetUnitsNN(generation.size());
                for (size_t j = 0; j < generation.size(); j++) {
                    targetUnitsNN[j] = generation[j].getUnitsUntyped(availableUnitTypes);
                    auto upgrades = generation[j].getUpgrades(availableUnitTypes);
                    upgrades.remove(startingBuildState->upgrades);
                    for (auto u : upgrades) targetUnitsNN[j].push_back({ (int)u + UPGRADE_ID_OFFSET, 1 });
                }

                vector<vector<float>> timesToProduceUnits = startingBuildState != nullptr && buildTimePredictor != nullptr ? buildTimePredictor->predictTimeToBuild(startingUnitsNN, startingBuildState->resources, targetUnitsNN) : vector<vector<float>>(generation.size(), vector<float>(3));

                for (size_t j = 0; j < generation.size(); j++) {
                    indices[j] = j;
                    fitness[j] = calculateFitnessFixedTime(predictor, opponent, availableUnitTypes, generation[j], timesToProduceUnits[j]);
                }
            }
        }

        sortByValueDescending<int, float>(indices, [&](int index) { return fitness[index]; });
        // for (int j = 0; j < indices.size(); j++) {
        //     cout << " " << fitness[indices[j]];
        // }
        // cout << endl;
        vector<CompositionGene> nextGeneration;
        // Add the N best performing genes
        for (int j = 0; j < 5; j++) {
            nextGeneration.push_back(generation[indices[j]]);
        }
        // Add a random one as well
        nextGeneration.push_back(generation[uniform_int_distribution<int>(0, indices.size() - 1)(rnd)]);

        uniform_int_distribution<int> randomParentIndex(0, nextGeneration.size() - 1);
        while (nextGeneration.size() < POOL_SIZE) {
            nextGeneration.push_back(CompositionGene::crossover(generation[randomParentIndex(rnd)], generation[randomParentIndex(rnd)], rnd));
        }

        // Note: do not mutate the first gene
        for (size_t i = 1; i < nextGeneration.size(); i++) {
            nextGeneration[i].mutate(mutationRate, rnd, availableUnitTypes);
        }

        swap(generation, nextGeneration);

        /*if (i == 49) cout << "Best fitness " << fitness[indices[0]] << " time to produce: " << timesToProduceUnits[indices[0]] << endl;
        if (i == 49) {
            for (auto u : targetUnitsNN[indices[0]]) {
                cout << "Target unit: " << UnitTypeToName((UNIT_TYPEID)u.first) << " " << u.second << endl;
            }
        }*/
    }

    // CombatState testState = opponent;
    // generation[0].addToState(testState, availableUnitTypes, 2);
    // logRecordings(testState, predictor);

    ArmyComposition result;
    result.unitCounts = generation[0].getUnits(availableUnitTypes);
    result.upgrades = generation[0].getUpgrades(availableUnitTypes);
    return result;
}

void unitTestSurround() {
    // One unit can be surrounded by 6 melee units and attacked by all 6 at the same time
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_MARINE), 2) * M_PI * 1, 1).maxAttackersPerDefender == 6);
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_MARINE), 2) * M_PI * 1, 1).maxMeleeAttackers == 6);

    // Two units can be surrounded by 8 melee units, but each one can only be attacked by at most 4
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_MARINE), 2) * M_PI * 2, 2).maxAttackersPerDefender == 4);
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_MARINE), 2) * M_PI * 2, 2).maxMeleeAttackers == 8);

    // Two units can be surrounded by 9 melee units, but each one can only be attacked by at most 3
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_MARINE), 2) * M_PI * 3, 3).maxAttackersPerDefender == 3);
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_MARINE), 2) * M_PI * 3, 3).maxMeleeAttackers == 9);

    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_MARINE), 2) * M_PI * 4, 4).maxAttackersPerDefender == 3);
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_MARINE), 2) * M_PI * 4, 4).maxMeleeAttackers == 10);

    // One thor can be attacked by 10 melee units at a time.
    // This seems to be slightly incorrect, the real number is only 9, but it's approximately correct at least
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_THOR), 2) * M_PI * 1, 1).maxAttackersPerDefender == 10);
    assert(maxSurround(pow(unitRadius(UNIT_TYPEID::TERRAN_THOR), 2) * M_PI * 1, 1).maxMeleeAttackers == 10);
}

void CombatPredictor::unitTest(const BuildOptimizerNN& buildTimePredictor) const {
    unitTestSurround();

    auto u1 = makeUnit(1, UNIT_TYPEID::TERRAN_BATTLECRUISER);
    auto u2 = makeUnit(1, UNIT_TYPEID::TERRAN_THOR);
    cout << "DPS1 " << defaultCombatEnvironment.calculateDPS(u1, u2) << endl;
    cout << "DPS2 " << defaultCombatEnvironment.calculateDPS(u2, u1) << endl;

    assert(testCombat(*this, {{
		makeUnit(1, UNIT_TYPEID::TERRAN_VIKINGFIGHTER),
        makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
	}}) == 1);

	assert(testCombat(*this, {{
		makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
		makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
		makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
		makeUnit(2, UNIT_TYPEID::TERRAN_MEDIVAC),
		makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
	}}) == 1);

	// Medivacs are pretty good (this is a very narrow win though)
	assert(testCombat(*this, {{
		makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
		makeUnit(1, UNIT_TYPEID::TERRAN_MEDIVAC),
		makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
		makeUnit(2, UNIT_TYPEID::TERRAN_MARINE),
	}}) == 1);

	// 1 marine wins against 1 zergling
	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_MARINE, 50, false),
		CombatUnit(2, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
	}}) == 1);

	// Symmetric
	assert(testCombat(*this, {{
		CombatUnit(2, UNIT_TYPEID::TERRAN_MARINE, 50, false),
		CombatUnit(1, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
	}}) == 2);

	// 4 marines win against 4 zerglings
	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_MARINE, 50, false),
		CombatUnit(1, UNIT_TYPEID::TERRAN_MARINE, 50, false),
		CombatUnit(1, UNIT_TYPEID::TERRAN_MARINE, 50, false),
		CombatUnit(1, UNIT_TYPEID::TERRAN_MARINE, 50, false),
		CombatUnit(2, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
		CombatUnit(2, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
		CombatUnit(2, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
		CombatUnit(2, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
	}}) == 1);

	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_MARINE, 50, false),
		CombatUnit(2, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
		CombatUnit(2, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
		CombatUnit(2, UNIT_TYPEID::ZERG_ZERGLING, 35, false),
	}}) == 2);

	assert(testCombat(*this, {{
		makeUnit(1, UNIT_TYPEID::ZERG_SPORECRAWLER),
		makeUnit(1, UNIT_TYPEID::ZERG_SPORECRAWLER),
        makeUnit(1, UNIT_TYPEID::ZERG_SPORECRAWLER),
        makeUnit(2, UNIT_TYPEID::TERRAN_REAPER),
        makeUnit(2, UNIT_TYPEID::TERRAN_REAPER),
        makeUnit(2, UNIT_TYPEID::TERRAN_REAPER),
	}}) == 2);

	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
	}}) == 1);

    // TODO: Splash?
	// assert(testCombat(*this, {{
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	// }}) == 1);

	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_CYCLONE, 180, true),
		CombatUnit(2, UNIT_TYPEID::PROTOSS_IMMORTAL, 200, true),
	}}) == 2);

	assert(testCombat(*this, {{
		makeUnit(1, UNIT_TYPEID::TERRAN_BATTLECRUISER),
		makeUnit(2, UNIT_TYPEID::TERRAN_THOR),
	}}) == 1);

    assert(testCombat(*this, {{
		makeUnit(1, UNIT_TYPEID::ZERG_INFESTOR),
		makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
	}}) == 1);

	// Wins due to splash damage
	// Really depends on microing though...
	// assert(testCombat(*this, {{
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// 	CombatUnit(2, UNIT_TYPEID::ZERG_MUTALISK, 120, true),
	// }}) == 1);

	// Colossus can be attacked by air weapons
	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS, 200, false),
	}}) == 1);

    {
        CombatSettings settings;
        settings.maxTime = 3;
        auto result = predict_engage({{
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE),
            makeUnit(2, UNIT_TYPEID::PROTOSS_IMMORTAL),
            makeUnit(2, UNIT_TYPEID::PROTOSS_IMMORTAL),
        }}, settings, nullptr, 1);

        cout << result.state.str() << endl;
    }

	// Do not assume all enemies will just target the most beefy unit and leave the banshee alone
	// while it takes out the hydras
	assert(testCombat(*this, {{
		makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
		makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
		makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
		makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
		makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
		makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
		makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
		makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
		makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
		makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
		makeUnit(1, UNIT_TYPEID::ZERG_ZERGLING),
		makeUnit(1, UNIT_TYPEID::ZERG_ZERGLING),
		makeUnit(1, UNIT_TYPEID::ZERG_ZERGLING),
		makeUnit(1, UNIT_TYPEID::ZERG_ZERGLING),
		makeUnit(1, UNIT_TYPEID::ZERG_ZERGLING),
		makeUnit(1, UNIT_TYPEID::ZERG_ZERGLING),
		makeUnit(1, UNIT_TYPEID::ZERG_ZERGLING),
		makeUnit(2, UNIT_TYPEID::TERRAN_BANSHEE),
		makeUnit(2, UNIT_TYPEID::TERRAN_THOR),
	}}) == 1);

    CombatState state1 = { {
        makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
        makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
        makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),

        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
    } };

    CombatState state2 = { {
        makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
        makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
        makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
        makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
        makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
        makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
        makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
        makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
        makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
        makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
        makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
    } };

    // CombatResult newResult = this->predict_engage(state2, true);

    // cout << "Score1 " << mineralScore(state1, this->predict_engage(state1), 2) << endl;
    // cout << "Score2 " << mineralScore(state2, this->predict_engage(state2), 2) << endl;

    // createState(state2);

    // TODO: Melee units should get a negative effect due to not being able to focus fire
    // TODO: For ground units, add time offsets due to the army size because every unit cannot be at the frontlines
    // TODO: Long range units can take down static defenses without taking any damage (e.g. tempest vs missile turret)
    // TODO: Shield armor != armor
    // TODO: Kiting approximation?

    for (int i = 0; i < 1; i++) {
        // Problematic
        // findBestCompositionGenetic(*this, availableUnitTypesProtoss, {{
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // 	makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        // }});

        BuildState startingBuildState({
            { UNIT_TYPEID::PROTOSS_NEXUS, 1},
            { UNIT_TYPEID::PROTOSS_PROBE, 24 },
            { UNIT_TYPEID::PROTOSS_ASSIMILATOR, 2 },
            { UNIT_TYPEID::PROTOSS_PYLON, 3 },
            // { UNIT_TYPEID::PROTOSS_STARGATE, 2 },
            { UNIT_TYPEID::PROTOSS_GATEWAY, 2 },
            // { UNIT_TYPEID::TERRAN_STARPORT, 1 },
        });
        startingBuildState.resources.vespene = 200;
        startingBuildState.race = Race::Protoss;

        CombatState opponentUnits = { {
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_ARCHON),
            makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),
            makeUnit(1, UNIT_TYPEID::PROTOSS_SENTRY),




		} };

        CombatState otherUnits = { {
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
            makeUnit(2, UNIT_TYPEID::ZERG_ZERGLING),
		} };

		auto res = findBestCompositionGenetic(*this, getAvailableUnitsForRace(Race::Protoss, UnitCategory::ArmyCompositionOptions), opponentUnits, &buildTimePredictor, &startingBuildState);

        CombatState resState = opponentUnits;
        for (auto u : res.unitCounts) for (int i = 0; i < u.second; i++) resState.units.push_back(makeUnit(2, u.first));
        // createState(resState);

        cout << "Counter" << endl;
        for (auto u : res.unitCounts) {
            cout << UnitTypeToName(u.first) << " " << u.second << endl;
            // for (int i = 0; i < u.second; i++) {
            //     opponentUnits.units.push_back(makeUnit(1, u.first));
            // }
        }
        cout << "Upgrades: ";
        for (auto u : res.upgrades) {
            cout << UpgradeIDToName(u) << " ";
        }
        cout << endl;
        cout << res.upgrades.upgrades << endl;

        for (auto& u : opponentUnits.units) otherUnits.units.push_back(u);

        
        CombatResult newResult = predict_engage(otherUnits, false, false, nullptr);

        logRecordings(resState, *this, 0, "alt_recording");

        /*CombatState ourUnits1 = { {
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
        }};
        CombatState ourUnits2 = { {
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER),
            makeUnit(2, UNIT_TYPEID::PROTOSS_PHOENIX),
            makeUnit(2, UNIT_TYPEID::PROTOSS_PHOENIX),
        }};
        for (auto& u : opponentUnits.units) ourUnits1.units.push_back(u);
        for (auto& u : opponentUnits.units) ourUnits2.units.push_back(u);

        for (int k = 0; k < 2; k++) {
            auto& ourUnits = k == 0 ? ourUnits1 : ourUnits2;
            vector<pair<int,int>> startingUnitsNN;
            for (auto u : startingBuildState.units) startingUnitsNN.push_back({(int)u.type, u.units});

            vector<pair<UNIT_TYPEID, int>> targetUnits;
            for (auto u : ourUnits.units) if (u.owner == 2) targetUnits.push_back({ u.type, 1 });

            auto gene = CompositionGene(availableUnitTypesProtoss, targetUnits);

            vector<float> timeToBuild = buildTimePredictor.predictTimeToBuild(startingUnitsNN, startingBuildState.resources, { gene.getUnitsUntyped(availableUnitTypesProtoss) });
            cout << endl;
            cout << "Time to build " << timeToBuild[0] << endl;
            auto combatResult = predict_engage(ourUnits, false, false, nullptr, 1);
            cout << "Combat time " << combatResult.time << endl;
            for (auto& u : combatResult.state.units) {
                cout << UnitTypeToName(u.type) << " " << u.health << " " << u.shield << endl;
            }
            cout << "Mineral score " << mineralScore(ourUnits, combatResult, 2, timeToBuild[0]) << endl;
        }*/


        // auto buildOrder = findBestBuildOrderGenetic(startingBuildState, res);


	}

    /*CombatState state3 = {{
		makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
		makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
		makeUnit(1, UNIT_TYPEID::PROTOSS_SHIELDBATTERY),
		makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_PHOTONCANNON),
		makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
		makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
		makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
		makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
		makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
		makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
		makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
		makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
		makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
		makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
		makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
		makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY),
	}};
	createState(state3);*/

    // Learn targeting score :: unit -> unit -> float
    // Simulate lots of battles with random target scores
    // target score for units += score in battle * result of battle (+1,-1)
}
