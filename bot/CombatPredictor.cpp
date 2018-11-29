#include "CombatPredictor.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <random>
#include "Bot.h"
#include "utilities/mappings.h"
#include "utilities/profiler.h"
#include "utilities/stdutils.h"

using namespace std;
using namespace sc2;

static UnitTypes unitTypes;

inline bool canBeAttackedByAirWeapons(UNIT_TYPEID type) {
    return isFlying(type) || type == UNIT_TYPEID::PROTOSS_COLOSSUS;
}

float calculateDPS(const Weapon& weapon, UNIT_TYPEID target) {
    // canBeAttackedByAirWeapons is primarily for coloussus.
    if (weapon.type == Weapon::TargetType::Any || (weapon.type == Weapon::TargetType::Air ? canBeAttackedByAirWeapons(target) : !isFlying(target))) {
        float dmg = weapon.damage_;
        for (auto& b : weapon.damage_bonus) {
            if (contains(unitTypes[(int)target].attributes, b.attribute)) {
                dmg += b.bonus;
            }
        }

        int armor = unitTypes[(int)target].armor;
        return max(0.0f, dmg - armor) * weapon.attacks / weapon.speed;
    }

    return 0;
}

struct WeaponInfo {
   private:
    mutable vector<float> dpsCache;
    float baseDPS;

   public:
    bool available;
    float splash;
    const Weapon* weapon;

    float getDPS() const {
        return baseDPS;
    }

    float getDPS(UNIT_TYPEID target) const {
        if ((int)target >= dpsCache.size()) {
            if (!available)
                return 0;
            for (int i = dpsCache.size(); i <= (int)target; i++) {
                dpsCache.push_back(calculateDPS(*weapon, (UNIT_TYPEID)i));
            }
        }

        return dpsCache[(int)target];
    }

    float range() const {
        return weapon != nullptr ? weapon->range : 0;
    }

    WeaponInfo()
        : baseDPS(0), available(false), splash(0), weapon(nullptr) {
    }

    WeaponInfo(const Weapon& weapon) {
        available = true;
        splash = 0;
        this->weapon = &weapon;
        baseDPS = weapon.damage_ * weapon.attacks / weapon.speed;
    }
};

struct UnitCombatInfo {
    WeaponInfo groundWeapon;
    WeaponInfo airWeapon;

    UnitCombatInfo(UNIT_TYPEID type) {
        auto& data = unitTypes[(int)type];
        for (const Weapon& weapon : data.weapons) {
            if (weapon.type == Weapon::TargetType::Any || weapon.type == Weapon::TargetType::Air) {
                if (airWeapon.available) {
                    cerr << "For unit type " << UnitTypeToName(type) << endl;
                    cerr << "Weapon slot is already used";
                    assert(false);
                }
                airWeapon = WeaponInfo(weapon);
            }
            if (weapon.type == Weapon::TargetType::Any || weapon.type == Weapon::TargetType::Ground) {
                if (groundWeapon.available) {
                    cerr << "For unit type " << UnitTypeToName(type) << endl;
                    cerr << "Weapon slot is already used";
                    assert(false);
                }
                groundWeapon = WeaponInfo(weapon);
            }
        }
    }
};

struct CombatRecordingFrame {
    int tick;
    vector<tuple<UNIT_TYPEID, int, float>> healths;
    void add(UNIT_TYPEID type, int owner, float health, float shield) {
        for (int i = 0; i < healths.size(); i++) {
            if (get<0>(healths[i]) == type && get<1>(healths[i]) == owner) {
                healths[i] = make_tuple(type, owner, get<2>(healths[i]) + health + shield);
                return;
            }
        }
        healths.push_back(make_tuple(type, owner, health + shield));
    }
};

struct CombatRecording {
    vector<CombatRecordingFrame> frames;

    void writeCSV(string filename) {
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
                    output << unitTypes[(int)get<0>(p)].name << "\t";
                }
            }
            output << endl;
            for (auto& f : frames) {
                output << ((f.tick - frames[0].tick) / 22.4f) << "\t";
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
};

void CombatRecorder::tick() {
    vector<Unit> units;
    for (auto u : agent.Observation()->GetUnits()) {
        units.push_back(*u);
    }
    frames.push_back(make_pair(agent.Observation()->GetGameLoop(), units));
}

void CombatRecorder::finalize() {
    map<Tag, float> totalHealths;
    set<Tag> inCombat;
    int firstFrame = frames.size() - 1;
    int lastFrame = 0;
    for (int i = 0; i < frames.size(); i++) {
        auto& p = frames[i];
        for (auto& u : p.second) {
            if (totalHealths.find(u.tag) != totalHealths.end()) {
                float minDist = 10000;
                if (!inCombat.count(u.tag)) {
                    for (auto& u2 : p.second) {
                        if (inCombat.count(u2.tag)) {
                            minDist = min(minDist, DistanceSquared2D(u2.pos, u.pos));
                        }
                    }
                }
                if (u.health + u.shield != totalHealths[u.tag] || u.engaged_target_tag != NullTag || minDist < 5 * 5) {
                    inCombat.insert(u.tag);
                    firstFrame = min(firstFrame, i - 1);
                    lastFrame = max(lastFrame, i);
                }
            }

            totalHealths[u.tag] = u.health + u.shield;
        }
    }

    CombatRecording recording;
    for (int i = firstFrame; i <= lastFrame; i++) {
        auto& p = frames[i];
        CombatRecordingFrame frame;
        frame.tick = p.first;
        for (auto& u : p.second) {
            if (inCombat.count(u.tag)) {
                frame.add(u.unit_type, u.owner, u.health, u.shield);
            }
        }
        recording.frames.push_back(frame);
    }

    recording.writeCSV("recording.csv");
};

static vector<UnitCombatInfo> combatInfo;

void CombatPredictor::init(const ObservationInterface* observation) {
    unitTypes = observation->GetUnitTypeData();
    for (int i = 0; i < unitTypes.size(); i++) {
        combatInfo.push_back(UnitCombatInfo((UNIT_TYPEID)i));
    }

    combatInfo[(int)UNIT_TYPEID::TERRAN_LIBERATOR].airWeapon.splash = 3.0;
    combatInfo[(int)UNIT_TYPEID::TERRAN_MISSILETURRET].airWeapon.splash = 3.0;
    combatInfo[(int)UNIT_TYPEID::TERRAN_SIEGETANKSIEGED].groundWeapon.splash = 4.0;
    combatInfo[(int)UNIT_TYPEID::TERRAN_HELLION].groundWeapon.splash = 2;
    combatInfo[(int)UNIT_TYPEID::TERRAN_HELLIONTANK].groundWeapon.splash = 3;
    combatInfo[(int)UNIT_TYPEID::ZERG_MUTALISK].groundWeapon.splash = 1.44f;
    combatInfo[(int)UNIT_TYPEID::ZERG_MUTALISK].airWeapon.splash = 1.44f;
    combatInfo[(int)UNIT_TYPEID::TERRAN_THOR].airWeapon.splash = 3;
    combatInfo[(int)UNIT_TYPEID::PROTOSS_ARCHON].groundWeapon.splash = 2;
    combatInfo[(int)UNIT_TYPEID::PROTOSS_ARCHON].airWeapon.splash = 2;
    combatInfo[(int)UNIT_TYPEID::PROTOSS_COLOSSUS].groundWeapon.splash = 3;
};

// TODO: Air?
float attackRange(UNIT_TYPEID type) {
    return max(combatInfo[(int)type].airWeapon.range(), combatInfo[(int)type].groundWeapon.range());
}

bool isMelee(UNIT_TYPEID type) {
    return attackRange(type) <= 2;
}

float calculateDPS(UNIT_TYPEID type, bool air) {
    return air ? combatInfo[(int)type].airWeapon.getDPS() : combatInfo[(int)type].groundWeapon.getDPS();
}

float calculateDPS(const vector<CombatUnit>& units, bool air) {
    float dps = 0;
    for (auto& u : units)
        dps += calculateDPS(u.type, air);
    return dps;
}

float calculateDPS(CombatUnit& unit1, CombatUnit& unit2) {
    auto& info = combatInfo[(int)unit1.type];
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

float targetScore(const CombatUnit& unit, bool hasGround, bool hasAir) {
    const float VESPENE_MULTIPLIER = 1.5f;
    float cost = unitTypes[(int)unit.type].mineral_cost + VESPENE_MULTIPLIER * unitTypes[(int)unit.type].vespene_cost;

    float score = 0;

    float airDPS = calculateDPS(unit.type, true);
    float groundDPS = calculateDPS(unit.type, false);

    score += 0.01 * cost;
    score += 1000 * max(groundDPS, airDPS) / (1 + unit.health + unit.shield);
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

// Owner = 1 is the defender, Owner != 1 is an attacker
CombatResult CombatPredictor::predict_engage(const CombatState& inputState, bool debug, bool badMicro, CombatRecording* recording) const {
    // cout << "Predicting combat" << endl;
    // Sort units by target score
    // sides attack sqrt(n) units with highest score evenly

    // Copy state
    CombatResult result;
    result.state = inputState;
    CombatState& state = result.state;

    // TODO: Is it 1 and 2?
    auto units1 = filterByOwner(state.units, 1);
    auto units2 = filterByOwner(state.units, 2);

    auto rng = std::default_random_engine{};
    shuffle(begin(units1), end(units1), rng);
    shuffle(begin(units2), end(units2), rng);

    // sortByValueDescending<CombatUnit*>(units1, [=] (auto u) { return targetScore(*u, true, true); });
    // sortByValueDescending<CombatUnit*>(units2, [=] (auto u) { return targetScore(*u, true, true); });

    float maxRangeDefender = 0;
    float fastestAttackerSpeed = 0;
    for (auto& u : units1) {
        maxRangeDefender = max(maxRangeDefender, attackRange(u->type));
    }
    for (auto& u : units2) {
        fastestAttackerSpeed = max(fastestAttackerSpeed, unitTypes[(int)u->type].movement_speed);
    }

    bool changed = true;
    // Note: required in case of healers on both sides to avoid inf loop
    const int MAX_ITERATIONS = 100;
    float time = 0;
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
            }
        }
        for (auto u : units2) {
            if (u->health > 0) {
                hasAir2 += canBeAttackedByAirWeapons(u->type);
                hasGround2 += !u->is_flying;
                float r = unitRadius(u->type);
                groundArea2 += r * r;
            }
        }

        if (recording != nullptr) {
            CombatRecordingFrame frame;
            frame.tick = (int)round(time * 22.4f);
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
            cout << "Iteration " << it << " " << dt << endl;
        changed = false;
        for (int group = 0; group < 2; group++) {
            // TODO: Group 1 always attacks first
            auto& g1 = group == 0 ? units1 : units2;
            auto& g2 = group == 0 ? units2 : units1;
            SurroundInfo surround = group == 0 ? surroundInfo1 : surroundInfo2;
            // The melee units furthest back in the group have to move around the whole group and around the whole enemy group
            // (very very approximative)
            float maxExtraMeleeDistance = sqrt(groundArea1 / M_PI) * M_PI + sqrt(groundArea2 / M_PI) * M_PI;

            int numMeleeUnitsUsed = 0;

            // Only a single healer can heal a given unit at a time
            // (holds for medivacs and shield batteries at least)
            vector<bool> hasBeenHealed(g1.size());
            // How many melee units that have attacked a particular enemy so far
            vector<int> meleeUnitAttackCount(g2.size());

            if (debug)
                cout << "Max meleee attackers: " << surround.maxMeleeAttackers << " " << surround.maxAttackersPerDefender << endl;

            for (int i = 0; i < g1.size(); i++) {
                auto& unit = *g1[i];

                if (unit.health == 0)
                    continue;

                auto& unitTypeData = unitTypes[(int)unit.type];
                float airDPS = calculateDPS(unit.type, true);
                float groundDPS = calculateDPS(unit.type, false);

                if (debug)
                    cout << "Processing " << UnitTypeToName(unit.type) << " " << unit.health << "+" << unit.shield << " "
                         << "e=" << unit.energy << endl;

                if (unit.type == UNIT_TYPEID::TERRAN_MEDIVAC) {
                    if (unit.energy > 0) {
                        // Pick a random target
                        int offset = rand() % g1.size();
                        const float HEALING_PER_NORMAL_SPEED_SECOND = 12.6 / 1.4f;
                        for (int j = 0; j < g1.size(); j++) {
                            int index = (j + offset) % g1.size();
                            auto& other = *g1[index];
                            if (index != i && !hasBeenHealed[index] && other.health > 0 && other.health < other.health_max && contains(unitTypes[(int)other.type].attributes, Attribute::Biological)) {
                                other.modifyHealth(HEALING_PER_NORMAL_SPEED_SECOND * dt);
                                hasBeenHealed[index] = true;
                                break;
                            }
                        }
                    }
                    continue;
                }

                if (unit.type == UNIT_TYPEID::PROTOSS_SHIELDBATTERY) {
                    if (unit.energy > 0) {
                        // Pick a random target
                        int offset = rand() % g1.size();
                        const float SHIELDS_PER_NORMAL_SPEED_SECOND = 50.4 / 1.4f;
                        const float ENERGY_USE_PER_SHIELD = 1.0f / 3.0f;
                        for (int j = 0; j < g1.size(); j++) {
                            int index = (j + offset) % g1.size();
                            auto& other = *g1[index];
                            if (index != i && !hasBeenHealed[index] && other.health > 0 && other.shield < other.shield_max) {
                                float delta = min(min(other.shield_max - other.shield, SHIELDS_PER_NORMAL_SPEED_SECOND * dt), unit.energy / ENERGY_USE_PER_SHIELD);
                                assert(delta >= 0);
                                other.shield += delta;
                                assert(other.shield >= 0 && other.shield <= other.shield_max + 0.001f);
                                unit.energy -= delta * ENERGY_USE_PER_SHIELD;
                                hasBeenHealed[index] = true;
                                // cout << "Restoring " << delta << " shields. New health is " << other.health << "+" << other.shield << endl;
                                break;
                            }
                        }
                    }
                    continue;
                }

                if (airDPS == 0 && groundDPS == 0)
                    continue;

                bool isUnitMelee = isMelee(unit.type);
                if (isUnitMelee && numMeleeUnitsUsed >= surround.maxMeleeAttackers)
                    continue;

                if (group == 1) {
                    // Attacker (move until we are within range of enemy)
                    float distanceToEnemy = maxRangeDefender;
                    if (isUnitMelee) {
                        // Make later melee units take longer to reach the enemy.
                        distanceToEnemy += maxExtraMeleeDistance * (i / (float)g1.size());
                    }
                    float timeToReachEnemy = unitTypeData.movement_speed > 0 ? max(0.0f, distanceToEnemy - attackRange(unit.type)) / unitTypeData.movement_speed : 100000;
                    if (time < timeToReachEnemy) {
                        changed = true;
                        continue;
                    }
                } else {
                    // Defender (stay put until attacker comes within range)
                    float timeToReachEnemy = fastestAttackerSpeed > 0 ? (maxRangeDefender - attackRange(unit.type)) / fastestAttackerSpeed : 100000;
                    if (time < timeToReachEnemy) {
                        changed = true;
                        continue;
                    }
                }

                // if (debug) cout << "DPS: " << groundDPS << " " << airDPS << endl;

                CombatUnit* bestTarget = nullptr;
                int bestTargetIndex = -1;
                float bestScore = 0;
                const WeaponInfo* bestWeapon = nullptr;

                for (int j = 0; j < g2.size(); j++) {
                    auto& other = *g2[j];
                    if (other.health == 0)
                        continue;

                    if (((canBeAttackedByAirWeapons(other.type) && airDPS > 0) || (!other.is_flying && groundDPS > 0))) {
                        auto& info = combatInfo[(int)unit.type];
                        auto& otherData = unitTypes[(int)other.type];

                        float airDPS2 = info.airWeapon.getDPS(other.type);
                        float groundDPS2 = info.groundWeapon.getDPS(other.type);

                        auto dps = max(groundDPS2, airDPS2);
                        float score = dps * targetScore(other, group == 0 ? hasGround1 : hasGround2, group == 0 ? hasAir1 : hasAir2);
                        if (group == 1 && badMicro)
                            score = -score;

                        if (isUnitMelee) {
                            if (meleeUnitAttackCount[j] >= surround.maxAttackersPerDefender) {
                                // Can't attack this unit, too many melee units are attacking it already
                                continue;
                            }

                            // Assume the enemy has positioned its units in the most annoying way possible
                            // so that we have to attack the ones we really do not want to attack first.
                            // TODO: Add pessimistic option?
                            if (!badMicro)
                                score = -score;

                            // Melee units should attack other melee units first, and have no splash against ranged units (assume reasonable micro)
                            if (isMelee(other.type))
                                score += 1000;
                            // Check for kiting, hard to attack units with a higher movement speed
                            else if (unitTypeData.movement_speed < 1.05f * otherData.movement_speed)
                                score -= 500;
                        }
                        // if (debug) cout << "Potential target: " << UnitTypeToName(other.type) << " score: " << score << endl;

                        if (bestTarget == nullptr || score > bestScore) {
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
                    auto dps = bestWeapon->getDPS(other.type) * min(1.0f, remainingSplash);
                    other.modifyHealth(-dps * dt);
                    // cout << "Picked target " << (-dps*dt) << " " << other.health << endl;

                    remainingSplash -= 1;
                    // Splash is only applied when a melee unit attacks another melee unit
                    // or a non-melee unit does splash damage.
                    // If it is a melee unit, then splash is only applied to other melee units.
                    // TODO: Better rule: units only apply splash to other units that have a shorter range than themselves, or this unit has a higher movement speed than the other one
                    if (remainingSplash > 0.001f && (!isUnitMelee || isMelee(other.type))) {
                        // Apply remaining splash to other random melee units
                        int offset = rand() % g2.size();
                        for (int j = 0; j < g2.size() && remainingSplash > 0.001f; j++) {
                            auto* splashOther = g2[(j + offset) % g2.size()];
                            if (splashOther != bestTarget && splashOther->health > 0 && (!isUnitMelee || isMelee(splashOther->type))) {
                                auto dps = bestWeapon->getDPS(splashOther->type) * min(1.0f, remainingSplash);
                                if (dps > 0) {
                                    splashOther->modifyHealth(-dps * dt);
                                    remainingSplash -= 1.0f;
                                }
                            }
                        }
                    }
                }
            }

            if (debug)
                cout << "Meleee attackers used: " << numMeleeUnitsUsed << endl;
        }

        time += dt;
    }

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

vector<UNIT_TYPEID> availableUnitTypes = {
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

vector<UNIT_TYPEID> availableUnitTypesProtoss = {
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ARCHON,
    UNIT_TYPEID::PROTOSS_CARRIER,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    UNIT_TYPEID::PROTOSS_DARKTEMPLAR,
    UNIT_TYPEID::PROTOSS_DISRUPTOR,
    UNIT_TYPEID::PROTOSS_HIGHTEMPLAR,
    UNIT_TYPEID::PROTOSS_IMMORTAL,
    UNIT_TYPEID::PROTOSS_MOTHERSHIP,
    UNIT_TYPEID::PROTOSS_OBSERVER,
    UNIT_TYPEID::PROTOSS_ORACLE,
    UNIT_TYPEID::PROTOSS_PHOENIX,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_SENTRY,
    UNIT_TYPEID::PROTOSS_STALKER,
    UNIT_TYPEID::PROTOSS_TEMPEST,
    UNIT_TYPEID::PROTOSS_VOIDRAY,
    UNIT_TYPEID::PROTOSS_WARPPRISM,
    UNIT_TYPEID::PROTOSS_ZEALOT,
};

vector<UNIT_TYPEID> availableUnitTypesZerg = {
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

float mineralScore(const CombatState& initialState, const CombatResult& combatResult, int player) {
    assert(initialState.units.size() == combatResult.state.units.size());
    float totalScore = 0;
    for (int i = 0; i < initialState.units.size(); i++) {
        auto& unit1 = initialState.units[i];
        auto& unit2 = combatResult.state.units[i];
        assert(unit1.type == unit2.type);

        float healthDiff = (unit2.health - unit1.health) + (unit2.shield - unit1.shield);
        float damageTakenFraction = -healthDiff / (unit1.health_max + unit1.shield_max);
        auto& unitTypeData = unitTypes[(int)unit1.type];
        float score = (unitTypeData.mineral_cost + 2 * unitTypeData.vespene_cost) * (unit1.owner == player ? -(1 + damageTakenFraction) : (-100 * (1 - damageTakenFraction) + 1 + damageTakenFraction));
        totalScore += score;
    }

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

void createState(const CombatState& state) {
    for (auto& u : state.units) {
        agent.Debug()->DebugCreateUnit(u.type, Point2D(40 + 20 * (u.owner == 1 ? -1 : 1), 20), u.owner);
    }
}

void findBestComposition(const CombatPredictor& predictor, const CombatState& opponent) {
    // Find composition which has
    // 1. lowest cost
    // 2. takes the least amount of damage
    // 3. kills the enemy in the shortest amount of time
    //
    // possibly:
    // minimize sum_allies unit.cost*(1 + unit.damage_taken_fraction) - sum_enemies unit.cost*unit.damage_taken_fraction = expected delta in mineral cost

    Stopwatch watch;
    CombatState state = opponent;
    int ourUnitsStartIndex = state.units.size();
    float bestScore = -100000000;

    // Starting guess
    do {
        // Add 3 marines at a time until we win
        for (int i = 0; i < 3; i++) {
            state.units.push_back(makeUnit(2, UNIT_TYPEID::TERRAN_MARINE));
        }
    } while (predictor.predict_engage(state).state.owner_with_best_outcome() == 1);

    for (int i = 0; i < 1000; i++) {
        CombatState newState = state;
        // int action = rand() % 3;
        int ourUnitCount = newState.units.size() - ourUnitsStartIndex;
        int toRemove = ourUnitCount > 0 ? min(ourUnitCount, rand() % ourUnitCount) : 0;
        int toAdd = rand() % max(5, ourUnitCount);
        if (toRemove == 0 && toAdd == 0)
            toAdd = 1;

        for (int j = 0; j < toRemove; j++) {
            int ourUnitCount2 = newState.units.size() - ourUnitsStartIndex;
            // Overwrite this unit with the last unit and then delete the last unit.
            // Effectively deletes this unit because the order is not important.
            newState.units[(rand() % ourUnitCount2) + ourUnitsStartIndex] = newState.units[newState.units.size() - 1];
            newState.units.pop_back();
        }

        // Add unit
        for (int j = 0; j < toAdd; j++) {
            auto unitType = availableUnitTypes[rand() % availableUnitTypes.size()];
            newState.units.push_back(makeUnit(2, unitType));
        }

        /*if (action == 1) {
			// Remove unit
			if (newState.units.size() > ourUnitsStartIndex) {
				for (int j = 0; j < stepSize && newState.units.size() > ourUnitsStartIndex; j++) {
					int ourUnitCount = newState.units.size() - ourUnitsStartIndex;
					// Overwrite this unit with the last unit and then delete the last unit.
					// Effectively deletes this unit because the order is not important.
					newState.units[(rand() % ourUnitCount) + ourUnitsStartIndex] = newState.units[newState.units.size()-1];
					newState.units.pop_back();
				}
			} else {
				// Add unit instead
				action = 0;
			}
		}

		if (action == 2) {
			// Replace unit
			if (newState.units.size() > ourUnitsStartIndex) {
				for (int j = 0; j < stepSize; j++) {
					int ourUnitCount = newState.units.size() - ourUnitsStartIndex;
					// Overwrite this unit
					auto unitType = availableUnitTypes[rand() % availableUnitTypes.size()];
					newState.units[(rand() % ourUnitCount) + ourUnitsStartIndex] = CombatUnit(2, unitType, maxHealth(unitType), isFlying(unitType));
				}
			} else {
				// Add unit instead
				action = 0;
			}
		}

		if (action == 0) {
			// Add unit
			for (int j = 0; j < stepSize; j++) {
				auto unitType = availableUnitTypes[rand() % availableUnitTypes.size()];
				newState.units.push_back(CombatUnit(2, unitType, maxHealth(unitType), isFlying(unitType)));
			}

			/CombatState newState2 = newState;
			for (int i = 0; i < 20; i++) {
				auto unitType = availableUnitTypes[rand() % availableUnitTypes.size()];
				newState2.units.push_back(CombatUnit(2, unitType, maxHealth(unitType), isFlying(unitType)));

				CombatResult newResult2 = predictor.predict_engage(newState2);
				float newScore2 = mineralScore(newState2, newResult2, 2);
				cout << " " << newScore2;
			}
			cout << endl;*
		}*/

        CombatResult newResult1 = predictor.predict_engage(newState);
        CombatResult newResult2 = predictor.predict_engage(newState);
        CombatResult newResult3 = predictor.predict_engage(newState);
        float score1 = mineralScore(newState, newResult1, 2);
        float score2 = mineralScore(newState, newResult2, 2);
        float score3 = mineralScore(newState, newResult3, 2);

        float newScore = (score1 + score2 + score3) / 3.0f;
        cout << (int)score1 << " " << (int)score2 << " " << (int)score3 << endl;

        float temperature = 500.0f / (i + 1);
        float randValue = (rand() % 10000) / 10000.0f;
        if (randValue < exp((newScore - bestScore) / temperature)) {
            state = newState;
            if (newScore > bestScore) {
                cout << "New record " << newScore << endl;
                for (auto& u : state.units) {
                    if (u.owner == 2) {
                        cout << '\t' << UnitTypeToName(u.type) << endl;
                    }
                }
            } else {
                cout << "New score " << newScore << endl;
                for (auto& u : state.units) {
                    if (u.owner == 2) {
                        cout << '\t' << UnitTypeToName(u.type) << endl;
                    }
                }
            }
            bestScore = newScore;
        } else {
            cout << "Discarded score " << newScore << endl;
        }
        // Actions: add unit, replace unit, remove unit?
    }

    watch.stop();
    cout << "Duration " << watch.millis() << " ms" << endl;
    vector<int> mineralCosts(3);
    vector<int> vespeneCosts(3);
    for (auto u : state.units) {
        mineralCosts[u.owner] += unitTypes[(int)u.type].mineral_cost;
        vespeneCosts[u.owner] += unitTypes[(int)u.type].vespene_cost;
    }

    cout << "Team 1 costs: " << mineralCosts[1] << "+" << vespeneCosts[1] << endl;
    cout << "Team 2 costs: " << mineralCosts[2] << "+" << vespeneCosts[2] << endl;
    createState(state);

    CombatResult newResult = predictor.predict_engage(state, true);
}

struct Gene {
   private:
    // Indices are into the availableUnitTypes list
    vector<int> unitCounts;

   public:
    void addToState(CombatState& state, const vector<UNIT_TYPEID>& availableUnitTypes, int owner) const {
        assert(unitCounts.size() == availableUnitTypes.size());
        for (int i = 0; i < unitCounts.size(); i++) {
            for (int j = 0; j < unitCounts[i]; j++) {
                state.units.push_back(makeUnit(owner, availableUnitTypes[i]));
            }
        }
    }

    void mutate(float amount, default_random_engine& seed) {
        bernoulli_distribution shouldMutate(amount);
        for (int i = 0; i < unitCounts.size(); i++) {
            if (shouldMutate(seed)) {
                exponential_distribution<float> dist(1.0f / max(1, unitCounts[i]));
                unitCounts[i] = (int)round(dist(seed));
            }
        }
    }

    Gene()
        : Gene(0) {}

    Gene(int units)
        : unitCounts(units) {
    }

    void scale(float scale) {
        int diff = 0;
        for (int i = 0; i < unitCounts.size(); i++) {
            if (unitCounts[i] > 100)
                continue;
            auto o = unitCounts[i];
            unitCounts[i] = (int)round(unitCounts[i] * scale);
            diff += unitCounts[i] - o;
        }
        if (diff == 0) {
            unitCounts[rand() % unitCounts.size()] += 1;
        }
    }

    static Gene crossover(const Gene& parent1, const Gene& parent2, default_random_engine& seed) {
        assert(parent1.unitCounts.size() == parent2.unitCounts.size());
        bernoulli_distribution dist(0.5);
        Gene gene(parent1.unitCounts.size());
        for (int i = 0; i < gene.unitCounts.size(); i++) {
            gene.unitCounts[i] = dist(seed) ? parent1.unitCounts[i] : parent2.unitCounts[i];
        }
        return gene;
    }

    Gene(const vector<UNIT_TYPEID>& availableUnitTypes, int meanTotalCount, default_random_engine& seed)
        : unitCounts(availableUnitTypes.size()) {
        float mean = meanTotalCount / availableUnitTypes.size();
        exponential_distribution<float> dist(1.0f / mean);
        for (int i = 0; i < unitCounts.size(); i++) {
            unitCounts[i] = (int)round(dist(seed));
        }
    }
};

float calculateFitness(const CombatPredictor& predictor, const CombatState& opponent, const vector<UNIT_TYPEID>& availableUnitTypes, Gene& gene) {
    CombatState state = opponent;
    gene.addToState(state, availableUnitTypes, 2);
    int its = 0;
    while (predictor.predict_engage(state, false, false).state.owner_with_best_outcome() == 1) {
        its++;
        if (its > 4)
            break;
        gene.scale(1.5f);
        state = opponent;
        gene.addToState(state, availableUnitTypes, 2);
    }
    return mineralScore(state, predictor.predict_engage(state, false, false), 2);  // + mineralScore(state, predictor.predict_engage(state, false, true), 2)) * 0.5f;
}

void findBestCompositionGenetic(const CombatPredictor& predictor, const vector<UNIT_TYPEID>& availableUnitTypes, const CombatState& opponent) {
    Stopwatch watch;

    const int POOL_SIZE = 20;
    const float mutationRate = 0.1f;
    vector<Gene> generation(20);
    default_random_engine rnd(time(0));
    for (int i = 0; i < POOL_SIZE; i++) {
        generation[i] = Gene(availableUnitTypes, 10, rnd);
    }
    for (int i = 0; i < 100; i++) {
        vector<float> fitness(generation.size());
        vector<int> indices(generation.size());
        for (int j = 0; j < generation.size(); j++) {
            indices[j] = j;
            fitness[j] = calculateFitness(predictor, opponent, availableUnitTypes, generation[j]);
        }

        sortByValueDescending<int, float>(indices, [=](int index) { return fitness[index]; });
        for (int j = 0; j < indices.size(); j++) {
            cout << " " << fitness[indices[j]];
        }
        cout << endl;
        vector<Gene> nextGeneration;
        // Add the N best performing genes
        for (int j = 0; j < 5; j++) {
            nextGeneration.push_back(generation[indices[j]]);
        }
        // Add a random one as well
        nextGeneration.push_back(generation[uniform_int_distribution<int>(0, indices.size() - 1)(rnd)]);

        uniform_int_distribution<int> randomParentIndex(0, nextGeneration.size() - 1);
        while (nextGeneration.size() < POOL_SIZE) {
            nextGeneration.push_back(Gene::crossover(generation[randomParentIndex(rnd)], generation[randomParentIndex(rnd)], rnd));
        }

        // Note: do not mutate the first gene
        for (int i = 1; i < nextGeneration.size(); i++) {
            nextGeneration[i].mutate(mutationRate, rnd);
        }

        swap(generation, nextGeneration);

        cout << "Best fitness " << fitness[indices[0]] << endl;
    }

    CombatState testState = opponent;
    generation[0].addToState(testState, availableUnitTypes, 2);
    watch.stop();
    cout << "Duration " << watch.millis() << " ms" << endl;
    vector<int> mineralCosts(3);
    vector<int> vespeneCosts(3);
    for (auto u : testState.units) {
        mineralCosts[u.owner] += unitTypes[(int)u.type].mineral_cost;
        vespeneCosts[u.owner] += unitTypes[(int)u.type].vespene_cost;
    }

    cout << "Team 1 costs: " << mineralCosts[1] << "+" << vespeneCosts[1] << endl;
    cout << "Team 2 costs: " << mineralCosts[2] << "+" << vespeneCosts[2] << endl;
    createState(testState);
    CombatRecording recording;
    CombatResult newResult = predictor.predict_engage(testState, true, false, &recording);
    recording.writeCSV("recording2.csv");

    CombatRecording recording2;
    CombatResult newResult2 = predictor.predict_engage(testState, false, true, &recording2);
    recording2.writeCSV("recording3.csv");
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

void CombatPredictor::unitTest() const {
    unitTestSurround();

    /*
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
		CombatUnit(1, UNIT_TYPEID::ZERG_SPORECRAWLER, 100, false),
		CombatUnit(1, UNIT_TYPEID::ZERG_SPORECRAWLER, 100, false),
		CombatUnit(1, UNIT_TYPEID::ZERG_SPORECRAWLER, 100, false),
		CombatUnit(2, UNIT_TYPEID::TERRAN_REAPER, 65, false),
		CombatUnit(2, UNIT_TYPEID::TERRAN_REAPER, 65, false),
		CombatUnit(2, UNIT_TYPEID::TERRAN_REAPER, 65, false),
	}}) == 2);

	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_BROODLORD, 225, true),
	}}) == 1);

	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(1, UNIT_TYPEID::TERRAN_LIBERATOR, 180, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
		CombatUnit(2, UNIT_TYPEID::ZERG_CORRUPTOR, 225, true),
	}}) == 1);

	assert(testCombat(*this, {{
		CombatUnit(1, UNIT_TYPEID::TERRAN_CYCLONE, 180, true),
		CombatUnit(2, UNIT_TYPEID::PROTOSS_IMMORTAL, 200, true),
	}}) == 2);

	assert(testCombat(*this, {{
		makeUnit(1, UNIT_TYPEID::TERRAN_BATTLECRUISER),
		makeUnit(2, UNIT_TYPEID::TERRAN_THOR),
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
	}}) == 1);*/

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

		findBestCompositionGenetic(*this, availableUnitTypesZerg, { {
			makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
            makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
            makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
			makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
            makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
            makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
			makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
            makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
            makeUnit(1, UNIT_TYPEID::PROTOSS_TEMPEST),
		} });
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
