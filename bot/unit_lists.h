#pragma once
#include "sc2api/sc2_interfaces.h"
#include "BuildOptimizerGenetic.h"

/*struct AvailableUnitType {
    sc2::UNIT_TYPEID type;
    bool allowInBuildOrderWhenNotExplicitlyRequested;

    AvailableUnitType(sc2::UNIT_TYPEID type, bool allowInBuildOrderWhenNotExplicitlyRequested = false) : type(type), allowInBuildOrderWhenNotExplicitlyRequested(allowInBuildOrderWhenNotExplicitlyRequested), allowChronoBoost(allowChronoBoost) {}
    AvailableUnitType(sc2::UPGRADE_ID type, bool allowInBuildOrderWhenNotExplicitlyRequested = false) : type((sc2::UNIT_TYPEID)((int)type + UPGRADE_ID_OFFSET)), allowInBuildOrderWhenNotExplicitlyRequested(allowInBuildOrderWhenNotExplicitlyRequested), allowChronoBoost(allowChronoBoost) {}
};*/

struct AvailableUnitTypes {
    std::vector<BuildOrderItem> index2item;
    std::vector<int> type2index;
    std::vector<int> upgrade2index;
    std::map<int, int> arbitraryType2index;
  private:
    std::vector<sc2::UNIT_TYPEID> unitTypes;
  public:

    size_t size() const {
        return index2item.size();
    }

    const std::vector<sc2::UNIT_TYPEID>& getUnitTypes() const {
        return unitTypes;
    }


    AvailableUnitTypes(std::initializer_list<BuildOrderItem> types);

    int getIndex (sc2::UPGRADE_ID upgrade) const {
        assert((int)upgrade < (int)upgrade2index.size());
        auto res = upgrade2index[(int)upgrade];
        assert(res != -1);
        return res;
    }

    int getIndex (sc2::UNIT_TYPEID unit) const {
        assert((int)unit < (int)type2index.size());
        auto res = type2index[(int)unit];
        assert(res != -1);
        return res;
    }

    int getIndexMaybe (sc2::UPGRADE_ID unit) const {
        auto it = arbitraryType2index.find((int)unit);
        if (it != arbitraryType2index.end()) {
            return it->second;
        }
        return -1;
    }

    int getIndexMaybe (sc2::UNIT_TYPEID unit) const {
        // Fast path
        if ((int)unit < (int)type2index.size()) {
            return type2index[(int)unit];
        }
        // Slow path
        auto it = arbitraryType2index.find((int)unit);
        if (it != arbitraryType2index.end()) {
            return it->second;
        }
        return -1;
    }

    // Note: does not work for upgrades
    bool contains (sc2::UNIT_TYPEID unit) const {
        return getIndexMaybe(unit) != -1;
    }

    // Reasonable maximum for the unit/upgrade in an army composition.
    // This is primarily used to prevent upgrade counts higher than possible
    int armyCompositionMaximum(int index) const {
        assert(index < (int)index2item.size());
        if (!index2item[index].isUnitType()) {
            // This is an upgrade
            switch(index2item[index].upgradeID()) {
            case sc2::UPGRADE_ID::TERRANINFANTRYWEAPONSLEVEL1:
            case sc2::UPGRADE_ID::TERRANINFANTRYARMORSLEVEL1:
            case sc2::UPGRADE_ID::TERRANVEHICLEWEAPONSLEVEL1:
            case sc2::UPGRADE_ID::TERRANSHIPWEAPONSLEVEL1:
            case sc2::UPGRADE_ID::PROTOSSGROUNDWEAPONSLEVEL1:
            case sc2::UPGRADE_ID::PROTOSSGROUNDARMORSLEVEL1:
            case sc2::UPGRADE_ID::PROTOSSSHIELDSLEVEL1:
            case sc2::UPGRADE_ID::ZERGMELEEWEAPONSLEVEL1:
            case sc2::UPGRADE_ID::ZERGGROUNDARMORSLEVEL1:
            case sc2::UPGRADE_ID::ZERGMISSILEWEAPONSLEVEL1:
            case sc2::UPGRADE_ID::ZERGFLYERWEAPONSLEVEL1:
            case sc2::UPGRADE_ID::ZERGFLYERARMORSLEVEL1:
            case sc2::UPGRADE_ID::PROTOSSAIRWEAPONSLEVEL1:
            case sc2::UPGRADE_ID::PROTOSSAIRARMORSLEVEL1:
            case sc2::UPGRADE_ID::TERRANVEHICLEANDSHIPARMORSLEVEL1:
                // Enumerable upgrade, there are 3 upgrade levels
                return 3;
            default:
                return 1;
            }
        }

        // Essentially infinity
        return 1000;
    }

    sc2::UNIT_TYPEID getUnitType(int index) const {
        assert(index < (int)index2item.size());
        if (!index2item[index].isUnitType()) return sc2::UNIT_TYPEID::INVALID;
        return index2item[index].typeID();
    }

    bool canBeChronoBoosted (int index) const;

    BuildOrderItem getBuildOrderItem (int index) const {
        assert(index < (int)index2item.size());
        return index2item[index];
    }

    BuildOrderItem getBuildOrderItem (const GeneUnitType item) const {
        auto itm = index2item[item.type];
        itm.chronoBoosted = item.chronoBoosted;
        return itm;
    }

    GeneUnitType getGeneItem (const BuildOrderItem item) const {
        return GeneUnitType(arbitraryType2index.at((int)item.rawType()), item.chronoBoosted);
    }

    friend int remapAvailableUnitIndex(int index, const AvailableUnitTypes& from, const AvailableUnitTypes& to) {
        assert(index < (int)from.index2item.size());
        return to.arbitraryType2index.at((int)from.index2item[index].rawType());
    }
};


enum class UnitCategory {
    Economic,
    ArmyCompositionOptions,
    BuildOrderOptions,
};

const AvailableUnitTypes& getAvailableUnitsForRace (sc2::Race race);
const AvailableUnitTypes& getAvailableUnitsForRace (sc2::Race race, UnitCategory category);

extern const AvailableUnitTypes availableUpgrades;