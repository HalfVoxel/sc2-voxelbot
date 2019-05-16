import numpy as np
from typing import List
from generated_mappings import terranUnits, zergUnits, protossUnits, Unit, terranUpgrades, zergUpgrades, protossUpgrades

allUnits = terranUnits + zergUnits + protossUnits
ignoredUnits = {
    1942,  # ParasiticBombRelayDummy
    830,  # TERRAN_KD8CHARGE
    824,  # ZERG_PARASITICBOMBDUMMY
    11,  # TERRAN_POINTDEFENSEDRONE
}

upgrades = [
    # ("INVALID", 0),
    ("CARRIERLAUNCHSPEEDUPGRADE", 1),
    ("GLIALRECONSTITUTION", 2),
    ("TUNNELINGCLAWS", 3),
    ("CHITINOUSPLATING", 4),
    ("HISECAUTOTRACKING", 5),
    ("TERRANBUILDINGARMOR", 6),
    ("TERRANINFANTRYWEAPONSLEVEL1", 7),
    ("TERRANINFANTRYWEAPONSLEVEL2", 8),
    ("TERRANINFANTRYWEAPONSLEVEL3", 9),
    ("NEOSTEELFRAME", 10),
    ("TERRANINFANTRYARMORSLEVEL1", 11),
    ("TERRANINFANTRYARMORSLEVEL2", 12),
    ("TERRANINFANTRYARMORSLEVEL3", 13),
    ("STIMPACK", 15),
    ("SHIELDWALL", 16),
    ("PUNISHERGRENADES", 17),
    ("HIGHCAPACITYBARRELS", 19),
    ("BANSHEECLOAK", 20),
    ("RAVENCORVIDREACTOR", 22),
    ("PERSONALCLOAKING", 25),
    ("TERRANVEHICLEWEAPONSLEVEL1", 30),
    ("TERRANVEHICLEWEAPONSLEVEL2", 31),
    ("TERRANVEHICLEWEAPONSLEVEL3", 32),
    ("TERRANSHIPWEAPONSLEVEL1", 36),
    ("TERRANSHIPWEAPONSLEVEL2", 37),
    ("TERRANSHIPWEAPONSLEVEL3", 38),
    ("PROTOSSGROUNDWEAPONSLEVEL1", 39),
    ("PROTOSSGROUNDWEAPONSLEVEL2", 40),
    ("PROTOSSGROUNDWEAPONSLEVEL3", 41),
    ("PROTOSSGROUNDARMORSLEVEL1", 42),
    ("PROTOSSGROUNDARMORSLEVEL2", 43),
    ("PROTOSSGROUNDARMORSLEVEL3", 44),
    ("PROTOSSSHIELDSLEVEL1", 45),
    ("PROTOSSSHIELDSLEVEL2", 46),
    ("PROTOSSSHIELDSLEVEL3", 47),
    ("OBSERVERGRAVITICBOOSTER", 48),
    ("GRAVITICDRIVE", 49),
    ("EXTENDEDTHERMALLANCE", 50),
    ("PSISTORMTECH", 52),
    ("ZERGMELEEWEAPONSLEVEL1", 53),
    ("ZERGMELEEWEAPONSLEVEL2", 54),
    ("ZERGMELEEWEAPONSLEVEL3", 55),
    ("ZERGGROUNDARMORSLEVEL1", 56),
    ("ZERGGROUNDARMORSLEVEL2", 57),
    ("ZERGGROUNDARMORSLEVEL3", 58),
    ("ZERGMISSILEWEAPONSLEVEL1", 59),
    ("ZERGMISSILEWEAPONSLEVEL2", 60),
    ("ZERGMISSILEWEAPONSLEVEL3", 61),
    ("OVERLORDSPEED", 62),
    ("BURROW", 64),
    ("ZERGLINGATTACKSPEED", 65),
    ("ZERGLINGMOVEMENTSPEED", 66),
    ("ZERGFLYERWEAPONSLEVEL1", 68),
    ("ZERGFLYERWEAPONSLEVEL2", 69),
    ("ZERGFLYERWEAPONSLEVEL3", 70),
    ("ZERGFLYERARMORSLEVEL1", 71),
    ("ZERGFLYERARMORSLEVEL2", 72),
    ("ZERGFLYERARMORSLEVEL3", 73),
    ("INFESTORENERGYUPGRADE", 74),
    ("CENTRIFICALHOOKS", 75),
    ("BATTLECRUISERENABLESPECIALIZATIONS", 76),
    ("PROTOSSAIRWEAPONSLEVEL1", 78),
    ("PROTOSSAIRWEAPONSLEVEL2", 79),
    ("PROTOSSAIRWEAPONSLEVEL3", 80),
    ("PROTOSSAIRARMORSLEVEL1", 81),
    ("PROTOSSAIRARMORSLEVEL2", 82),
    ("PROTOSSAIRARMORSLEVEL3", 83),
    ("WARPGATERESEARCH", 84),
    ("CHARGE", 86),
    ("BLINKTECH", 87),
    ("PHOENIXRANGEUPGRADE", 99),
    ("NEURALPARASITE", 101),
    ("TERRANVEHICLEANDSHIPARMORSLEVEL1", 116),
    ("TERRANVEHICLEANDSHIPARMORSLEVEL2", 117),
    ("TERRANVEHICLEANDSHIPARMORSLEVEL3", 118),
    ("DRILLCLAWS", 122),
    ("ADEPTPIERCINGATTACK", 130),
    ("MAGFIELDLAUNCHERS", 133),
    ("EVOLVEGROOVEDSPINES", 134),
    ("EVOLVEMUSCULARAUGMENTS", 135),
    ("BANSHEESPEED", 136),
    ("RAVENRECALIBRATEDEXPLOSIVES", 138),
    ("MEDIVACINCREASESPEEDBOOST", 139),
    ("LIBERATORAGRANGEUPGRADE", 140),
    ("DARKTEMPLARBLINKUPGRADE", 141),
    ("SMARTSERVOS", 289),
    ("RAPIDFIRELAUNCHERS", 291),
    ("ENHANCEDMUNITIONS", 292),
]

nonMilitaryUnits = {"PROTOSS_PROBE", "TERRAN_SCV", "ZERG_DRONE", "ZERG_OVERLORD", "ZERG_OVERSEER"}
workerUnits = {"PROTOSS_PROBE", "TERRAN_SCV", "ZERG_DRONE"}


class UnitLookup:
    def __init__(self, units: List[Unit]):
        self.units = units
        self.num_units = len(units)
        self.military_units_mask = np.zeros(self.num_units)
        self.all_unit_indices = list(range(0, self.num_units))
        self.unit_index_map = {}
        self.reverse_unit_index_map = {}
        self.upgrade_index_map = {}
        self.num_upgrades = len(upgrades)
        self.unit_index_to_race = {}
        self.workerUnitTypes = set()
        for i, u in enumerate(upgrades):
            self.upgrade_index_map[u[1]] = i

        for i, unit in enumerate(units):
            self.military_units_mask[i] = 1 if unit.is_army else 0
            self.reverse_unit_index_map[i] = unit.type_ids[0]
            self.unit_index_to_race[i] = unit.race
            for index in unit.type_ids:
                self.unit_index_map[index] = i
                if unit.name in workerUnits:
                    self.workerUnitTypes.add(index)

        self.military_units_mask_indices = np.where(self.military_units_mask)[0]
        self.non_military_units_mask_indices = np.where(self.military_units_mask == False)[0]

        self.movable_units_mask = self.military_units_mask.copy()
        for i, u in enumerate(units):
            if u[0] in nonMilitaryUnits:
                self.movable_units_mask[i] = 1
    
    def findByName(self, name):
        for u in self.units:
            if u.name == name:
                return u
        
        return None

    def __contains__(self, index):
        return index in self.unit_index_map

    def __getitem__(self, index):
        return self.units[self.unit_index_map[index]]
    
    def __len__(self):
        return len(self.units)
