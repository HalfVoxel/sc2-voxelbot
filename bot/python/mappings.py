import numpy as np

terranUnits = [
    ("TERRAN_TECHLAB", False, [5], "terran"),
    ("TERRAN_REACTOR", False, [6], "terran"),
    # ("TERRAN_POINTDEFENSEDRONE", False, [11], "terran"),
    ("TERRAN_COMMANDCENTER", False, [18, 36], "terran"),
    ("TERRAN_SUPPLYDEPOT", False, [19, 47], "terran"),
    ("TERRAN_REFINERY", False, [20], "terran"),
    ("TERRAN_BARRACKS", False, [21, 46], "terran"),
    ("TERRAN_ENGINEERINGBAY", False, [22], "terran"),
    ("TERRAN_MISSILETURRET", False, [23], "terran"),
    ("TERRAN_BUNKER", False, [24], "terran"),
    ("TERRAN_SENSORTOWER", False, [25], "terran"),
    ("TERRAN_GHOSTACADEMY", False, [26], "terran"),
    ("TERRAN_FACTORY", False, [27, 43], "terran"),
    ("TERRAN_STARPORT", False, [28, 44], "terran"),
    ("TERRAN_ARMORY", False, [29], "terran"),
    ("TERRAN_FUSIONCORE", False, [30], "terran"),
    ("TERRAN_AUTOTURRET", False, [31], "terran"),
    ("TERRAN_SIEGETANK", True, [33, 32], "terran"),
    ("TERRAN_VIKINGFIGHTER", True, [35, 34], "terran"),
    ("TERRAN_BARRACKSTECHLAB", False, [37], "terran"),
    ("TERRAN_BARRACKSREACTOR", False, [38], "terran"),
    ("TERRAN_FACTORYTECHLAB", False, [39], "terran"),
    ("TERRAN_FACTORYREACTOR", False, [40], "terran"),
    ("TERRAN_STARPORTTECHLAB", False, [41], "terran"),
    ("TERRAN_STARPORTREACTOR", False, [42], "terran"),
    ("TERRAN_SCV", False, [45], "terran"),
    ("TERRAN_MARINE", True, [48], "terran"),
    ("TERRAN_REAPER", True, [49], "terran"),
    ("TERRAN_GHOST", True, [50], "terran"),
    ("TERRAN_MARAUDER", True, [51], "terran"),
    ("TERRAN_THOR", True, [52, 691], "terran"),
    ("TERRAN_HELLION", True, [53], "terran"),
    ("TERRAN_MEDIVAC", True, [54], "terran"),
    ("TERRAN_BANSHEE", True, [55], "terran"),
    ("TERRAN_RAVEN", True, [56], "terran"),
    ("TERRAN_BATTLECRUISER", True, [57], "terran"),
    ("TERRAN_NUKE", False, [58], "terran"),
    ("TERRAN_PLANETARYFORTRESS", False, [130], "terran"),
    ("TERRAN_ORBITALCOMMAND", False, [132, 134], "terran"),
    ("TERRAN_MULE", False, [268], "terran"),
    ("TERRAN_HELLIONTANK", True, [484], "terran"),
    ("TERRAN_WIDOWMINE", True, [498, 500], "terran"),
    ("TERRAN_LIBERATOR", True, [689, 734], "terran"),
    ("TERRAN_CYCLONE", True, [692], "terran"),
    # ("TERRAN_KD8CHARGE", False, [830], "terran"),
]

zergUnits = [
    ("ZERG_INFESTORTERRAN", True, [7], "zerg"),
    ("ZERG_BANELINGCOCOON", True, [8], "zerg"),
    ("ZERG_BANELING", True, [9, 115], "zerg"),
    ("ZERG_CHANGELING", True, [12, 13, 14, 15, 16, 17], "zerg"),
    ("ZERG_HATCHERY", False, [86], "zerg"),
    ("ZERG_CREEPTUMOR", False, [87, 137, 138], "zerg"),
    ("ZERG_EXTRACTOR", False, [88], "zerg"),
    ("ZERG_SPAWNINGPOOL", False, [89], "zerg"),
    ("ZERG_EVOLUTIONCHAMBER", False, [90], "zerg"),
    ("ZERG_HYDRALISKDEN", False, [91], "zerg"),
    ("ZERG_SPIRE", False, [92], "zerg"),
    ("ZERG_ULTRALISKCAVERN", False, [93], "zerg"),
    ("ZERG_INFESTATIONPIT", False, [94], "zerg"),
    ("ZERG_NYDUSNETWORK", False, [95], "zerg"),
    ("ZERG_BANELINGNEST", False, [96], "zerg"),
    ("ZERG_ROACHWARREN", False, [97], "zerg"),
    ("ZERG_SPINECRAWLER", False, [98, 139], "zerg"),
    ("ZERG_SPORECRAWLER", False, [99, 140], "zerg"),
    ("ZERG_LAIR", False, [100], "zerg"),
    ("ZERG_HIVE", False, [101], "zerg"),
    ("ZERG_GREATERSPIRE", False, [102], "zerg"),
    ("ZERG_EGG", False, [103], "zerg"),
    ("ZERG_DRONE", False, [104, 116], "zerg"),
    ("ZERG_ZERGLING", True, [105, 119], "zerg"),
    ("ZERG_OVERLORD", False, [106], "zerg"),
    ("ZERG_HYDRALISK", True, [107, 117], "zerg"),
    ("ZERG_MUTALISK", True, [108], "zerg"),
    ("ZERG_ULTRALISK", True, [109], "zerg"),
    ("ZERG_ROACH", True, [110, 118], "zerg"),
    ("ZERG_INFESTOR", True, [111, 127], "zerg"),
    ("ZERG_CORRUPTOR", True, [112], "zerg"),
    ("ZERG_BROODLORDCOCOON", True, [113], "zerg"),
    ("ZERG_BROODLORD", True, [114], "zerg"),
    ("ZERG_QUEEN", False, [126, 125], "zerg"),
    ("ZERG_OVERLORDCOCOON", True, [128], "zerg"),
    ("ZERG_OVERSEER", True, [129], "zerg"),
    ("ZERG_NYDUSCANAL", False, [142], "zerg"),
    ("ZERG_INFESTEDTERRANSEGG", True, [150], "zerg"),
    ("ZERG_LARVA", False, [151], "zerg"),
    ("ZERG_BROODLING", True, [289], "zerg"),
    ("ZERG_LOCUSTMP", True, [489, 693], "zerg"),
    ("ZERG_SWARMHOSTMP", True, [494, 493], "zerg"),
    ("ZERG_VIPER", True, [499], "zerg"),
    ("ZERG_LURKERMPEGG", True, [501], "zerg"),
    ("ZERG_LURKERMP", True, [502, 503], "zerg"),
    ("ZERG_LURKERDENMP", False, [504], "zerg"),
    ("ZERG_RAVAGERCOCOON", True, [687], "zerg"),
    ("ZERG_RAVAGER", True, [688], "zerg"),
    # ("ZERG_PARASITICBOMBDUMMY", True, [824], "zerg"),
    ("ZERG_TRANSPORTOVERLORDCOCOON", True, [892], "zerg"),
    ("ZERG_OVERLORDTRANSPORT", True, [893], "zerg"),
]

protossUnits = [
    ("PROTOSS_COLOSSUS", True, [4], "protoss"),
    ("PROTOSS_MOTHERSHIP", True, [10], "protoss"),
    ("PROTOSS_NEXUS", False, [59], "protoss"),
    ("PROTOSS_PYLON", False, [60, 894], "protoss"),
    ("PROTOSS_ASSIMILATOR", False, [61], "protoss"),
    ("PROTOSS_GATEWAY", False, [62], "protoss"),
    ("PROTOSS_FORGE", False, [63], "protoss"),
    ("PROTOSS_FLEETBEACON", False, [64], "protoss"),
    ("PROTOSS_TWILIGHTCOUNCIL", False, [65], "protoss"),
    ("PROTOSS_PHOTONCANNON", False, [66], "protoss"),
    ("PROTOSS_STARGATE", False, [67], "protoss"),
    ("PROTOSS_TEMPLARARCHIVE", False, [68], "protoss"),
    ("PROTOSS_DARKSHRINE", False, [69], "protoss"),
    ("PROTOSS_ROBOTICSBAY", False, [70], "protoss"),
    ("PROTOSS_ROBOTICSFACILITY", False, [71], "protoss"),
    ("PROTOSS_CYBERNETICSCORE", False, [72], "protoss"),
    ("PROTOSS_ZEALOT", True, [73], "protoss"),
    ("PROTOSS_STALKER", True, [74], "protoss"),
    ("PROTOSS_HIGHTEMPLAR", True, [75], "protoss"),
    ("PROTOSS_DARKTEMPLAR", True, [76], "protoss"),
    ("PROTOSS_SENTRY", True, [77], "protoss"),
    ("PROTOSS_PHOENIX", True, [78], "protoss"),
    ("PROTOSS_CARRIER", True, [79], "protoss"),
    ("PROTOSS_VOIDRAY", True, [80], "protoss"),
    ("PROTOSS_WARPPRISM", True, [81, 136], "protoss"),
    ("PROTOSS_OBSERVER", True, [82], "protoss"),
    ("PROTOSS_IMMORTAL", True, [83], "protoss"),
    ("PROTOSS_PROBE", False, [84], "protoss"),
    ("PROTOSS_INTERCEPTOR", True, [85], "protoss"),
    ("PROTOSS_WARPGATE", False, [133], "protoss"),
    ("NEUTRAL_FORCEFIELD", True, [135], "protoss"),
    ("PROTOSS_ARCHON", True, [141], "protoss"),
    ("PROTOSS_ADEPT", True, [311, 801], "protoss"),
    ("PROTOSS_MOTHERSHIPCORE", True, [488], "protoss"),
    ("PROTOSS_ORACLE", True, [495], "protoss"),
    ("PROTOSS_TEMPEST", True, [496], "protoss"),
    ("PROTOSS_DISRUPTOR", True, [694], "protoss"),
    ("PROTOSS_ORACLESTASISTRAP", False, [732], "protoss"),
    ("PROTOSS_DISRUPTORPHASED", True, [733], "protoss"),
    ("PROTOSS_SHIELDBATTERY", False, [1910], "protoss"),
]

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
    def __init__(self, units):
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
            self.military_units_mask[i] = 1 if unit[1] else 0
            self.reverse_unit_index_map[i] = unit[2][0]
            self.unit_index_to_race[i] = unit[3]
            for index in unit[2]:
                self.unit_index_map[index] = i
                if unit[0] in workerUnits:
                    self.workerUnitTypes.add(index)

        self.military_units_mask_indices = np.where(self.military_units_mask)[0]
        self.non_military_units_mask_indices = np.where(self.military_units_mask == False)[0]

        self.movable_units_mask = self.military_units_mask.copy()
        for i, u in enumerate(units):
            if u[0] in nonMilitaryUnits:
                self.movable_units_mask[i] = 1
