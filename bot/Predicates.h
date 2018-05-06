#include "Bot.h"

struct IsAttackable {
	bool operator()(const sc2::Unit& unit);
};
struct IsFlying {
	bool operator()(const sc2::Unit& unit);
};
struct IsArmy {
	const sc2::ObservationInterface* observation_;
	//Ignores Overlords, workers, and structures
	IsArmy(const sc2::ObservationInterface* obs) : observation_(obs) {}
	bool operator()(const sc2::Unit& unit);
};
struct IsTownHall {
	bool operator()(const sc2::Unit& unit);
};
struct IsVespeneGeyser {
	bool operator()(const sc2::Unit& unit);
};
struct IsStructure {
	const sc2::ObservationInterface* observation_;
	IsStructure(const sc2::ObservationInterface* obs) : observation_(obs) {};
	bool operator()(const sc2::Unit& unit);
};
