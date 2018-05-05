#include "UnitNodes.h"
#include "bot.h"
#include "bot_examples.h"

using namespace BOT;
using namespace std;
using namespace sc2;

bool GetRandomUnit(const Unit*& unit_out, const ObservationInterface* observation, UnitTypeID unit_type) {
    Units my_units = observation->GetUnits(Unit::Alliance::Self);
    std::random_shuffle(my_units.begin(), my_units.end()); // Doesn't work, or doesn't work well.
    for (const auto unit : my_units) {
        if (unit->unit_type == unit_type) {
            unit_out = unit;
            return true;
        }
    }
    return false;
}

Status BuildUnit::Tick() {
	const ObservationInterface* observation = bot.Observation();

    //If we are at supply cap, don't build anymore units, unless its an overlord.
    if (observation->GetFoodUsed() >= observation->GetFoodCap() && abilityType != ABILITY_ID::TRAIN_OVERLORD) {
        return Status::Failure;
    }
    const Unit* unit = nullptr;
    if (!GetRandomUnit(unit, observation, unitType)) {
        return Status::Failure;
    }

    if (!unit->orders.empty()) {
        return Status::Failure;
    }

    if (unit->build_progress != 1) {
        return Status::Failure;
    }

    bot.Actions()->UnitCommand(unit, abilityType);
    return Status::Success;
}