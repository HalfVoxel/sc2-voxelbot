#pragma once
#include "bot.h"

struct DependencyAnalyzer {
	void analyze(const sc2::ObservationInterface* obs);
};