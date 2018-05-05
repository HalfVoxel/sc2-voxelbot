#pragma once
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_map_info.h"

namespace BOT {

	class Bot : public sc2::Agent {
	public:
	    void OnGameStart() override final;
	    void OnStep() override final;
		//void OnUnitDestroyed(const sc2::Unit* unit) override;
	};

};