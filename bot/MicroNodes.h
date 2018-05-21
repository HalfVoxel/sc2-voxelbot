#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"

struct UnitContext : BOT::Context {
    const sc2::Unit* unit;
    UnitContext(const sc2::Unit* unit) : unit(unit) {}
};

struct MicroNode : BOT::ContextAwareActionNode {
    const sc2::Unit* GetUnit() const { return ((UnitContext*)context)->unit; }
    MicroNode(const sc2::Unit* unit) : BOT::ContextAwareActionNode(new UnitContext(unit)){}
};

struct MicroBattleCruiser : MicroNode {
public:
    MicroBattleCruiser(const sc2::Unit* unit) : MicroNode(unit){}
    BOT::Status OnTick() override;
};

struct MicroLiberator : MicroNode {
public:
    sc2::Point2D defensive_point;
    MicroLiberator(const sc2::Unit* unit) : MicroNode(unit){}
    BOT::Status OnTick() override;
};

struct MicroOrbitalCommand : MicroNode {
public:
    MicroOrbitalCommand(const sc2::Unit* unit) : MicroNode(unit){}
    BOT::Status OnTick() override;
};

bool IsAbilityReady (const sc2::Unit* unit, sc2::ABILITY_ID ability);
void TickMicro ();