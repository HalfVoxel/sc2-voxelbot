#include "BehaviorTree.h"
using namespace BOT;

Status ParallelNode::Tick() {
    for (const auto& child : children) {
        child->Tick();
    }
    return Status::Success;
}
