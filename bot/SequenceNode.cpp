#include "BehaviorTree.h"
using namespace BOT;

Status SequenceNode::Tick() {
    for (const auto& child : children) {
        Status status = child->Tick();
        if (status == Status::Running || status == Status::Failure) {
            return status;
        }
    }

    return Status::Success;
}
