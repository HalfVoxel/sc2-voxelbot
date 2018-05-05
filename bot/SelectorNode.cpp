#include "BehaviorTree.h"
using namespace BOT;

Status SelectorNode::Tick() {
    for (const auto& child : children) {
        Status status = child->Tick();
        if (status == Status::Running || status == Status::Success) {
            return status;
        }
    }
    return Status::Failure;
}
