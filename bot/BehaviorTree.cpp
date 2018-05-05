#include "BehaviorTree.h"
using namespace BOT;
using namespace std;

void ControlFlowNode::Add(unique_ptr<TreeNode> node) {
    children.push_back(move(node));
}

Status ParallelNode::Tick() {
    for (const auto& child : children) {
        child->Tick();
    }
    return Status::Success;
}

Status SelectorNode::Tick() {
    for (const auto& child : children) {
        Status status = child->Tick();
        if (status == Status::Running || status == Status::Success) {
            return status;
        }
    }
    return Status::Failure;
}

Status SequenceNode::Tick() {
    for (const auto& child : children) {
        Status status = child->Tick();
        if (status == Status::Running || status == Status::Failure) {
            return status;
        }
    }

    return Status::Success;
}
