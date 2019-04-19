#include "BehaviorTree.h"
using namespace BOT;
using namespace std;
#include <iostream>
#include <typeinfo>
#include <algorithm>

void ControlFlowNode::Add(shared_ptr<TreeNode> node) {
    children.push_back(move(node));
}

void ControlFlowNode::Remove(std::shared_ptr<TreeNode> node) {
    auto it = find(children.begin(), children.end(), node);
    if (it != children.end()) children.erase(it);
}

int depth = 0;
Status TreeNode::Tick() {
    // depth++;
    auto res = OnTick();
    // depth--;
    return res;
}

Status ParallelNode::OnTick() {
    for (const auto& child : children) {
        child->Tick();
    }
    return Status::Success;
}

Status SelectorNode::OnTick() {
    for (const auto& child : children) {
        Status status = child->Tick();
        if (status == Status::Running || status == Status::Success) {
            return status;
        }
    }
    return Status::Failure;
}

Status SequenceNode::OnTick() {
    for (const auto& child : children) {
        Status status = child->Tick();
        if (status == Status::Running || status == Status::Failure) {
            return status;
        }
    }

    return Status::Success;
}

Status Not::OnTick() {
    Status s = child->Tick();
    switch (s) {
        case Failure:
            return Success;
        case Success:
            return Failure;
        default:
            throw invalid_argument("child node returned a status which was neither success nor failure");
    }
}