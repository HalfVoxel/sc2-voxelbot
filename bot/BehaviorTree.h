#pragma once
#include <vector>

namespace BOT {

enum Status { Running, Success, Failure, Idle, Halted };

class TreeNode {
   public:
    virtual ~TreeNode() = default;
    virtual BOT::Status Tick() = 0;
};

class ControlFlowNode : public TreeNode {
   public:
    void Add(std::unique_ptr<TreeNode> node);

   private:
   protected:
    std::vector<std::unique_ptr<TreeNode>> children;
};

class ActionNode : public TreeNode {};

class ConditionNode : public TreeNode {};

class ParallelNode : public ControlFlowNode {
   public:
    BOT::Status Tick() override;
};

class SelectorNode : public ControlFlowNode {
   public:
    BOT::Status Tick() override;
};

class SequenceNode : public ControlFlowNode {
   public:
    BOT::Status Tick() override;
};
}  // namespace BOT
