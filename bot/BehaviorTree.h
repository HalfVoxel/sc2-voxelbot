#pragma once
#include <vector>

namespace BOT {

enum Status { Running, Success, Failure, Idle, Halted };

class TreeNode {
   public:
    virtual ~TreeNode() = default;
    virtual BOT::Status Tick() = 0;
};

static std::vector<std::unique_ptr<TreeNode>> convertChildren(std::initializer_list<TreeNode*> ls) {
    std::vector<std::unique_ptr<TreeNode>> result(ls.size());
    int i = 0;
    for (auto* node : ls) {
        result[i] = std::unique_ptr<TreeNode>(node);
        i++;
    }
    return result;
}

class ControlFlowNode : public TreeNode {
   public:
    void Add(std::unique_ptr<TreeNode> node);

   private:
   protected:
    std::vector<std::unique_ptr<TreeNode>> children;
    ControlFlowNode(std::initializer_list<TreeNode*> ls) : children(convertChildren(ls)) {}
};

class ActionNode : public TreeNode {};

class ConditionNode : public TreeNode {};

class ParallelNode : public ControlFlowNode {
   public:
    ParallelNode(std::initializer_list<TreeNode*> ls) : ControlFlowNode(ls) {}
    BOT::Status Tick() override;
};

class SelectorNode : public ControlFlowNode {
   public:
    SelectorNode(std::initializer_list<TreeNode*> ls) : ControlFlowNode(ls) {}
    BOT::Status Tick() override;
};

class SequenceNode : public ControlFlowNode {
   public:
    SequenceNode(std::initializer_list<TreeNode*> ls) : ControlFlowNode(ls) {}
    BOT::Status Tick() override;
};
}  // namespace BOT
