#pragma once
#include <memory>
#include <vector>
namespace BOT {

const bool Debug = true;

enum Status { Running, Success, Failure, Idle, Halted };

class Context{};

class TreeNode {
   protected:
    virtual BOT::Status OnTick() = 0;

   public:
    virtual ~TreeNode() = default;
    virtual BOT::Status Tick();
};

class ContextAwareTreeNode : public TreeNode {
public:
    Context* context;
public:
    ContextAwareTreeNode(Context* context): context(context) {}
};

class ContextAwareActionNode : public ContextAwareTreeNode {
public:
    ContextAwareActionNode(Context* context) : ContextAwareTreeNode(context){}
};

class ContextAwareConditionNode : public ContextAwareTreeNode {
public:
    ContextAwareConditionNode(Context* context) : ContextAwareTreeNode(context) {}
};

static std::vector<std::shared_ptr<TreeNode>> convertChildren(std::initializer_list<TreeNode*> ls) {
    std::vector<std::shared_ptr<TreeNode>> result(ls.size());
    int i = 0;
    for (auto* node : ls) {
        result[i] = std::shared_ptr<TreeNode>(node);
        i++;
    }
    return result;
}

class ControlFlowNode : public TreeNode {
   public:
    void Add(std::shared_ptr<TreeNode> node);

   private:
   protected:
    std::vector<std::shared_ptr<TreeNode>> children;
    ControlFlowNode(std::initializer_list<TreeNode*> ls) : children(convertChildren(ls)) {}
};

class ActionNode : public TreeNode {};

class ConditionNode : public TreeNode {};

class ParallelNode : public ControlFlowNode {
   public:
    ParallelNode(std::initializer_list<TreeNode*> ls) : ControlFlowNode(ls) {}
    BOT::Status OnTick() override;
};

class SelectorNode : public ControlFlowNode {
   public:
    SelectorNode(std::initializer_list<TreeNode*> ls) : ControlFlowNode(ls) {}
    BOT::Status OnTick() override;
};

class SequenceNode : public ControlFlowNode {
   public:
    SequenceNode(std::initializer_list<TreeNode*> ls) : ControlFlowNode(ls) {}
    BOT::Status OnTick() override;
};

class Not : public BOT::TreeNode {
    std::unique_ptr<TreeNode> child;

   public:
    Not(TreeNode* node) : child(node) {}
    BOT::Status OnTick() override;
};
}  // namespace BOT
