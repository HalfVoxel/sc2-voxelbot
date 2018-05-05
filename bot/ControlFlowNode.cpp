#include "BehaviorTree.h"


void BOT::ControlFlowNode::Add(TreeNode* node) {
	children.push_back(node);
}
