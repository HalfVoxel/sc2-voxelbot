#include "BehaviorTree.h"
using namespace BOT;

void ControlFlowNode::Add(TreeNode* node) {
	children.push_back(node);
}
