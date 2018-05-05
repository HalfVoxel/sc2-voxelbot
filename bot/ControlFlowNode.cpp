#include "BehaviorTree.h"
using namespace BOT;
using namespace std;

void ControlFlowNode::Add(unique_ptr<TreeNode> node) {
	children.push_back(move(node));
}
