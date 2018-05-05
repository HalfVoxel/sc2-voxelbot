#include "BehaviorTree.h"
using namespace BOT;

Status ParralelNode::Tick(){
	for (const auto& child : children){
		child->Tick();
	}
	return Status::Success;
}
