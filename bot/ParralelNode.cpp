#include "BehaviorTree.h"

BOT::Status BOT::ParralelNode::Tick(){
	for (const auto& child : children){
		child->Tick();
	}
	return BOT::Status::Success;
}
