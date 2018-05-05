#include "BehaviorTree.h"

BOT::Status BOT::SequenceNode::Tick() {
	for (const auto& child : children) {
		BOT::Status status = child->Tick();
		if (status == BOT::Status::Running || status == BOT::Status::Failure ) {
			return status;
		}
	}

	return BOT::Status::Success;
}
