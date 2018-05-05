#include "BehaviorTree.h""

BOT::Status BOT::SelectorNode::Tick() {
	for (const auto& child : children) {
		BOT::Status status = child->Tick();
		if (status == BOT::Status::Running || status == BOT::Status::Success) {
			return status;
		}
	}
	return BOT::Status::Failure;
}
