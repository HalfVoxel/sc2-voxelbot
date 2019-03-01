#pragma once
#include "sc2api/sc2_interfaces.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

struct MLMovement {
	pybind11::object predictShouldMoveFn;
	pybind11::object stepper;
	pybind11::object stepperTarget;
	void OnGameStart();
	void Tick(const sc2::ObservationInterface* observation);
};
