#include "CameraController.h"
#include "sc2api/sc2_api.h"
#include "Mappings.h"
#include "Bot.h"
#include "Predicates.h"
#include <vector>
#include <tuple>
#include <iostream>

using namespace sc2;
using namespace std;

map<Tag, double> interestDecay;
map<Tag, double> interest;
map<Tag, double> hysterisis;

Tag lastTag = NullTag;
double hysterisisBonus = 0;

Point2D position;
Point2D targetPoint;
void CameraController::OnStep() {
	auto observation = bot.Observation();
	if (position == Point2D(0,0)) {
		targetPoint = position = observation->GetCameraPos();
	}

	map<UNIT_TYPEID,int> unitTypeCounts;
	for (auto unit : observation->GetUnits(Unit::Alliance::Self)) {
		if (unit->build_progress >= 1) {
			unitTypeCounts[simplifyUnitType(unit->unit_type)]++;
		}
	}

	// Score different interesting points (mostly unit positions)
	// and then pick the most interesting one to move the camera to.
	vector<tuple<double, pair<double,double>, Tag>> points;
	for (auto unit : observation->GetUnits()) {
		double score = 0;
		Point2D pos = unit->pos;
		if (unit->alliance == Unit::Alliance::Self) {
			if (unit->orders.size() > 0) {
				auto order = unit->orders[0];
				auto& ability = observation->GetAbilityData()[order.ability_id];

				UnitTypeID unitType = simplifyUnitType(abilityToUnit(order.ability_id));
				if (unitType != UNIT_TYPEID::INVALID) {
					const auto& unitData = observation->GetUnitTypeData()[unitType];
					double multiplier = (unitData.mineral_cost + unitData.vespene_cost) / 100.0;

					if (order.target_pos != Point2D(0,0)) {
						if (Distance2D(pos, order.target_pos) < 5) {
							// When building, just use the building's position
							pos = order.target_pos;
						} else if (Distance2D(pos, order.target_pos) < 30) {
							// When relatively close to the target, look at the middle between the target position and the unit's position
							pos = (pos + order.target_pos) * 0.5;
						}
					}

					// Higher scores for unit types that we haven't seen before
					score += multiplier * (1 + order.progress) / ((1 + unitTypeCounts[unitType]) * (1 + unitTypeCounts[unitType]));
				}
			}

			if (unit->engaged_target_tag != NullTag) {
				score += 5;
				auto* enemy = observation->GetUnit(unit->engaged_target_tag);
				if (enemy != nullptr) {
					pos = (pos + enemy->pos) * 0.5;
				}
			}

			if (IsArmy(observation)(*unit)) {
				score += 0.1;
			}
		} else if (unit->alliance == Unit::Alliance::Enemy){
			if (IsArmy(observation)(*unit)) {
				score += 1;
			} else {
				score += 0.2;
			}
		} else {
			// Neutral or something
			continue;
		}

		interest[unit->tag] += score;
		if (unit->is_on_screen) {
			interestDecay[unit->tag] += 0.05;
		}
		interest[unit->tag] *= 0.995;
		interestDecay[unit->tag] *= 0.995;

		double finalScore = interest[unit->tag] / (1 + interestDecay[unit->tag]);
		if (unit->tag == lastTag) finalScore *= hysterisisBonus;
		points.push_back(make_tuple(finalScore, make_pair(pos.x, pos.y), unit->tag));
	}

	if (points.size() == 0) return;

	sort(points.begin(), points.end());
	auto target = points[points.size()-1];
	auto newTargetPoint = Point2D(get<1>(target).first, get<1>(target).second);

	Point2D sum (0,0);
	double weight = 0;
	for (auto tup : points) {
		auto p = Point2D(get<1>(tup).first, get<1>(tup).second);
		if (Distance2D(p, newTargetPoint) < 15) {
			double w = get<0>(tup) + 0.0001;
			sum += w * p;
			weight += w;
		}
	}

	if (weight > 0) newTargetPoint = sum / weight;

	double alpha2 = 0.8;
	targetPoint = targetPoint * alpha2 + newTargetPoint * (1-alpha2);

	double alpha = 0.96;

	position = position * alpha + targetPoint * (1-alpha);
	bot.Debug()->DebugMoveCamera(position);

	if (get<2>(target) != lastTag) {
		hysterisisBonus = 10;
	}
	hysterisisBonus *= 0.997;
	lastTag = get<2>(target);
}