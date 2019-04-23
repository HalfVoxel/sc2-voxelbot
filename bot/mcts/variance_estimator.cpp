#include "variance_estimator.h"
#include <cmath>
#include <cassert>

struct UnitTestVarianceEstimator {
    UnitTestVarianceEstimator() {
        VarianceEstimator est;
        est.add(1);
        est.add(2);
        est.add(3);
        assert(std::abs(est.mean() - 2) < 0.0001f);
        assert(std::abs(est.variance() - (1*1 + 0 + 1*1)/3.0f) < 0.0001f);

        est = VarianceEstimator();
        est.add(4123);
        assert(std::abs(est.variance() - 0) < 0.0001f);

        est = VarianceEstimator();
        est.add(0);
        est.add(0);
        est.add(-412);
        est.add(5);
        est.add(534);
        est.add(5343);
        for (int i = 0; i < 100; i++) est.add(123);
        float mean = 167.641509434f;
        float variance = (pow(0 - mean, 2)*2 + pow(-412 - mean, 2) + pow(5 - mean, 2) + pow(534 - mean, 2) + pow(5343 - mean, 2) + pow(123 - mean, 2)*100)/106.0f;
        assert(std::abs(est.mean() - mean) < 0.0001f);
        assert(std::abs(est.variance() - variance) < 0.2f);
    }
};

// Ugly way of always running unit tests... but whatever
static UnitTestVarianceEstimator unitTestVarianceEstimator;
