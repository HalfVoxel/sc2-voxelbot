#pragma once

// See https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
struct VarianceEstimator {
    float m = 0;
    float v = 0;
    float k = 0;

    void add(float x) {
        k++;
        float mNew = m + (x - m)/k;
        float vNew = v + (x - m)*(x - mNew);
        m = mNew;
        v = vNew;
    }

    float mean() const {
        return m;
    }

    float variance() const {
        return v/k;
    }
};
