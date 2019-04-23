#include "tensor.h"
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>

using namespace std;
using namespace sc2;

Tensor& Tensor::operator+=(const Tensor& other) {
    assert(w == other.w);
    assert(h == other.h);
    for (auto p : npZip(weights, other.weights)) p.first += p.second;
    return (*this);
}

Tensor& Tensor::operator+=(float other) {
    for (auto& x : npIterate(weights)) x += other;
    return (*this);
}

Tensor Tensor::operator+(const Tensor& other) const {
    auto ret = (*this);
    return (ret += other);
}

Tensor& Tensor::operator-=(const Tensor& other) {
    assert(w == other.w);
    assert(h == other.h);

    for (auto p : npZip(weights, other.weights)) p.first -= p.second;
    return (*this);
}

Tensor Tensor::operator-(const Tensor& other) const {
    auto ret = (*this);
    return (ret -= other);
}

Tensor& Tensor::operator*=(const Tensor& other) {
    assert(w == other.w);
    assert(h == other.h);
    for (auto p : npZip(weights, other.weights)) p.first *= p.second;
    return (*this);
}

Tensor Tensor::operator*(const Tensor& other) const {
    auto ret = (*this);
    return (ret *= other);
}

Tensor& Tensor::operator/=(const Tensor& other) {
    assert(w == other.w);
    assert(h == other.h);

    for (auto p : npZip(weights, other.weights)) p.first /= p.second;
    return (*this);
}

Tensor Tensor::operator/(const Tensor& other) const {
    auto ret = (*this);
    return (ret /= other);
}

Tensor Tensor::operator+(float factor) const {
    Tensor ret = Tensor(w, h);
    for (auto p : npZip(ret.weights, weights)) p.first = p.second + factor;
    return ret;
}

Tensor Tensor::operator-(float factor) const {
    Tensor ret = Tensor(w, h);
    for (auto p : npZip(ret.weights, weights)) p.first = p.second - factor;
    return ret;
}

Tensor Tensor::operator*(float factor) const {
    Tensor ret = Tensor(w, h);
    for (auto p : npZip(ret.weights, weights)) p.first = p.second * factor;
    return ret;
}

void Tensor::operator*=(float factor) {
    for (auto& x : npIterate(weights)) x *= factor;
}

void Tensor::threshold(float value) {
    for (auto& x : npIterate(weights)) x = x >= value ? 1 : 0;
}

float Tensor::sum() const {
    float ret = 0.0;
    for (auto x : npIterate(weights)) ret += x;
    return ret;
}

float Tensor::max() const {
    float ret = 0.0;
    for (auto x : npIterate(weights)) ret = std::max(ret, x);
    return ret;
}

float Tensor::maxFinite() const {
    float ret = 0.0;
    for (auto x : npIterate(weights)) if (isfinite(x)) ret = std::max(ret, x);
    return ret;
}

void Tensor::print() const {
    auto data = weights.unchecked();
    for (int y = 0; y < w; y++) {
        for (int x = 0; x < h; x++) {
            cout << setfill(' ') << setw(6) << setprecision(1) << fixed << data(x, y) << " ";
        }
        cout << endl;
    }
}

/*Point2DI Tensor::argmax() const {
    float mx = -100000;
    Point2DI best(0, 0);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            if (weights[y * w + x] > mx && isfinite(weights[y * w + x])) {
                mx = weights[y * w + x];
                best = Point2DI(x, y);
            }
        }
    }
    return best;
}*/

/*
const int scale = 2;
void Tensor::render(int x0, int y0) const {
    ImageGrayscale((float*)weights.data(), w, h, scale * x0 * (w + 5), scale * y0 * (h + 5), scale, false);
}

void Tensor::renderNormalized(int x0, int y0) const {
    ImageGrayscale((float*)weights.data(), w, h, scale * x0 * (w + 5), scale * y0 * (h + 5), scale, true);
}

default_random_engine generator(time(0));


Point2DI Tensor::samplePointFromProbabilityDistribution() const {
    uniform_real_distribution<float> distribution(0.0, sum());
    float picked = distribution(generator);

    for (int i = 0; i < w * h; i++) {
        picked -= weights[i];
        if (picked <= 0)
            return Point2DI(i % w, i / w);
    }

    // Should in theory not happen, but it may actually happen because of floating point errors
    return Point2DI(w - 1, h - 1);
}
*/