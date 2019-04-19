
#pragma once
#include "sc2api/sc2_api.h"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pybind11_numpy.h"
#include <iostream>

struct Tensor {
    pybind11::array_t<float> weights;
    int w, h;

    Tensor() {
    }

    Tensor(std::vector<int> dims) : weights(dims) {
        w = dims[0];
        h = dims[1];
        npZero(weights);
    }

    Tensor(int width, int height) : Tensor(std::vector<int>{width, height}) {}

    inline float& operator()(int x, int y) {
        return weights.mutable_at(x, y);
    }

    inline float operator()(int x, int y) const {
        return weights.at(x, y);
    }

    inline float& operator()(sc2::Point2DI p) {
        return (*this)(p.x, p.y);
    }

    inline float operator()(sc2::Point2DI p) const {
        return (*this)(p.x, p.y);
    }

    inline float& operator()(sc2::Point2D p) {
        return (*this)(std::min(h-1, std::max(0, (int)round(p.y))), std::min(w-1, std::max(0, (int)round(p.x))));
    }

    inline float operator()(sc2::Point2D p) const {
        return (*this)(std::min(h-1, std::max(0, (int)round(p.y))), std::min(w-1, std::max(0, (int)round(p.x))));
    }

    inline float& operator[](int index) {
        return (*this)(index, 0);
    }

    // inline float operator[](int index) const {
    //     return weights[index];
    // }

    Tensor& operator+= (const Tensor& other);

    Tensor& operator+= (float other);

    Tensor operator+ (const Tensor& other) const;

    Tensor& operator-= (const Tensor& other);

    Tensor operator- (const Tensor& other) const;

    Tensor& operator*= (const Tensor& other);

    Tensor operator* (const Tensor& other) const;

    Tensor& operator/= (const Tensor& other);

    Tensor operator/ (const Tensor& other) const;

    Tensor operator+ (float factor) const;

    Tensor operator- (float factor) const;

    Tensor operator* (float factor) const;

    void operator*= (float factor);

    void threshold(float value);

    float sum() const;

    float max() const;
    float maxFinite() const;

    // sc2::Point2DI argmax() const;

    void print() const;
};
