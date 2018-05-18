#pragma once
#include "sc2api/sc2_api.h"
#include <vector>

struct InfluenceMap {
    std::vector<double> weights;
    int w, h;

    InfluenceMap() {
    }

    InfluenceMap(int width, int height) {
        w = width;
        h = height;
        weights = std::vector<double>(width*height, 0.0);
    }

    InfluenceMap(sc2::ImageData map);

    double& operator()(int x, int y) {
        return weights[y*w + x];
    }

    double operator()(int x, int y) const {
        return weights[y*w + x];
    }

    double& operator()(sc2::Point2DI p) {
        return weights[p.y*w + p.x];
    }

    double operator()(sc2::Point2DI p) const {
        return weights[p.y*w + p.x];
    }

    double& operator()(sc2::Point2D p) {
        return weights[std::min(h-1, std::max(0, (int)round(p.y)))*w + std::min(w-1, std::max(0, (int)round(p.x)))];
    }

    double operator()(sc2::Point2D p) const {
        return weights[std::min(h-1, std::max(0, (int)round(p.y)))*w + std::min(w-1, std::max(0, (int)round(p.x)))];
    }

    double& operator[](int index) {
        return weights[index];
    }

    double operator[](int index) const {
        return weights[index];
    }

    InfluenceMap& operator+= (const InfluenceMap& other);

    InfluenceMap& operator+= (double other);

    InfluenceMap operator+ (const InfluenceMap& other) const;

    InfluenceMap& operator-= (const InfluenceMap& other);

    InfluenceMap operator- (const InfluenceMap& other) const;

    InfluenceMap& operator*= (const InfluenceMap& other);

    InfluenceMap operator* (const InfluenceMap& other) const;

    InfluenceMap& operator/= (const InfluenceMap& other);

    InfluenceMap operator/ (const InfluenceMap& other) const;

    InfluenceMap operator+ (double factor) const;

    InfluenceMap operator- (double factor) const;

    InfluenceMap operator* (double factor) const;

    void operator*= (double factor);

    void threshold(double value);

    double sum() const;

    double max() const;
    double maxFinite() const;

    sc2::Point2DI argmax() const;

    InfluenceMap replace_nonzero(double with) const;

    void addInfluence(double influence, sc2::Point2DI pos);
    void addInfluence(double influence, sc2::Point2D pos);

    void addInfluence(const std::vector<std::vector<double> >& influence, sc2::Point2D);
    
    void addInfluenceMultiple(const std::vector<std::vector<double> >& influence, sc2::Point2D, double factor);

    void maxInfluence(const std::vector<std::vector<double> >& influence, sc2::Point2D);

    void maxInfluenceMultiple(const std::vector<std::vector<double> >& influence, sc2::Point2D, double factor);

    void propagateMax(double decay, double speed, const InfluenceMap& traversable);
    void propagateSum(double decay, double speed, const InfluenceMap& traversable);
    sc2::Point2DI samplePointFromProbabilityDistribution() const;

    void print() const;

    void render(int x0, int y0) const;
    void renderNormalized(int x0, int y0) const;
};