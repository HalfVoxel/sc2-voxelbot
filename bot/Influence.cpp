#include "Influence.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include "Renderer.h"

using namespace std;
using namespace sc2;

InfluenceMap::InfluenceMap(sc2::ImageData map) : InfluenceMap(map.width, map.height) {
    assert(map.bits_per_pixel == 8);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            weights[y*w+x] = (uint8_t)map.data[(h-y-1)*w+x] == 255 ? 0.0 : 1.0;
        }
    }
}

pair<int,int> round_point(Point2D p) {
    return make_pair((int)round(p.x), (int)round(p.y));
}

InfluenceMap& InfluenceMap::operator+= (const InfluenceMap& other) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            weights[y*w+x] += other.weights[y*w+x];
        }
    }
    return (*this);
}

InfluenceMap& InfluenceMap::operator+= (double other) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            weights[y*w+x] += other;
        }
    }
    return (*this);
}

InfluenceMap InfluenceMap::operator+ (const InfluenceMap& other) const {
    auto ret = (*this);
    return (ret += other);
}

InfluenceMap& InfluenceMap::operator-= (const InfluenceMap& other) {
    assert(w == other.w);
    assert(h == other.h);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            weights[y*w+x] -= other.weights[y*w+x];
        }
    }
    return (*this);
}

InfluenceMap InfluenceMap::operator- (const InfluenceMap& other) const {
    auto ret = (*this);
    return (ret -= other);
}

InfluenceMap& InfluenceMap::operator*= (const InfluenceMap& other) {
    assert(w == other.w);
    assert(h == other.h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            weights[y*w+x] *= other.weights[y*w+x];
        }
    }
    return (*this);
}

InfluenceMap InfluenceMap::operator* (const InfluenceMap& other) const {
    auto ret = (*this);
    return (ret *= other);
}

InfluenceMap& InfluenceMap::operator/= (const InfluenceMap& other) {
    assert(w == other.w);
    assert(h == other.h);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            weights[y*w+x] /= other.weights[y*w+x];
        }
    }
    return (*this);
}

InfluenceMap InfluenceMap::operator/ (const InfluenceMap& other) const {
    auto ret = (*this);
    return (ret /= other);
}

InfluenceMap InfluenceMap::operator+ (double factor) const {
    InfluenceMap ret = InfluenceMap(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ret.weights[y*w+x] = weights[y*w+x] + factor;
        }
    }
    return ret;
}

InfluenceMap InfluenceMap::operator- (double factor) const {
    InfluenceMap ret = InfluenceMap(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ret.weights[y*w+x] = weights[y*w+x] - factor;
        }
    }
    return ret;
}

InfluenceMap InfluenceMap::operator* (double factor) const {
    InfluenceMap ret = InfluenceMap(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ret.weights[y*w+x] = weights[y*w+x] * factor;
        }
    }
    return ret;
}

void InfluenceMap::operator*= (double factor) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            weights[y*w+x] *= factor;
        }
    }
}

double InfluenceMap::sum() const {
    double ret = 0.0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ret += weights[y*w+x];
        }
    }
    return ret;
}

double InfluenceMap::max() const {
    double ret = 0.0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            ret = std::max(ret, weights[y*w+x]);
        }
    }
    return ret;
}

void InfluenceMap::addInfluence(double influence, Point2D pos) {
    auto p = round_point(pos);
    weights[p.second*w + p.first] += influence;
}

void InfluenceMap::addInfluence(const vector<vector<double> >& influence, Point2D pos) {
    int x0, y0;
    tie(x0, y0) = round_point(pos);

    int r = influence.size() / 2;
    for (int dx = -r; dx <= r; dx++) {
        for (int dy = -r; dy <= r; dy++) {
            int x = x0 + dx;
            int y = y0 + dy; 
            if (x >= 0 && y >= 0 && x < w && y < h) {
                weights[y*w+x] += influence[dx+r][dy+r];
            }
        }
    }
}

void InfluenceMap::addInfluenceMultiple(const vector<vector<double> >& influence, Point2D pos, double factor) {
    int x0, y0;
    tie(x0, y0) = round_point(pos);

    int r = influence.size() / 2;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = x0 + dx;
            int y = y0 + dy; 
            if (x >= 0 && y >= 0 && x < w && y < h) {
                weights[y*w+x] += influence[dx+r][dy+r] * factor;
            }
        }
    }
}

void InfluenceMap::maxInfluence(const vector<vector<double> >& influence, Point2D pos) {
    int x0, y0;
    tie(x0, y0) = round_point(pos);

    int r = influence.size() / 2;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = x0 + dx;
            int y = y0 + dy; 
            if (x >= 0 && y >= 0 && x < w && y < h) {
                weights[y*w+x] = std::max(weights[y*w+x], influence[dx+r][dy+r]);
            }
        }
    }
}

void InfluenceMap::maxInfluenceMultiple(const vector<vector<double> >& influence, Point2D pos, double factor) {
    int x0, y0;
    tie(x0, y0) = round_point(pos);

    int r = influence.size() / 2;
    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
            int x = x0 + dx;
            int y = y0 + dy; 
            if (x >= 0 && y >= 0 && x < w && y < h) {
                weights[y*w+x] = std::max(weights[y*w+x], influence[dx+r][dy+r] * factor);
            }
        }
    }
}

void InfluenceMap::propagateMax(double decay, double speed) {
    double factor = 1 - decay;
    // Diagonal decay
    double factor2 = pow(factor, 1.41);

    vector<double> newWeights (weights.size());
    for (int y = 0; y < h; y++) {
        weights[y*w+0] = weights[y*w+1];
        weights[y*w+w-1] = weights[y*w+w-2];
    }
    for (int x = 0; x < w; x++) {
        weights[x] = weights[x+w];
        weights[(h-1)*w+x] = weights[(h-2)*w+x];
    }

    for (int y = 1; y < h-1; y++) {
        int yw = y*w;
        for (int x = 1; x < w-1; x++) {
            int i = yw + x;
            double c = 0;
            c = std::max(c, weights[i-1]);
            c = std::max(c, weights[i+1]);
            c = std::max(c, weights[i-w]);
            c = std::max(c, weights[i+w]);
            c *= factor;

            double c2 = 0;
            c2 = std::max(c2, weights[i-w-1]);
            c2 = std::max(c2, weights[i-w+1]);
            c2 = std::max(c2, weights[i+w-1]);
            c2 = std::max(c2, weights[i+w+1]);
            c2 *= factor2;
            c = std::max(c, c2);

            c = c*speed + (1-speed)*weights[yw+x];
            newWeights[yw+x] = c;
        }
    }

    weights = newWeights;
}

void InfluenceMap::propagateSum(double decay, double speed) {
    double factor = 1 - decay;
    // Diagonal decay
    double factor2 = pow(factor, 1.41);

    vector<double> newWeights (weights.size());
    for (int y = 0; y < h; y++) {
        weights[y*w+0] = weights[y*w+1];
        weights[y*w+w-1] = weights[y*w+w-2];
    }
    for (int x = 0; x < w; x++) {
        weights[x] = weights[x+w];
        weights[(h-1)*w+x] = weights[(h-2)*w+x];
    }

    for (int y = 1; y < h-1; y++) {
        int yw = y*w;
        for (int x = 1; x < w-1; x++) {
            int i = yw + x;
            double c = 0;
            c += weights[i-1];
            c += weights[i+1];
            c += weights[i-w];
            c += weights[i+w];
            c *= factor;

            double c2 = 0;
            c2 += weights[i-w-1];
            c2 += weights[i-w+1];
            c2 += weights[i+w-1];
            c2 += weights[i+w+1];
            c2 *= factor2;
            c += c2;

            // To prevent the total weight values from increasing unbounded
            c *= 0.125;

            c = c*speed + (1-speed)*weights[i];
            newWeights[i] = c;
        }
    }

    weights = newWeights;
}

void InfluenceMap::print() const {
    for (int y = 0; y < w; y++) {
        for (int x = 0; x < h; x++) {
            cout << setfill(' ') << setw(6) << setprecision(1) << fixed << weights[y*w + x] << " ";
        }
        cout << endl;
    }
}

void InfluenceMap::render(int x0, int y0, int scale) const {
    ImageGrayscale((double*)weights.data(), w, h, x0, y0, scale);
}
