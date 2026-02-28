#pragma once

#include "components/simple_scene.h"

#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

namespace perlin {

    // Fade function (Perlin's smoothing function)
    inline float fade(float t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    // Linear interpolation
    inline float lerp(float a, float b, float t) {
        return a + t * (b - a);
    }

    // Gradient function for 2D
    inline float grad(int hash, float x, float y) {
        int h = hash & 3;
        float u = h < 2 ? x : y;
        float v = h < 2 ? y : x;
        return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
    }

    class Perlin2D {
    public:
        Perlin2D(unsigned int seed = 0) {
            p.resize(256);
            std::iota(p.begin(), p.end(), 0);
            std::mt19937 gen(seed);
            std::shuffle(p.begin(), p.end(), gen);

            // Duplicate permutation vector
            p.insert(p.end(), p.begin(), p.end());
        }

        float perlin(float x, float y) const {
            int X = (int)std::floor(x) & 255;
            int Y = (int)std::floor(y) & 255;

            x -= std::floor(x);
            y -= std::floor(y);

            float u = fade(x);
            float v = fade(y);

            int aa = p[p[X] + Y];
            int ab = p[p[X] + Y + 1];
            int ba = p[p[X + 1] + Y];
            int bb = p[p[X + 1] + Y + 1];

            float res = lerp(
                lerp(grad(aa, x, y), grad(ba, x - 1, y), u),
                lerp(grad(ab, x, y - 1), grad(bb, x - 1, y - 1), u),
                v
            );

            return (res + 1.0f) * 0.5f;
        }

        float perlin_fractal(float x, float y, int octaves = 4, float persistence = 0.5f) const {
            float total = 0.0f;
            float frequency = 1.0f;
            float amplitude = 1.0f;
            float maxValue = 0.0f;

            for (int i = 0; i < octaves; i++) {
                total += perlin(x * frequency, y * frequency) * amplitude;
                maxValue += amplitude;
                amplitude *= persistence;
                frequency *= 2.0f; 
            }

            return total / maxValue;
        }

    private:
        std::vector<int> p;
    };

}


