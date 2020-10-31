#ifndef WASM_SIMD128_EMULATION_H
#define WASM_SIMD128_EMULATION_H

union v128_t {
    unsigned int uint32x4[4];
    float float32x4[4];

    v128_t() { 
        float32x4[0] = 0.0f;
        float32x4[1] = 0.0f;
        float32x4[2] = 0.0f;
        float32x4[3] = 0.0f;
    }
};

inline v128_t wasm_f32x4_splat(float f) {
    v128_t result;
    result.float32x4[0] = f;
    result.float32x4[1] = f;
    result.float32x4[2] = f;
    result.float32x4[3] = f;
    return result;
}

inline v128_t wasm_f32x4_make(float f0, float f1, float f2, float f3) {
    v128_t result;
    result.float32x4[0] = f0;
    result.float32x4[1] = f1;
    result.float32x4[2] = f2;
    result.float32x4[3] = f3;
    return result;
}

inline v128_t wasm_v128_load(float const * f) {
    v128_t result;
    result.float32x4[0] = f[0];
    result.float32x4[1] = f[1];
    result.float32x4[2] = f[2];
    result.float32x4[3] = f[3];
    return result;
}

inline void wasm_v128_store(float* f, const v128_t &v) {
    f[0] = v.float32x4[0];
    f[1] = v.float32x4[1];
    f[2] = v.float32x4[2];
    f[3] = v.float32x4[3];
}

inline v128_t wasm_f32x4_add(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.float32x4[0] = v1.float32x4[0] + v2.float32x4[0];
    result.float32x4[1] = v1.float32x4[1] + v2.float32x4[1];
    result.float32x4[2] = v1.float32x4[2] + v2.float32x4[2];
    result.float32x4[3] = v1.float32x4[3] + v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_sub(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.float32x4[0] = v1.float32x4[0] - v2.float32x4[0];
    result.float32x4[1] = v1.float32x4[1] - v2.float32x4[1];
    result.float32x4[2] = v1.float32x4[2] - v2.float32x4[2];
    result.float32x4[3] = v1.float32x4[3] - v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_mul(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.float32x4[0] = v1.float32x4[0] * v2.float32x4[0];
    result.float32x4[1] = v1.float32x4[1] * v2.float32x4[1];
    result.float32x4[2] = v1.float32x4[2] * v2.float32x4[2];
    result.float32x4[3] = v1.float32x4[3] * v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_div(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.float32x4[0] = v1.float32x4[0] / v2.float32x4[0];
    result.float32x4[1] = v1.float32x4[1] / v2.float32x4[1];
    result.float32x4[2] = v1.float32x4[2] / v2.float32x4[2];
    result.float32x4[3] = v1.float32x4[3] / v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_eq(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.uint32x4[0] = v1.float32x4[0] == v2.float32x4[0];
    result.uint32x4[1] = v1.float32x4[1] == v2.float32x4[1];
    result.uint32x4[2] = v1.float32x4[2] == v2.float32x4[2];
    result.uint32x4[3] = v1.float32x4[3] == v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_ne(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.uint32x4[0] = v1.float32x4[0] != v2.float32x4[0];
    result.uint32x4[1] = v1.float32x4[1] != v2.float32x4[1];
    result.uint32x4[2] = v1.float32x4[2] != v2.float32x4[2];
    result.uint32x4[3] = v1.float32x4[3] != v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_lt(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.uint32x4[0] = v1.float32x4[0] < v2.float32x4[0];
    result.uint32x4[1] = v1.float32x4[1] < v2.float32x4[1];
    result.uint32x4[2] = v1.float32x4[2] < v2.float32x4[2];
    result.uint32x4[3] = v1.float32x4[3] < v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_le(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.uint32x4[0] = v1.float32x4[0] <= v2.float32x4[0];
    result.uint32x4[1] = v1.float32x4[1] <= v2.float32x4[1];
    result.uint32x4[2] = v1.float32x4[2] <= v2.float32x4[2];
    result.uint32x4[3] = v1.float32x4[3] <= v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_gt(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.uint32x4[0] = v1.float32x4[0] > v2.float32x4[0];
    result.uint32x4[1] = v1.float32x4[1] > v2.float32x4[1];
    result.uint32x4[2] = v1.float32x4[2] > v2.float32x4[2];
    result.uint32x4[3] = v1.float32x4[3] > v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_ge(const v128_t &v1, const v128_t &v2) {
    v128_t result;
    result.uint32x4[0] = v1.float32x4[0] >= v2.float32x4[0];
    result.uint32x4[1] = v1.float32x4[1] >= v2.float32x4[1];
    result.uint32x4[2] = v1.float32x4[2] >= v2.float32x4[2];
    result.uint32x4[3] = v1.float32x4[3] >= v2.float32x4[3];
    return result;
}

inline v128_t wasm_f32x4_abs(const v128_t &v) {
    v128_t result;
    result.float32x4[0] = fabs(v.float32x4[0]);
    result.float32x4[1] = fabs(v.float32x4[1]);
    result.float32x4[2] = fabs(v.float32x4[2]);
    result.float32x4[3] = fabs(v.float32x4[3]);
    return result;
}

inline v128_t wasm_v128_bitselect(const v128_t &v1, const v128_t &v2, const v128_t &mask) {
    v128_t result;
    result.float32x4[0] = mask.uint32x4[0] ? v1.float32x4[0] : v2.float32x4[0];
    result.float32x4[1] = mask.uint32x4[1] ? v1.float32x4[1] : v2.float32x4[1];
    result.float32x4[2] = mask.uint32x4[2] ? v1.float32x4[2] : v2.float32x4[2];
    result.float32x4[3] = mask.uint32x4[3] ? v1.float32x4[3] : v2.float32x4[3];
    return result;
}
#endif