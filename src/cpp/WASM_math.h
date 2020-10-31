#pragma once

#include <iomanip>
#include <vector>
#include <memory>

#include <malloc.h>

#include "wasm_simd128_emulation.h"

#include "EigenTypes.h"

// ----------------------------------------------------------------------------------------------
//vector of 4 float values to represent 4 scalars
class Scalarf4
{
public:
	v128_t v;

	Scalarf4() {}

	Scalarf4(float f) {
		v = wasm_f32x4_splat(f);
	}

	Scalarf4(float f0, float f1, float f2, float f3) {
		v = wasm_f32x4_make(f0, f1, f2, f3);
	}

	Scalarf4(v128_t const & x) {
		v = x;
	}

	Scalarf4 & operator = (v128_t const & x) {
		v = x;
		return *this;
	}

	Scalarf4& load(float const * p) {
		v = wasm_v128_load(p);
		return *this;
	}
	
	void store(float * p) const {
		wasm_v128_store(p, v);
	}
};

static inline Scalarf4 operator + (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_add(a.v, b.v);
}

static inline Scalarf4 & operator += (Scalarf4 & a, Scalarf4 const & b) {
	a.v = wasm_f32x4_add(a.v, b.v);
	return a;
}

static inline Scalarf4 operator - (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_sub(a.v, b.v);
}

static inline Scalarf4 & operator -= (Scalarf4 & a, Scalarf4 const & b) {
	a.v = wasm_f32x4_sub(a.v, b.v);
	return a;
}

static inline Scalarf4 operator * (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_mul(a.v, b.v);
}

static inline Scalarf4 & operator *= (Scalarf4 & a, Scalarf4 const & b) {
	a.v = wasm_f32x4_mul(a.v, b.v);
	return a;
}

static inline Scalarf4 operator / (Scalarf4 const & a, Scalarf4 const & b) {
    return wasm_f32x4_div(a.v, b.v);
}

static inline Scalarf4 operator == (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_eq(a.v, b.v);
}

static inline Scalarf4 operator != (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_ne(a.v, b.v);
}

static inline Scalarf4 operator < (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_lt(a.v, b.v);
}

static inline Scalarf4 operator <= (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_le(a.v, b.v);
}

static inline Scalarf4 operator > (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_gt(a.v, b.v);
}

static inline Scalarf4 operator >= (Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_f32x4_ge(a.v, b.v);
}

static inline Scalarf4 abs(Scalarf4 const & a) {
	return wasm_f32x4_abs(a.v);
}

//does the same as for (int i = 0; i < 4; i++) result[i] = c[i] ? a[i] : b[i];
//the elemets in c must be either 0 (false) or 0xFFFFFFFF (true)
static inline Scalarf4 blend(Scalarf4 const & c, Scalarf4 const & a, Scalarf4 const & b) {
	return wasm_v128_bitselect(a.v, b.v, c.v);
}

// ----------------------------------------------------------------------------------------------
//3 dimensional vector of Scalar4f to represent 4 3d vectors
class Vector3f4
{
public:

	Scalarf4 v[3];

	Vector3f4() { v[0] = 0.0; v[1] = 0.0; v[2] = 0.0; }
	Vector3f4(Scalarf4 x, Scalarf4 y, Scalarf4 z) { v[0] = x; v[1] = y; v[2] = z; }
	Vector3f4(Scalarf4 x) { v[0] = v[1] = v[2] = x; }

	inline Scalarf4& operator [] (int i) { return v[i]; }
	inline Scalarf4 operator [] (int i) const { return v[i]; }

	inline Scalarf4& x() { return v[0]; }
	inline Scalarf4& y() { return v[1]; }
	inline Scalarf4& z() { return v[2]; }

	inline Scalarf4 x() const { return v[0]; }
	inline Scalarf4 y() const { return v[1]; }
	inline Scalarf4 z() const { return v[2]; }
	
	inline Scalarf4 dot(const Vector3f4& a) const {
		return v[0] * a.v[0] + v[1] * a.v[1] + v[2] * a.v[2];
	}

	//dot product
	inline Scalarf4 operator * (const Vector3f4& a) const {
		return v[0] * a.v[0] + v[1] * a.v[1] + v[2] * a.v[2];
	}

	inline void cross(const Vector3f4& a, const Vector3f4& b) {
		v[0] = a.v[1] * b.v[2] - a.v[2] * b.v[1];
		v[1] = a.v[2] * b.v[0] - a.v[0] * b.v[2];
		v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];
	}

	//cross product
	inline const Vector3f4 operator % (const Vector3f4& a) const {
		return Vector3f4(v[1] * a.v[2] - v[2] * a.v[1],
			v[2] * a.v[0] - v[0] * a.v[2],
			v[0] * a.v[1] - v[1] * a.v[0]);
	}

	inline const Vector3f4 operator * (Scalarf4 s) const {
		return Vector3f4(v[0] * s, v[1] * s, v[2] * s);
	}

	inline Vector3f4& operator *= (Scalarf4 s) {
		v[0] *= s;
		v[1] *= s;
		v[2] *= s;
		return *this;
	}

	inline const Vector3f4 operator / (Scalarf4 s) const {
		return Vector3f4(v[0] / s, v[1] / s, v[2] / s);
	}

	inline Vector3f4& operator /= (Scalarf4 s) {
		v[0] = v[0] / s;
		v[1] = v[1] / s;
		v[2] = v[2] / s;
		return *this;
	}

	inline const Vector3f4 operator + (const Vector3f4& a) const {
		return Vector3f4(v[0] + a.v[0], v[1] + a.v[1], v[2] + a.v[2]);
	}

	inline Vector3f4& operator += (const Vector3f4& a) {
		v[0] += a.v[0];
		v[1] += a.v[1];
		v[2] += a.v[2];
		return *this;
	}

	inline const Vector3f4 operator - (const Vector3f4& a) const {
		return Vector3f4(v[0] - a.v[0], v[1] - a.v[1], v[2] - a.v[2]);
	}

	inline Vector3f4& operator -= (const Vector3f4& a) {
		v[0] -= a.v[0];
		v[1] -= a.v[1];
		v[2] -= a.v[2];
		return *this;
	}

	inline const Vector3f4 operator - () const {
		return Vector3f4(Scalarf4(-1.0) * v[0], Scalarf4(-1.0) * v[1], Scalarf4(-1.0) * v[2]);
	}

	inline Scalarf4 lengthSquared() const {
		return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	}

	//does the same as for (int i = 0; i < 4; i++) result[i] = c[i] ? a[i] : b[i];
	//the elemets in c must be either 0 (false) or 0xFFFFFFFF (true)
	static inline Vector3f4 blend(Scalarf4 const & c, Vector3f4 const & a, Vector3f4 const & b) {
		Vector3f4 result;
		result.x() = wasm_v128_bitselect(a.x().v, b.x().v, c.v);
		result.y() = wasm_v128_bitselect(a.y().v, b.y().v, c.v);
		result.z() = wasm_v128_bitselect(a.z().v, b.z().v, c.v);
		return result;
	}
};


// ----------------------------------------------------------------------------------------------
//3x3 dimensional matrix of Scalar8f to represent 4 3x3 matrices
class Matrix3f4
{
public:
	Scalarf4 m[3][3];

	Matrix3f4() {  }

	//constructor to create matrix from 3 column vectors
	Matrix3f4(const Vector3f4& m1, const Vector3f4& m2, const Vector3f4& m3)
	{
		m[0][0] = m1.x();
		m[1][0] = m1.y();
		m[2][0] = m1.z();

		m[0][1] = m2.x();
		m[1][1] = m2.y();
		m[2][1] = m2.z();

		m[0][2] = m3.x();
		m[1][2] = m3.y();
		m[2][2] = m3.z();
	}

	inline Scalarf4& operator()(int i, int j) { return m[i][j]; }

	inline void setCol(int i, const Vector3f4& v)
	{
		m[0][i] = v.x();
		m[1][i] = v.y();
		m[2][i] = v.z();
	}

	inline void setCol(int i, const Scalarf4& x, const Scalarf4& y, const Scalarf4& z)
	{
		m[0][i] = x;
		m[1][i] = y;
		m[2][i] = z;
	}

	inline Vector3f4 operator * (const Vector3f4 &b) const
	{
		Vector3f4 A;

		A.v[0] = m[0][0] * b.v[0] + m[0][1] * b.v[1] + m[0][2] * b.v[2];
		A.v[1] = m[1][0] * b.v[0] + m[1][1] * b.v[1] + m[1][2] * b.v[2];
		A.v[2] = m[2][0] * b.v[0] + m[2][1] * b.v[1] + m[2][2] * b.v[2];

		return A;
	}

	inline Matrix3f4 operator * (const Matrix3f4 &b) const
	{
		Matrix3f4 A;

		A.m[0][0] = m[0][0] * b.m[0][0] + m[0][1] * b.m[1][0] + m[0][2] * b.m[2][0];
		A.m[0][1] = m[0][0] * b.m[0][1] + m[0][1] * b.m[1][1] + m[0][2] * b.m[2][1];
		A.m[0][2] = m[0][0] * b.m[0][2] + m[0][1] * b.m[1][2] + m[0][2] * b.m[2][2];

		A.m[1][0] = m[1][0] * b.m[0][0] + m[1][1] * b.m[1][0] + m[1][2] * b.m[2][0];
		A.m[1][1] = m[1][0] * b.m[0][1] + m[1][1] * b.m[1][1] + m[1][2] * b.m[2][1];
		A.m[1][2] = m[1][0] * b.m[0][2] + m[1][1] * b.m[1][2] + m[1][2] * b.m[2][2];

		A.m[2][0] = m[2][0] * b.m[0][0] + m[2][1] * b.m[1][0] + m[2][2] * b.m[2][0];
		A.m[2][1] = m[2][0] * b.m[0][1] + m[2][1] * b.m[1][1] + m[2][2] * b.m[2][1];
		A.m[2][2] = m[2][0] * b.m[0][2] + m[2][1] * b.m[1][2] + m[2][2] * b.m[2][2];

		return A;
	}

	inline Matrix3f4 transpose() const
	{
		Matrix3f4 A;
		A.m[0][0] = m[0][0]; A.m[0][1] = m[1][0]; A.m[0][2] = m[2][0];
		A.m[1][0] = m[0][1]; A.m[1][1] = m[1][1]; A.m[1][2] = m[2][1];
		A.m[2][0] = m[0][2]; A.m[2][1] = m[1][2]; A.m[2][2] = m[2][2];

		return A;
	}

	inline Scalarf4 determinant() const
	{
		return  m[0][1] * m[1][2] * m[2][0] - m[0][2] * m[1][1] * m[2][0] + m[0][2] * m[1][0] * m[2][1] 
			  - m[0][0] * m[1][2] * m[2][1] - m[0][1] * m[1][0] * m[2][2] + m[0][0] * m[1][1] * m[2][2];
	}

	inline void store(std::vector<EigenMatrix3>& Mf) const
	{
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				float val[4];
				m[i][j].store(val);
				for (int k = 0; k < 4; k++)
					Mf[k](i, j) = val[k];
			}
		}
	}
};

// ----------------------------------------------------------------------------------------------
//4 dimensional vector of Scalar8f to represent 4 quaternions
class Quaternion4f 
{
public:

	Scalarf4  q[4];

	inline Quaternion4f() { q[0] = 0.0; q[1] = 0.0; q[2] = 0.0; q[3] = 1.0; }

	inline Quaternion4f(Scalarf4 x, Scalarf4 y, Scalarf4 z, Scalarf4 w) {
		q[0] = x; q[1] = y; q[2] = z; q[3] = w;
	}

	inline Quaternion4f(Vector3f4& v) {
		q[0] = v[0]; q[1] = v[1]; q[2] = v[2]; q[3] = 0.0;
	}

	inline Scalarf4 & operator [] (int i) { return q[i]; }
	inline Scalarf4   operator [] (int i) const { return q[i]; }

	inline Scalarf4 & x() { return q[0]; }
	inline Scalarf4 & y() { return q[1]; }
	inline Scalarf4 & z() { return q[2]; }
	inline Scalarf4 & w() { return q[3]; }

	inline Scalarf4 x() const { return q[0]; }
	inline Scalarf4 y() const { return q[1]; }
	inline Scalarf4 z() const { return q[2]; }
	inline Scalarf4 w() const { return q[3]; }

	inline const Quaternion4f operator*(const Quaternion4f& a) const {
		return
			Quaternion4f(q[3] * a.q[0] + q[0] * a.q[3] + q[1] * a.q[2] - q[2] * a.q[1],
				q[3] * a.q[1] - q[0] * a.q[2] + q[1] * a.q[3] + q[2] * a.q[0],
				q[3] * a.q[2] + q[0] * a.q[1] - q[1] * a.q[0] + q[2] * a.q[3],
				q[3] * a.q[3] - q[0] * a.q[0] - q[1] * a.q[1] - q[2] * a.q[2]);
	}

	inline void toRotationMatrix(Matrix3f4& R)
	{
		const Scalarf4 tx = Scalarf4(2.0) * q[0];
		const Scalarf4 ty = Scalarf4(2.0) * q[1];
		const Scalarf4 tz = Scalarf4(2.0) * q[2];
		const Scalarf4 twx = tx*q[3];
		const Scalarf4 twy = ty*q[3];
		const Scalarf4 twz = tz*q[3];
		const Scalarf4 txx = tx*q[0];
		const Scalarf4 txy = ty*q[0];
		const Scalarf4 txz = tz*q[0];
		const Scalarf4 tyy = ty*q[1];
		const Scalarf4 tyz = tz*q[1];
		const Scalarf4 tzz = tz*q[2];

	    R.m[0][0] = Scalarf4(1.0) - (tyy + tzz);
		R.m[0][1] = txy - twz;
		R.m[0][2] = txz + twy;
		R.m[1][0] = txy + twz;
		R.m[1][1] = Scalarf4(1.0) - (txx + tzz);
		R.m[1][2] = tyz - twx;
		R.m[2][0] = txz - twy;
		R.m[2][1] = tyz + twx;
		R.m[2][2] = Scalarf4(1.0) - (txx + tyy);
	}

	inline void toRotationMatrix(Vector3f4& R1, Vector3f4& R2, Vector3f4& R3)
	{
		const Scalarf4 tx = Scalarf4(2.0) * q[0];
		const Scalarf4 ty = Scalarf4(2.0) * q[1];
		const Scalarf4 tz = Scalarf4(2.0) * q[2];
		const Scalarf4 twx = tx*q[3];
		const Scalarf4 twy = ty*q[3];
		const Scalarf4 twz = tz*q[3];
		const Scalarf4 txx = tx*q[0];
		const Scalarf4 txy = ty*q[0];
		const Scalarf4 txz = tz*q[0];
		const Scalarf4 tyy = ty*q[1];
		const Scalarf4 tyz = tz*q[1];
		const Scalarf4 tzz = tz*q[2];

		R1[0] = Scalarf4(1.0) - (tyy + tzz);
		R2[0] = txy - twz;
		R3[0] = txz + twy;
		R1[1] = txy + twz;
		R2[1] = Scalarf4(1.0) - (txx + tzz);
		R3[1] = tyz - twx;
		R1[2] = txz - twy;
		R2[2] = tyz + twx;
		R3[2] = Scalarf4(1.0) - (txx + tyy);
	}

	inline void store(std::vector<EigenQuaternion>& qf) const
	{
		float x[4], y[4], z[4], w[4];
		q[0].store(x);
		q[1].store(y);
		q[2].store(z);
		q[3].store(w);

		for (int i = 0; i < 4; i++)
		{
			qf[i].x() = x[i];
			qf[i].y() = y[i];
			qf[i].z() = z[i];
			qf[i].w() = w[i];
		}
	}

	inline void set(const std::vector<EigenQuaternion>& qf)
	{
		float x[4], y[4], z[4], w[4];
		for(int i=0; i<4; i++)
		{
			x[i] = static_cast<float>(qf[i].x());
			y[i] = static_cast<float>(qf[i].y());
			z[i] = static_cast<float>(qf[i].z());
			w[i] = static_cast<float>(qf[i].w());
		}
		Scalarf4 s;
		s.load(x);
		q[0] = s;
		s.load(y);
		q[1] = s; 
		s.load(z);
		q[2] = s; 
		s.load(w);
		q[3] = s;
	}
};

// ----------------------------------------------------------------------------------------------
//alligned allocator so that vectorized types can be used in std containers
//from: https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
template <typename T, std::size_t N = 32>
class AlignmentAllocator {
public:
	typedef T value_type;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	typedef T * pointer;
	typedef const T * const_pointer;

	typedef T & reference;
	typedef const T & const_reference;

public:
	inline AlignmentAllocator() throw () { }

	template <typename T2>
	inline AlignmentAllocator(const AlignmentAllocator<T2, N> &) throw () { }

	inline ~AlignmentAllocator() throw () { }

	inline pointer adress(reference r) {
		return &r;
	}

	inline const_pointer adress(const_reference r) const {
		return &r;
	}

	inline pointer allocate(size_type n) {
		return (pointer) memalign(N, n * sizeof(value_type));
	}

	inline void deallocate(pointer p, size_type) {
		free(p);
	}

	inline void construct(pointer p, const value_type & wert) {
		new (p) value_type(wert);
	}

	inline void destroy(pointer p) {
		p->~value_type();
	}

	inline size_type max_size() const throw () {
		return size_type(-1) / sizeof(value_type);
	}

	template <typename T2>
	struct rebind {
		typedef AlignmentAllocator<T2, N> other;
	};

	bool operator!=(const AlignmentAllocator<T, N>& other) const {
		return !(*this == other);
	}

	// Returns true if and only if storage allocated from *this
	// can be deallocated from other, and vice versa.
	// Always returns true for stateless allocators.
	bool operator==(const AlignmentAllocator<T, N>& other) const {
		return true;
	}
};