#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/nerf.h>
#include <math.h>

using namespace Eigen;

NGP_NAMESPACE_BEGIN

typedef Eigen::Matrix<float, 16, 1> FeatureVector;
typedef Eigen::Matrix<double, 16, 1> FeatureVectorFp;

// size of the density/occupancy grid in number of cells along an axis.
inline constexpr __device__ uint32_t NERF_GRIDSIZE() {
	return 128;
}

inline constexpr __device__ uint32_t NERF_GRIDVOLUME() {
	return NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE();
}

inline constexpr __device__ float NERF_RENDERING_NEAR_DISTANCE() { return 0.05f; }
inline constexpr __device__ uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length
inline constexpr __device__ uint32_t NERF_CASCADES() { return 5; }

inline constexpr __device__ float SQRT3() { return 1.73205080757f; }
inline constexpr __device__ float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch
inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __device__ float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

__host__ __device__ Eigen::Vector3f warp_position(const Eigen::Vector3f& pos, const BoundingBox& aabb);

__host__ __device__ Eigen::Vector3f unwarp_position(const Eigen::Vector3f& pos, const BoundingBox& aabb);

// Applied to make sure that direction coordinates stay between 0 and 1
__host__ __device__ Vector3f warp_direction(const Vector3f& dir);

__host__ __device__ Vector3f unwarp_direction(const Vector3f& dir);

__device__ float warp_dt(float dt);

__device__ float unwarp_dt(float dt);

__device__ float network_to_rgb(float val, ENerfActivation activation);

__device__ Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, ENerfActivation activation);

__device__ float network_to_density(float val, ENerfActivation activation);

__device__ FeatureVector network_to_feature(const tcnn::vector_t<tcnn::network_precision_t, 16>& local_network_output);

__host__ __device__ uint32_t grid_mip_offset(uint32_t mip);

__host__ __device__ float calc_cone_angle(float cosine, const Eigen::Vector2f& focal_length, float cone_angle_constant);

__host__ __device__ float calc_dt(float t, float cone_angle);

__device__ float distance_to_next_voxel(const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res);

__device__ float advance_to_next_voxel(float t, float cone_angle, const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res);

__device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip);

__device__ bool density_grid_occupied_at(const Vector3f& pos, const uint8_t* density_grid_bitfield, uint32_t mip);

__host__ __device__ bool get_bitfield_at(const uint32_t cell_idx, const uint32_t level, const uint8_t* bitfield);

__host__ __device__ void set_bitfield_at(const uint32_t cell_idx, const uint32_t level, const bool value, uint8_t* bitfield);

__device__ float cascaded_grid_at(Vector3f pos, const float* cascaded_grid, uint32_t mip);

__device__ float& cascaded_grid_at(Vector3f pos, float* cascaded_grid, uint32_t mip);

__device__ int mip_from_pos(const Vector3f& pos);

__device__ int mip_from_dt(float dt, const Vector3f& pos);

__global__ void generate_grid_samples_nerf_nonuniform(const uint32_t n_elements, default_rng_t rng, const uint32_t step, BoundingBox aabb, const float* __restrict__ grid_in, NerfPosition* __restrict__ out, uint32_t* __restrict__ indices, uint32_t n_cascades, float thresh);

NetworkDims network_dims_nerf();

__host__ __device__ Eigen::Vector3f evaluate_sh9(const SH9RGB& sh, const Eigen::Vector3f dir);

__host__ __device__ float evaluate_sh9(const SH9Scalar& sh, const Eigen::Vector3f dir);

// Rotate using ZXZXZ factorization (112 multiplications in theory)
__host__ __device__ SH9Scalar rotate_sh9_scalar_zxzxz(const SH9Scalar& sh, const Matrix3f& R);

__host__ __device__ SH9RGB rotate_sh9_zxzxz(const SH9RGB& sh, const Matrix3f& R);

// Rotate using the method in: http://filmicworlds.com/blog/simple-and-fast-spherical-harmonic-rotation/
__host__ __device__ SH9Scalar rotate_sh9_scalar_fast(const SH9Scalar& sh, const Matrix3f& R);

__host__ __device__ SH9RGB rotate_sh9_fast(const SH9RGB& sh, const Matrix3f& R);

__host__ __device__ SH9Scalar diffuse_reflection_sh9_scalar(const SH9Scalar& L);

__host__ __device__ SH9RGB diffuse_reflection_sh9(const SH9RGB& L);

// See: https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
__host__ __device__ Eigen::Vector3f evaluate_irradiance_sh9(const SH9RGB& L, const Eigen::Vector3f& dir);

NGP_NAMESPACE_END
