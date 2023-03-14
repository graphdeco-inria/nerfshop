#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_nerf.h>

#include <queue>

NGP_NAMESPACE_BEGIN

static __device__ float scalar_tp(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) {
	return a.dot(b.cross(c));
}

static __device__ Eigen::Vector4f bary_tet(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c, const Eigen::Vector3f& d, const Eigen::Vector3f& p) {
	Eigen::Vector3f vap = p - a;
	Eigen::Vector3f vbp = p - b;

	Eigen::Vector3f vab = b - a;
	Eigen::Vector3f vac = c - a;
	Eigen::Vector3f vad = d - a;

	Eigen::Vector3f vbc = c - b;
	Eigen::Vector3f vbd = d - b;
	// ScTP computes the scalar triple product
	float va6 = scalar_tp(vbp, vbd, vbc);
	float vb6 = scalar_tp(vap, vac, vad);
	float vc6 = scalar_tp(vap, vad, vab);
	float vd6 = scalar_tp(vap, vab, vac);
	float v6 = 1. / scalar_tp(vab, vac, vad);
	return Eigen::Vector4f(va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6);
}

template<typename float_t, class point_t>
inline __host__ __device__ bool same_side_tet(const point_t& v1, const point_t& v2, const point_t& v3, const point_t& v4, point_t& p) {
	point_t normal = (v2 - v1).cross(v3 - v1);
	float_t dotV4 = normal.dot(v4 -v1);
	float_t dotP = normal.dot(p - v1);
	return signbit(dotV4) == signbit(dotP);
}

template<typename float_t, class point_t>
__host__ __device__ bool point_in_tet(const point_t& v1, const point_t& v2, const point_t& v3, const point_t& v4, point_t& p) {
	return  same_side_tet<float_t, point_t>(v1, v2, v3, v4, p) &&
			same_side_tet<float_t, point_t>(v2, v3, v4, v1, p) &&
			same_side_tet<float_t, point_t>(v3, v4, v1, v2, p) &&
			same_side_tet<float_t, point_t>(v4, v1, v2, v3, p);  
}

bool is_boundary(const uint32_t cell_idx);

void add_neighbours(std::queue<uint32_t>& growing_queue, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t level);

void add_upper_levels(std::queue<uint32_t>& growing_queue, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t level);

uint32_t get_upper_cell_idx(const uint32_t cell_idx, const uint32_t target_level);

Eigen::Vector3f get_cell_pos(const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t level);

Eigen::Vector3i get_cell_at_pos(Eigen::Vector3f pos, const uint32_t level);

bool check_boundary(const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t cell_idx);

Eigen::Vector3f get_boundary_normal(const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t cell_idx);

std::vector<float> rescale_densities(const std::vector<float>& densities);

void update_selection_attributes(
	const std::vector<uint8_t>& selection_grid_bitfield, 
	const std::vector<float>& density_grid_host, 
	std::vector<Eigen::Vector3f>& selection_points, 
	std::vector<uint8_t>& selection_levels,
	std::vector<float>& selection_densities, 
	std::vector<float>& scaled_selection_densities, 
	std::vector<uint32_t>& selection_cell_idx);

enum class EOperatorMM : int {
	Max,
	Min
};

bool check_neighbour(const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t level, const EOperatorMM operator_mm);

NGP_NAMESPACE_END
