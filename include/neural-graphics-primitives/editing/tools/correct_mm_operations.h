#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_nerf.h>
#include <neural-graphics-primitives/editing/tools/mm_operations.h>

#include <tiny-cuda-nn/common.h>

#include <vector>
#include <queue>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

enum class ESEType : int {
	Cube,
	Sphere,
};

static constexpr const char* SETypeStr = "Cube\0Sphere\0\0";

inline bool in_grid (int x, int y, int z) {
    return x >= 0 && y >= 0 && z >= 0 && x < NERF_GRIDSIZE() && y < NERF_GRIDSIZE() && z < NERF_GRIDSIZE();
}

class StructuringElement {
public:
    virtual void imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) {}

    virtual bool fit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const = 0;

    virtual bool hit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const = 0;
};

class CubeSE : public StructuringElement {
public:
    CubeSE(int se_radius) : m_se_radius{se_radius} {}

    void imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) override;

    bool fit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const override;

    bool hit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const override;

private:
    int m_se_radius;
};

class SphereSE : public StructuringElement {
public:
    SphereSE(int se_radius);

    void imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) override;

    bool fit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const override;

    bool hit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const override;

private:
    void update_support();
    int m_se_radius;
    std::vector<std::tuple<int, int, int>> m_se_support;
};


class CorrectMMOperations : public MMOperations {
public:
    CorrectMMOperations() : m_dilation_se{new CubeSE(2)}, m_erosion_se{new SphereSE(2)} {}

    void imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center);

    std::vector<uint8_t> dilate(
        const std::vector<uint8_t>& selection_grid_bitfield, 
        int growing_level, 
        std::vector<Eigen::Vector3f>& selected_points, std::vector<uint32_t>& selected_cell_idx
    ) override;

    std::vector<uint8_t> erode(
        const std::vector<uint8_t>& selection_grid_bitfield, 
        int growing_level, 
        std::vector<Eigen::Vector3f>& selected_points, std::vector<uint32_t>& selected_cell_idx
    ) override;

private:
    ESEType m_dilation_se_type = ESEType::Cube;
    ESEType m_erosion_se_type = ESEType::Sphere;
    std::shared_ptr<StructuringElement> m_dilation_se;
    std::shared_ptr<StructuringElement> m_erosion_se;
};

NGP_NAMESPACE_END