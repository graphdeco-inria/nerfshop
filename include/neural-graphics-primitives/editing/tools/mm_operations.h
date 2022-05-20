#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/common.h>

#include <vector>
#include <queue>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

class MMOperations {
public:
    MMOperations() = default;

    virtual void imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) {}

    virtual std::vector<uint8_t> dilate(
        const std::vector<uint8_t>& selection_grid_bitfield, 
        int growing_level, 
        std::vector<Eigen::Vector3f>& selected_points,
        std::vector<uint32_t>& selected_cell_idx
    ) = 0;

    virtual std::vector<uint8_t> erode(
        const std::vector<uint8_t>& selection_grid_bitfield, 
        int growing_level, 
        std::vector<Eigen::Vector3f>& selected_points,
        std::vector<uint32_t>& selected_cell_idx
    ) = 0;
};

NGP_NAMESPACE_END