#include <neural-graphics-primitives/editing/tools/correct_mm_operations.h>
#include <neural-graphics-primitives/editing/tools/selection_utils.h>
#include <tiny-cuda-nn/common_device.h>


#ifdef NGP_GUI
#  include <imgui/imgui.h>
#  include <imgui/backends/imgui_impl_glfw.h>
#  include <imgui/backends/imgui_impl_opengl3.h>
#endif


NGP_NAMESPACE_BEGIN

void CubeSE::imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) {
    ImGui::SliderInt("SE Radius", &m_se_radius, 1, 10);
}

bool CubeSE::fit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const {
    for (int i = -m_se_radius; i <= m_se_radius; i++) {
        for (int j = -m_se_radius; j <= m_se_radius; j++) {
            for (int k = -m_se_radius; k <= m_se_radius; k++) {
                if(in_grid(x+i, y+j, z+k) && !get_bitfield_at(tcnn::morton3D(x+i, y+j, z+k), level, selection_grid_bitfield.data())) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool CubeSE::hit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const {
    for (int i = -m_se_radius; i <= m_se_radius; i++) {
        for (int j = -m_se_radius; j <= m_se_radius; j++) {
            for (int k = -m_se_radius; k <= m_se_radius; k++) {
                if(in_grid(x+i, y+j, z+k) && get_bitfield_at(tcnn::morton3D(x+i, y+j, z+k), level, selection_grid_bitfield.data())) {
                    return true;
                }
            }
        }
    }
    return false;
}

SphereSE::SphereSE(int se_radius) : m_se_radius{se_radius} {
    update_support();
}

void SphereSE::imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) {
    if(ImGui::SliderInt("SE Radius", &m_se_radius, 1, 10)) {
        update_support();
    }
}

bool SphereSE::fit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const {
    for (const auto& coord: m_se_support) {
        int i = std::get<0>(coord);
        int j = std::get<1>(coord);
        int k = std::get<2>(coord);
        if(in_grid(x+i, y+j, z+k) && !get_bitfield_at(tcnn::morton3D(x+i, y+j, z+k), level, selection_grid_bitfield.data())) {
            return false;
        }
    }
    return true;
}

bool SphereSE::hit (const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const int level) const {
    for (const auto& coord: m_se_support) {
        int i = std::get<0>(coord);
        int j = std::get<1>(coord);
        int k = std::get<2>(coord);
        if(in_grid(x+i, y+j, z+k) && get_bitfield_at(tcnn::morton3D(x+i, y+j, z+k), level, selection_grid_bitfield.data())) {
            return true;
        }
    }
    return false;
}

void SphereSE::update_support() {
    float se_radius_sq = m_se_radius * m_se_radius;
    for (int i = -m_se_radius; i <= m_se_radius; i++) {
        for (int j = -m_se_radius; j <= m_se_radius; j++) {
            for (int k = -m_se_radius; k <= m_se_radius; k++) {
                if (i*i+j*j+k*k <= se_radius_sq) {
                    m_se_support.emplace_back(i, k, j);
                }
            }
        }
    }
}

void CorrectMMOperations::imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) {
    if (ImGui::Combo("Dilation SE", (int*)&(m_dilation_se_type), SETypeStr)) {
        if (m_dilation_se_type == ESEType::Cube) {
            m_dilation_se = std::make_shared<CubeSE>(2);
        } else if (m_dilation_se_type == ESEType::Sphere) {
            m_dilation_se = std::make_shared<SphereSE>(2);
        }
    }
    ImGui::PushID(1);
    m_dilation_se->imgui(resolution, focal_length, 
    camera_matrix, screen_center);
    ImGui::PopID();
    if (ImGui::Combo("Erosion SE", (int*)&(m_erosion_se_type), SETypeStr)) {
        if (m_erosion_se_type == ESEType::Cube) {
            m_erosion_se = std::make_shared<CubeSE>(2);
        } else if (m_erosion_se_type == ESEType::Sphere) {
            m_erosion_se = std::make_shared<SphereSE>(2);
        }
    }
    ImGui::PushID(2);
    m_erosion_se->imgui(resolution, focal_length, 
    camera_matrix, screen_center);
    ImGui::PopID();
}

std::vector<uint8_t> CorrectMMOperations::dilate(
    const std::vector<uint8_t>& selection_grid_bitfield, 
    int growing_level, 
    std::vector<Eigen::Vector3f>& selected_points, std::vector<uint32_t>& selected_cell_idx
) {

    if (!m_dilation_se) {
        std::cout << "No dilation SE set..." << std::endl;
        return selection_grid_bitfield;
    }

    // Copy the input bitfield
    std::vector<uint8_t> dilated_bitfield = std::vector<uint8_t>(selection_grid_bitfield.size(), 0);
    const uint32_t level = growing_level;

    selected_points.clear();
    selected_cell_idx.clear();

    for (uint32_t x = 0; x < NERF_GRIDSIZE(); x++) {
        for (uint32_t y = 0; y < NERF_GRIDSIZE(); y++) {
            for (uint32_t z = 0; z < NERF_GRIDSIZE(); z++) {
                uint32_t pos_idx = tcnn::morton3D(x, y, z);
                uint32_t current_cell = level * NERF_GRIDVOLUME() + pos_idx;
                if (m_dilation_se->hit(selection_grid_bitfield, x, y, z, level)) {
                    Eigen::Vector3f pos = get_cell_pos(x, y, z, level);
                    selected_points.push_back(pos);
                    selected_cell_idx.push_back(current_cell);
                    set_bitfield_at(pos_idx, level, true, dilated_bitfield.data());
                }
            }
        }
    }

    return dilated_bitfield;
}

std::vector<uint8_t> CorrectMMOperations::erode(
    const std::vector<uint8_t>& selection_grid_bitfield, 
    int growing_level, 
    std::vector<Eigen::Vector3f>& selected_points, std::vector<uint32_t>& selected_cell_idx
) {

    if (!m_erosion_se) {
        std::cout << "No erosion SE set..." << std::endl;
        return selection_grid_bitfield;
    }

    // Copy the input bitfield
    std::vector<uint8_t> eroded_bitfield = std::vector<uint8_t>(selection_grid_bitfield.size(), 0);
    const uint32_t level = growing_level;

    selected_points.clear();
    selected_cell_idx.clear();

    for (uint32_t x = 0; x < NERF_GRIDSIZE(); x++) {
        for (uint32_t y = 0; y < NERF_GRIDSIZE(); y++) {
            for (uint32_t z = 0; z < NERF_GRIDSIZE(); z++) {
                uint32_t pos_idx = tcnn::morton3D(x, y, z);
                uint32_t current_cell = level * NERF_GRIDVOLUME() + pos_idx;
                if (m_erosion_se->fit(selection_grid_bitfield, x, y, z, level)) {
                    Eigen::Vector3f pos = get_cell_pos(x, y, z, level);
                    selected_points.push_back(pos);
                    selected_cell_idx.push_back(current_cell);
                    set_bitfield_at(pos_idx, level, true, eroded_bitfield.data());
                }
            }
        }
    }

    return eroded_bitfield;
}


NGP_NAMESPACE_END
