#include <neural-graphics-primitives/editing/tools/default_mm_operations.h>
#include <neural-graphics-primitives/editing/tools/selection_utils.h>
#include <neural-graphics-primitives/common_nerf.h>
// #include <tiny-cuda-nn/common_device.h>

#ifdef NGP_GUI
#  include <imgui/imgui.h>
#  include <imgui/backends/imgui_impl_glfw.h>
#  include <imgui/backends/imgui_impl_opengl3.h>
#endif

NGP_NAMESPACE_BEGIN

void DefaultMMOperations::imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) {
    ImGui::SliderInt("Dilation steps", &m_dilation_steps, 0, 20, "%d");
}

std::vector<uint8_t> DefaultMMOperations::dilate(
    const std::vector<uint8_t>& selection_grid_bitfield, 
    int growing_level, 
    std::vector<Eigen::Vector3f>& selected_points, std::vector<uint32_t>& selected_cell_idx
) {

    // Copy the input bitfield
    std::vector<uint8_t> dilated_bitfield = selection_grid_bitfield;
    const uint32_t level = growing_level;

    for (int i = 0; i < m_dilation_steps; i++) {
        std::vector<uint32_t> new_cells;

        for (uint32_t x = 0; x < NERF_GRIDSIZE(); x++) {
            for (uint32_t y = 0; y < NERF_GRIDSIZE(); y++) {
                for (uint32_t z = 0; z < NERF_GRIDSIZE(); z++) {
                    uint32_t pos_idx = tcnn::morton3D(x, y, z);
                    uint32_t current_cell = level * NERF_GRIDVOLUME() + pos_idx;
                    // Make sure it as not already been set!
                    if (!get_bitfield_at(pos_idx, level, dilated_bitfield.data()) && check_neighbour(dilated_bitfield, x, y, z, level, EOperatorMM::Max)) {
                        Eigen::Vector3f pos = get_cell_pos(x, y, z, level);
                        selected_points.push_back(pos);
                        selected_cell_idx.push_back(current_cell);
                        new_cells.push_back(pos_idx);
                    }
                }
            }
        }

        // Update the bitfield with the new cells
        for (int j = 0; j < new_cells.size(); j++) {
            set_bitfield_at(new_cells[j], growing_level, true, dilated_bitfield.data());
        }
    }

    return dilated_bitfield;
}

std::vector<uint8_t> DefaultMMOperations::erode(
    const std::vector<uint8_t>& selection_grid_bitfield, 
    int growing_level, 
    std::vector<Eigen::Vector3f>& selected_points, std::vector<uint32_t>& selected_cell_idx
) {
    // Copy the input bitfield
    std::vector<uint8_t> eroded_bitfield = selection_grid_bitfield;
    const uint32_t level = growing_level;

    for (int i = 0; i < m_dilation_steps; i++) {
        std::vector<uint32_t> new_cells;

        for (uint32_t x = 0; x < NERF_GRIDSIZE(); x++) {
            for (uint32_t y = 0; y < NERF_GRIDSIZE(); y++) {
                for (uint32_t z = 0; z < NERF_GRIDSIZE(); z++) {
                    uint32_t pos_idx = tcnn::morton3D(x, y, z);
                    uint32_t current_cell = level * NERF_GRIDVOLUME() + pos_idx;
                    // Make sure it as not already been set!
                    if (get_bitfield_at(pos_idx, level, eroded_bitfield.data()) && check_neighbour(eroded_bitfield, x, y, z, level, EOperatorMM::Min)) {
                        Eigen::Vector3f pos = get_cell_pos(x, y, z, level);
                        new_cells.push_back(pos_idx);
                    }
                }
            }
        }

        // Update the bitfield with the new cells
        for (int j = 0; j < new_cells.size(); j++) {
            set_bitfield_at(new_cells[j], growing_level, false, eroded_bitfield.data());
        }
    }

    // Update selected points and selected cells accordingly
    selected_points.clear();
    selected_cell_idx.clear();

    for (uint32_t x = 0; x < NERF_GRIDSIZE(); x++) {
        for (uint32_t y = 0; y < NERF_GRIDSIZE(); y++) {
            for (uint32_t z = 0; z < NERF_GRIDSIZE(); z++) {
                uint32_t pos_idx = tcnn::morton3D(x, y, z);
                uint32_t current_cell = level * NERF_GRIDVOLUME() + pos_idx;
                // Make sure it as not already been set!
                if (get_bitfield_at(pos_idx, level, eroded_bitfield.data())) {
                    Eigen::Vector3f pos = get_cell_pos(x, y, z, level);
                    selected_points.push_back(pos);
                    selected_cell_idx.push_back(current_cell);
                }
            }
        }
    }

    return eroded_bitfield;
}

NGP_NAMESPACE_END