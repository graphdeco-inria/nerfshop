// #pragma once

// #include <neural-graphics-primitives/common.h>
// #include <neural-graphics-primitives/bounding_box.cuh>
// #include <neural-graphics-primitives/editing/tools/affine_bounding_box.cuh>
// #include <neural-graphics-primitives/editing/edit_operator.h>
// #include <neural-graphics-primitives/json_binding.h>

// #ifdef NGP_GUI
// #include <neural-graphics-primitives/editing/tools/visualization_utils.h>
// #include <neural-graphics-primitives/camera_path.h>

// #include <imgui/imgui.h>
// #include <imguizmo/ImGuizmo.h>
// #endif

// NGP_NAMESPACE_BEGIN

// class TwistOperator : public EditOperator {
// public:

//     TwistOperator(AffineBoundingBox selection_box, float angle, BoundingBox aabb, Eigen::Vector3f twist_axis = Eigen::Vector3f(0.f, 1.0f, 0.f)) : m_selection_box{selection_box}, m_angle{angle}, m_scene_aabb{aabb}, m_twist_axis{twist_axis} {
//         update_warped();
//     }

//     TwistOperator(nlohmann::json operator_json, BoundingBox aabb) : m_scene_aabb{aabb} {
//         from_json(operator_json["selection_box"], m_selection_box);
//         from_json(operator_json["twist_axis"], m_twist_axis);
//         m_angle = operator_json["angle"];

//         update_warped();
//     }

//     void map_rays(cudaStream_t stream, tcnn::PitchedPtr<NerfCoordinate> nerf_coords, tcnn::GPUMatrixDynamic<bool>& density_mask, uint32_t n_elements) const override;

    
//     void map_positions(cudaStream_t stream, tcnn::PitchedPtr<NerfPosition> nerf_pos, tcnn::GPUMatrixDynamic<bool>& density_mask, uint32_t n_elements) const override;

// #ifdef NGP_GUI

//     bool imgui(bool& delete_operator, const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) override;

//     bool handle_keyboard() override;

//     bool visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time) override;
// #endif

//     void draw_gl (
//         const Eigen::Vector2i& resolution, 
//         const Eigen::Vector2f& focal_length, 
//         const Eigen::Matrix<float, 3, 4>& camera_matrix, 
//         const Eigen::Vector2f& screen_center
//     ) override {}

//     nlohmann::json to_json() override;

// private:

//     void update_warped() {
//         m_warped_selection_box = m_selection_box;
//         m_warped_selection_box.warp_box(m_scene_aabb);
//     }

//     BoundingBox m_scene_aabb;
//     AffineBoundingBox m_selection_box;
//     AffineBoundingBox m_warped_selection_box;
//     float m_angle;
//     Eigen::Vector3f m_twist_axis;

//     // GUI
//     bool m_header_visible = true;

//     ImGuizmo::MODE m_gizmo_mode = ImGuizmo::LOCAL;
// 	ImGuizmo::OPERATION m_gizmo_op = ImGuizmo::TRANSLATE;
// };

// NGP_NAMESPACE_END
