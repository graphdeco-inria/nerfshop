#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/editing/edit_operator.h>
#include <neural-graphics-primitives/json_binding.h>

#ifdef NGP_GUI
#include <neural-graphics-primitives/editing/tools/visualization_utils.h>
#include <neural-graphics-primitives/camera_path.h>

#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>
#endif

NGP_NAMESPACE_BEGIN

enum BoxMode {
    SELECTION_BOX,
    DESTINATION_BOX
};

class AffineDuplication : public EditOperator {
public:

    AffineDuplication(AffineBoundingBox selection_box, Eigen::Vector3f translation, BoundingBox aabb) : m_selection_box{selection_box}, m_translation{translation}, m_scene_aabb{aabb}, m_scale{1.0f, 1.0f, 1.0f}, m_rotation_matrix{Eigen::Matrix3f::Identity()}, m_id{++ID} {
        
        update_destination();
    }

    AffineDuplication(nlohmann::json operator_json, BoundingBox aabb) : m_scene_aabb{aabb}, m_id{++ID} {
        from_json(operator_json["selection_box"], m_selection_box);
        from_json(operator_json["translation"], m_translation);
        from_json(operator_json["scale"], m_scale);
        from_json(operator_json["rotation_matrix"], m_rotation_matrix);
        m_hide_original = operator_json["hide_original"];
        m_correct_dir = operator_json["correct_dir"];

        update_destination();
    }

    void map_rays(cudaStream_t stream, tcnn::PitchedPtr<NerfCoordinate> nerf_coords, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const override;

    void map_positions(cudaStream_t stream, tcnn::PitchedPtr<NerfPosition> nerf_pos, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const override;

#ifdef NGP_GUI

    bool imgui(bool& delete_operator, const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center, bool& auto_clean) override;

    bool handle_keyboard() override;

    bool visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time) override;
#endif

    void draw_gl(
        const Eigen::Vector2i& resolution, 
        const Eigen::Vector2f& focal_length, 
        const Eigen::Matrix<float, 3, 4>& camera_matrix, 
        const Eigen::Vector2f& screen_center
    ) override {}

    nlohmann::json to_json() override;

	Distiller* gpu_distiller = nullptr;

	virtual Distiller* getDistiller() override;

private:
    static int ID;
    int m_id;

    void update_destination() {
        m_destination_box = m_selection_box;
        m_destination_box.translate(m_translation);
        m_destination_box.scale_with_vector(m_scale);
        m_destination_box.rotate(m_rotation_matrix);

        m_warped_destination_box = m_destination_box;
        m_warped_destination_box.warp_box(m_scene_aabb);
        m_warped_translation = m_translation.cwiseQuotient(m_scene_aabb.diag());

        m_warped_selection_box = m_selection_box;
        m_warped_selection_box.warp_box(m_scene_aabb);
    }

    BoundingBox m_scene_aabb;
    AffineBoundingBox m_selection_box;
    AffineBoundingBox m_destination_box;
    AffineBoundingBox m_warped_destination_box;
    AffineBoundingBox m_warped_selection_box;
    Eigen::Vector3f m_translation;
    Eigen::Vector3f m_warped_translation;
    Eigen::Vector3f m_scale;
    Eigen::Matrix3f m_rotation_matrix;
    bool m_hide_original = false;
    bool m_correct_dir = true;

    // GUI
    bool m_display_warped = false; // DEBUG only
    float m_translation_scale = 0.1f;
    Eigen::Vector3f m_scaling_vector = Eigen::Vector3f(1.1f, 1.1f, 1.1f);
    Eigen::Matrix3f m_rotation_transform = Eigen::AngleAxisf(10.f * M_PI / 180.f, Vector3f::UnitY()).toRotationMatrix();

    ImGuizmo::MODE m_gizmo_mode = ImGuizmo::LOCAL;
	ImGuizmo::OPERATION m_gizmo_op = ImGuizmo::TRANSLATE;

    BoxMode m_box_mode = BoxMode::SELECTION_BOX;
};

NGP_NAMESPACE_END
