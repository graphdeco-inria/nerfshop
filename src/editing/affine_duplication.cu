#include <neural-graphics-primitives/editing/affine_duplication.h>
#include <neural-graphics-primitives/editing/tools/visualization_utils.h>
#include <neural-graphics-primitives/editing/distiller.h>

NGP_NAMESPACE_BEGIN

__device__ Vector3f warp_direction_ad(const Vector3f& dir) {
	return (dir + Vector3f::Ones()) * 0.5f;
}

__device__ Vector3f unwarp_direction_ad(const Vector3f& dir) {
	return dir * 2.0f - Vector3f::Ones();
}


class AffineDuplicationDistiller : public Distiller
{
public:

	BoundingBox aabb;
	AffineBoundingBox destination_bbox;
	AffineBoundingBox original_bbox;
	Eigen::Vector3f scale;
	Eigen::Vector3f translation;
	Eigen::Matrix3f rotation;

	virtual __device__ bool in_source(Eigen::Vector3f& coord, bool warp) override
	{
		Eigen::Vector3f unwarped_pos = coord;
		if (warp)
			unwarped_pos = unwarp_position(coord, aabb);

		if (original_bbox.contains(coord))
		{
			return true;
		}
		return false;
	}

	virtual __device__ bool in_target(Eigen::Vector3f& coord, bool warp) override
	{
		Eigen::Vector3f unwarped_pos = coord;
		if (warp)
			unwarped_pos = unwarp_position(coord, aabb);

		if (destination_bbox.contains(coord))
		{
			return true;
		}
		return false;
	}

	virtual __device__ Vector3f map(Eigen::Vector3f coord, bool warp) override
	{
		Eigen::Vector3f unwarped_pos = coord;
		if (warp)
			unwarped_pos = unwarp_position(coord, aabb);

		if (original_bbox.contains(coord)) {
			// First, remap to center to perform the rotation
			coord = rotation * (coord - original_bbox.center).cwiseProduct(scale) + original_bbox.center;
			// Then translate
			coord += translation;
		}
		return coord;
	};
};

__global__ void translate_in_box_pos(
	const uint32_t n_elements,
	tcnn::PitchedPtr<NerfPosition> nerf_coords,
    bool* __restrict__ empty_mask,
    const AffineBoundingBox selection_bbox,
    const AffineBoundingBox destination_bbox,
	const Eigen::Vector3f translation,
    const Eigen::Vector3f scale,
    const Eigen::Matrix3f rotation
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    if (destination_bbox.contains(nerf_coords(i)->p)) {
        // First, remap to center to perform the rotation
        nerf_coords(i)->p = rotation.transpose()* (nerf_coords(i)->p-destination_bbox.center).cwiseQuotient(scale)+destination_bbox.center;
        // Then translate
        nerf_coords(i)->p -= translation;
    } else if (empty_mask && selection_bbox.contains(nerf_coords(i)->p)) {
        empty_mask[i] = true;
    }
}

__global__ void translate_in_box(
	const uint32_t n_elements,
	tcnn::PitchedPtr<NerfCoordinate> nerf_coords,
    bool* __restrict__ empty_mask,
    const AffineBoundingBox selection_bbox,
    const AffineBoundingBox destination_bbox,
	const Eigen::Vector3f translation,
    const Eigen::Vector3f scale,
    const Eigen::Matrix3f rotation,
    const bool correct_dir
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    if (destination_bbox.contains(nerf_coords(i)->pos.p)) {
        // First, remap to center
        nerf_coords(i)->pos.p = rotation.transpose() * (nerf_coords(i)->pos.p-destination_bbox.center).cwiseQuotient(scale)+destination_bbox.center;
        // Then translate
        nerf_coords(i)->pos.p -= translation;
        // Rotate the direction as well (if requested)
        if (correct_dir) {
            nerf_coords(i)->dir.d = warp_direction_ad(rotation.transpose() * unwarp_direction_ad(nerf_coords(i)->dir.d));
        }
    } else if (empty_mask && selection_bbox.contains(nerf_coords(i)->pos.p)) {
        empty_mask[i] = true;
    }
}

int AffineDuplication::ID = 0;

void AffineDuplication::map_rays(cudaStream_t stream, tcnn::PitchedPtr<NerfCoordinate> nerf_coords, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const {
    
    // Perform the backward translation
    tcnn::linear_kernel(translate_in_box, 0, stream,
        n_elements,
        nerf_coords,
        m_hide_original ? empty_mask.data() : nullptr,
        m_warped_selection_box,
        m_warped_destination_box,
        m_warped_translation,
        m_scale,
        m_rotation_matrix,
        m_correct_dir
    );
}

void AffineDuplication::map_positions(cudaStream_t stream, tcnn::PitchedPtr<NerfPosition> nerf_pos, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const {
    
    // Perform the backward translation
    tcnn::linear_kernel(translate_in_box_pos, 0, stream,
        n_elements,
        nerf_pos,
        m_hide_original ? empty_mask.data() : nullptr,
        m_warped_selection_box,
        m_warped_destination_box,
        m_warped_translation,
        m_scale,
        m_rotation_matrix
    );
}

#ifdef NGP_GUI

bool AffineDuplication::imgui(bool& delete_operator, const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center, bool& auto_clean) {
    bool update_transformation = false;
    if (ImGui::CollapsingHeader("Affine Duplication", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Hide original", &m_hide_original);
        ImGui::SameLine();
        update_transformation |= ImGui::Checkbox("Correct direction", &m_correct_dir);
        // ImGui::SliderFloat("Translation scale", &m_translation_scale, 0.f, m_scene_aabb.diag().minCoeff());
        // ImGui::SliderFloat("Scaling factor x", &m_scaling_vector.x(), 1.0f, 10.f);
        // ImGui::SliderFloat("Scaling factor y", &m_scaling_vector.y(), 1.0f, 10.f);
        // ImGui::SliderFloat("Scaling factor z", &m_scaling_vector.z(), 1.0f, 10.f);

        // Guizmo control
        if (ImGui::RadioButton("Translate", m_gizmo_op == ImGuizmo::TRANSLATE))
            m_gizmo_op = ImGuizmo::TRANSLATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", m_gizmo_op == ImGuizmo::ROTATE))
            m_gizmo_op = ImGuizmo::ROTATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", m_gizmo_op == ImGuizmo::SCALE))
            m_gizmo_op = ImGuizmo::SCALE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Local", m_gizmo_mode == ImGuizmo::LOCAL))
            m_gizmo_mode = ImGuizmo::LOCAL;
        ImGui::SameLine();
        if (ImGui::RadioButton("World", m_gizmo_mode == ImGuizmo::WORLD))
            m_gizmo_mode = ImGuizmo::WORLD;

        if (ImGui::RadioButton("Selection Box", m_box_mode == BoxMode::SELECTION_BOX))
            m_box_mode = BoxMode::SELECTION_BOX;
        ImGui::SameLine();
        if (ImGui::RadioButton("Destination Box", m_box_mode == BoxMode::DESTINATION_BOX))
            m_box_mode = BoxMode::DESTINATION_BOX;
        ImGui::Checkbox("Display warped bbox", &m_display_warped);
        
        // Delete operator
        if (imgui_colored_button2("Delete Operator", 0)) {
            delete_operator = true;
        }
    }
    return update_transformation | delete_operator;
}

__global__ void constructDuplicationDistiller(
	AffineDuplicationDistiller* distiller,
	BoundingBox aabb,
	AffineBoundingBox bbox,
	AffineBoundingBox original_bbox,
	Eigen::Vector3f translation,
	Eigen::Vector3f scale,
	Eigen::Matrix3f rotation
)
{
	new (distiller) AffineDuplicationDistiller();
	distiller->emptying = false;
	distiller->aabb = aabb;
	distiller->destination_bbox = bbox;
	distiller->original_bbox = original_bbox;
	distiller->translation = translation;
	distiller->scale = scale;
	distiller->rotation = rotation;
}

Distiller* AffineDuplication::getDistiller()
{
	if (gpu_distiller == nullptr)
	{
		cudaMalloc(&gpu_distiller, sizeof(AffineDuplicationDistiller));
	}
	constructDuplicationDistiller << <1, 1 >> > (
		(AffineDuplicationDistiller*)gpu_distiller,
		m_scene_aabb,
		m_destination_box,
		m_selection_box,
		m_translation,
		m_scale,
		m_rotation_matrix
		);
	return gpu_distiller;
}

bool AffineDuplication::handle_keyboard() {
    // Handle axis-aligned translation
    // if (ImGui::IsKeyDown(ImGuiKey_PageUp)) {
    //     m_translation += m_translation_scale * Eigen::Vector3f{0.f, 1.f, 0.f};
    //     update_destination();
    //     return true;
    // }
    // if (ImGui::IsKeyDown(ImGuiKey_PageDown)) {
    //     m_translation += m_translation_scale * Eigen::Vector3f{0.f, -1.f, 0.f};
    //     update_destination();
    //     return true;
    // }
    // if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) {
    //     m_translation += m_translation_scale * Eigen::Vector3f{0.f, 0.f, 1.f};
    //     update_destination();
    //     return true;
    // }
    // if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) {
    //     m_translation += m_translation_scale * Eigen::Vector3f{0.f, 0.f, -1.f};
    //     update_destination();
    //     return true;
    // }
    // if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) {
    //     m_translation += m_translation_scale * Eigen::Vector3f{-1.f, 0.f, 0.f};
    //     update_destination();
    //     return true;
    // }
    // if (ImGui::IsKeyDown(ImGuiKey_RightArrow)) {
    //     m_translation += m_translation_scale * Eigen::Vector3f{1.f, 0.f, 0.f};
    //     update_destination();
    //     return true;
    // }
    // if (ImGui::IsKeyDown('U')) {
    //     if (m_box_mode == BoxMode::SELECTION_BOX)
    //         m_selection_box.scale(m_scaling_vector);
    //     if (m_box_mode == BoxMode::DESTINATION_BOX)
    //         m_scale = m_scale.cwiseProduct(m_scaling_vector);
    //     update_destination();
    //     return true;
    // }
    // if (ImGui::IsKeyDown('I')) {
    //     if (m_box_mode == BoxMode::SELECTION_BOX)
    //         m_selection_box.scale(m_scaling_vector.cwiseInverse());
    //     if (m_box_mode == BoxMode::DESTINATION_BOX)
    //         m_scale = m_scale.cwiseQuotient(m_scaling_vector);
    //     update_destination();
    //     return true;
    // }
    // if (ImGui::IsKeyDown('Y')) {
    //     m_rotation_matrix = m_rotation_transform * m_rotation_matrix;
    //     update_destination();
    //     return true;
    // }
    return false;
}

bool AffineDuplication::visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time) {
    ImDrawList* list = ImGui::GetForegroundDrawList();

    visualize_bbox(world2proj, list, m_selection_box, 0xff0000ff);
    visualize_bbox(world2proj, list, m_destination_box, 0xffff00ff);

    if (m_display_warped) {
        visualize_bbox(world2proj, list, m_warped_selection_box, 0xff0000ff);
        visualize_bbox(world2proj, list, m_warped_destination_box, 0xffff00ff);
    }

    // Guizmo visualization and editing
    bool edited_guizmo = false;
    float flx = focal.x();
    float fly = focal.y();
    Matrix<float, 4, 4> view2proj_guizmo;
    float zfar = 100.f;
    float znear = 0.1f;
    view2proj_guizmo <<
        fly*2.f/aspect, 0, 0, 0,
        0, -fly*2.f, 0, 0,
        0, 0, (zfar+znear)/(zfar-znear), -(2.f*zfar*znear) / (zfar-znear),
        0, 0, 1, 0;

    ImGuiIO& io = ImGui::GetIO();
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

    Eigen::Matrix4f edit_matrix;
    if (m_box_mode == BoxMode::SELECTION_BOX) {
        compose_imguizmo_matrix<Eigen::Matrix3f, Eigen::Vector3f, float>(edit_matrix, m_selection_box.rot_matrix, m_selection_box.center, m_selection_box.scale);
        
        if (ImGuizmo::Manipulate((const float*)&world2view, (const float*)&view2proj_guizmo, (ImGuizmo::OPERATION)m_gizmo_op, (ImGuizmo::MODE)m_gizmo_mode, (float*)&edit_matrix, NULL, NULL)) {
            edited_guizmo = true;
            Eigen::Matrix3f guizmo_rotation;
            Eigen::Vector3f guizmo_translation;
            Eigen::Vector3f guizmo_scale;
            decompose_imguizmo_matrix<Eigen::Matrix3f, Eigen::Vector3f, float>(edit_matrix, guizmo_rotation, guizmo_translation, guizmo_scale);
            m_selection_box.set_center(guizmo_translation);
            m_selection_box.set_rotation(guizmo_rotation);
            m_selection_box.set_scale(guizmo_scale);
        }
    } else if (m_box_mode == BoxMode::DESTINATION_BOX) {
        compose_imguizmo_matrix<Eigen::Matrix3f, Eigen::Vector3f, float>(edit_matrix, m_destination_box.rot_matrix, m_destination_box.center, m_destination_box.scale);
        
        if (ImGuizmo::Manipulate((const float*)&world2view, (const float*)&view2proj_guizmo, (ImGuizmo::OPERATION)m_gizmo_op, (ImGuizmo::MODE)m_gizmo_mode, (float*)&edit_matrix, NULL, NULL)) {
            edited_guizmo = true;
            Eigen::Matrix3f guizmo_rotation;
            Eigen::Vector3f guizmo_translation;
            Eigen::Vector3f guizmo_scale;
            decompose_imguizmo_matrix<Eigen::Matrix3f, Eigen::Vector3f, float>(edit_matrix, guizmo_rotation, guizmo_translation, guizmo_scale);
            m_translation = guizmo_translation - m_selection_box.center;
            m_rotation_matrix = guizmo_rotation * m_selection_box.rot_matrix.transpose();
            // std::cout << m_rotation_matrix.determinat() << std::endl;
            m_scale = guizmo_scale.cwiseQuotient(m_selection_box.scale);
        }
    }
    
    if (edited_guizmo) {
        update_destination();
    }

    return edited_guizmo;
}
#endif

nlohmann::json AffineDuplication::to_json() {
    nlohmann::json j;
    j["type"] = "affine_duplication";

    j["selection_box"] = m_selection_box;

    j["translation"] = m_translation;
    j["scale"] = m_scale;
    j["rotation_matrix"] = m_rotation_matrix;
    j["hide_original"] = m_hide_original;
    j["correct_dir"] = m_correct_dir;

    return j;
}


NGP_NAMESPACE_END
