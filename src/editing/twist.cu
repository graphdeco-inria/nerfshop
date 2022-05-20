#include <neural-graphics-primitives/editing/twist.h>

NGP_NAMESPACE_BEGIN

__device__ void get_rotation_matrix(
    const Eigen::Vector3f& point, 
    const AffineBoundingBox& bbox, 
    const float angle, 
    const Eigen::Vector3f& twist_axis,
    Eigen::Matrix3f& rot_matrix) {

    // Get the amplitude in the chosen direction of the twist
    Eigen::Vector3f rescaled_pt = point - (bbox.center - bbox.diag().dot(twist_axis) / 2.f * twist_axis);

    float t = rescaled_pt.dot(twist_axis) / bbox.diag().dot(twist_axis);
    rot_matrix = AngleAxisf(-angle*t, twist_axis).toRotationMatrix();
}

__global__ void twist_kernel(
	const uint32_t n_elements,
	tcnn::PitchedPtr<NerfCoordinate> nerf_coords,
    const AffineBoundingBox bbox,
    const float angle,
    const Eigen::Vector3f twist_axis
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    // TODO: change the test to take into account the twist!
    // TODO: add an enclosing bbox to decrease the compute overhead
    if (bbox.contains(nerf_coords(i)->pos.p)) {
        Eigen::Matrix3f rot_matrix;
        get_rotation_matrix(nerf_coords(i)->pos.p, bbox, angle, twist_axis, rot_matrix);
        nerf_coords(i)->pos.p = rot_matrix * (nerf_coords(i)->pos.p - bbox.center) + bbox.center;
    }
}

__global__ void twist_kernel_pos(
	const uint32_t n_elements,
	tcnn::PitchedPtr<NerfPosition> nerf_coords,
    const AffineBoundingBox bbox,
    const float angle,
    const Eigen::Vector3f twist_axis
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    // TODO: change the test to take into account the twist!
    // TODO: add an enclosing bbox to decrease the compute overhead
    if (bbox.contains(nerf_coords(i)->p)) {
        Eigen::Matrix3f rot_matrix;
        get_rotation_matrix(nerf_coords(i)->p, bbox, angle, twist_axis, rot_matrix);
        nerf_coords(i)->p = rot_matrix * (nerf_coords(i)->p - bbox.center) + bbox.center;
    }
}

void TwistOperator::map_rays(cudaStream_t stream, tcnn::PitchedPtr<NerfCoordinate> nerf_coords, tcnn::GPUMatrixDynamic<bool>& density_mask, uint32_t n_elements) const {
        
    // Perform the backward translation
    tcnn::linear_kernel(twist_kernel, 0, stream,
        n_elements,
        nerf_coords,
        m_warped_selection_box,
        m_angle,
        m_twist_axis
    );
}

    
void TwistOperator::map_positions(cudaStream_t stream, tcnn::PitchedPtr<NerfPosition> nerf_pos, tcnn::GPUMatrixDynamic<bool>& density_mask, uint32_t n_elements) const {
    
    // Perform the backward translation
    tcnn::linear_kernel(twist_kernel_pos, 0, stream,
        n_elements,
        nerf_pos,
        m_warped_selection_box,
        m_angle,
        m_twist_axis
    );
}

#ifdef NGP_GUI

bool TwistOperator::imgui(bool& delete_operator, const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) {
    bool edited_gui = false;
    if (ImGui::CollapsingHeader("Twist Operator", &m_header_visible, ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::SliderFloat("Twist angle", &m_angle, -M_PI, M_PI)) {
            edited_gui = true;
        }
        if (ImGui::InputFloat3("Twist direction", &m_twist_axis(0))) {
            m_twist_axis.normalize();
            // TODO: return edit to update
            edited_gui = true;
        }

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

        // Delete operator
        if (ImGui::Button("Delete Operator")) {
            delete_operator = true;
        }
    }
    return edited_gui;
}

bool TwistOperator::handle_keyboard() {
    return false;
}

bool TwistOperator::visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time) {
    ImDrawList* list = ImGui::GetForegroundDrawList();

    // Draw the selection box
    visualize_bbox(world2proj, list, m_selection_box, 0xff0000ff);

    // Draw the twist direction 
    add_debug_line(world2proj, list, m_selection_box.center-0.5f*m_twist_axis.cwiseProduct(m_selection_box.diag()),  m_selection_box.center+0.5f*m_twist_axis.cwiseProduct(m_selection_box.diag()), 0xffff00ff);
    
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
    if (edited_guizmo) {
        update_warped();
    }

    return edited_guizmo;
}
#endif


nlohmann::json TwistOperator::to_json() {
    nlohmann::json j;
    j["type"] = "twist";

    j["selection_box"] = m_selection_box;

    j["twist_axis"] = m_twist_axis;
    j["angle"] = m_angle;

    return j;
}

NGP_NAMESPACE_END
