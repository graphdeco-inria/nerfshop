#include <neural-graphics-primitives/editing/tools/visualization_utils.h>

#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>

NGP_NAMESPACE_BEGIN

using namespace Eigen;

// template<class matrix3_t, class point_t, typename float_t>
// void decompose_imguizmo_matrix(const Eigen::Matrix4f& edit_matrix, matrix3_t& rotation, point_t& translation, point_t& scale) {
// 	rotation = edit_matrix.block<3, 3>(0, 0).cast<float_t>();
// 	// Orthonormalize the vectors
// 	rotation.block<3, 1>(0, 0) /= rotation.template block<3, 1>(0, 0).norm();
// 	rotation.block<3, 1>(0, 1) /= rotation.template block<3, 1>(0, 1).norm();
// 	rotation.block<3, 1>(0, 2) /= rotation.template block<3, 1>(0, 2).norm();
// 	translation = edit_matrix.block<3, 1>(0, 3).cast<float_t>();
// 	scale = point_t{edit_matrix.block<3,1>(0, 0).norm(), edit_matrix.block<3, 1>(0, 1).norm(), edit_matrix.block<3, 1>(0, 2).norm()};
// }

// template<class matrix3_t, class point_t, typename float_t>
// void compose_imguizmo_matrix(Eigen::Matrix4f& edit_matrix, const matrix3_t& rotation, const point_t& translation, const point_t& scale) {
// 	edit_matrix = Eigen::Matrix4f::Identity();
// 	edit_matrix.block<3, 3>(0, 0) = rotation.template cast<float>();
// 	edit_matrix.block<3, 1>(0, 3) = translation.template cast<float>();
// 	edit_matrix.block<3, 1>(0, 0) *= scale.x();
// 	edit_matrix.block<3, 1>(0, 1) *= scale.y();
// 	edit_matrix.block<3, 1>(0, 2) *= scale.z();
// }


// template void decompose_imguizmo_matrix<Eigen::Matrix3f, Eigen::Vector3f, float>(const Eigen::Matrix4f& edit_matrix, Eigen::Matrix3f& rotation, Eigen::Vector3f& translation, Eigen::Vector3f& scale);
// template void decompose_imguizmo_matrix<Eigen::Matrix3d, Eigen::Vector3d, double>(const Eigen::Matrix4f& edit_matrix, Eigen::Matrix3d& rotation, Eigen::Vector3d& translation, Eigen::Vector3d& scale);
// template void compose_imguizmo_matrix<Eigen::Matrix3f, Eigen::Vector3f, float>(Eigen::Matrix4f& edit_matrix, const Eigen::Matrix3f& rotation, const Eigen::Vector3f& translation, const Eigen::Vector3f& scale);
// template void compose_imguizmo_matrix<Eigen::Matrix3d, Eigen::Vector3d, double>(Eigen::Matrix4f& edit_matrix, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation, const Eigen::Vector3d& scale);

void visualize_bbox(const Matrix<float, 4, 4>& world2proj, ImDrawList* list, const AffineBoundingBox& bbox, uint32_t col) {

	add_debug_line(world2proj, list, bbox.min,  bbox.min + bbox.w, col); // Z
	add_debug_line(world2proj, list, bbox.min + bbox.u, bbox.min + bbox.u + bbox.w, col);
	add_debug_line(world2proj, list, bbox.min + bbox.v, bbox.min + bbox.v + bbox.w, col);
	add_debug_line(world2proj, list, bbox.min + bbox.u + bbox.v, bbox.max, col);

	add_debug_line(world2proj, list, bbox.min, bbox.min + bbox.u, col); // X
	add_debug_line(world2proj, list, bbox.min + bbox.v, bbox.min + bbox.u + bbox.v, col);
    add_debug_line(world2proj, list, bbox.min + bbox.w, bbox.min + bbox.u + bbox.w, col);
	add_debug_line(world2proj, list, bbox.min + bbox.v + bbox.w, bbox.max, col);

	add_debug_line(world2proj, list, bbox.min, bbox.min + bbox.v, col); // Y
	add_debug_line(world2proj, list, bbox.min + bbox.u, bbox.min + bbox.u + bbox.v, col);
	add_debug_line(world2proj, list, bbox.min + bbox.w, bbox.min + bbox.v + bbox.w, col);
	add_debug_line(world2proj, list, bbox.min + bbox.u + bbox.w, bbox.max, col);
}

void visualize_quad(const Matrix<float, 4, 4>& world2proj, const Vector3f pos, const Vector3f x, const Vector3f y)
{
	ImDrawList* list = ImGui::GetForegroundDrawList();

	add_debug_line(world2proj, list, pos - x - y, pos - x + y, 0xffff4040);
	add_debug_line(world2proj, list, pos - x + y, pos + x + y, 0xffff4040);
	add_debug_line(world2proj, list, pos + x + y, pos + x - y, 0xffff4040);
	add_debug_line(world2proj, list, pos + x - y, pos - x - y, 0xffff4040);
}

void visualize_level_cube(const Matrix<float, 4, 4>& world2proj, const uint32_t level) {
	ImDrawList* list = ImGui::GetForegroundDrawList();

	const float scale = scalbnf(1.0f, level-1);

	add_debug_line(world2proj, list, scale * (Vector3f{0.f,0.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{0.f,0.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffff4040); // Z
	add_debug_line(world2proj, list, scale * (Vector3f{1.f,0.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{1.f,0.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);
	add_debug_line(world2proj, list, scale * (Vector3f{0.f,1.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{0.f,1.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);
	add_debug_line(world2proj, list, scale * (Vector3f{1.f,1.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{1.f,1.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);

	add_debug_line(world2proj, list, scale * (Vector3f{0.f,0.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{1.f,0.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xff4040ff); // X
	add_debug_line(world2proj, list, scale * (Vector3f{0.f,1.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{1.f,1.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);
	add_debug_line(world2proj, list, scale * (Vector3f{0.f,0.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{1.f,0.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);
	add_debug_line(world2proj, list, scale * (Vector3f{0.f,1.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{1.f,1.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);

	add_debug_line(world2proj, list, scale * (Vector3f{0.f,0.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{0.f,1.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xff40ff40); // Y
	add_debug_line(world2proj, list, scale * (Vector3f{1.f,0.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{1.f,1.f,0.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);
	add_debug_line(world2proj, list, scale * (Vector3f{0.f,0.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{0.f,1.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);
	add_debug_line(world2proj, list, scale * (Vector3f{1.f,0.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), scale * (Vector3f{1.f,1.f,1.f} - Eigen::Vector3f::Constant(0.5f)) + Eigen::Vector3f::Constant(0.5f), 0xffffffff);
}

#ifdef NGP_GUI

bool imgui_colored_button2(const char* name, float hue) {
	ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(hue, 0.6f, 0.6f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(hue, 0.7f, 0.7f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(hue, 0.8f, 0.8f));
	bool rv = ImGui::Button(name);
	ImGui::PopStyleColor(3);
	return rv;
}

#endif

NGP_NAMESPACE_END
