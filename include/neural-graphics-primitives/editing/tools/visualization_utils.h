#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/editing/tools/affine_bounding_box.cuh>
#include <neural-graphics-primitives/camera_path.h>

NGP_NAMESPACE_BEGIN

using namespace Eigen;

template<class matrix3_t, class point_t, typename float_t>
void decompose_imguizmo_matrix(const Eigen::Matrix4f& edit_matrix, matrix3_t& rotation, point_t& translation, point_t& scale) {
    rotation = edit_matrix.block<3, 3>(0, 0).cast<float_t>();
	// Orthonormalize the vectors
	rotation.template block<3, 1>(0, 0) /= rotation.template block<3, 1>(0, 0).norm();
	rotation.template block<3, 1>(0, 1) /= rotation.template block<3, 1>(0, 1).norm();
	rotation.template block<3, 1>(0, 2) /= rotation.template block<3, 1>(0, 2).norm();
	translation = edit_matrix.block<3, 1>(0, 3).cast<float_t>();
	scale = point_t{edit_matrix.block<3,1>(0, 0).norm(), edit_matrix.block<3, 1>(0, 1).norm(), edit_matrix.block<3, 1>(0, 2).norm()};
}

template<class matrix3_t, class point_t, typename float_t>
void compose_imguizmo_matrix(Eigen::Matrix4f& edit_matrix, const matrix3_t& rotation, const point_t& translation, const point_t& scale) {
    edit_matrix = Eigen::Matrix4f::Identity();
	edit_matrix.template block<3, 3>(0, 0) = rotation.template cast<float>();
	edit_matrix.template block<3, 1>(0, 3) = translation.template cast<float>();
	edit_matrix.template block<3, 1>(0, 0) *= scale.x();
	edit_matrix.template block<3, 1>(0, 1) *= scale.y();
	edit_matrix.template block<3, 1>(0, 2) *= scale.z();
}

void visualize_bbox(const Matrix<float, 4, 4>& world2proj, ImDrawList* list, const AffineBoundingBox& bbox, uint32_t col=0xffffffff);

void visualize_level_cube(const Matrix<float, 4, 4>& world2proj, const uint32_t level);

void visualize_quad(const Matrix<float, 4, 4>& world2proj, const Vector3f pos, const Vector3f x, const Vector3f y);

#ifdef NGP_GUI

bool imgui_colored_button2(const char* name, float hue);

#endif

NGP_NAMESPACE_END
