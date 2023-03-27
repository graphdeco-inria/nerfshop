#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/editing/edit_operator.h>
#include <neural-graphics-primitives/editing/distiller.h>
#include <neural-graphics-primitives/editing/tools/growing_selection.h>
#include <neural-graphics-primitives/editing/datastructures/tet_mesh.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/common_nerf.h>

#ifdef NGP_GUI

#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>
#endif

TCNN_NAMESPACE_BEGIN
template <typename T> class PitchedPtr;
TCNN_NAMESPACE_END

using namespace Eigen;

NGP_NAMESPACE_BEGIN

class CageDeformation : public EditOperator {
public:

    CageDeformation(
        BoundingBox aabb, 
        cudaStream_t stream,
        const std::shared_ptr<NerfNetwork<precision_t>> nerf_network, 
        const tcnn::GPUMemory<float>& density_grid, 
        const tcnn::GPUMemory<uint8_t>& density_grid_bitfield,
        const float cone_angle_constant,
        const ENerfActivation rgb_activation,
        const ENerfActivation density_activation,
        const Eigen::Vector3f light_dir,
        const std::string default_envmap_path,
        const uint32_t max_cascade
    ) : 
        m_scene_aabb{aabb},
        m_growing_selection(aabb, stream, nerf_network, density_grid, density_grid_bitfield, cone_angle_constant, rgb_activation, density_activation, light_dir, default_envmap_path, max_cascade) 
    {
    }

    CageDeformation(
        nlohmann::json operator_json,
        BoundingBox aabb, 
        cudaStream_t stream,
        const std::shared_ptr<NerfNetwork<precision_t>> nerf_network, 
        const tcnn::GPUMemory<float>& density_grid, 
        const tcnn::GPUMemory<uint8_t>& density_grid_bitfield,
        const float cone_angle_constant,
        const ENerfActivation rgb_activation,
        const ENerfActivation density_activation,
        const Eigen::Vector3f light_dir,
        const std::string default_envmap_path,
        const uint32_t max_cascade
    ) : 
        m_scene_aabb{aabb},
        m_growing_selection(operator_json, aabb, stream, nerf_network, density_grid, density_grid_bitfield, cone_angle_constant, rgb_activation, density_activation, light_dir, default_envmap_path, max_cascade) 
    {
    }

    void map_rays(cudaStream_t stream, tcnn::PitchedPtr<NerfCoordinate> nerf_coords, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const override;

	virtual void kill_empty_density(cudaStream_t stream,
		uint32_t n_elements,
		PitchedPtr<NerfPosition> output,
		tcnn::GPUMatrixDynamic<bool>& empty_mask,
		tcnn::network_precision_t* density_network_output
	) const override;

    void map_positions(cudaStream_t stream, tcnn::PitchedPtr<NerfPosition> nerf_pos, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const override;

	Distiller* gpu_distiller = nullptr;

	virtual Distiller* getDistiller() override;

	//struct MiniGPUInfo
	//{
	//	bool emptying;
	//	BoundingBox aabb;
	//	BoundingBox bbox;
	//	BoundingBox original_bbox;
	//	uint32_t* original_tet_lut_idx;
	//	uint32_t* original_tet_lut_offsets;
	//	uint32_t* tet_lut_idx;
	//	uint32_t* tet_lut_offsets;
	//	uint32_t* tets;
	//	Eigen::Vector3f* vertices;
	//	Eigen::Vector3f* original_vertices;
	//	Eigen::Matrix3f* local_rotations;
	//	uint8_t* original_bitfield_gpu;
	//};

	//MiniGPUInfo toGPUInfo()
	//{
	//	MiniGPUInfo info;
	//	info.emptying = !m_growing_selection.m_copy;
	//	info.aabb = m_scene_aabb;
	//	info.bbox = m_growing_selection.tet_interpolation_mesh->bbox;
	//	info.original_bbox = m_growing_selection.tet_interpolation_mesh->original_bbox;
	//	info.original_tet_lut_idx = m_growing_selection.tet_interpolation_mesh->original_tet_lut_idx.data();
	//	info.tet_lut_idx = m_growing_selection.tet_interpolation_mesh->tet_lut_idx.data();
	//	info.original_tet_lut_offsets = m_growing_selection.tet_interpolation_mesh->original_tet_lut_offsets.data();
	//	info.tet_lut_offsets = m_growing_selection.tet_interpolation_mesh->tet_lut_offsets.data();
	//	info.tets = m_growing_selection.tet_interpolation_mesh->tets_gpu.data();
	//	info.vertices = m_growing_selection.tet_interpolation_mesh->vertices_gpu.data();
	//	info.original_vertices = m_growing_selection.tet_interpolation_mesh->original_vertices_gpu.data();

	//	info.local_rotations = (m_growing_selection.tet_interpolation_mesh->local_rotations_gpu.size() > 0) ? m_growing_selection.tet_interpolation_mesh->local_rotations_gpu.data() : nullptr;

	//	info.original_bitfield_gpu = m_growing_selection.tet_interpolation_mesh->original_bitfield_gpu.data();
	//	return info;
	//}

    void compute_poisson_residual_density(
        cudaStream_t stream,  
        const uint32_t n_elements,
        tcnn::PitchedPtr<NerfPosition> input_position,
        network_precision_t* density_network_output
    ) const override;

    void compute_poisson_full_residuals(
        cudaStream_t stream,  
        const uint32_t n_elements,
        NerfPayload* payloads,
        tcnn::PitchedPtr<NerfCoordinate> network_input,
		//SH9RGB* __restrict__ sh_in_boundary,
        SH9RGB* __restrict__ sh_boundary,
		//float* __restrict__ in_density_boundary,
        float* __restrict__ out_density_boundary,
        float* __restrict__ residual_density_boundary
    ) const override;

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
    ) override;

    nlohmann::json to_json() override;

	GrowingSelection m_growing_selection;
private:

    BoundingBox m_scene_aabb;
    BoundingBox m_warped_bbox;

    bool m_apply_residuals = true;
    bool m_apply_poisson = false;
    float m_residual_amplitude = 1.0f;
};

NGP_NAMESPACE_END
