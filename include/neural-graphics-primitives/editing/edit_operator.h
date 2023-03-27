#pragma once

#include <nvfunctional>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <exception>
#include <json/json.hpp>

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif

TCNN_NAMESPACE_BEGIN
template <typename T> class PitchedPtr;
TCNN_NAMESPACE_END

NGP_NAMESPACE_BEGIN

class Distiller;

class EditOperator {
public:

	virtual ~EditOperator() { }

    virtual bool imgui(bool& delete_operator, const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length,  const Eigen::Matrix<float, 3, 4>& camera_matrix, const Eigen::Vector2f& screen_center, bool& auto_clean) = 0;

    virtual bool visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time) = 0;

    virtual void draw_gl(
        const Eigen::Vector2i& resolution, 
        const Eigen::Vector2f& focal_length, 
        const Eigen::Matrix<float, 3, 4>& camera_matrix, 
        const Eigen::Vector2f& screen_center) = 0;
    
    // Returns true if the occupancy grid must be updated
    virtual bool handle_keyboard() = 0;

    virtual void map_rays(cudaStream_t stream, tcnn::PitchedPtr<NerfCoordinate> nerf_coords, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const = 0;

    virtual void map_positions(cudaStream_t stream, tcnn::PitchedPtr<NerfPosition> nerf_coords, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const = 0;

     // Compute the interpolated incoming radiance
    virtual void compute_interpolated_radiance(
        cudaStream_t stream,  
        const uint32_t n_elements,
        NerfPayload* payloads,
        tcnn::PitchedPtr<NerfCoordinate> network_input,
        SH9RGB* __restrict__ sh_initial,
        SH9RGB* __restrict__ sh_new) const {}

	virtual void kill_empty_density(cudaStream_t stream,
		uint32_t n_elements,
		tcnn::PitchedPtr<NerfPosition> output,
		tcnn::GPUMatrixDynamic<bool>& empty_mask,
		tcnn::network_precision_t* density_network_output
		) const
	{
		std::cout << "Not implemented!" << std::endl;
		//throw std::runtime_error("Not implemented!");
	}

    // Compute the interpolated residual density for Poisson processing
    virtual void compute_poisson_residual_density(
        cudaStream_t stream,  
        const uint32_t n_elements,
        tcnn::PitchedPtr<NerfPosition> input_position,
        tcnn::network_precision_t* density_network_output
    ) const {}

	virtual Distiller* getDistiller()
	{
		throw std::runtime_error("Not implemented!");
	}

    // Compute both the interpolated residual density, outside density and SHs for Poisson processing
    virtual void compute_poisson_full_residuals(
        cudaStream_t stream,  
        const uint32_t n_elements,
        NerfPayload* payloads,
        tcnn::PitchedPtr<NerfCoordinate> network_input,
		//SH9RGB* __restrict__ sh_in_boundary,
        SH9RGB* __restrict__ sh_boundary,
		//float* __restrict__ in_density_boundary,
        float* __restrict__ out_density_boundary,
        float* __restrict__ residual_density_boundary
    ) const {}

    virtual nlohmann::json to_json() = 0;
};

NGP_NAMESPACE_END
