#include <neural-graphics-primitives/editing/cage_deformation.h>
#include <neural-graphics-primitives/editing/tools/visualization_utils.h>
#include <neural-graphics-primitives/editing/distiller.h>

using namespace Eigen;

NGP_NAMESPACE_BEGIN

//__device__ float scalar_tp(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) {
//    return a.dot(b.cross(c));
//}
//
//__device__ Eigen::Vector4f bary_tet(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c, const Eigen::Vector3f& d, const Eigen::Vector3f& p) {
//    Eigen::Vector3f vap = p - a;
//    Eigen::Vector3f vbp = p - b;
//
//    Eigen::Vector3f vab = b - a;
//    Eigen::Vector3f vac = c - a;
//    Eigen::Vector3f vad = d - a;
//
//    Eigen::Vector3f vbc = c - b;
//    Eigen::Vector3f vbd = d - b;
//    // ScTP computes the scalar triple product
//    float va6 = scalar_tp(vbp, vbd, vbc);
//    float vb6 = scalar_tp(vap, vac, vad);
//    float vc6 = scalar_tp(vap, vad, vab);
//    float vd6 = scalar_tp(vap, vab, vac);
//    float v6 = 1. / scalar_tp(vab, vac, vad);
//    return Eigen::Vector4f(va6*v6, vb6*v6, vc6*v6, vd6*v6);
//}

class CageDeformationDistiller : public Distiller
{
public:

	BoundingBox aabb;
	BoundingBox bbox;
	BoundingBox original_bbox;
	uint32_t* original_tet_lut_idx;
	uint32_t* original_tet_lut_offsets;
	uint32_t* tet_lut_idx;
	uint32_t* tet_lut_offsets;
	uint32_t* tets;
	Eigen::Vector3f* vertices;
	Eigen::Vector3f* original_vertices;
	Eigen::Matrix3f* local_rotations;
	uint8_t* original_bitfield_gpu;

	virtual __device__ bool in_source(Eigen::Vector3f& coord, bool warp) override
	{
		Eigen::Vector3f unwarped_pos = coord;
		if (warp)
			unwarped_pos = unwarp_position(coord, aabb);

		if (original_bbox.contains_base(coord))
		{
			int level = mip_from_pos(unwarped_pos);
			uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(unwarped_pos, level);

			// If cell contains a triangle, get it(/them)
			for (uint32_t j = original_tet_lut_offsets[cell_idx]; j < original_tet_lut_offsets[cell_idx + 1]; j++) {
				uint32_t tet_idx = original_tet_lut_idx[j];
				// If point is actually in the selected tet
				if (point_in_tet<float, Eigen::Vector3f>(original_vertices[tets[4 * tet_idx]], original_vertices[tets[4 * tet_idx + 1]], original_vertices[tets[4 * tet_idx + 2]], original_vertices[tets[4 * tet_idx + 3]], unwarped_pos))
				{
					return true;
				}
			}
		}
		return false;
	}

	virtual __device__ bool in_target(Eigen::Vector3f& coord, bool warp) override
	{
		Eigen::Vector3f unwarped_pos = coord;
		if (warp)
			unwarped_pos = unwarp_position(coord, aabb);

		if (bbox.contains_base(coord))
		{
			int level = mip_from_pos(unwarped_pos);
			uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(unwarped_pos, level);

			// If cell contains a triangle, get it(/them)
			for (uint32_t j = tet_lut_offsets[cell_idx]; j < tet_lut_offsets[cell_idx + 1]; j++) {
				uint32_t tet_idx = tet_lut_idx[j];
				// If point is actually in the selected tet
				if (point_in_tet<float, Eigen::Vector3f>(vertices[tets[4 * tet_idx]], vertices[tets[4 * tet_idx + 1]], vertices[tets[4 * tet_idx + 2]], vertices[tets[4 * tet_idx + 3]], unwarped_pos))
				{
					return true;
				}
			}
		}
		return false;
	}

	virtual __device__ Vector3f map(Eigen::Vector3f coord, bool warp) override
	{
		Eigen::Vector3f unwarped_pos = coord;
		if (warp)
			unwarped_pos = unwarp_position(coord, aabb);

		// Test the bounding box first
		if (original_bbox.contains_base(coord))
		{
			int level = mip_from_pos(unwarped_pos);

			uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(unwarped_pos, level);

			// If cell contains a triangle, get it(/them)
			for (uint32_t j = original_tet_lut_offsets[cell_idx]; j < original_tet_lut_offsets[cell_idx + 1]; j++) {
				uint32_t tet_idx = original_tet_lut_idx[j];
				// If point is actually in the selected tet
				if (point_in_tet<float, Eigen::Vector3f>(original_vertices[tets[4 * tet_idx]], original_vertices[tets[4 * tet_idx + 1]], original_vertices[tets[4 * tet_idx + 2]], original_vertices[tets[4 * tet_idx + 3]], unwarped_pos))
				{
					// Compute barycentric coordinates
					Eigen::Vector4f bary_coord = bary_tet(original_vertices[tets[4 * tet_idx]], original_vertices[tets[4 * tet_idx + 1]], original_vertices[tets[4 * tet_idx + 2]], original_vertices[tets[4 * tet_idx + 3]], unwarped_pos);

					coord = bary_coord.x() * vertices[tets[4 * tet_idx]]
						+ bary_coord.y() * vertices[tets[4 * tet_idx + 1]]
						+ bary_coord.z() * vertices[tets[4 * tet_idx + 2]]
						+ bary_coord.w() * vertices[tets[4 * tet_idx + 3]];

					if (warp)
						coord = warp_position(coord, aabb);

					break;
				}
			}
		}
		return coord;
	};
};


__global__ void interpolate_tet_pos(
	const uint32_t n_elements,
	tcnn::PitchedPtr<NerfPosition> nerf_coords,
    bool* __restrict__ empty_mask,
    const BoundingBox aabb,
    const BoundingBox deformed_bbox,
    const BoundingBox original_bbox, // TODO: remove this when handling MIPs
    const uint32_t* __restrict__ tet_lut_idx,
    const uint32_t* __restrict__ tet_lut_offsets,
    const uint32_t* __restrict__ tets,
    const Eigen::Vector3f* __restrict__ vertices,
    const Eigen::Vector3f* __restrict__ original_vertices,
    const uint8_t* original_bitfield
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    bool in_deformed = false;

    // Test the bounding box first
    if (deformed_bbox.contains(nerf_coords(i)->p)) {
    
        Eigen::Vector3f unwarped_pos = unwarp_position(nerf_coords(i)->p, aabb);
        int level = mip_from_pos(unwarped_pos);

        uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(unwarped_pos, level);
        // If cell contains a triangle, get it(/them)
        for (uint32_t j = tet_lut_offsets[cell_idx]; j < tet_lut_offsets[cell_idx+1]; j++) {
            uint32_t tet_idx = tet_lut_idx[j];
            // If point is actually in the selected tet
            if (point_in_tet<float, Eigen::Vector3f>(vertices[tets[4*tet_idx]], vertices[tets[4*tet_idx+1]], vertices[tets[4*tet_idx+2]], vertices[tets[4*tet_idx+3]], unwarped_pos)) {
                // Compute barycentric coordinates
                Eigen::Vector4f bary_coord = bary_tet(vertices[tets[4*tet_idx]], vertices[tets[4*tet_idx+1]], vertices[tets[4*tet_idx+2]], vertices[tets[4*tet_idx+3]], unwarped_pos);
                Eigen::Vector3f canonical_pos = bary_coord.x() * original_vertices[tets[4*tet_idx]]
                                            + bary_coord.y() * original_vertices[tets[4*tet_idx+1]]
                                            + bary_coord.z() * original_vertices[tets[4*tet_idx+2]]
                                            + bary_coord.w() * original_vertices[tets[4*tet_idx+3]];
                nerf_coords(i)->p = warp_position(canonical_pos, aabb);
                in_deformed = true;
                break;
            }
        } 
    }

    // If not in deformed, test the bitfield to potentially update the empty mask
    // TODO: remove second condition when handling MIPs
    // !!!!!!!!!!!!!!!!
    if (!in_deformed && original_bbox.contains(nerf_coords(i)->p)) {
        Eigen::Vector3f unwarped_pos = unwarp_position(nerf_coords(i)->p, aabb);
        int level = mip_from_pos(unwarped_pos);

        uint32_t pos_idx = cascaded_grid_idx_at(unwarped_pos, level);
        if (get_bitfield_at(pos_idx, level, original_bitfield))  {
            empty_mask[i] = true;
        }
    }
}

//__device__ int sample_tested;
//__device__ int sample_num_samples;

__global__ void interpolate_tet(
	const uint32_t n_elements,
	tcnn::PitchedPtr<NerfCoordinate> nerf_coords,
    bool* __restrict__ empty_mask,
	bool copy,
    const BoundingBox aabb,
    const BoundingBox deformed_bbox,
    const BoundingBox original_bbox, // TODO: remove this when handling MIPs
    const uint32_t* __restrict__ tet_lut_idx,
    const uint32_t* __restrict__ tet_lut_offsets,
    const uint32_t* __restrict__ tets,
    const Eigen::Vector3f* __restrict__ vertices,
    const Eigen::Vector3f* __restrict__ original_vertices,
    const Eigen::Matrix3f* __restrict__ local_rotations,
    const uint8_t* original_bitfield
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    bool in_deformed = false;

    // Test the bounding box first
    if (deformed_bbox.contains(nerf_coords(i)->pos.p)) {

        Eigen::Vector3f unwarped_pos = unwarp_position(nerf_coords(i)->pos.p, aabb);
        int level = mip_from_pos(unwarped_pos);

        uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(unwarped_pos, level);

		// int tested = 0;
        // If cell contains a triangle, get it(/them)
        for (uint32_t j = tet_lut_offsets[cell_idx]; j < tet_lut_offsets[cell_idx+1]; j++) {
            uint32_t tet_idx = tet_lut_idx[j];
            // If point is actually in the selected tet
			//tested++;
            if (point_in_tet<float, Eigen::Vector3f>(vertices[tets[4*tet_idx]], vertices[tets[4*tet_idx+1]], vertices[tets[4*tet_idx+2]], vertices[tets[4*tet_idx+3]], unwarped_pos)) {
                // Compute barycentric coordinates
                Eigen::Vector4f bary_coord = bary_tet(vertices[tets[4*tet_idx]], vertices[tets[4*tet_idx+1]], vertices[tets[4*tet_idx+2]], vertices[tets[4*tet_idx+3]], unwarped_pos);
                Eigen::Vector3f canonical_pos = bary_coord.x() * original_vertices[tets[4*tet_idx]]
                                            + bary_coord.y() * original_vertices[tets[4*tet_idx+1]]
                                            + bary_coord.z() * original_vertices[tets[4*tet_idx+2]]
                                            + bary_coord.w() * original_vertices[tets[4*tet_idx+3]];
                nerf_coords(i)->pos.p = warp_position(canonical_pos, aabb);
                if (local_rotations) {
                    Eigen::Vector3f unwarped_dir = unwarp_direction(nerf_coords(i)->dir.d);
                    unwarped_dir = local_rotations[tet_idx] * unwarped_dir;
                    nerf_coords(i)->dir.d = warp_direction(unwarped_dir);
                }
                in_deformed = true;
                break;
            }
        }

		//atomicAdd(&sample_tested, tested);
		//atomicAdd(&sample_num_samples, 1);
    }

    // If not in deformed, test the bitfield to potentially update the empty mask
    // TODO: remove second condition when handling MIPs
	if (!copy)
	{
		if (!in_deformed && original_bbox.contains(nerf_coords(i)->pos.p)) {
			Eigen::Vector3f unwarped_pos = unwarp_position(nerf_coords(i)->pos.p, aabb);
			int level = mip_from_pos(unwarped_pos);

			uint32_t pos_idx = cascaded_grid_idx_at(unwarped_pos, level);

			if (get_bitfield_at(pos_idx, level, original_bitfield)) {
				empty_mask[i] = true;
			}
		}
	}
}

__global__ void compute_shs_kernel(
    const uint32_t n_elements,
    NerfPayload* payloads,
    tcnn::PitchedPtr<NerfCoordinate> network_input,
    SH9RGB* __restrict__ initial_shs,
    SH9RGB* __restrict__ new_shs,
    const BoundingBox aabb,
    const BoundingBox original_bbox, // TODO: remove this when handling MIPs
    const uint32_t* __restrict__ original_tet_lut_idx,
    const uint32_t* __restrict__ original_tet_lut_offsets,
    const uint32_t* __restrict__ tets,
    const Eigen::Vector3f* __restrict__ original_vertices,
    const SH9RGB* __restrict__ initial_shs_tet,
    const SH9RGB* __restrict__ new_shs_tet,
    const float residual_amplitude
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    NerfPayload& payload = payloads[i];

    uint32_t actual_n_steps = payload.n_steps;
	uint32_t k = 0;

    for (; k < actual_n_steps; ++k) {
        const NerfCoordinate* input = network_input(i + k * n_elements);
		Vector3f warped_pos = input->pos.p;
		Vector3f pos = unwarp_position(warped_pos, aabb);

        // Test the bounding box first
        if (original_bbox.contains(pos)) {

            int level = mip_from_pos(pos);

            uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(pos, level);

            // If cell contains a triangle, get it(/them)
            for (uint32_t j = original_tet_lut_offsets[cell_idx]; j < original_tet_lut_offsets[cell_idx+1]; j++) {
                uint32_t tet_idx = original_tet_lut_idx[j];
                // If point is actually in the selected tet
                if (point_in_tet<float, Eigen::Vector3f>(original_vertices[tets[4*tet_idx]], original_vertices[tets[4*tet_idx+1]], original_vertices[tets[4*tet_idx+2]], original_vertices[tets[4*tet_idx+3]], pos)) {
                    // Compute barycentric coordinates
                    Eigen::Vector4f bary_coord = bary_tet(original_vertices[tets[4*tet_idx]], original_vertices[tets[4*tet_idx+1]], original_vertices[tets[4*tet_idx+2]], original_vertices[tets[4*tet_idx+3]], pos);
                    // DEBUG
                    // Array3f local_residual = vertices_residuals[tets[4*tet_idx]];
                    // local_residual = local_residual.cwiseMax(vertices_residuals[tets[4*tet_idx+1]]);
                    // local_residual = local_residual.cwiseMax(vertices_residuals[tets[4*tet_idx+2]]);
                    // local_residual = local_residual.cwiseMax(vertices_residuals[tets[4*tet_idx+3]]);
                    SH9RGB local_initial = bary_coord.x() * initial_shs_tet[tets[4*tet_idx]]
                                                + bary_coord.y() * initial_shs_tet[tets[4*tet_idx+1]]
                                                + bary_coord.z() * initial_shs_tet[tets[4*tet_idx+2]]
                                                + bary_coord.w() * initial_shs_tet[tets[4*tet_idx+3]];
                    SH9RGB local_outside = bary_coord.x() * new_shs_tet[tets[4*tet_idx]]
                                                + bary_coord.y() * new_shs_tet[tets[4*tet_idx+1]]
                                                + bary_coord.z() * new_shs_tet[tets[4*tet_idx+2]]
                                                + bary_coord.w() * new_shs_tet[tets[4*tet_idx+3]];
                    // rgb_residuals[i + k * n_elements] = residual_amplitude * Array3f(1.0f, 0.0f, 0.0f);
                    initial_shs[i + k * n_elements] = local_initial;
                    new_shs[i + k * n_elements] = residual_amplitude * local_outside;
                    // if (local_residual.x()*local_residual.x() + local_residual.y()*local_residual.y() + local_residual.z()*local_residual.z() > 0.1f) {
                    //     printf("Color: %.4f %.4f %.4f\n", local_residual.x(), local_residual.y(), local_residual.z()); 
                    // }
                    // printf("Bary coord: %.4f %.4f %.4f %.4f\n", bary_coord.x(), bary_coord.y(), bary_coord.z(), bary_coord.w()); 
                    break;
                }
            }
        }
    }
}

__global__ void compute_poisson_residual_density_kernel(
    const uint32_t n_elements,
    tcnn::PitchedPtr<NerfPosition> nerf_coords,
    network_precision_t* __restrict__ density_network_output,
    const BoundingBox aabb,
	const Vector3f plane_pos,
	const Vector3f plane_dir,
    const BoundingBox bbox, // TODO: remove this when handling MIPs
    const uint32_t* __restrict__ tet_lut_idx,
    const uint32_t* __restrict__ tet_lut_offsets,
    const uint32_t* __restrict__ tets,
    const Eigen::Vector3f* __restrict__ vertices,
    const float* __restrict__ boundary_residual_density_tet
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    Vector3f warped_pos = nerf_coords(i)->p;
    Vector3f pos = unwarp_position(warped_pos, aabb);

    // Test the bounding box first
    if (bbox.contains(pos)) {

        int level = mip_from_pos(pos);

        uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(pos, level);

        // If cell contains a triangle, get it(/them)
        for (uint32_t j = tet_lut_offsets[cell_idx]; j < tet_lut_offsets[cell_idx+1]; j++) {
            uint32_t tet_idx = tet_lut_idx[j];
            // If point is actually in the selected tet
            if (point_in_tet<float, Eigen::Vector3f>(vertices[tets[4*tet_idx]], vertices[tets[4*tet_idx+1]], vertices[tets[4*tet_idx+2]], vertices[tets[4*tet_idx+3]], pos)) {
                // Compute barycentric coordinates
                Eigen::Vector4f bary_coord = bary_tet(vertices[tets[4*tet_idx]], vertices[tets[4*tet_idx+1]], vertices[tets[4*tet_idx+2]], vertices[tets[4*tet_idx+3]], pos);
                float density_residual = bary_coord.x() * boundary_residual_density_tet[tets[4*tet_idx]]
                                            + bary_coord.y() * boundary_residual_density_tet[tets[4*tet_idx+1]]
                                            + bary_coord.z() * boundary_residual_density_tet[tets[4*tet_idx+2]]
                                            + bary_coord.w() * boundary_residual_density_tet[tets[4*tet_idx+3]];
                density_network_output[i] += (network_precision_t)density_residual;
                break;
            }
        }
    }
}

__global__ void kill_empty_density_kernel(
	const uint32_t n_elements,
	bool* __restrict__ empty_mask,
	tcnn::PitchedPtr<NerfPosition> nerf_coords,
	network_precision_t* __restrict__ density_network_output,
	const BoundingBox aabb,
	const Vector3f plane_pos,
	const Vector3f plane_dir,
	const BoundingBox original_bbox, // TODO: remove this when handling MIPs
	const uint32_t* __restrict__ original_tet_lut_idx,
	const uint32_t* __restrict__ original_tet_lut_offsets,
	const uint32_t* __restrict__ tets,
	const Eigen::Vector3f* __restrict__ original_vertices,
	float user_off) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	if (!empty_mask[i])
		return;

	Vector3f warped_pos = nerf_coords(i)->p;
	Vector3f pos = unwarp_position(warped_pos, aabb);

	// Test the bounding box first
	if (original_bbox.contains(pos)) {

		int level = mip_from_pos(pos);

		uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(pos, level);

		// If cell contains a triangle, get it(/them)
		for (uint32_t j = original_tet_lut_offsets[cell_idx]; j < original_tet_lut_offsets[cell_idx + 1]; j++) {
			uint32_t tet_idx = original_tet_lut_idx[j];
			// If point is actually in the selected tet
			if (point_in_tet<float, Eigen::Vector3f>(original_vertices[tets[4 * tet_idx]], original_vertices[tets[4 * tet_idx + 1]], original_vertices[tets[4 * tet_idx + 2]], original_vertices[tets[4 * tet_idx + 3]], pos)) {

				auto diff = pos - plane_pos;
				if(diff.dot(plane_dir) > user_off)
					density_network_output[i] = 0;
				break;
			}
		}
	}
}

__global__ void compute_residual_poisson_kernel(
    const uint32_t n_elements,
    NerfPayload* payloads,
    tcnn::PitchedPtr<NerfCoordinate> network_input,
	//SH9RGB* __restrict__ boundary_shs_in,
    SH9RGB* __restrict__ boundary_shs,
	//float* __restrict__ boundary_inside_density,
    float* __restrict__ boundary_outside_density,
    float* __restrict__ boundary_residual_density,
    const BoundingBox aabb,
	const BoundingBox original_bbox, // TODO: remove this when handling MIPs
    const BoundingBox bbox, // TODO: remove this when handling MIPs
	const uint32_t* __restrict__ tet_original_lut_idx,
    const uint32_t* __restrict__ tet_lut_idx,
	const uint32_t* __restrict__ tet_original_lut_offsets,
    const uint32_t* __restrict__ tet_lut_offsets,
    const uint32_t* __restrict__ tets,
	const Eigen::Vector3f* __restrict__ original_vertices,
    const Eigen::Vector3f* __restrict__ vertices,
	//const SH9RGB* __restrict__ boundary_shs_in_tet,
    const SH9RGB* __restrict__ boundary_shs_tet,
	//const float* __restrict__ boundary_inside_density_tet,
    const float* __restrict__ boundary_outside_density_tet,
    const float* __restrict__ boundary_residual_density_tet,
    const float residual_amplitude
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

    NerfPayload& payload = payloads[i];

    uint32_t actual_n_steps = payload.n_steps;
	uint32_t k = 0;

    for (; k < actual_n_steps; ++k) {
        const NerfCoordinate* input = network_input(i + k * n_elements);
		Vector3f warped_pos = input->pos.p;
		Vector3f pos = unwarp_position(warped_pos, aabb);

        // Test the bounding box first
		if (bbox.contains(pos)) {

			int level = mip_from_pos(pos);

			uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(pos, level);

			// If cell contains a triangle, get it(/them)
			for (uint32_t j = tet_lut_offsets[cell_idx]; j < tet_lut_offsets[cell_idx + 1]; j++) {
				uint32_t tet_idx = tet_lut_idx[j];
				// If point is actually in the selected tet
				if (point_in_tet<float, Eigen::Vector3f>(vertices[tets[4 * tet_idx]], vertices[tets[4 * tet_idx + 1]], vertices[tets[4 * tet_idx + 2]], vertices[tets[4 * tet_idx + 3]], pos)) {
					// Compute barycentric coordinates
					Eigen::Vector4f bary_coord = bary_tet(vertices[tets[4 * tet_idx]], vertices[tets[4 * tet_idx + 1]], vertices[tets[4 * tet_idx + 2]], vertices[tets[4 * tet_idx + 3]], pos);

					SH9RGB local_sh = bary_coord.x() * boundary_shs_tet[tets[4 * tet_idx]]
						+ bary_coord.y() * boundary_shs_tet[tets[4 * tet_idx + 1]]
						+ bary_coord.z() * boundary_shs_tet[tets[4 * tet_idx + 2]]
						+ bary_coord.w() * boundary_shs_tet[tets[4 * tet_idx + 3]];
					float local_outside_density = bary_coord.x() * boundary_outside_density_tet[tets[4 * tet_idx]]
						+ bary_coord.y() * boundary_outside_density_tet[tets[4 * tet_idx + 1]]
						+ bary_coord.z() * boundary_outside_density_tet[tets[4 * tet_idx + 2]]
						+ bary_coord.w() * boundary_outside_density_tet[tets[4 * tet_idx + 3]];
					float local_residual_density = bary_coord.x() * boundary_residual_density_tet[tets[4 * tet_idx]]
						+ bary_coord.y() * boundary_residual_density_tet[tets[4 * tet_idx + 1]]
						+ bary_coord.z() * boundary_residual_density_tet[tets[4 * tet_idx + 2]]
						+ bary_coord.w() * boundary_residual_density_tet[tets[4 * tet_idx + 3]];
					// rgb_residuals[i + k * n_elements] = residual_amplitude * Array3f(1.0f, 0.0f, 0.0f);
					boundary_shs[i + k * n_elements] = local_sh;
					boundary_outside_density[i + k * n_elements] = residual_amplitude * local_outside_density;
					boundary_residual_density[i + k * n_elements] = residual_amplitude * local_residual_density;
					// if (local_residual.x()*local_residual.x() + local_residual.y()*local_residual.y() + local_residual.z()*local_residual.z() > 0.1f) {
					//     printf("Color: %.4f %.4f %.4f\n", local_residual.x(), local_residual.y(), local_residual.z()); 
					// }
					// printf("Bary coord: %.4f %.4f %.4f %.4f\n", bary_coord.x(), bary_coord.y(), bary_coord.z(), bary_coord.w()); 
					break;
				}
			}
		}
		//if (original_bbox.contains(pos))
		//{
		//	int level = mip_from_pos(pos);

		//	uint32_t cell_idx = level * NERF_GRIDVOLUME() + cascaded_grid_idx_at(pos, level);

		//	for (uint32_t j = tet_original_lut_offsets[cell_idx]; j < tet_original_lut_offsets[cell_idx + 1]; j++) {
		//		
		//		uint32_t tet_idx = tet_original_lut_idx[j];
		//		// If point is actually in the selected tet
		//		if (point_in_tet<float, Eigen::Vector3f>(original_vertices[tets[4 * tet_idx]], original_vertices[tets[4 * tet_idx + 1]], original_vertices[tets[4 * tet_idx + 2]], original_vertices[tets[4 * tet_idx + 3]], pos)) {
		//			// Compute barycentric coordinates
		//			Eigen::Vector4f bary_coord = bary_tet(original_vertices[tets[4 * tet_idx]], original_vertices[tets[4 * tet_idx + 1]], original_vertices[tets[4 * tet_idx + 2]], original_vertices[tets[4 * tet_idx + 3]], pos);

		//			SH9RGB local_sh_in = bary_coord.x() * boundary_shs_in_tet[tets[4 * tet_idx]]
		//				+ bary_coord.y() * boundary_shs_in_tet[tets[4 * tet_idx + 1]]
		//				+ bary_coord.z() * boundary_shs_in_tet[tets[4 * tet_idx + 2]]
		//				+ bary_coord.w() * boundary_shs_in_tet[tets[4 * tet_idx + 3]];

		//			float local_inside_density = bary_coord.x() * boundary_inside_density_tet[tets[4 * tet_idx]]
		//				+ bary_coord.y() * boundary_inside_density_tet[tets[4 * tet_idx + 1]]
		//				+ bary_coord.z() * boundary_inside_density_tet[tets[4 * tet_idx + 2]]
		//				+ bary_coord.w() * boundary_inside_density_tet[tets[4 * tet_idx + 3]];
		//			// rgb_residuals[i + k * n_elements] = residual_amplitude * Array3f(1.0f, 0.0f, 0.0f);
		//			boundary_shs_in[i + k * n_elements] = local_sh_in;
		//			boundary_inside_density[i + k * n_elements] = local_inside_density;

		//			break;
		//		}
		//	}
  //      }
    }
}

//static int collected_samples_tested = 0;
//static int collected_samples_num = 0;
//static std::chrono::time_point<std::chrono::system_clock> last_check = std::chrono::system_clock::now();

void CageDeformation::map_rays(cudaStream_t stream, tcnn::PitchedPtr<NerfCoordinate> nerf_coords, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const {
    
    if (!m_growing_selection.tet_interpolation_mesh || m_growing_selection.tet_interpolation_mesh->tets_gpu.size() == 0) {
        return;
    }

	//int zero = 0;
	//cudaMemcpyToSymbol(sample_tested, &zero, sizeof(int));
	//cudaMemcpyToSymbol(sample_num_samples, &zero, sizeof(int));

    // Perform the backward translation
    tcnn::linear_kernel(interpolate_tet, 0, stream,
        n_elements,
        nerf_coords,
        empty_mask.data(),
		m_growing_selection.m_copy,
        m_scene_aabb,
        m_growing_selection.tet_interpolation_mesh->warped_bbox,
        m_growing_selection.tet_interpolation_mesh->original_warped_bbox,
        m_growing_selection.tet_interpolation_mesh->tet_lut_idx.data(),
        m_growing_selection.tet_interpolation_mesh->tet_lut_offsets.data(),
        m_growing_selection.tet_interpolation_mesh->tets_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->vertices_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->original_vertices_gpu.data(),
        (m_growing_selection.tet_interpolation_mesh->local_rotations_gpu.size() > 0) ? m_growing_selection.tet_interpolation_mesh->local_rotations_gpu.data() : nullptr,
        m_growing_selection.tet_interpolation_mesh->original_bitfield_gpu.data()
    );

	//int samples_tested, samples_num_samples;
	//cudaMemcpyFromSymbol(&samples_tested, sample_tested, sizeof(int));
	//cudaMemcpyFromSymbol(&samples_num_samples, sample_num_samples, sizeof(int));

	//if (samples_num_samples > 0)
	//{
	//	collected_samples_tested += samples_tested;
	//	collected_samples_num += samples_num_samples;
	//}

	//if ((std::chrono::system_clock::now() - last_check) > std::chrono::seconds(1))
	//{
	//	last_check = std::chrono::system_clock::now();
	//	std::cout << "Avg. tests: " << collected_samples_tested / ((float)collected_samples_num) << std::endl;;
	//	collected_samples_tested = 0;
	//	collected_samples_num = 0;
	//}
}

void CageDeformation::kill_empty_density(cudaStream_t stream,
	uint32_t n_elements,
	PitchedPtr<NerfPosition> output,
	tcnn::GPUMatrixDynamic<bool>& empty_mask,
	tcnn::network_precision_t* density_network_output) const {

	if (!m_growing_selection.tet_interpolation_mesh || m_growing_selection.tet_interpolation_mesh->tets_gpu.size() == 0) {
		return;
	}

	if (!m_growing_selection.m_plane_dir.isZero())
	{
		tcnn::linear_kernel(kill_empty_density_kernel, 0, stream,
			n_elements,
			empty_mask.data(),
			output,
			density_network_output,
			m_scene_aabb,
			m_growing_selection.m_plane_pos,
			m_growing_selection.m_plane_dir,
			m_growing_selection.tet_interpolation_mesh->original_bbox,
			m_growing_selection.tet_interpolation_mesh->original_tet_lut_idx.data(),
			m_growing_selection.tet_interpolation_mesh->original_tet_lut_offsets.data(),
			m_growing_selection.tet_interpolation_mesh->tets_gpu.data(),
			m_growing_selection.tet_interpolation_mesh->original_vertices_gpu.data(),
			m_growing_selection.m_plane_offset
		);
	}
}

void CageDeformation::map_positions(cudaStream_t stream, tcnn::PitchedPtr<NerfPosition> nerf_pos, tcnn::GPUMatrixDynamic<bool>& empty_mask, uint32_t n_elements) const {
    
    if (!m_growing_selection.tet_interpolation_mesh || m_growing_selection.tet_interpolation_mesh->tets_gpu.size() == 0) {
        return;
    }

    // Perform the backward translation
    tcnn::linear_kernel(interpolate_tet_pos, 0, stream,
        n_elements,
        nerf_pos,
        empty_mask.data(),
        m_scene_aabb,
        m_growing_selection.tet_interpolation_mesh->warped_bbox,
        m_growing_selection.tet_interpolation_mesh->original_warped_bbox,
        m_growing_selection.tet_interpolation_mesh->tet_lut_idx.data(),
        m_growing_selection.tet_interpolation_mesh->tet_lut_offsets.data(),
        m_growing_selection.tet_interpolation_mesh->tets_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->vertices_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->original_vertices_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->original_bitfield_gpu.data()
    );
}

void CageDeformation::compute_poisson_residual_density(
    cudaStream_t stream,  
    const uint32_t n_elements,
    tcnn::PitchedPtr<NerfPosition> input_position,
    network_precision_t* density_network_output
) const {
    if (!m_apply_poisson || !m_growing_selection.tet_interpolation_mesh || m_growing_selection.tet_interpolation_mesh->tets_gpu.size() == 0) {
        return;
    }

    // Perform the residual computation
    tcnn::linear_kernel(compute_poisson_residual_density_kernel, 0, stream,
        n_elements,
        input_position,
        density_network_output,
        m_scene_aabb,
		m_growing_selection.m_plane_pos,
		m_growing_selection.m_plane_dir,
        m_growing_selection.tet_interpolation_mesh->bbox,
        m_growing_selection.tet_interpolation_mesh->tet_lut_idx.data(),
        m_growing_selection.tet_interpolation_mesh->tet_lut_offsets.data(),
        m_growing_selection.tet_interpolation_mesh->tets_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->vertices_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->boundary_residual_density_gpu.data()
    );
}


void CageDeformation::compute_poisson_full_residuals(
    cudaStream_t stream,  
    const uint32_t n_elements,
    NerfPayload* payloads,
    tcnn::PitchedPtr<NerfCoordinate> network_input,
	//SH9RGB* __restrict__ sh_in_boundary,
    SH9RGB* __restrict__ sh_boundary,
	//float* __restrict__ in_density_boundary,
    float* __restrict__ out_density_boundary,
    float* __restrict__ residual_density_boundary
) const {
    if (!m_apply_poisson || !m_growing_selection.tet_interpolation_mesh || m_growing_selection.tet_interpolation_mesh->tets_gpu.size() == 0) {
        return;
    }

    // Perform the residual computation
    tcnn::linear_kernel(compute_residual_poisson_kernel, 0, stream,
        n_elements,
        payloads,
        network_input,
		//sh_in_boundary,
        sh_boundary,
		//in_density_boundary,
        out_density_boundary,
        residual_density_boundary,
        m_scene_aabb,
		m_growing_selection.tet_interpolation_mesh->original_bbox,
        m_growing_selection.tet_interpolation_mesh->bbox,
		m_growing_selection.tet_interpolation_mesh->original_tet_lut_idx.data(),
        m_growing_selection.tet_interpolation_mesh->tet_lut_idx.data(),
		m_growing_selection.tet_interpolation_mesh->original_tet_lut_offsets.data(),
        m_growing_selection.tet_interpolation_mesh->tet_lut_offsets.data(),
        m_growing_selection.tet_interpolation_mesh->tets_gpu.data(),
		m_growing_selection.tet_interpolation_mesh->original_vertices_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->vertices_gpu.data(),
		//m_growing_selection.tet_interpolation_mesh->boundary_shs_in_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->boundary_shs_gpu.data(),
		//m_growing_selection.tet_interpolation_mesh->boundary_inside_density_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->boundary_outside_density_gpu.data(),
        m_growing_selection.tet_interpolation_mesh->boundary_residual_density_gpu.data(),
        m_residual_amplitude
    );
}

#ifdef NGP_GUI

bool CageDeformation::imgui(bool& delete_operator, const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center) {
    bool update_transformation = false;
    if (ImGui::CollapsingHeader("Cage Deformation", ImGuiTreeNodeFlags_DefaultOpen)) {
       
        update_transformation |= m_growing_selection.imgui(resolution, focal_length, camera_matrix, screen_center);

        if (m_growing_selection.tet_interpolation_mesh) {
            //update_transformation |= ImGui::Checkbox("Apply residuals", &m_apply_residuals);
            update_transformation |= ImGui::Checkbox("Apply Membrane Correction (alpha)", &m_apply_poisson);
            update_transformation |= ImGui::SliderFloat("Residuals Amplitude", &m_residual_amplitude, 0.f, 10.f);
        }

        // Delete operator
        if (imgui_colored_button2("Delete Operator", 0)) {
            delete_operator = true;
        }
    }
    return update_transformation | delete_operator;
}

__global__ void constructDeformationDistiller(
	CageDeformationDistiller* distiller,
	bool emptying,
	BoundingBox aabb,
	BoundingBox bbox,
	BoundingBox original_bbox,
	uint32_t* original_tet_lut_idx,
	uint32_t* original_tet_lut_offsets,
	uint32_t* tet_lut_idx,
	uint32_t* tet_lut_offsets,
	uint32_t* tets,
	Eigen::Vector3f* vertices,
	Eigen::Vector3f* original_vertices,
	Eigen::Matrix3f* local_rotations,
	uint8_t* original_bitfield_gpu
	)
{
	new (distiller) CageDeformationDistiller();
	distiller->emptying = emptying;
	distiller->aabb = aabb;
	distiller->bbox = bbox;
	distiller->original_bbox = original_bbox;
	distiller->original_tet_lut_idx = original_tet_lut_idx;
	distiller->original_tet_lut_offsets = original_tet_lut_offsets;
	distiller->tet_lut_idx = tet_lut_idx;
	distiller->tet_lut_offsets = tet_lut_offsets;
	distiller->tets = tets;
	distiller->vertices = vertices;
	distiller->original_vertices = original_vertices;
	distiller->local_rotations = local_rotations;
	distiller->original_bitfield_gpu = original_bitfield_gpu;
}

Distiller* CageDeformation::getDistiller()
{
	if (gpu_distiller == nullptr)
	{
		cudaMalloc(&gpu_distiller, sizeof(CageDeformationDistiller));
	}
	constructDeformationDistiller<<<1,1>>>(
		(CageDeformationDistiller*)gpu_distiller,
		!m_growing_selection.m_copy,
		m_scene_aabb,
		m_growing_selection.tet_interpolation_mesh->bbox,
		m_growing_selection.tet_interpolation_mesh->original_bbox,
		m_growing_selection.tet_interpolation_mesh->original_tet_lut_idx.data(),
		m_growing_selection.tet_interpolation_mesh->original_tet_lut_offsets.data(),
		m_growing_selection.tet_interpolation_mesh->tet_lut_idx.data(),
		m_growing_selection.tet_interpolation_mesh->tet_lut_offsets.data(),
		m_growing_selection.tet_interpolation_mesh->tets_gpu.data(),
		m_growing_selection.tet_interpolation_mesh->vertices_gpu.data(),
		m_growing_selection.tet_interpolation_mesh->original_vertices_gpu.data(),
		(m_growing_selection.tet_interpolation_mesh->local_rotations_gpu.size() > 0) ? m_growing_selection.tet_interpolation_mesh->local_rotations_gpu.data() : nullptr,
		m_growing_selection.tet_interpolation_mesh->original_bitfield_gpu.data()
	);
	return gpu_distiller;
}

bool CageDeformation::handle_keyboard() {
    return false;
}

bool CageDeformation::visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time) {
    return m_growing_selection.visualize_edit_gui(view2proj, world2proj, world2view, focal, aspect, time);;
}
#endif

void CageDeformation::draw_gl(
    const Eigen::Vector2i& resolution, 
    const Eigen::Vector2f& focal_length, 
    const Eigen::Matrix<float, 3, 4>& camera_matrix, 
    const Eigen::Vector2f& screen_center
) {
    m_growing_selection.draw_gl(resolution, focal_length, camera_matrix, screen_center);    
}

nlohmann::json CageDeformation::to_json() {
    nlohmann::json j;

    j["type"] = "cage_deformation";

    m_growing_selection.to_json(j);

    return j;
}


NGP_NAMESPACE_END
