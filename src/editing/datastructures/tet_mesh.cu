#include <neural-graphics-primitives/common_gl.h>
#include <neural-graphics-primitives/editing/datastructures/tet_mesh.h>
#include <neural-graphics-primitives/editing/tools/svd3.h>
#include <tiny-cuda-nn/common.h>
#include <chrono>
#include <thread>
#include <future>
#include <tuple>

NGP_NAMESPACE_BEGIN

template<typename float_t, class point_t>
void TetMesh<float_t, point_t>::post_update_vertices() {
	bbox = BoundingBox();
	for (int i = 0; i < vertices.size(); i++) {
		bbox.enlarge(vertices[i].template cast<float>());
	}
	warped_bbox = bbox;
	warped_bbox.warp_box(m_scene_aabb);
}

// Used to display inner triangles of the tet mesh
template<typename float_t, class point_t>
void TetMesh<float_t, point_t>::update_all_indices()
{
	all_indices.clear();
	all_indices.resize(tets.size() * 3);
	for (int i = 0; i < tets.size() / 4; i++) {
		for (int j = 0; j < 4; j++) {
			all_indices[3 * 4 * i + 3 * j] = tets[4 * i + j];
			all_indices[3 * 4 * i + 3 * j + 1] = tets[4 * i + (j + 1) % 4];
			all_indices[3 * 4 * i + 3 * j + 2] = tets[4 * i + (j + 2) % 4];
		}
	}
}

template<typename float_t, class point_t>
void TetMesh<float_t, point_t>::update_local_rotations(cudaStream_t stream) {
	if (vertices.size() == 0 || vertices.size() != original_vertices.size()) {
		std::cout << "Deformed and canonical vertices are not identical..." << std::endl;
		return;
	}

	const uint32_t n_tets = tets.size() / 4;

	std::vector<Eigen::Matrix3f> local_rotations_host;
	local_rotations_host.reserve(n_tets);

	for (int i = 0; i < n_tets; i++) {
		// Compute centroids
		Eigen::Vector3f canonical_ci = Eigen::Vector3f::Zero();
		Eigen::Vector3f deformed_ci = Eigen::Vector3f::Zero();
		for (int j = 0; j < 4; j++) {
			canonical_ci += original_vertices[tets[4 * i + j]].template cast<float>();
			deformed_ci += vertices[tets[4 * i + j]].template cast<float>();
		}
		canonical_ci /= 4.f;
		deformed_ci /= 4.f;
		// Compute correlation matrix
		Eigen::Matrix3f corr_mat = Eigen::Matrix3f::Zero();
		for (int j = 0; j < 4; j++) {
			corr_mat += (original_vertices[tets[4 * i + j]].template cast<float>() - canonical_ci) * (vertices[tets[4 * i + j]].template cast<float>() - deformed_ci).transpose();
		}
		// Perform SVD
		Eigen::Matrix3f U, S, V;
		svd_eigen(corr_mat, U, S, V);
		// Estimate R
		Eigen::Matrix3f R = U * V.transpose();
		local_rotations_host.push_back(R);
	}

	local_rotations_gpu.resize(local_rotations_host.size());
	local_rotations_gpu.copy_from_host(local_rotations_host);
}

template<typename float_t, class point_t>
void TetMesh<float_t, point_t>::build_original_tet_grid(cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_CASCADES();
	const uint32_t n_tets = tets.size() / 4;

	// Reset tet counts
	std::vector<uint8_t> tet_counts(n_elements, 0);
	uint32_t tet_sum = 0;
	max_tet_lookup = 0;

	// Set a temporary bitfield grid 
	std::vector<bool> original_occupancy_grid(n_elements, false);
	std::vector<uint8_t> original_bitfield(n_elements / 8, 0);

	// FIRST PASS
	// Go through all tets (in deformed space)
	uint32_t marked_tet = 0;
	uint32_t intersection_marked = 0;
	for (int i = 0; i < n_tets; i++) {
		// Define bounding cube
		point_t min = point_t::Constant(std::numeric_limits<float>::infinity());
		point_t max = point_t::Constant(-std::numeric_limits<float>::infinity());
		for (int j = 0; j < 4; j++) {
			min = min.cwiseMin(original_vertices[tets[4 * i + j]]);
			max = max.cwiseMax(original_vertices[tets[4 * i + j]]);
		}
		// Update the lookup at every-level
		for (int level = 0; level < NERF_CASCADES(); level++) {
			float scale = scalbnf(1.0f, level);
			// For each point of the grid inside the bb and the tet overwrite the tet index
			Eigen::Vector3i min_i = get_cell_at_pos(min.template cast<float>(), level);
			Eigen::Vector3i max_i = get_cell_at_pos(max.template cast<float>(), level);
			for (int x = min_i.x(); x <= max_i.x(); x++) {
				for (int y = min_i.y(); y <= max_i.y(); y++) {
					for (int z = min_i.z(); z <= max_i.z(); z++) {
						point_t pos_sample = get_cell_pos(x, y, z, level).template cast<float_t>();
						bool is_inside = false;
						// First test inside
						for (auto& corner_offset : corner_offsets) {
							point_t offseted_pos_sample = pos_sample + corner_offset.cast<float_t>() * scale / NERF_GRIDSIZE();
							if (point_in_tet<float_t, point_t>(original_vertices[tets[4 * i]], original_vertices[tets[4 * i + 1]], original_vertices[tets[4 * i + 2]], original_vertices[tets[4 * i + 3]], offseted_pos_sample)) {
								is_inside = true;
								break;
							}
						}

						// If not, then test intesection
						if (!is_inside) {
							BoundingBox cube_bbox;
							cube_bbox.enlarge(pos_sample.template cast<float>() - Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
							cube_bbox.enlarge(pos_sample.template cast<float>() + Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
							// For each triangle
							for (int j = 0; j < 4; j++) {
								Eigen::Vector3f v0 = original_vertices[tets[4 * i + j]].template cast <float>();
								Eigen::Vector3f v1 = original_vertices[tets[4 * i + (j + 1) % 4]].template cast <float>();
								Eigen::Vector3f v2 = original_vertices[tets[4 * i + (j + 2) % 4]].template cast <float>();
								Triangle t{ v0, v1, v2 };
								// Test intersection with box
								if (cube_bbox.intersects(t)) {
									is_inside = true;
									intersection_marked++;
									break;
								}
							}
						}

						// If intersecting, then mark it!                        
						if (is_inside) {
							uint32_t pos_idx = tcnn::morton3D(x, y, z);
							uint32_t cell_idx = level * NERF_GRIDVOLUME() + pos_idx;
							tet_counts[cell_idx]++;
							tet_sum++;
							marked_tet++;
						}
					}
				}
			}
		}
	}
	std::vector<uint32_t> tet_lut_host(tet_sum, 0);
	original_tet_lut_idx.resize(tet_sum);
	std::vector<uint32_t> tet_lut_offsets_host(n_elements + 1, 0);
	original_tet_lut_offsets.resize(n_elements + 1);
	uint32_t counter = 0;
	for (uint32_t i = 0; i < n_elements; i++) {
		tet_lut_offsets_host[i] = counter;
		counter += tet_counts[i];
		if (tet_counts[i] > max_tet_lookup) {
			max_tet_lookup = tet_counts[i];
		}
	}
	tet_lut_offsets_host[n_elements] = tet_sum;

	// SECOND PASS
	std::vector<uint8_t> tet_allocated(n_elements, 0);
	for (int i = 0; i < n_tets; i++) {
		// Define bounding cube
		point_t min = point_t::Constant(std::numeric_limits<float>::infinity());
		point_t max = point_t::Constant(-std::numeric_limits<float>::infinity());
		for (int j = 0; j < 4; j++) {
			min = min.cwiseMin(original_vertices[tets[4 * i + j]]);
			max = max.cwiseMax(original_vertices[tets[4 * i + j]]);
		}
		// Update the lookup at every-level
		for (int level = 0; level < NERF_CASCADES(); level++) {
			float scale = scalbnf(1.0f, level);
			// For each point of the grid inside the bb and the tet overwrite the tet index
			Eigen::Vector3i min_i = get_cell_at_pos(min.template cast<float>(), level);
			Eigen::Vector3i max_i = get_cell_at_pos(max.template cast<float>(), level);
			for (int x = min_i.x(); x <= max_i.x(); x++) {
				for (int y = min_i.y(); y <= max_i.y(); y++) {
					for (int z = min_i.z(); z <= max_i.z(); z++) {
						point_t pos_sample = get_cell_pos(x, y, z, level).template cast<float_t>();
						bool is_inside = false;
						// First test inside
						for (auto& corner_offset : corner_offsets) {
							point_t offseted_pos_sample = pos_sample + corner_offset.cast<float_t>() * scale / NERF_GRIDSIZE();
							if (point_in_tet<float_t, point_t>(original_vertices[tets[4 * i]], original_vertices[tets[4 * i + 1]], original_vertices[tets[4 * i + 2]], original_vertices[tets[4 * i + 3]], offseted_pos_sample)) {
								is_inside = true;
								break;
							}
						}

						// If not, then test intesection
						if (!is_inside) {
							BoundingBox cube_bbox;
							cube_bbox.enlarge(pos_sample.template cast<float>() - Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
							cube_bbox.enlarge(pos_sample.template cast<float>() + Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
							// For each triangle
							for (int j = 0; j < 4; j++) {
								Eigen::Vector3f v0 = original_vertices[tets[4 * i + j]].template cast <float>();
								Eigen::Vector3f v1 = original_vertices[tets[4 * i + (j + 1) % 4]].template cast <float>();
								Eigen::Vector3f v2 = original_vertices[tets[4 * i + (j + 2) % 4]].template cast <float>();
								Triangle t{ v0, v1, v2 };
								// Test intersection with box
								if (cube_bbox.intersects(t)) {
									is_inside = true;
									intersection_marked++;
									break;
								}
							}
						}

						// If intersecting, then mark it!                        
						if (is_inside) {
							uint32_t pos_idx = tcnn::morton3D(x, y, z);
							uint32_t cell_idx = level * NERF_GRIDVOLUME() + pos_idx;
							tet_lut_host[tet_lut_offsets_host[cell_idx] + tet_allocated[cell_idx]] = i;
							tet_allocated[cell_idx]++;
						}
					}
				}
			}
		}
	}

	original_tet_lut_idx.copy_from_host(tet_lut_host.data());
	original_tet_lut_offsets.copy_from_host(tet_lut_offsets_host.data());

}

static std::vector<std::vector<int2>> clear_cells;

__global__ void clear_grid(int P, int N, int2* to_clear, float* density_grid, uint8_t* bf)
{
	for (int i = 0; i < N; i++)
	{
		int loc = to_clear[i].x;
		int lvl = to_clear[i].y;
		density_grid[lvl * NERF_GRIDVOLUME() + loc] = 0.0f;
		set_bitfield_at(loc, lvl, 0, bf);
	}
}

template<typename float_t, class point_t>
void TetMesh<float_t, point_t>::vanish(float* density_grid_gpu, uint8_t* density_grid_bitfield_gpu, cudaStream_t stream) {
	// If the canonical tet lut wasn't constructed, do it!
	if (original_tet_lut_idx.size() == 0) {
		build_original_tet_grid(stream);
	}

	const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_CASCADES();
	const uint32_t n_tets = tets.size() / 4;

	// Set a temporary bitfield grid 
	std::vector<uint8_t> original_bitfield(n_elements / 8, 0);

	// FIRST PASS
	// Go through all tets (in deformed space)

	int threads_running = std::min(n_tets, 32U);
	clear_cells.resize(threads_running);

	std::vector<std::future<void>> threads(threads_running);
	for (int tt = 0; tt < threads_running; tt++)
	{
		threads[tt] = std::async(std::launch::async, [&, tt]() {

			int beginn = (n_tets / threads_running) * tt;
			int endingg = (tt == threads_running - 1) ? n_tets : (n_tets / threads_running) * (tt + 1);
			clear_cells[tt].clear();

			for (int i = beginn; i < endingg; i++) {
				// Define bounding cube
				point_t min = point_t::Constant(std::numeric_limits<float>::infinity());
				point_t max = point_t::Constant(-std::numeric_limits<float>::infinity());
				for (int j = 0; j < 4; j++) {
					min = min.cwiseMin(vertices[tets[4 * i + j]]);
					max = max.cwiseMax(vertices[tets[4 * i + j]]);
				}
				// Update the lookup at every-level
				for (int level = 0; level < NERF_CASCADES(); level++) {
					float scale = scalbnf(1.0f, level);
					// For each point of the grid inside the bb and the tet overwrite the tet index
					Eigen::Vector3i min_i = get_cell_at_pos(min.template cast<float>(), level);
					Eigen::Vector3i max_i = get_cell_at_pos(max.template cast<float>(), level);
					for (int x = min_i.x(); x <= max_i.x(); x++) {
						for (int y = min_i.y(); y <= max_i.y(); y++) {
							for (int z = min_i.z(); z <= max_i.z(); z++) {
								point_t pos_sample = get_cell_pos(x, y, z, level).template cast<float_t>();
								bool is_inside = false;
								// First test inside
								for (auto& corner_offset : corner_offsets) {
									point_t offseted_pos_sample = pos_sample + corner_offset.cast<float_t>() * scale / NERF_GRIDSIZE();
									if (point_in_tet<float_t, point_t>(vertices[tets[4 * i]], vertices[tets[4 * i + 1]], vertices[tets[4 * i + 2]], vertices[tets[4 * i + 3]], offseted_pos_sample)) {
										is_inside = true;
										break;
									}
								}

								// If not, then test intesection
								if (!is_inside) {
									BoundingBox cube_bbox;
									cube_bbox.enlarge(pos_sample.template cast<float>() - Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
									cube_bbox.enlarge(pos_sample.template cast<float>() + Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
									// For each triangle
									for (int j = 0; j < 4; j++) {
										Eigen::Vector3f v0 = vertices[tets[4 * i + j]].template cast <float>();
										Eigen::Vector3f v1 = vertices[tets[4 * i + (j + 1) % 4]].template cast <float>();
										Eigen::Vector3f v2 = vertices[tets[4 * i + (j + 2) % 4]].template cast <float>();
										Triangle t{ v0, v1, v2 };
										// Test intersection with box
										if (cube_bbox.intersects(t)) {
											is_inside = true;
											break;
										}
									}
								}

								// If intersecting, then mark it!                        
								if (is_inside) {
									uint32_t pos_idx = tcnn::morton3D(x, y, z);
									clear_cells[tt].push_back(int2{ (int)pos_idx, level });
								}
							}
						}
					}
				}
			}
			});
	}

	for (int tt = 0; tt < threads_running; tt++)
		threads[tt].get();

	std::vector<int2> toclear;
	for (int tt = 0; tt < threads_running; tt++)
	{
		for (int i = 0; i < clear_cells[tt].size(); i++)
		{
			toclear.push_back(clear_cells[tt][i]);
		}
	}

	int2* toclear_gpu;
	cudaMalloc(&toclear_gpu, toclear.size() * sizeof(int2));
	cudaMemcpy(toclear_gpu, toclear.data(), toclear.size() * sizeof(int2), cudaMemcpyHostToDevice);

	tcnn::linear_kernel(clear_grid, 0, stream,
		1,
		toclear.size(),
		toclear_gpu,
		density_grid_gpu,
		density_grid_bitfield_gpu
	);

	cudaFree(toclear_gpu);
}

static std::vector<std::vector<std::tuple<int, int, int, int>>> up_ids;
static std::vector<int> tet_sums;

template<typename float_t, class point_t>
void TetMesh<float_t, point_t>::build_tet_grid(cudaStream_t stream) {
	// If the canonical tet lut wasn't constructed, do it!
	if (original_tet_lut_idx.size() == 0) {
		build_original_tet_grid(stream);
	}

	const uint32_t n_elements = NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_CASCADES();
	const uint32_t n_tets = tets.size() / 4;

	// Reset tet counts
	std::vector<uint8_t> tet_counts(n_elements, 0);
	uint32_t tet_sum = 0;
	max_tet_lookup = 0;

	// Set a temporary bitfield grid 
	std::vector<bool> original_occupancy_grid(n_elements, false);
	std::vector<uint8_t> original_bitfield(n_elements / 8, 0);

	// FIRST PASS
	// Go through all tets (in deformed space)
	uint32_t intersection_marked = 0;

	auto before = std::chrono::system_clock::now();

	int threads_running = std::min(n_tets, 32U);
	up_ids.resize(threads_running);
	tet_sums.resize(threads_running);

	std::vector<std::future<void>> threads(threads_running);
	for (int tt = 0; tt < threads_running; tt++)
	{
		threads[tt] = std::async(std::launch::async, [&, tt]() {

			int beginn = (n_tets / threads_running) * tt;
			int endingg = (tt == threads_running - 1) ? n_tets : (n_tets / threads_running) * (tt + 1);
			up_ids[tt].clear();
			tet_sums[tt] = 0;

			for (int i = beginn; i < endingg; i++) {
				// Define bounding cube
				point_t min = point_t::Constant(std::numeric_limits<float>::infinity());
				point_t max = point_t::Constant(-std::numeric_limits<float>::infinity());
				for (int j = 0; j < 4; j++) {
					min = min.cwiseMin(vertices[tets[4 * i + j]]);
					max = max.cwiseMax(vertices[tets[4 * i + j]]);
				}
				// Update the lookup at every-level
				for (int level = 0; level < NERF_CASCADES(); level++) {
					float scale = scalbnf(1.0f, level);
					// For each point of the grid inside the bb and the tet overwrite the tet index
					Eigen::Vector3i min_i = get_cell_at_pos(min.template cast<float>(), level);
					Eigen::Vector3i max_i = get_cell_at_pos(max.template cast<float>(), level);
					for (int x = min_i.x(); x <= max_i.x(); x++) {
						for (int y = min_i.y(); y <= max_i.y(); y++) {
							for (int z = min_i.z(); z <= max_i.z(); z++) {
								point_t pos_sample = get_cell_pos(x, y, z, level).template cast<float_t>();
								bool is_inside = false;
								// First test inside
								for (auto& corner_offset : corner_offsets) {
									point_t offseted_pos_sample = pos_sample + corner_offset.cast<float_t>() * scale / NERF_GRIDSIZE();
									if (point_in_tet<float_t, point_t>(vertices[tets[4 * i]], vertices[tets[4 * i + 1]], vertices[tets[4 * i + 2]], vertices[tets[4 * i + 3]], offseted_pos_sample)) {
										is_inside = true;
										break;
									}
								}

								// If not, then test intesection
								if (!is_inside) {
									BoundingBox cube_bbox;
									cube_bbox.enlarge(pos_sample.template cast<float>() - Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
									cube_bbox.enlarge(pos_sample.template cast<float>() + Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
									// For each triangle
									for (int j = 0; j < 4; j++) {
										Eigen::Vector3f v0 = vertices[tets[4 * i + j]].template cast <float>();
										Eigen::Vector3f v1 = vertices[tets[4 * i + (j + 1) % 4]].template cast <float>();
										Eigen::Vector3f v2 = vertices[tets[4 * i + (j + 2) % 4]].template cast <float>();
										Triangle t{ v0, v1, v2 };
										// Test intersection with box
										if (cube_bbox.intersects(t)) {
											is_inside = true;
											intersection_marked++;
											break;
										}
									}
								}

								// If intersecting, then mark it!                        
								if (is_inside) {
									uint32_t pos_idx = tcnn::morton3D(x, y, z);
									uint32_t cell_idx = level * NERF_GRIDVOLUME() + pos_idx;
									//tet_counts[cell_idx]++;
									up_ids[tt].push_back(std::make_tuple(cell_idx, i, pos_idx, level));
									tet_sums[tt]++;
									//marked_tet++;
								}
							}
						}
					}
				}
			}
			});
	}

	for (int tt = 0; tt < threads_running; tt++)
		threads[tt].get();

	for (int tt = 0; tt < threads_running; tt++)
	{
		tet_sum += tet_sums[tt];
		for (int i = 0; i < up_ids[tt].size(); i++)
			tet_counts[std::get<0>(up_ids[tt][i])]++;
	}

	std::vector<uint32_t> tet_lut_host(tet_sum, 0);
	tet_lut_idx.resize(tet_sum);
	std::vector<uint32_t> tet_lut_offsets_host(n_elements + 1, 0);
	tet_lut_offsets.resize(n_elements + 1);
	uint32_t counter = 0;
	for (uint32_t i = 0; i < n_elements; i++) {
		tet_lut_offsets_host[i] = counter;
		counter += tet_counts[i];
		if (tet_counts[i] > max_tet_lookup) {
			max_tet_lookup = tet_counts[i];
		}
	}
	tet_lut_offsets_host[n_elements] = tet_sum;

	auto after1 = std::chrono::system_clock::now();

	// SECOND PASS
	std::vector<uint8_t> tet_allocated(n_elements, 0);
	for (int tt = 0; tt < threads_running; tt++)
	{
		for (int i = 0; i < up_ids[tt].size(); i++)
		{
			auto p = up_ids[tt][i];
			int cell_idx = std::get<0>(p);
			int tet = std::get<1>(p);
			int pos_idx = std::get<2>(p);
			int level = std::get<3>(p);
			tet_lut_host[tet_lut_offsets_host[cell_idx] + tet_allocated[cell_idx]] = tet;
			tet_allocated[cell_idx]++;
		}
	}

	/*
	for (int i = 0; i < n_tets; i++) {
		// Define bounding cube
		point_t min = point_t::Constant(std::numeric_limits<float>::infinity());
		point_t max = point_t::Constant(-std::numeric_limits<float>::infinity());
		for (int j = 0; j < 4; j++) {
			min = min.cwiseMin(vertices[tets[4*i+j]]);
			max = max.cwiseMax(vertices[tets[4*i+j]]);
		}
		// Update the lookup at every-level
		for (int level = 0; level < NERF_CASCADES(); level++) {
			float scale = scalbnf(1.0f, level);
			// For each point of the grid inside the bb and the tet overwrite the tet index
			Eigen::Vector3i min_i = get_cell_at_pos(min.template cast<float>(), level);
			Eigen::Vector3i max_i = get_cell_at_pos(max.template cast<float>(), level);
			for (int x = min_i.x(); x <= max_i.x(); x++) {
				for (int y = min_i.y(); y <= max_i.y(); y++) {
					for (int z = min_i.z(); z <= max_i.z(); z++) {
						point_t pos_sample = get_cell_pos(x, y, z, level).template cast<float_t>();
						bool is_inside = false;
						// First test inside
						for (auto& corner_offset : corner_offsets) {
							point_t offseted_pos_sample = pos_sample + corner_offset.cast<float_t>() * scale/NERF_GRIDSIZE();
							if(point_in_tet<float_t, point_t>(vertices[tets[4*i]], vertices[tets[4*i+1]], vertices[tets[4*i+2]], vertices[tets[4*i+3]], offseted_pos_sample)) {
								is_inside = true;
								break;
							}
						}

						// If not, then test intesection
						if (!is_inside) {
							BoundingBox cube_bbox;
							cube_bbox.enlarge(pos_sample.template cast<float>() - Eigen::Vector3f::Constant(0.5f)*scale / NERF_GRIDSIZE());
							cube_bbox.enlarge(pos_sample.template cast<float>() + Eigen::Vector3f::Constant(0.5f)*scale / NERF_GRIDSIZE());
							// For each triangle
							for (int j = 0; j < 4; j++) {
								Eigen::Vector3f v0 = vertices[tets[4*i + j]].template cast <float>();
								Eigen::Vector3f v1 = vertices[tets[4*i + (j+1)%4]].template cast <float>();
								Eigen::Vector3f v2 = vertices[tets[4*i + (j+2)%4]].template cast <float>();
								Triangle t {v0, v1, v2};
								// Test intersection with box
								if (cube_bbox.intersects(t)) {
									is_inside = true;
									intersection_marked++;
									break;
								}
							}
						}

						// If intersecting, then mark it!
						if (is_inside) {
							uint32_t pos_idx = tcnn::morton3D(x, y, z);
							uint32_t cell_idx = level * NERF_GRIDVOLUME() + pos_idx;
							tet_lut_host[tet_lut_offsets_host[cell_idx] + tet_allocated[cell_idx]] = i;
							tet_allocated[cell_idx]++;
						}
					}
				}
			}
		}
	}
	*/
	auto after2 = std::chrono::system_clock::now();

	for (int tt = 0; tt < threads_running; tt++)
	{
		threads[tt] = std::async(std::launch::async, [&, tt]() {

			int beginn = (n_tets / threads_running) * tt;
			int endingg = (tt == threads_running - 1) ? n_tets : (n_tets / threads_running) * (tt + 1);
			// Do roughly the same to update the occupancy grid in the canonical tet mesh
			for (int i = beginn; i < endingg; i++) {
				// Define bounding cube
				point_t min = point_t::Constant(std::numeric_limits<float>::infinity());
				point_t max = point_t::Constant(-std::numeric_limits<float>::infinity());
				for (int j = 0; j < 4; j++) {
					min = min.cwiseMin(original_vertices[tets[4 * i + j]]);
					max = max.cwiseMax(original_vertices[tets[4 * i + j]]);
				}
				// Update the lookup at every-level
				for (int level = 0; level < NERF_CASCADES(); level++) {
					float scale = scalbnf(1.0f, level);
					// For each point of the grid inside the bb and the tet overwrite the tet index
					Eigen::Vector3i min_i = get_cell_at_pos(min.template cast<float>(), level);
					Eigen::Vector3i max_i = get_cell_at_pos(max.template cast<float>(), level);
					for (int x = min_i.x(); x <= max_i.x(); x++) {
						for (int y = min_i.y(); y <= max_i.y(); y++) {
							for (int z = min_i.z(); z <= max_i.z(); z++) {
								point_t pos_sample = get_cell_pos(x, y, z, level).template cast<float_t>();
								bool is_inside = false;
								// First test inside
								for (auto& corner_offset : corner_offsets) {
									point_t offseted_pos_sample = pos_sample + corner_offset.cast<float_t>() * scale / NERF_GRIDSIZE();
									if (point_in_tet<float_t, point_t>(original_vertices[tets[4 * i]], original_vertices[tets[4 * i + 1]], original_vertices[tets[4 * i + 2]], original_vertices[tets[4 * i + 3]], offseted_pos_sample)) {
										is_inside = true;
										break;
									}
								}

								// If not, then test intesection
								if (!is_inside) {
									BoundingBox cube_bbox;
									cube_bbox.enlarge(pos_sample.template cast<float>() - Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
									cube_bbox.enlarge(pos_sample.template cast<float>() + Eigen::Vector3f::Constant(0.5f) * scale / NERF_GRIDSIZE());
									// For each triangle
									for (int j = 0; j < 4; j++) {
										Eigen::Vector3f v0 = original_vertices[tets[4 * i + j]].template cast <float>();
										Eigen::Vector3f v1 = original_vertices[tets[4 * i + (j + 1) % 4]].template cast <float>();
										Eigen::Vector3f v2 = original_vertices[tets[4 * i + (j + 2) % 4]].template cast <float>();
										Triangle t{ v0, v1, v2 };
										// Test intersection with box
										if (cube_bbox.intersects(t)) {
											is_inside = true;
											break;
										}
									}
								}

								// If intersecting, then mark it!                        
								if (is_inside) {
									uint32_t pos_idx = tcnn::morton3D(x, y, z);
									set_bitfield_at(pos_idx, level, true, original_bitfield.data()); //hoho, a race
								}
							}
						}
					}
				}
			}
			});
	}

	for (int tt = 0; tt < threads_running; tt++)
		threads[tt].get();

	auto after3 = std::chrono::system_clock::now();

	// Copy to GPU
	tets_gpu.resize(tets.size());
	vertices_gpu.resize(vertices.size());
	original_vertices_gpu.resize(original_vertices.size());
	original_bitfield_gpu.resize(n_elements / 8);
	// initial_shs_gpu.resize(vertices.size());
	// new_shs_gpu.resize(vertices.size());
	boundary_residual_density_gpu.resize(vertices.size());
	// boundary_inside_density_gpu.resize(vertices.size());
	boundary_outside_density_gpu.resize(vertices.size());
	// boundary_shs_in_gpu.resize(vertices.size());
	boundary_shs_gpu.resize(vertices.size());
	tets_gpu.copy_from_host(tets);
	vertices_gpu.copy_from_host(vertices);
	original_vertices_gpu.copy_from_host(original_vertices);
	original_bitfield_gpu.copy_from_host(original_bitfield);
	tet_lut_idx.copy_from_host(tet_lut_host.data());
	tet_lut_offsets.copy_from_host(tet_lut_offsets_host.data());
	//std::cout << "Marked tets: " << marked_tet << " including " << intersection_marked << " marked with intersection test" << std::endl;

	auto after4 = std::chrono::system_clock::now();

	//std::cout << (after1 - before).count() << " " << (after2 - before).count() << " " << (after3 - before).count() << " " << (after4 - before).count() << "\n";
}

template<typename float_t, class point_t>
void TetMesh<float_t, point_t>::draw_gl(
	const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center,
	const bool display_in
) {
	if (vertices.size() == 0 || indices.size() == 0) {
		return;
	}

	if (!VAO) {
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);
	}
	if (vbosize != vertices.size()) {
		// If necessary, delete the VBO
		for (int i = 0; i < 4; ++i) {
			if (VBO[i]) {
				glDeleteBuffers(1, &VBO[i]);
			}
		}
		// VBO for positions
		glGenBuffers(1, &VBO[0]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(point_t), &vertices[0], GL_STATIC_DRAW);
		// VBO for labels
		glGenBuffers(1, &VBO[1]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
		glBufferData(GL_ARRAY_BUFFER, labels.size() * sizeof(uint8_t), &labels[0], GL_STATIC_DRAW);
		// VBO for colors
		glGenBuffers(1, &VBO[2]);
		glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f), &colors[0], GL_STATIC_DRAW);
	}
	std::vector<uint32_t>& displayed_indices = display_in ? all_indices : indices;
	if (ebosize != displayed_indices.size()) {
		if (EBO) {
			glDeleteBuffers(1, &EBO);
		}
		glGenBuffers(1, &EBO);
		ebosize = displayed_indices.size();
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, ebosize * sizeof(uint32_t), &displayed_indices[0], GL_STATIC_DRAW);
	}

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(point_t), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, labels.size() * sizeof(uint8_t), &labels[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(Eigen::Vector3f), &colors[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, ebosize * sizeof(uint32_t), &displayed_indices[0], GL_STATIC_DRAW);

	if (!program) {
		vs = compile_shader(false, R"foo(
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 nor;
layout (location = 2) in int label;
layout (location = 3) in vec3 col;
out vec3 vtxcol;
flat out int fLabel;
uniform mat4 camera;
uniform vec2 f;
uniform ivec2 res;
uniform vec2 cen;
uniform int mode;
void main()
{
	vec4 p = camera * vec4(pos, 1.0);
	p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
	p.w = p.z;
	p.z = p.z - 0.1;
	p.xy += cen * p.w;
    if (mode == 2) {
        vtxcol = col;
    }
	else {
		vtxcol = vec3(1.0, 0.0, 0.0);
	}
	gl_Position = p;
    fLabel = label;
}
)foo");
		ps = compile_shader(true, R"foo(
layout (location = 0) out vec4 o;
in vec3 vtxcol;
flat in int fLabel;
uniform int mode;
void main() {
	if (mode == 2) {
		if (fLabel == 1) {
            o = vec4(0.0, 0.0, 1.0, 1.0);
        } else {
            o = vec4(0.0, 1.0, 0.0, 1.0);
        }
	} else {
		o = vec4(vtxcol, 1.0);
	}
}
)foo");
		program = glCreateProgram();
		glAttachShader(program, vs);
		glAttachShader(program, ps);
		glLinkProgram(program);
		if (!check_shader(program, "shader program", true)) {
			glDeleteProgram(program);
			program = 0;
		}
	}
	Eigen::Matrix4f view2world = Eigen::Matrix4f::Identity();
	view2world.block<3, 4>(0, 0) = camera_matrix;
	Eigen::Matrix4f world2view = view2world.inverse();
	glBindVertexArray(VAO);
	glUseProgram(program);
	glUniformMatrix4fv(glGetUniformLocation(program, "camera"), 1, GL_FALSE, (GLfloat*)&world2view);
	glUniform2f(glGetUniformLocation(program, "f"), focal_length.x(), focal_length.y());
	glUniform2f(glGetUniformLocation(program, "cen"), screen_center.x() * 2.f - 1.f, screen_center.y() * -2.f + 1.f);
	glUniform2i(glGetUniformLocation(program, "res"), resolution.x(), resolution.y());
	glUniform1i(glGetUniformLocation(program, "mode"), (int)render_mode);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	GLuint posat = (GLuint)glGetAttribLocation(program, "pos");
	GLuint labat = (GLuint)glGetAttribLocation(program, "label");
	GLuint colat = (GLuint)glGetAttribLocation(program, "col");
	glEnableVertexAttribArray(posat);
	glEnableVertexAttribArray(labat);
	glEnableVertexAttribArray(colat);
	GLenum float_enum = std::is_same<float_t, float>::value ? GL_FLOAT : GL_DOUBLE;
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(posat, 3, float_enum, GL_FALSE, 3 * sizeof(float_t), 0);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glVertexAttribIPointer(labat, 1, GL_UNSIGNED_BYTE, sizeof(uint8_t), 0);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glVertexAttribPointer(colat, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawElements(GL_TRIANGLES, (GLsizei)displayed_indices.size(), GL_UNSIGNED_INT, (GLvoid*)0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_CULL_FACE);

	glUseProgram(0);
}

template class TetMesh<float, Eigen::Vector3f>;
template class TetMesh<double, Eigen::Vector3d>;

NGP_NAMESPACE_END
