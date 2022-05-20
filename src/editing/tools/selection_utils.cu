#include <neural-graphics-primitives/editing/tools/selection_utils.h>
#include <cfloat>

using namespace Eigen;

NGP_NAMESPACE_BEGIN




void add_neighbours(std::queue<uint32_t>& growing_queue, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t level) {
	if (x > 0) {
		growing_queue.push(level*NERF_GRIDVOLUME() + tcnn::morton3D(x-1, y, z));
	}
	if (y > 0) {
		growing_queue.push(level*NERF_GRIDVOLUME() + tcnn::morton3D(x, y-1, z));	
	}
	if (z > 0) {
		growing_queue.push(level*NERF_GRIDVOLUME() + tcnn::morton3D(x, y, z-1));	
	}
	if (x < (int)NERF_GRIDSIZE()-1) {
		growing_queue.push(level*NERF_GRIDVOLUME() + tcnn::morton3D(x+1, y, z));
	}
	if (y < (int)NERF_GRIDSIZE()-1) {
		growing_queue.push(level*NERF_GRIDVOLUME() + tcnn::morton3D(x, y+1, z));
	}
	if (z < (int)NERF_GRIDSIZE()-1) {
		growing_queue.push(level*NERF_GRIDVOLUME() + tcnn::morton3D(x, y, z+1));
	}
}

uint32_t get_upper_cell_idx(const uint32_t cell_idx, const uint32_t target_level) {
	const uint32_t init_level = cell_idx / NERF_GRIDVOLUME();
	const uint32_t init_pos_idx = cell_idx % NERF_GRIDVOLUME();
	uint32_t x = tcnn::morton3D_invert(init_pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(init_pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(init_pos_idx>>2);
	for (int i = init_level; i < target_level; i++) {
		x = x/2+NERF_GRIDSIZE()/4;
		y = y/2+NERF_GRIDSIZE()/4;
		z = z/2+NERF_GRIDSIZE()/4;
	}
	return target_level * NERF_GRIDVOLUME() + tcnn::morton3D(x, y, z);
}

void add_upper_levels(std::queue<uint32_t>& growing_queue, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t level) {
	uint32_t up_x = x;
	uint32_t up_y = z;
	uint32_t up_z = z;
	for (int i = level; i < NERF_CASCADES() - 1; i++) {
		up_x = up_x/2+NERF_GRIDSIZE()/4;
		up_y = up_y/2+NERF_GRIDSIZE()/4;
		up_z = up_z/2+NERF_GRIDSIZE()/4;
		growing_queue.push((i+1)*NERF_GRIDVOLUME() + tcnn::morton3D(up_x, up_y, up_z));
	}
}



// TODO: handle MIP
Eigen::Vector3f get_cell_pos(const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t level) {
	Eigen::Vector3f pos = ((Eigen::Vector3f{(float)x+0.5f, (float)y+0.5f, (float)z+0.5f}) / NERF_GRIDSIZE() - Eigen::Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Eigen::Vector3f::Constant(0.5f);
	return pos;
}

Eigen::Vector3i get_cell_at_pos(Eigen::Vector3f pos, const uint32_t level) {
	float mip_scale = scalbnf(1.0f, -level);
	pos -= Vector3f::Constant(0.5f);
	pos *= mip_scale;
	pos += Vector3f::Constant(0.5f);

	Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

	return Eigen::Vector3i{
		tcnn::clamp(i.x(), 0, (int)NERF_GRIDSIZE()-1),
		tcnn::clamp(i.y(), 0, (int)NERF_GRIDSIZE()-1),
		tcnn::clamp(i.z(), 0, (int)NERF_GRIDSIZE()-1)
	};
}

bool check_boundary(const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t cell_idx) {
	bool is_boundary = false;
	uint32_t x = tcnn::morton3D_invert(cell_idx>>0);
	uint32_t y = tcnn::morton3D_invert(cell_idx>>1);
	uint32_t z = tcnn::morton3D_invert(cell_idx>>2);
	if (x > 0) {
		// std::cout << "x" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x-1, y, z), 0, selection_grid_bitfield) << std::endl;
		is_boundary |= !get_bitfield_at(tcnn::morton3D(x-1, y, z), 0, selection_grid_bitfield.data());
	}
	if (y > 0) {
		// std::cout << "y" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x, y-1, z), 0, selection_grid_bitfield) << std::endl;
		is_boundary |= !get_bitfield_at(tcnn::morton3D(x, y-1, z), 0, selection_grid_bitfield.data());
	}
	if (z > 0) {
		// std::cout << "z" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x, y, z-1), 0, selection_grid_bitfield) << std::endl;
		is_boundary |= !get_bitfield_at(tcnn::morton3D(x, y, z-1), 0, selection_grid_bitfield.data());
	}
	if (x < (int)NERF_GRIDSIZE()-1) {
		// std::cout << "x+" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x+1, y, z), 0, selection_grid_bitfield) << std::endl;
		is_boundary |= !get_bitfield_at(tcnn::morton3D(x+1, y, z), 0, selection_grid_bitfield.data());
	}
	if (y < (int)NERF_GRIDSIZE()-1) {
		// std::cout << "y+" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x, y+1, z), 0, selection_grid_bitfield) << std::endl;
		is_boundary |= !get_bitfield_at(tcnn::morton3D(x, y+1, z), 0, selection_grid_bitfield.data());
	}
	if (z < (int)NERF_GRIDSIZE()-1) {
		// std::cout << "z+" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x, y, z+1), 0, selection_grid_bitfield) << std::endl;
		is_boundary |= !get_bitfield_at(tcnn::morton3D(x, y, z+1), 0, selection_grid_bitfield.data());
	}
	return is_boundary;
}

Eigen::Vector3f get_boundary_normal(const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t cell_idx) {
	Eigen::Vector3f boundary_normal = Eigen::Vector3f::Zero();
	uint32_t x = tcnn::morton3D_invert(cell_idx>>0);
	uint32_t y = tcnn::morton3D_invert(cell_idx>>1);
	uint32_t z = tcnn::morton3D_invert(cell_idx>>2);
	if (x > 0 && !get_bitfield_at(tcnn::morton3D(x-1, y, z), 0, selection_grid_bitfield.data())) {
		boundary_normal += Eigen::Vector3f(-1.0f, 0.0f, 0.0f);
	}
	if (y > 0 && !get_bitfield_at(tcnn::morton3D(x, y-1, z), 0, selection_grid_bitfield.data())) {
		boundary_normal += Eigen::Vector3f(0.0f, -1.0f, 0.0f);
	}
	if (z > 0 && !get_bitfield_at(tcnn::morton3D(x, y, z-1), 0, selection_grid_bitfield.data())) {
		boundary_normal += Eigen::Vector3f(0.0f, 0.0f, -1.0f);
	}
	if (x < (int)NERF_GRIDSIZE()-1) {
		boundary_normal += Eigen::Vector3f(1.0f, 0.0f, 0.0f);
	}
	if (y < (int)NERF_GRIDSIZE()-1) {
		boundary_normal += Eigen::Vector3f(0.0f, 1.0f, 0.0f);
	}
	if (z < (int)NERF_GRIDSIZE()-1) {
		boundary_normal += Eigen::Vector3f(0.0f, 0.0f, 1.0f);
	}
	boundary_normal.normalize();
	return boundary_normal;
}

std::vector<float> rescale_densities(const std::vector<float>& densities) {
	float min_density = FLT_MAX;
	float max_density = FLT_MIN;
	for (const auto& density: densities) {
		if (density < min_density) {
			min_density = density;
		}
		if (density > max_density) {
			max_density = density;
		}
	};
	std::vector<float> scaled_densities;
	if (max_density > min_density) {
		scaled_densities.reserve(densities.size());
		for (const auto& density: densities) {
			// float scaled_density = (density < 0.f) ? 0.f : ((density>0.5f) ? 0.5f : density);
			scaled_densities.push_back((density - min_density)/(max_density-min_density));
		}
	} else {
		scaled_densities = densities;
	}
	
	return scaled_densities;
}

void update_selection_attributes(const std::vector<uint8_t>& selection_grid_bitfield, 
	const std::vector<float>& density_grid_host, 
	std::vector<Eigen::Vector3f>& selection_points, 
	std::vector<uint8_t>& selection_labels,
	std::vector<float>& selection_densities, 
	std::vector<float>& scaled_selection_densities, 
	std::vector<uint32_t>& selection_cell_idx) 
{
	selection_points.clear();
	selection_labels.clear();
	selection_densities.clear();
	scaled_selection_densities.clear();
	selection_cell_idx.clear();
	for (uint32_t level = 0; level < NERF_CASCADES(); level++) {
		for (uint32_t x = 0; x < NERF_GRIDSIZE(); x++) {
			for (uint32_t y = 0; y < NERF_GRIDSIZE(); y++) {
				for (uint32_t z = 0; z < NERF_GRIDSIZE(); z++) {
					uint32_t pos_idx = tcnn::morton3D(x, y, z);
					uint32_t current_cell = level * NERF_GRIDVOLUME() + pos_idx;
					// Make sure it as not already been set!
					if (get_bitfield_at(pos_idx, level, selection_grid_bitfield.data())) {
						Eigen::Vector3f pos = get_cell_pos(x, y, z, level);
						selection_points.push_back(pos);
						selection_labels.push_back(0);
						// TODO: change this arbitrary density
						selection_densities.push_back(density_grid_host[current_cell]);
						selection_cell_idx.push_back(current_cell); 
					}
				}
			}
		}

	}
	scaled_selection_densities = rescale_densities(selection_densities);
}

bool check_neighbour(const std::vector<uint8_t>& selection_grid_bitfield, const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t level, const EOperatorMM operator_mm) {
	bool check_value = false;
	if (x > 0) {
		// std::cout << "x" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x-1, y, z), 0, selection_grid_bitfield) << std::endl;
		check_value |= !((operator_mm == EOperatorMM::Max) ^ get_bitfield_at(tcnn::morton3D(x-1, y, z), level, selection_grid_bitfield.data()));
	}
	if (y > 0) {
		// std::cout << "y" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x, y-1, z), 0, selection_grid_bitfield) << std::endl;
		check_value |= !((operator_mm == EOperatorMM::Max) ^ get_bitfield_at(tcnn::morton3D(x, y-1, z), level, selection_grid_bitfield.data()));
	}
	if (z > 0) {
		// std::cout << "z" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x, y, z-1), 0, selection_grid_bitfield) << std::endl;
		check_value |= !((operator_mm == EOperatorMM::Max) ^ get_bitfield_at(tcnn::morton3D(x, y, z-1), level, selection_grid_bitfield.data()));
	}
	if (x < (int)NERF_GRIDSIZE()-1) {
		// std::cout << "x+" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x+1, y, z), 0, selection_grid_bitfield) << std::endl;
		check_value |= !((operator_mm == EOperatorMM::Max) ^ get_bitfield_at(tcnn::morton3D(x+1, y, z), level, selection_grid_bitfield.data()));
	}
	if (y < (int)NERF_GRIDSIZE()-1) {
		// std::cout << "y+" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x, y+1, z), 0, selection_grid_bitfield) << std::endl;
		check_value |= !((operator_mm == EOperatorMM::Max) ^ get_bitfield_at(tcnn::morton3D(x, y+1, z), level, selection_grid_bitfield.data()));
	}
	if (z < (int)NERF_GRIDSIZE()-1) {
		// std::cout << "z+" << std::endl;
		// std::cout << get_bitfield_at(tcnn::morton3D(x, y, z+1), 0, selection_grid_bitfield) << std::endl;
		check_value |= !((operator_mm == EOperatorMM::Max) ^ get_bitfield_at(tcnn::morton3D(x, y, z+1), level, selection_grid_bitfield.data()));
	}
	return check_value;
}

NGP_NAMESPACE_END
