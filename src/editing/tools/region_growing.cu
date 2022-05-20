#include <neural-graphics-primitives/editing/tools/region_growing.h>
#include <neural-graphics-primitives/editing/tools/selection_utils.h>
#include <neural-graphics-primitives/common_nerf.h>

#include <tiny-cuda-nn/common_device.h>

NGP_NAMESPACE_BEGIN

// Reset the growing selection grid
void RegionGrowing::reset_growing(const std::vector<uint32_t>& selected_cells, int growing_level) {
    // Copy the density grid
    m_density_grid_host.resize(m_density_grid.size());
    m_density_grid.copy_to_host(m_density_grid_host);

    // Reset the selection grid (0 empty, 1 selected)	
    m_selection_grid_bitfield = std::vector<uint8_t>(m_density_grid_bitfield.size(), 0);

    // Reset the growing queue
    m_growing_queue = std::queue<uint32_t>();

    uint32_t n_rays = selected_cells.size();

    // Reset the points (used for visualization)
    m_selection_points.clear();
    m_selection_cell_idx.clear();
    m_selection_points.reserve(n_rays);
    m_selection_cell_idx.reserve(n_rays);

    for (int i = 0; i < n_rays; i++) {
        uint32_t cell_idx = selected_cells[i];
        uint32_t level = cell_idx / NERF_GRIDVOLUME();

        // If it's bigger than the requested level, discard it
        if (level > growing_level) {
            continue;
        }
        // If it is smaller then uplift!
        if (level < growing_level) {
            cell_idx = get_upper_cell_idx(cell_idx, growing_level);
        };
        level = cell_idx / NERF_GRIDVOLUME();

        // Add all pixels to their reprojected coordinate in the queue
        m_growing_queue.push(cell_idx);

        // Add visualization points
        // Invert morton coordinates to get xyz
        uint32_t pos_idx = cell_idx % NERF_GRIDVOLUME();
        uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
        uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
        uint32_t z = tcnn::morton3D_invert(pos_idx>>2);
        m_selection_points.push_back(get_cell_pos(x, y, z, level));
        m_selection_cell_idx.push_back(cell_idx);
    }
}

void RegionGrowing::grow_region(float density_threshold, ERegionGrowingMode region_growing_mode, int growing_level, int growing_steps) {
    // Make sure we can actually grow!
    if (m_growing_queue.empty()) {
        std::cout << "Growing queue is empty!" << std::endl;
        return;
    }

    int i = 1;

    if (region_growing_mode == ERegionGrowingMode::Manual) {
        while (!m_growing_queue.empty() && i <= growing_steps) {
            uint32_t current_cell = m_growing_queue.front();
            float current_density = m_density_grid_host[current_cell];
            m_growing_queue.pop();

            // Get position (with corresponding level) to fetch neighbours
            const uint32_t level = current_cell / (NERF_GRIDVOLUME());
            const uint32_t pos_idx = current_cell % (NERF_GRIDVOLUME());

            // Sample accepted only if at requested level, statisfying density threshold and not already selected!
            if (!get_bitfield_at(pos_idx, level, m_selection_grid_bitfield.data()) && current_density >= density_threshold && level == growing_level) {
                // Invert morton coordinates to get xyz
                uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
                uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
                uint32_t z = tcnn::morton3D_invert(pos_idx>>2);
                // Add possible neighbours
                add_neighbours(m_growing_queue, x, y, z, level);

                // Mark the current cell
                Eigen::Vector3f cell_pos = get_cell_pos(x, y, z, level);
                m_selection_points.push_back(cell_pos);
                m_selection_cell_idx.push_back(current_cell);
                set_bitfield_at(pos_idx, level, true, m_selection_grid_bitfield.data());
            }
            i++;
        }
    }
    // TODO: not supported yet!!!!!!! 
    else {
        // // Compute features and test distances for all the cells currently in the queue
        // std::vector<uint32_t> tentative_cells;
        // std::vector<NerfCoordinate> tentative_coordinates;
        // std::vector<FeatureVector> tentative_features;
        // // Compute the corresponding coordinates
        // for (int j = 0; j < tentative_cells.size(); j++) {

        // }
        // while (!growing_queue.empty() && i <= growing_steps) {
        //     uint32_t current_cell = growing_queue.front();
        //     float current_density = density_grid_host[current_cell];
        //     growing_queue.pop();
        //     // Test for density!
        //     if (current_density >= density_threshold) {

        //     }
        //     i++;
        // }
        // // Test 
    }
    std::cout << "Selected " << m_selection_points.size() << " points overall" << std::endl;
}

// Queue needs to be copied because we'll exhaust it
template <typename T>
inline void to_json_queue(nlohmann::json& j, std::queue<T> queue) {
	std::vector<T> tmp_vec;
    tmp_vec.reserve(queue.size());
    while (!queue.empty()) {
        tmp_vec.push_back(queue.front());
        queue.pop();
    }
	to_json(j, tmp_vec);
}

template <typename T>
inline void from_json_queue(const nlohmann::json& j, std::queue<T>& queue) {
	std::vector<T> tmp_vec = j.get<std::vector<T>>();
	for (auto item: tmp_vec) {
		queue.push(item);
	}
}

nlohmann::json RegionGrowing::to_json() {
        nlohmann::json j;

        j["selection_grid_bitfield"] = m_selection_grid_bitfield;
        j["selection_points"] = m_selection_points;
        j["selection_cell_idx"] = m_selection_cell_idx;
        j["density_grid_host"] = m_density_grid_host;
        // TODO: support saving of queue
        // to_json_queue<uint32_t>(j["growing_queue"], m_growing_queue);

        return j;
    }

void RegionGrowing::load_json(nlohmann::json& j) {
    std::cout << "most" << std::endl;
    from_json(j["selection_grid_bitfield"], m_selection_grid_bitfield);
    from_json(j["selection_points"], m_selection_points);
    from_json(j["selection_cell_idx"], m_selection_cell_idx);
    from_json(j["density_grid_host"], m_density_grid_host);
    // TODO: support reloading of the queue
    // from_json_queue<uint32_t>(j["growing_queue"], m_growing_queue);
}

NGP_NAMESPACE_END