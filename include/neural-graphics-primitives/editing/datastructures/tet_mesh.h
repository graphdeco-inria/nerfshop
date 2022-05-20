#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_nerf.h>
#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/editing/datastructures/mesh.h>
#include <neural-graphics-primitives/editing/tools/selection_utils.h>
#include <neural-graphics-primitives/common_gl.h>

#include <tiny-cuda-nn/common.h>

#include <vector>

#ifdef NGP_GUI
#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GL/glu.h>
#  include <GLFW/glfw3.h>
#  include <cuda_gl_interop.h>
#endif

NGP_NAMESPACE_BEGIN

enum class ETetMeshRenderMode : int {
	UniformColor,
	Labels,
	Colors,
};
static constexpr const char* TetMeshRenderModeStr = "UniformColor\0Labels\0Colors\0\0";

static const std::vector<Eigen::Vector3f> corner_offsets = {
    Eigen::Vector3f(-0.5f, -0.5f, -0.5f),
    Eigen::Vector3f(-0.5f, -0.5f, 0.5f),
    Eigen::Vector3f(-0.5f, 0.5f, -0.5f),
    Eigen::Vector3f(0.5f, -0.5f, -0.5f),
    Eigen::Vector3f(0.5f, 0.5f, -0.5f),
    Eigen::Vector3f(-0.5f, 0.5f, 0.5f),
    Eigen::Vector3f(0.5f, -0.5f, 0.5f),
    Eigen::Vector3f(0.5f, 0.5f, 0.5f),
};

static const Eigen::Vector3f default_tet_color = Eigen::Vector3f(1.f, 0.f, 0.f);

template<typename float_t, class point_t>
struct TetMesh : public Mesh<float_t, point_t> {
public:
    BoundingBox bbox;
    BoundingBox original_bbox;
    BoundingBox warped_bbox;
    BoundingBox original_warped_bbox;

    using Mesh<float_t, point_t>::vertices;
    using Mesh<float_t, point_t>::indices;
    // using Mesh<float_t, point_t>::normals;

    std::vector<point_t> original_vertices;
    std::vector<std::vector<float_t>> mvc_coordinates; 
    std::vector<std::vector<float_t>> gamma_coordinates;
    std::vector<uint32_t> tets;
    std::vector<uint8_t> labels;
    std::vector<Eigen::Vector3f> colors;

    // This includes inner vertices
    std::vector<uint32_t> all_indices;

    int max_tet_lookup = 0; // Maximum number of tets stored in a cell (DEBUG)

	std::vector<uint8_t> m_tet_counts;
	// Set a temporary bitfield grid 
	std::vector<bool> m_original_occupancy_grid;
	std::vector<uint8_t> m_original_bitfield;

    // ------------------------
    // GPU Memory
    // ------------------------
    
    tcnn::GPUMemory<uint32_t> tet_lut_idx;
    tcnn::GPUMemory<uint32_t> tet_lut_offsets;
    tcnn::GPUMemory<uint32_t> original_tet_lut_idx;
    tcnn::GPUMemory<uint32_t> original_tet_lut_offsets;
    tcnn::GPUMemory<uint32_t> tets_gpu;
    tcnn::GPUMemory<point_t> vertices_gpu;
    tcnn::GPUMemory<point_t> original_vertices_gpu;
    tcnn::GPUMemory<uint8_t> original_bitfield_gpu; // Set to 1 if intersects a tet from the original undeformed tetmesh

    tcnn::GPUMemory<Eigen::Matrix3f> local_rotations_gpu;
    
    // Poisson editing
    tcnn::GPUMemory<float> boundary_outside_density_gpu;
    tcnn::GPUMemory<float> boundary_residual_density_gpu;
    tcnn::GPUMemory<SH9RGB> boundary_shs_gpu;

    ETetMeshRenderMode render_mode = ETetMeshRenderMode::Colors;

    TetMesh() {}

    TetMesh(std::vector<point_t>& _vertices, std::vector<uint32_t>& _indices, std::vector<uint32_t>& _tets, BoundingBox aabb) : Mesh<float_t, point_t>{_vertices, _indices}, original_vertices{_vertices}, tets{_tets}, m_scene_aabb{aabb} {
        labels.resize(vertices.size());
        colors.resize(vertices.size(), default_tet_color);
        update_all_indices();
        post_update_vertices();
        original_bbox = bbox;
        original_warped_bbox = warped_bbox;
    }

    void draw_gl(
        const Eigen::Vector2i& resolution,
        const Eigen::Vector2f& focal_length,
        const Eigen::Matrix<float, 3, 4>& camera_matrix,
        const Eigen::Vector2f& screen_center,
        const bool display_in = false
    );

    void post_update_vertices();

    // Used to display inner triangles of the tet mesh
    void update_all_indices();

    void update_local_rotations(cudaStream_t stream);

    void build_original_tet_grid(cudaStream_t stream);

	void vanish(float* density_grid, uint8_t* bitfield, cudaStream_t stream);

    void build_tet_grid(cudaStream_t stream);

private:
    GLuint VAO = 0, VBO[4] = {}, EBO = 0, vbosize = 0, ebosize = 0, program = 0, vs = 0, ps = 0;
    BoundingBox m_scene_aabb;
};


template<typename float_t, class point_t>
inline void to_json(nlohmann::json& j, const TetMesh<float_t, point_t>& tet_mesh) {
    to_json(j["bbox"], tet_mesh.bbox);
    to_json(j["original_bbox"], tet_mesh.original_bbox);
    to_json(j["warped_bbox"], tet_mesh.warped_bbox);
    to_json(j["original_warped_bbox"], tet_mesh.original_warped_bbox);

	to_json(j["vertices"], tet_mesh.vertices);
	to_json(j["indices"], tet_mesh.indices);
    // No need to save normals!

    to_json(j["original_vertices"], tet_mesh.original_vertices);
    to_json(j["mvc_coordinates"], tet_mesh.mvc_coordinates);
    to_json(j["gamma_coordinates"], tet_mesh.gamma_coordinates);
    to_json(j["tets"], tet_mesh.tets);
    to_json(j["labels"], tet_mesh.labels);
    to_json(j["colors"], tet_mesh.colors);

    to_json(j["all_indices"], tet_mesh.all_indices);
}

template<typename float_t, class point_t>
inline void from_json(const nlohmann::json& j, TetMesh<float_t, point_t>& tet_mesh) {
    from_json(j.at("bbox"), tet_mesh.bbox);
    from_json(j.at("original_bbox"), tet_mesh.original_bbox);
    from_json(j.at("warped_bbox"), tet_mesh.warped_bbox);
    from_json(j.at("original_warped_bbox"), tet_mesh.original_warped_bbox);

	from_json(j.at("vertices"), tet_mesh.vertices);
	from_json(j.at("indices"), tet_mesh.indices);
    
    from_json(j.at("original_vertices"), tet_mesh.original_vertices);
    from_json(j.at("mvc_coordinates"), tet_mesh.mvc_coordinates);
    from_json(j.at("gamma_coordinates"), tet_mesh.gamma_coordinates);
    from_json(j.at("tets"), tet_mesh.tets);
    from_json(j.at("labels"), tet_mesh.labels);
    from_json(j.at("colors"), tet_mesh.colors);
    from_json(j.at("all_indices"), tet_mesh.all_indices);
}

NGP_NAMESPACE_END
