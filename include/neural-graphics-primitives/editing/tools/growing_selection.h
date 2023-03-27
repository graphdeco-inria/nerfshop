#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/editing/datastructures/cage.h>
#include <neural-graphics-primitives/camera_path.h>
#include <neural-graphics-primitives/editing/tools/fast_quadric.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/editing/datastructures/tet_mesh.h>
#include <neural-graphics-primitives/editing/tools/progressive_hulls.h>
#include <neural-graphics-primitives/editing/tools/region_growing.h>
#include <neural-graphics-primitives/editing/tools/mm_operations.h>

#include <tiny-cuda-nn/common.h>

#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>

#include <vector>
#include <queue>

#include <json/json.hpp>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

static constexpr const char SCREEN_SELECTION_KEY = 'B';
static constexpr const uint32_t DEBUG_CUBEMAP_WIDTH = 16;
static constexpr const uint32_t DEBUG_ENVMAP_WIDTH = 256;
static constexpr const uint32_t DEBUG_ENVMAP_HEIGHT = 128;

enum class ESelectionMode : int {
    PixelWise,
    Scribble
};

static constexpr const char* SelectionModeStr = "Pixelwise\0 Scribble\0\0";

enum class EPcRenderMode : int {
	Off,
	UniformColor,
	Labels,
};
static constexpr const char* PcRenderModeStr = "Off\0UniformColor\0Labels\0\0";

enum class EManipulationTarget : int {
	CageVerts,
	CursorCoords
};
static constexpr const char* targetStrings = "Cage Vertices\0Cursor Coordinate System";

enum class ESelectionRenderMode : int {
	Off,
	ScreenSelection,
    Projection,
	RegionGrowing,
	SelectionMesh,
	ProxyMesh,
	TetMesh,
};
static constexpr const char* SelectionRenderModeStr = "Off\0Screen Scribbles\0Projected Scribbles\0Grown Region\0Fine Cage\0Coarse Cage\0Tetrahedral Mesh\0\0";

enum class EDecimationAlgorithm : int {
	ShortestEdge,
    ProgressiveHullsQuadratic,
    ProgressiveHullsLinear,
};

static constexpr const char* DecimationAlgorithmStr = "Shortest Edge\0Progressive Hulls Quadratic\0Progressive Hulls Linear\0\0";

enum class ERadianceAlgorithm : int {
    ColorOnly,
    SH,
};

static constexpr const char* RadianceAlgorithmStr = "Color Only\0Spherical Harmonics\0\0";

enum class EProjectionThresholds : int {
    Low,
    Intermediate,
    High,
};
static constexpr const char* ProjectionThresholdsStr[3] = { "Low", "Intermediate", "High"};
static constexpr const float ProjectionThresholdsVal[3] = {1e-3f, 1e-1f, 1.f};

typedef float float_t;
typedef Eigen::Vector3f point_t;
typedef Eigen::Matrix3f matrix3_t;
typedef Eigen::Matrix4f matrix4_t;

struct CageEdition {
    std::vector<uint32_t> selected_vertices;
    point_t selection_barycenter;
    matrix3_t selection_rotation;
	point_t selection_scaling;
};

static Eigen::Vector3f DEBUG_COLORS_LUT[6] = {
    Eigen::Vector3f(1.0f, 0.0f, 0.0f),
    Eigen::Vector3f(0.0f, 1.0f, 0.0f),
    Eigen::Vector3f(0.0f, 0.0f, 1.0f),
    Eigen::Vector3f(1.0f, 0.75f, 0.0f),
    Eigen::Vector3f(0.9f, 0.55f, 0.9f),
    Eigen::Vector3f(0.9f, 0.9f, 0.9f),
};

struct GrowingSelection {

    // Fine mesh extracted from the region-grown points
    Mesh<float_t, point_t> selection_mesh;

    // Proxy cage obtained with decimation
    Cage<float_t, point_t> proxy_cage;
    bool display_in_tet = false;
    bool preserve_surface_mesh = true;
    float ideal_tet_edge_length;
    std::shared_ptr<TetMesh<float_t, point_t>> tet_interpolation_mesh;

	bool m_copy = false;
	bool m_bypass = false;

	EManipulationTarget m_target = EManipulationTarget::CageVerts;

    float transmittance_threshold = 1e-1f;
    float off_surface_projection = 0.01f;

	Eigen::Vector3f m_plane_pos = Eigen::Vector3f::Zero();
	Eigen::Vector3f m_plane_dir = Eigen::Vector3f::Zero();
	Eigen::Vector3f m_plane_dir1;
	Eigen::Vector3f m_plane_dir2;

    int proxy_size = 100;

    ESelectionRenderMode render_mode = ESelectionRenderMode::ScreenSelection;

    GrowingSelection(
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
    );

    GrowingSelection(
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
    );

	void find_plane();

	bool imgui(const Vector2i& resolution, const Vector2f& focal_length,  const Matrix<float, 3, 4>& camera_matrix, const Vector2f& screen_center, bool& auto_clean);

    bool visualize_edit_gui(const Eigen::Matrix<float, 4, 4> &view2proj, const Eigen::Matrix<float, 4, 4> &world2proj, const Eigen::Matrix<float, 4, 4> &world2view, const Eigen::Vector2f& focal, float aspect, float time);

    void draw_gl(
        const Eigen::Vector2i& resolution, 
        const Eigen::Vector2f& focal_length, 
        const Eigen::Matrix<float, 3, 4>& camera_matrix, 
        const Eigen::Vector2f& screen_center
    );

    void to_json(nlohmann::json& j);

	float m_plane_offset = 0.0f;

	bool m_use_morphological = true;

	void deform_proxy_from_file(std::string deformed_file);

	void proxy_mesh_from_file(std::string orig_file);

	bool m_refine_cage = false;

	void set_proxy_mesh(std::vector<point_t>& points, std::vector<uint32_t>& indices);

private:

    // Selection specifics
    ESelectionMode m_selection_mode = ESelectionMode::Scribble;
    std::vector<Eigen::Vector2i> m_selected_pixels;
    std::vector<ImVec2> m_selected_pixels_imgui;
    Eigen::Vector2i m_last_selected_pixel = Eigen::Vector2i(-1, -1);

    // Necessary for the kernel parts
    const BoundingBox m_aabb;
    const std::shared_ptr<NerfNetwork<precision_t>> m_nerf_network;
    const tcnn::GPUMemory<float>& m_density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
    const tcnn::GPUMemory<uint8_t>& m_density_grid_bitfield;
    const float m_cone_angle_constant = 1.f/256.f;
    const ENerfActivation m_rgb_activation;
    const ENerfActivation m_density_activation;
    const Eigen::Vector3f m_light_dir;

    cudaStream_t m_stream;

    // Cage rect-based screen-space selection
    bool selected_cage = false;
    bool currently_selecting_cage = false;
    ImVec2 mouse_clicked_selecting_cage;
    ImVec2 mouse_released_selecting_cage;
    ImGuizmo::MODE m_gizmo_mode = ImGuizmo::LOCAL;
	ImGuizmo::OPERATION m_gizmo_op = ImGuizmo::TRANSLATE;


    CageEdition cage_edition = {};

    EPcRenderMode m_pc_render_mode = EPcRenderMode::Labels;
    int m_pc_render_max_level = NERF_CASCADES();
    bool m_visualize_max_level_cube = false;
    bool m_automatic_max_level = true;
    uint32_t m_max_cascade;

    // Projected pixels
    std::vector<Eigen::Vector3f> m_projected_pixels;
    std::vector<uint8_t> m_projected_labels;
    std::vector<uint32_t> m_projected_cell_idx;
    // std::vector<FeatureVector> m_projected_features;

	float m_select_radius = 8;
	bool m_rigid_editing = false;
 
    // Region-grown points (+ MM operators)
    std::vector<Eigen::Vector3f> m_selection_points;
    std::vector<uint8_t> m_selection_labels;
    std::vector<uint32_t> m_selection_cell_idx;
    std::vector<uint8_t> m_selection_grid_bitfield;

    // Region-growing
    int m_growing_steps = 10000;
    int m_growing_level = 0;
    float m_density_threshold = 0.01f;
    ERegionGrowingMode m_region_growing_mode = ERegionGrowingMode::Manual;
    RegionGrowing m_region_growing;

    // Morphological operations
    std::shared_ptr<MMOperations> m_MM_operations;
    bool m_performed_closing = false;

    EDecimationAlgorithm m_decimation_algorithm = EDecimationAlgorithm::ProgressiveHullsQuadratic;
    
    ProgressiveHullsParams m_progressive_hulls_params;

    ERadianceAlgorithm m_radiance_algorithm = ERadianceAlgorithm::SH;
    int m_n_hemisphere_samples = 10;
    int m_hemisphere_width = 10;
    int m_projection_threshold_simple = (int)EProjectionThresholds::Intermediate;
    float m_transmittance_threshold_boundary = 1e-3f;
    float m_brush_color[3] = {0.0f, 1.0f, 0.0};
    float m_cage_color[3] = {0.9f, 0.9f, 0.98f};
    GLuint m_debug_cubemap_textures[6] = {0, 0, 0, 0, 0, 0};
    bool m_rotate_debug_cubemap = true;
    bool m_initial_debug_cubemap = false;
    int m_debug_ray_idx = 0;
    Eigen::Vector3f m_debug_rotation_normal = Eigen::Vector3f(1.f, 0.f, 0.f);
    float m_t_min_boundary = 0.05f; // This ensures that we don't get interferences from within the boundary
    tcnn::GPUMemory<float> m_envmap;
    Eigen::Vector2i m_envmap_resolution = Eigen::Vector2i::Constant(0.0f);
    std::string m_default_envmap_path = "";
    GLuint m_debug_envmap_texture = 0;

	float m_plane_radius = 0.1f;

    // Automatically update the tet when a manipulation is performed
    bool m_update_tet_manipulation = true;

    std::vector<Eigen::Vector3f> m_debug_points;
    std::vector<Eigen::Vector3f> m_debug_colors;

    // Poisson editing
    struct PoissonEditing {
        int sh_sampling_width = 10;
        GLuint sh_cubemap_textures[6] = {0, 0, 0, 0, 0, 0};
        float sh_sum_weights_threshold = 1e-4f;
        float inside_contribution = 1.f;
        float mvc_gamma = 1.0f;
    } m_poisson_editing;

    bool m_correct_direction = true;

    void clear();

    // ------------------------
    // Screen-space selection
    // ------------------------
    void project_selection_pixels(const std::vector<Eigen::Vector2i>& ray_pixels, const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length, const Eigen::Matrix<float, 3, 4>& camera_matrix, const Eigen::Vector2f& screen_center, cudaStream_t stream);

	inline bool is_near_mouse(const ImVec2& p);

    inline bool is_inside_rect(const ImVec2& p);

	void select_scribbling(const Eigen::Matrix<float, 4, 4>& world2proj);

    void select_cage_rect(const Eigen::Matrix<float, 4, 4>& world2proj);

    void reset_cage_selection();

    void delete_selected_projection();

    void delete_selected_growing();

    void color_selection();

    // ------------------------
    // Region growing
    // ------------------------

    // Initialize region growing
    void reset_growing();
    
    void upscale_growing();

    // Grow region (by user-selected steps)
    void grow_region();

    // ------------------------
    // Morphological Operators
    // ------------------------
	
    // MM dilation
    void dilate();
	
    // MM_erosion
    void erode();

    // ------------------------
    // Proxy mesh processing
    // ------------------------

    // Extract the fine mesh from the voxelized region selection (using marching cubes)
    void extract_fine_mesh();

    // Decimate fine mesh with linear bounding constraint
    void compute_proxy_mesh();



    // Not used in practice
    void fix_fine_mesh();

    // Fix proxy mesh with MeshFix
    void fix_proxy_mesh();

    // DEBUG: export the proxy mesh as a file
    void export_proxy_mesh();

    // ------------------------
    // Proxy mesh processing
    // ------------------------

    // Extract mesh with TetGen
    void extract_tet_mesh();

	void force_cage();

    // Initialize the tet mvc coordinates with the pre-computed cage
    void initialize_mvc();

    // Update the new position with MVC coordinates and update tet_lut
    void update_tet_mesh();

    // ------------------------
    // Poisson correction
    // ------------------------

    // Store view-dependent color of NeRF at the boundary proxies of the cage as SH
    void compute_poisson_boundary(const bool is_inside);

    // DEBUG: display incoming radiance SH as cube maps
    void generate_poisson_cube_map();

    // TODO: handle rotation
    // Interpolate poisson values at the boundary using MVC coordinates
    void interpolate_poisson_boundary();
};

void draw_selection_gl(const std::vector<Eigen::Vector3f>& points, const std::vector<uint8_t>& labels, const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length, const Eigen::Matrix<float, 3, 4>& camera_matrix, const Eigen::Vector2f& screen_center, const int pc_render_mode, const int max_label);

void draw_debug_gl(const std::vector<Eigen::Vector3f>& points, const std::vector<Eigen::Vector3f>& colors, const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length, const Eigen::Matrix<float, 3, 4>& camera_matrix, const Eigen::Vector2f& screen_center);

NGP_NAMESPACE_END

