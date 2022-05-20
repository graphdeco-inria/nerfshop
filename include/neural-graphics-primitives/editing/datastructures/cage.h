#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/editing/datastructures/mesh.h>
#include <neural-graphics-primitives/editing/datastructures/tet_mesh.h>
#include <neural-graphics-primitives/editing/tools/mvc.h>
#include <neural-graphics-primitives/common_gl.h>
#include <neural-graphics-primitives/json_binding.h>

#include <tiny-cuda-nn/common.h>

#include <vector>
#include <cmath>

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

enum class ECageRenderMode : int {
	UniformColor,
	//Normals,
	//Labels,
	Colors,
};
static constexpr const char* CageRenderModeStr = "Solid\0Colors\0\0";

template<typename float_t, class point_t>
struct Cage : public Mesh<float_t, point_t> {
public:
    BoundingBox bbox;

    using Mesh<float_t, point_t>::vertices;
    using Mesh<float_t, point_t>::indices;
    using Mesh<float_t, point_t>::normals;
    std::vector<point_t> initial_normals;

    // 0 stands for default
    // 1 stands for selected
    std::vector<uint8_t> labels;
    std::vector<point_t> original_vertices;
    std::vector<Eigen::Vector3f> colors;

    std::vector<Eigen::Vector3f> outside_colors;
    std::vector<Eigen::Vector3f> initial_colors;

    std::vector<SH9RGB> new_shs;
    std::vector<SH9RGB> initial_shs;

    // std::vector<float> boundary_density;
    // std::vector<SH9RGB> boundary_shs;
    std::vector<SH9RGB> inside_shs;
    std::vector<SH9RGB> outside_shs;
    std::vector<float> inside_density;
    std::vector<float> outside_density;

    ECageRenderMode render_mode = ECageRenderMode::Colors;

    Cage() {}

    Cage(std::vector<point_t>& _vertices, std::vector<uint32_t>& _indices) : Mesh<float_t, point_t>{_vertices, _indices}, original_vertices{_vertices} {
        for (int i = 0; i < vertices.size(); i++) {
            bbox.enlarge(vertices[i].template cast<float>());
        }
        initial_normals.resize(vertices.size());
        normals.resize(vertices.size());
        labels.resize(vertices.size());
        colors.resize(vertices.size());
    }

    void reset_original_vertices();

    void compute_mvc(const std::vector<point_t>& points, std::vector<std::vector<float_t>>& weights, std::vector<uint8_t>& labels, bool original, float gamma = 1.f);

    void interpolate_with_mvc(const std::vector<std::vector<float_t>>& weights, std::vector<point_t>& points);

    void interpolate_with_mvc(std::shared_ptr<TetMesh<float_t, point_t>> tet_mesh);

    void draw_gl(
        const Eigen::Vector2i& resolution,
        const Eigen::Vector2f& focal_length,
        const Eigen::Matrix<float, 3, 4>& camera_matrix,
        const Eigen::Vector2f& screen_center
    );

private:
    GLuint VAO = 0, VBO[4] = {}, EBO = 0, vbosize = 0, ebosize = 0, program = 0, vs = 0, ps = 0;
};

template<typename float_t, class point_t>
inline void to_json(nlohmann::json& j, const Cage<float_t, point_t>& cage) {
	to_json(j["vertices"], cage.vertices);
	to_json(j["indices"], cage.indices);
	to_json(j["normals"], cage.normals);
    to_json(j["initial_normals"], cage.initial_normals);

    to_json(j["labels"], cage.labels);
    to_json(j["original_vertices"], cage.original_vertices);
    to_json(j["colors"], cage.colors);

    to_json(j["outside_colors"], cage.outside_colors);
    to_json(j["initial_colors"], cage.initial_colors);

    to_json(j["new_shs"], cage.new_shs);
    to_json(j["initial_shs"], cage.initial_shs);

    to_json(j["inside_shs"], cage.inside_shs);
    to_json(j["outside_shs"], cage.outside_shs);

    to_json(j["inside_density"], cage.inside_density);
    to_json(j["outside_density"], cage.outside_density);
}

template<typename float_t, class point_t>
inline void from_json(const nlohmann::json& j, Cage<float_t, point_t>& cage) {
	from_json(j.at("vertices"), cage.vertices);
	from_json(j.at("indices"), cage.indices);
	from_json(j.at("normals"), cage.normals);
    from_json(j.at("initial_normals"), cage.initial_normals);

    from_json(j.at("labels"), cage.labels);
    from_json(j.at("original_vertices"), cage.original_vertices);
    from_json(j.at("colors"), cage.colors);

    from_json(j.at("outside_colors"), cage.outside_colors);
    from_json(j.at("initial_colors"), cage.initial_colors);

    from_json(j.at("new_shs"), cage.new_shs);
    from_json(j.at("initial_shs"), cage.initial_shs);

    from_json(j.at("inside_shs"), cage.inside_shs);
    from_json(j.at("outside_shs"), cage.outside_shs);
    from_json(j.at("inside_density"), cage.inside_density);
    from_json(j.at("outside_density"), cage.outside_density);
}

NGP_NAMESPACE_END
