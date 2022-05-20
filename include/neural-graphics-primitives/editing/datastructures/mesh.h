#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/json_binding.h>

#include <tiny-cuda-nn/common.h>

#include <json/json.hpp>

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

template<typename float_t, class point_t>
struct Mesh {
public: 
	std::vector<point_t> vertices;
    std::vector<uint32_t> indices;
    std::vector<point_t> normals;

	Mesh() = default;

	Mesh(std::vector<point_t>& _vertices, std::vector<uint32_t>& _indices) : vertices{_vertices}, indices{_indices} {}

	void clear() {
		indices={};
		vertices={};
		normals={};
	}

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
inline void to_json(nlohmann::json& j, const Mesh<float_t, point_t>& mesh) {
	to_json(j["vertices"], mesh.vertices);
	to_json(j["indices"], mesh.indices);
	to_json(j["normals"], mesh.normals);
}

template<typename float_t, class point_t>
inline void from_json(const nlohmann::json& j, Mesh<float_t, point_t>& mesh) {
	from_json(j.at("vertices"), mesh.vertices);
	from_json(j.at("indices"), mesh.indices);
	from_json(j.at("normals"), mesh.normals);
}

NGP_NAMESPACE_END
