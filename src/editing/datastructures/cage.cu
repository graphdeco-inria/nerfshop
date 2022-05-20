#include <neural-graphics-primitives/editing/datastructures/cage.h>
#include <neural-graphics-primitives/common_gl.h>

NGP_NAMESPACE_BEGIN

template<typename float_t, class point_t>
void Cage<float_t, point_t>::compute_mvc(const std::vector<point_t>& points, std::vector<std::vector<float_t>>& weights, std::vector<uint8_t>& labels, bool original, float gamma) {
	uint32_t n_vertices = vertices.size();
	uint32_t n_points = points.size();
	weights.clear();
	weights.resize(points.size(), std::vector<float_t>(n_vertices, 0.f));
	labels.clear();
	labels.resize(n_points, 0);
	std::vector<float_t> tmp_w_weights;
	for (int i = 0; i < n_points; i++) {
		// TODO: provide mask?
		// compute_mvc(points[i], weights[i], tmp_w_weights, original);
		bool success = MVC3D::computeCoordinatesCustomCode<uint32_t, float_t, point_t>(points[i], indices, original ? original_vertices : vertices, normals, weights[i], tmp_w_weights);
		if (!success) {
			labels[i] = 1;
		}
	}

	if (gamma > 1.f) {
		for (int i = 0; i < n_points; i++) {
			float sum_gamma_weights = 0.f;
			for (int j = 0; j < n_vertices; j++) {
				weights[i][j] = std::pow(weights[i][j], gamma);
				sum_gamma_weights += weights[i][j];
			}
			for (int j = 0; j < n_vertices; j++) {
				weights[i][j] /= sum_gamma_weights;
			}
		}
	}
}

template<typename float_t, class point_t>
void Cage<float_t, point_t>::interpolate_with_mvc(const std::vector<std::vector<float_t>>& weights, std::vector<point_t>& points) {
	uint32_t n_points = weights.size();
	uint32_t n_vertices = original_vertices.size();
	points.clear();
	points.resize(n_points, point_t::Zero());
	for (int i = 0; i < n_points; i++) {
		for (int v = 0; v < n_vertices; v++) {
			points[i] += weights[i][v] * vertices[v];
		}
	}
}

template<typename float_t, class point_t>
void Cage<float_t, point_t>::interpolate_with_mvc(std::shared_ptr<TetMesh<float_t, point_t>> tet_mesh) {
	interpolate_with_mvc(tet_mesh->mvc_coordinates, tet_mesh->vertices);
	tet_mesh->post_update_vertices();
}

template<typename float_t, class point_t>
void Cage<float_t, point_t>::reset_original_vertices() {
	original_vertices = vertices;
}

template<typename float_t, class point_t>
void Cage<float_t, point_t>::draw_gl(
    const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center) {
    if (vertices.size() == 0 || indices.size() == 0) {
		return;
	}

	if (!VAO) {
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);
	}
	if (vbosize != vertices.size()) {
        // If necessary, delete the VBO
		for (int i= 0; i < 4; ++i) {
			if (VBO[i]) {
				glDeleteBuffers(1, &VBO[i]);
			}
		}
        // VBO for positions
        glGenBuffers(1, &VBO[0]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
        glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(point_t), &vertices[0], GL_STATIC_DRAW);
        // VBO for normals
        glGenBuffers(1, &VBO[1]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
        glBufferData(GL_ARRAY_BUFFER, normals.size()*sizeof(point_t), &normals[0], GL_STATIC_DRAW);
         // VBO for labels
        glGenBuffers(1, &VBO[2]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
        glBufferData(GL_ARRAY_BUFFER, labels.size()*sizeof(uint8_t), &labels[0], GL_STATIC_DRAW);
        // VBO for colors
        glGenBuffers(1, &VBO[3]);
        glBindBuffer(GL_ARRAY_BUFFER, VBO[3]);
        glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(Eigen::Vector3f), &colors[0], GL_STATIC_DRAW);
    }
	if (ebosize != indices.size()) {
		if (EBO) {
			glDeleteBuffers(1, &EBO);
		}
		glGenBuffers(1, &EBO);
		ebosize = indices.size();
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, ebosize * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);
	}

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(point_t), &vertices[0], GL_STATIC_DRAW);
	
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, normals.size()*sizeof(point_t), &normals[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
    glBufferData(GL_ARRAY_BUFFER, labels.size()*sizeof(uint8_t), &labels[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[3]);
    glBufferData(GL_ARRAY_BUFFER, colors.size()*sizeof(Eigen::Vector3f), &colors[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ebosize * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);

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
    if (mode == 1) {
        vtxcol = col;
    // } else if (mode == 1) {
	// 	vtxcol = normalize(nor) * 0.5 + vec3(0.5); // visualize vertex normals
	} else {
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
	// if (mode == 2) {
	// 	if (fLabel == 1) {
    //         o = vec4(0.0, 0.0, 1.0, 1.0);
    //     } else {
    //         o = vec4(1.0, 0.0, 0.0, 1.0);
    //     }
	// } else {
		o = vec4(vtxcol, 1.0);
	// }
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
	Eigen::Matrix4f view2world=Eigen::Matrix4f::Identity();
	view2world.block<3,4>(0,0) = camera_matrix;
	Eigen::Matrix4f world2view = view2world.inverse();
	glBindVertexArray(VAO);
	glUseProgram(program);
	glUniformMatrix4fv(glGetUniformLocation(program, "camera"), 1, GL_FALSE, (GLfloat*)&world2view);
	glUniform2f(glGetUniformLocation(program, "f"), focal_length.x(), focal_length.y());
	glUniform2f(glGetUniformLocation(program, "cen"), screen_center.x()*2.f-1.f, screen_center.y()*-2.f+1.f);
	glUniform2i(glGetUniformLocation(program, "res"), resolution.x(), resolution.y());
	glUniform1i(glGetUniformLocation(program, "mode"), (int)render_mode);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	GLuint posat = (GLuint)glGetAttribLocation(program, "pos");
	GLuint norat = (GLuint)glGetAttribLocation(program, "nor");
	GLuint labat = (GLuint)glGetAttribLocation(program, "label");
    GLuint colat = (GLuint)glGetAttribLocation(program, "col");
	glEnableVertexAttribArray(posat);
	glEnableVertexAttribArray(norat);
	glEnableVertexAttribArray(labat);
    glEnableVertexAttribArray(colat);
	GLenum float_enum = std::is_same<float_t,float>::value ? GL_FLOAT : GL_DOUBLE;
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(posat, 3, float_enum, GL_FALSE, 3*sizeof(float_t), 0);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glVertexAttribPointer(norat, 3, float_enum, GL_FALSE, 3*sizeof(float_t), 0);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glVertexAttribIPointer(labat, 1, GL_UNSIGNED_BYTE, sizeof(uint8_t), 0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[3]);
	glVertexAttribPointer(colat, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT , (GLvoid*)0);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    glDisable(GL_CULL_FACE);

	glUseProgram(0);
}

template class Cage<float, Eigen::Vector3f>;
template class Cage<double, Eigen::Vector3d>;

NGP_NAMESPACE_END