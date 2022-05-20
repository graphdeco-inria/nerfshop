#include <neural-graphics-primitives/common_gl.h>
#include <neural-graphics-primitives/editing/datastructures/tet_mesh.h>

NGP_NAMESPACE_BEGIN

template<typename float_t, class point_t>
void Mesh<float_t, point_t>::draw_gl(
    const Eigen::Vector2i& resolution,
	const Eigen::Vector2f& focal_length,
	const Eigen::Matrix<float, 3, 4>& camera_matrix,
	const Eigen::Vector2f& screen_center
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
		for (int i= 0; i < 2; ++i) {
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
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ebosize * sizeof(uint32_t), &indices[0], GL_STATIC_DRAW);

	if (!program) {
		vs = compile_shader(false, R"foo(
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 nor;
out vec3 vtxcol;
uniform mat4 camera;
uniform vec2 f;
uniform ivec2 res;
uniform vec2 cen;
void main()
{
	vec4 p = camera * vec4(pos, 1.0);
	p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
	p.w = p.z;
	p.z = p.z - 0.1;
	p.xy += cen * p.w;
    vtxcol = normalize(nor) * 0.5 + vec3(0.5); // visualize vertex normals
	gl_Position = p;
}
)foo");
		ps = compile_shader(true, R"foo(
layout (location = 0) out vec4 o;
in vec3 vtxcol;
void main() {
	o = vec4(vtxcol, 1.0);
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
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	GLuint posat = (GLuint)glGetAttribLocation(program, "pos");
	GLuint norat = (GLuint)glGetAttribLocation(program, "nor");
	glEnableVertexAttribArray(posat);
	glEnableVertexAttribArray(norat);
	GLenum float_enum = std::is_same<float_t,float>::value ? GL_FLOAT : GL_DOUBLE;
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(posat, 3, float_enum, GL_FALSE, 3*sizeof(float_t), 0);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glVertexAttribPointer(norat, 3, float_enum, GL_FALSE, 3*sizeof(float_t), 0);
	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT , (GLvoid*)0);
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    glDisable(GL_CULL_FACE);

	glUseProgram(0);
}

template class Mesh<float, Eigen::Vector3f>;
template class Mesh<double, Eigen::Vector3d>;

NGP_NAMESPACE_END