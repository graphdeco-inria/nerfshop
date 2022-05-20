#pragma once

#include <neural-graphics-primitives/common.h>

#ifdef NGP_GUI

#  ifdef _WIN32
#    include <GL/gl3w.h>
#  else
#    include <GL/glew.h>
#  endif
#  include <GLFW/glfw3.h>
#  include <cuda_gl_interop.h>

NGP_NAMESPACE_BEGIN

void glCheckError(const char* file, unsigned int line);

bool check_shader(GLuint handle, const char* desc, bool program);

GLuint compile_shader(bool pixel, const char* code);

NGP_NAMESPACE_END

#endif