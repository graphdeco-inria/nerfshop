# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

cmake_minimum_required(VERSION 3.18)

PROJECT(instant-ngp
	VERSION 1.0
	DESCRIPTION "Instant neural graphics primitives"
	LANGUAGES C CXX CUDA
)
set(NGP_VERSION "${CMAKE_PROJECT_VERSION}")

# Necessary to be compatible with pytorch
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

if (NOT NGP_DEPLOY)
	set(NGP_VERSION "${NGP_VERSION}dev")
endif()

option(NGP_BUILD_WITH_GUI "Build with GUI support (requires GLFW and GLEW)?" ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to Release.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw/CMakeLists.txt")
	message(FATAL_ERROR
		"Some instant-ngp dependencies are missing. "
		"If you forgot the \"--recursive\" flag when cloning this project, "
		"this can be fixed by calling \"git submodule update --init --recursive\"."
	)
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

###############################################################################
# Dependencies
###############################################################################

##############
#    IGL     #
##############

# Options for libigl modules

option(LIBIGL_USE_STATIC_LIBRARY          "Use libigl as static library" OFF)
option(LIBIGL_COPYLEFT_COMISO            "Use CoMiso"                   OFF)
option(LIBIGL_EMBREE            "Use Embree"                   OFF)
option(LIBIGL_OPENGL            "Use OpenGL"                   OFF)
option(LIBIGL_OPENGL_GLFW       "Use GLFW"                     OFF)
option(LIBIGL_OPENGL_GLFW_IMGUI "Use ImGui"                    OFF)
option(LIBIGL_PNG               "Use PNG"                      OFF)
option(LIBIGL_COPYLEFT_TETGEN            "Use Tetgen"                   ON)
option(LIBIGL_TRIANGLE          "Use Triangle"                 OFF)
option(LIBIGL_PREDICATES        "Use exact predicates"         ON)
option(LIBIGL_XML               "Use XML"                      OFF)
set(BUILD_SHARED_LIBS OFF)
include(libigl)
set(BUILD_SHARED_LIBS ON)

################
# tiny-cuda-nn #
################
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(dependencies/tiny-cuda-nn)
include_directories("dependencies/tiny-cuda-nn/include")
include_directories("dependencies/tiny-cuda-nn/dependencies")

if (NGP_BUILD_WITH_GUI)
	##########
	# Vulkan #
	##########
	find_package(Vulkan)
	if (Vulkan_FOUND)
		list(APPEND NGP_DEFINITIONS -DNGP_VULKAN -DGLFW_INCLUDE_VULKAN)
		include_directories("${Vulkan_INCLUDE_DIRS}")
		set(NGP_LIBRARIES ${NGP_LIBRARIES} ${Vulkan_LIBRARIES})

		set(GUI_SOURCES
			${GUI_SOURCES}
			src/dlss.cu
		)

		#############################################
		# DLSS (depends on vulkan, so appears here) #
		#############################################
		include_directories("dependencies/dlss/include")
		if (MSVC)
			link_directories("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Windows_x86_64/x86_64")
			if (CMAKE_BUILD_TYPE STREQUAL "Debug")
				set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx_d_dbg)
			else()
				set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx_d)
			endif()
		else()
			link_directories("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Linux_x86_64")
			set(NGP_LIBRARIES ${NGP_LIBRARIES} nvsdk_ngx)
		endif()
	else()
		message(WARNING
			"Vulkan was not found. Neural graphics primitives will still compile "
			"and run correctly, but DLSS will not be supported."
		)
	endif()

	########
	# GLFW #
	########
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
	set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
	set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
	set(GLFW_USE_CHDIR OFF CACHE BOOL " " FORCE)
	set(GLFW_VULKAN_STATIC OFF CACHE BOOL " " FORCE)
	set(BUILD_SHARED_LIBS ON CACHE BOOL " " FORCE)

	add_subdirectory(dependencies/glfw)

	set_target_properties(glfw PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)

	mark_as_advanced(
		GLFW_BUILD_DOCS GLFW_BUILD_EXAMPLES GLFW_BUILD_INSTALL GLFW_BUILD_TESTS
		GLFW_DOCUMENT_INTERNALS GLFW_INSTALL GLFW_USE_CHDIR GLFW_USE_MENUBAR
		GLFW_USE_OSMESA GLFW_VULKAN_STATIC GLFW_USE_RETINA GLFW_USE_MIR
		BUILD_SHARED_LIBS USE_MSVC_RUNTIME_LIBRARY_DLL
	)

	include_directories("dependencies/glfw/include")
	include_directories("dependencies/imgui")

	#############
	# GL3W/GLEW #
	#############
	if (MSVC)
		include_directories("dependencies/gl3w")
		link_directories("dependencies")
		set(GUI_SOURCES ${GUI_SOURCES} "dependencies/gl3w/GL/gl3w.c")
		set(NGP_LIBRARIES ${NGP_LIBRARIES} opengl32 $<TARGET_OBJECTS:glfw_objects>)
	else()
		find_package(GLEW REQUIRED)
		include_directories(${GLEW_INCLUDE_DIRS})
		set(NGP_LIBRARIES ${NGP_LIBRARIES} GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects>)
	endif()

	set(GUI_SOURCES
		${GUI_SOURCES}
		dependencies/imguizmo/ImGuizmo.cpp
		dependencies/imgui/imgui.cpp
		dependencies/imgui/backends/imgui_impl_glfw.cpp
		dependencies/imgui/backends/imgui_impl_opengl3.cpp
		dependencies/imgui/imgui_draw.cpp
		dependencies/imgui/imgui_tables.cpp
		dependencies/imgui/imgui_widgets.cpp
	)

	list(APPEND NGP_DEFINITIONS -DNGP_GUI)
endif(NGP_BUILD_WITH_GUI)

include_directories("dependencies")

#########
# Eigen #
#########
include_directories("dependencies/eigen")
include_directories("dependencies/filesystem")

###########
# nanovdb #
###########
include_directories("dependencies/nanovdb")

##########
# OpenMP #
##########
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#########
# OPTIX #
#########
#find_package(OptiX)
# set OptiX_INSTALL_DIR via your environment if its not found automatically

if (OptiX_FOUND OR OptiX_INCLUDE)
	include_directories("${OptiX_INCLUDE}")
	list(APPEND NGP_DEFINITIONS -DNGP_OPTIX)
else()
	message(WARNING
		"OptiX was not found. Neural graphics primitives will still compile "
		"and run correctly. However, SDF training in raystab and pathescape "
		"modes will be significantly slower."
	)
endif()

##########
# Python #
##########
find_package(Python 3.7 COMPONENTS Interpreter Development)
if (Python_FOUND)
	add_subdirectory("dependencies/pybind11")
endif()


##############
# tinylogger #
##############
include_directories("dependencies/tinylogger")

##########
# Pytorch #
##########
if (DEFINED CACHE{TORCH_LOCATION})
	message(STATUS "Torch location provided at: ${TORCH_LOCATION}")
	# Include directories.
	if(EXISTS "${TORCH_LOCATION}/include")
	set(TORCH_INCLUDE_DIRS
		${TORCH_LOCATION}/include
		${TORCH_LOCATION}/include/torch/csrc/api/include)
	else()
	message( FATAL_ERROR "Torch include directory does not exists: ${TORCH_LOCATION}/include" )
	endif()
	if(EXISTS "${TORCH_LOCATION}/lib")
	set(TORCH_LIBRARY_DIR
		${TORCH_LOCATION}/lib)
	else()
	message( FATAL_ERROR "Torch lib directory does not exists: ${TORCH_LOCATION}/lib" )
	endif()
endif()

##########
#  CUDA  #
##########
# DEPRECATED
find_package(CUDA REQUIRED)
# ADDED TO LINK CUDA LIBS
# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
find_package(CUDAToolkit REQUIRED)

##############
#  fTetWild  #
##############
# set(FLOAT_TETWILD_USE_FLOAT OFF)
# add_subdirectory(dependencies/fTetWild)
# # # Include required geogram headers
# include_directories(${GEOGRAM_SOURCE_INCLUDE_DIR})

##############
#  MeshFix   #
##############

option(MESHFIX_WITH_LIBIGL "Use meshfix with libigl libigl" OFF)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory("dependencies/meshfix")
include_directories("dependencies/meshfix")
set(JMESHEXT_DIR "${PROJECT_SOURCE_DIR}/dependencies/meshfix/JMeshExt-1.0alpha_src")
set(JMESH_DIR "${JMESHEXT_DIR}/JMeshLib-1.2/")
include_directories("${JMESHEXT_DIR}/include")
include_directories("${JMESH_DIR}/include")
set(BUILD_SHARED_LIBS ON)

# Compile zlib (only on Windows)
if (WIN32)
	set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL " " FORCE)
	set(ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
	set(ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
	set(SKIP_INSTALL_ALL ON CACHE BOOL " " FORCE)
	add_subdirectory("dependencies/zlib")
	set_property(TARGET zlibstatic PROPERTY FOLDER "dependencies")

	set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/zlib" CACHE PATH " " FORCE)
	set(ZLIB_LIBRARY zlibstatic)

	include_directories(${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/dependencies/zlib")

	list(APPEND NGP_LIBRARIES zlibstatic)
endif()

add_subdirectory("dependencies/zstr")
list(APPEND NGP_LIBRARIES zstr::zstr)


###############################################################################
# Program
###############################################################################

list(APPEND NGP_DEFINITIONS -DNGP_VERSION="${NGP_VERSION}")

include_directories("include")

set(SOURCES
	${GUI_SOURCES}
	src/camera_path.cu
	src/common_device.cu
	src/common_nerf.cu
	src/common_gl.cu
	src/marching_cubes.cu
	src/nerf_loader.cu
	src/render_buffer.cu
	src/student_trainer.cu
	src/testbed.cu
	src/testbed_image.cu
	src/testbed_nerf.cu
	src/testbed_sdf.cu
	src/testbed_volume.cu
	src/thread_pool.cpp
	src/tinyexr_wrapper.cu
	src/tinyobj_loader_wrapper.cpp
	src/triangle_bvh.cu
	src/editing/affine_duplication.cu
	src/editing/cage_deformation.cu
	src/editing/twist.cu
	src/editing/tools/correct_mm_operations.cu
	src/editing/tools/default_mm_operations.cu
	src/editing/tools/fast_quadric.cu
	src/editing/tools/growing_selection.cu
	src/editing/tools/progressive_hulls.cu
	src/editing/tools/visualization_utils.cu
	src/editing/tools/selection_utils.cu
	src/editing/tools/sh_utils.cu
	src/editing/tools/region_growing.cu
	src/editing/datastructures/cage.cu
	src/editing/datastructures/mesh.cu
	src/editing/datastructures/tet_mesh.cu
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)

if (OptiX_FOUND OR OptiX_INCLUDE)
	add_library(optix_program OBJECT
		src/optix/pathescape.cu
		src/optix/raystab.cu
		src/optix/raytrace.cu
	)

	set_target_properties(optix_program PROPERTIES CUDA_PTX_COMPILATION ON)
	set_target_properties(optix_program PROPERTIES CUDA_ARCHITECTURES OFF)
	target_compile_definitions(optix_program PUBLIC ${NGP_DEFINITIONS} -DTCNN_MIN_GPU_ARCH=0)
	target_compile_options(optix_program PUBLIC "--expt-relaxed-constexpr")

	# OptiX programs will be compiled as PTX and packaged
	# as headers to be included from the binary dir.
	include_directories("${CMAKE_CURRENT_BINARY_DIR}")
	set(OPTIX_PTX_HEADER ${CMAKE_CURRENT_BINARY_DIR}/optix_ptx.h)

	find_program(bin_to_c NAMES bin2c PATHS ${CUDA_COMPILER_BIN})
	if (NOT bin_to_c)
		message(FATAL_ERROR
			"bin2c not found:\n"
			"  CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}\n"
			"  CUDA_COMPILER_BIN=${CUDA_COMPILER_BIN}\n"
		)
	endif()

	add_custom_command(
		OUTPUT "${OPTIX_PTX_HEADER}"
		COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${bin_to_c}"
		"-DOBJECTS=$<TARGET_OBJECTS:optix_program>"
		"-DOUTPUT=${OPTIX_PTX_HEADER}"
		-P ${PROJECT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
		VERBATIM
		DEPENDS optix_program
		COMMENT "Converting PTX files to a C header"
	)

	list(APPEND SOURCES ${OPTIX_PTX_HEADER})
endif()

add_library(ngp STATIC ${SOURCES})

set_target_properties(ngp PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(ngp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ngp PUBLIC ${NGP_LIBRARIES} tiny-cuda-nn)
target_link_libraries(ngp PRIVATE igl::core igl::predicates meshfix)
target_link_libraries(ngp PUBLIC igl_copyleft::tetgen)
# target_link_libraries(ngp PUBLIC FloatTetwild)
target_compile_options(ngp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(ngp PUBLIC ${NGP_DEFINITIONS} ${TCNN_DEFINITIONS})

add_executable(nerfshop src/main.cu)
target_link_libraries(nerfshop PUBLIC ngp)
target_compile_options(nerfshop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(nerfshop PUBLIC ${NGP_DEFINITIONS} ${TCNN_DEFINITIONS})

# Copy DLSS shared libraries
if (Vulkan_FOUND)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		set(NGX_BUILD_DIR "dev")
	else()
		set(NGX_BUILD_DIR "rel")
	endif()

	if (MSVC)
		add_custom_command(TARGET nerfshop POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Windows_x86_64/${NGX_BUILD_DIR}/nvngx_dlss.dll" $<TARGET_FILE_DIR:nerfshop>
			COMMAND_EXPAND_LISTS
		)
	else()
		file(GLOB DLSS_SOS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/dlss/lib/Linux_x86_64/${NGX_BUILD_DIR}/libnvidia-ngx-dlss.so.*")
		add_custom_command(TARGET nerfshop POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy ${DLSS_SOS} $<TARGET_FILE_DIR:nerfshop>
			COMMAND_EXPAND_LISTS
		)
	endif()
endif()


if (Python_FOUND)
	# add_library(pyngp_bindings SHARED src/python_api.cu)
	# set_target_properties(pyngp_bindings PROPERTIES CXX_VISIBILITY_PRESET "hidden")
	# set_target_properties(pyngp_bindings PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
	# target_link_libraries(pyngp_bindings PUBLIC ngp ${PYTHON_LIBRARIES} pybind11::module)

	# # TORCH specific options
	# message(STATUS ${TORCH_INCLUDE_DIRS})
	# message(STATUS ${TORCH_LIBRARY_DIR})
	# include_directories(${TORCH_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
	# target_link_directories(pyngp_bindings PUBLIC ${TORCH_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_DIR})
	# target_link_libraries(pyngp_bindings PUBLIC cuda cudadevrt cudart_static c10 torch torch_cpu torch_python cudart c10_cuda torch_cuda_cu torch_cuda_cpp)
	# target_compile_definitions(pyngp_bindings PUBLIC -DNGP_TORCH)

	# target_compile_options(pyngp_bindings PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

	# target_compile_definitions(pyngp_bindings PUBLIC -DNGP_PYTHON ${NGP_DEFINITIONS} ${TCNN_DEFINITIONS})
	# pybind11_extension(pyngp_bindings)

	# if (MSVC)
	# 	# Copy dlls needed at runtime
	# 	file(GLOB CUBLAS_DLLS "${CUDA_COMPILER_BIN}/cublas*.dll")
	# 	if (CUBLAS_DLLS)
	# 		add_custom_command(TARGET pyngp_bindings POST_BUILD
	# 			COMMAND ${CMAKE_COMMAND} -E copy ${CUBLAS_DLLS} $<TARGET_FILE_DIR:pyngp_bindings>
	# 			COMMAND_EXPAND_LISTS
	# 		)
	# 	endif()
	# endif()
endif()


set(CMAKE_VERBOSE_MAKEFILE ON)
