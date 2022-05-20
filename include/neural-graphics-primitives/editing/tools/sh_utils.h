#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/common.h>

using namespace Eigen;
using namespace tcnn;

NGP_NAMESPACE_BEGIN

Eigen::Vector2f concentric_sample_disk(const Eigen::Vector2f& u);

Eigen::Vector3f cosine_sample_hemisphere(const Eigen::Vector2f& u);

// Dir is provided in global coordinate system
SH9RGB project_sh9(const Eigen::Vector3f& dir, const Eigen::Vector3f& rgb, const float domega = 1.0f);

SH9Scalar project_sh9(const Eigen::Vector3f& dir, const float val, const float domega = 1.0f);

SH9RGB get_constant_sh9(const Eigen::Vector3f rgb);

NGP_NAMESPACE_END