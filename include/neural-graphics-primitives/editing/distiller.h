#pragma once


NGP_NAMESPACE_BEGIN

using namespace Eigen;

class Distiller
{
public:
	bool emptying;

	virtual __device__ bool in_source(Eigen::Vector3f& coord, bool warp) = 0;

	virtual __device__ bool in_target(Eigen::Vector3f& coord, bool warp) = 0;

	virtual __device__ Vector3f map(Eigen::Vector3f coord, bool warp) = 0;
};

NGP_NAMESPACE_END
