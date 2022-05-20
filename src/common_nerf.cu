#include <neural-graphics-primitives/common_nerf.h>

NGP_NAMESPACE_BEGIN

__host__ __device__ Eigen::Vector3f warp_position(const Eigen::Vector3f& pos, const BoundingBox& aabb) {
	// return {tcnn::logistic(pos.x() - 0.5f), tcnn::logistic(pos.y() - 0.5f), tcnn::logistic(pos.z() - 0.5f)};
	// return pos;

	return aabb.relative_pos(pos);
}

__host__ __device__ Eigen::Vector3f unwarp_position(const Eigen::Vector3f& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.min + pos.cwiseProduct(aabb.diag());
}

// Applied to make sure that direction coordinates stay between 0 and 1
__host__ __device__ Vector3f warp_direction(const Vector3f& dir) {
	return (dir + Vector3f::Ones()) * 0.5f;
}

__host__ __device__ Vector3f unwarp_direction(const Vector3f& dir) {
	return dir * 2.0f - Vector3f::Ones();
}

__device__ float warp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

__device__ float unwarp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}

__device__ float network_to_rgb(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, ENerfActivation activation) {
	return {
		network_to_rgb(float(local_network_output[0]), activation),
		network_to_rgb(float(local_network_output[1]), activation),
		network_to_rgb(float(local_network_output[2]), activation)
	};
}

__device__ float network_to_density(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(val);
		default: assert(false);
	}
	return 0.0f;
}

__device__ FeatureVector network_to_feature(const tcnn::vector_t<tcnn::network_precision_t, 16>& local_network_output) {
	FeatureVector fVec;
	for (int i = 0; i < 16; i++) {
		fVec(i) = float(local_network_output[i]);
	}
	return fVec;
}

__host__ __device__ uint32_t grid_mip_offset(uint32_t mip) {
	return (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE()) * mip;
}

__host__ __device__ float calc_cone_angle(float cosine, const Eigen::Vector2f& focal_length, float cone_angle_constant) {
	// Pixel size. Doesn't always yield a good performance vs. quality
	// trade off. Especially if training pixels have a much different
	// size than rendering pixels.
	// return cosine*cosine / focal_length.mean();

	return cone_angle_constant;
}

__host__ __device__ float calc_dt(float t, float cone_angle) {
	return tcnn::clamp(t*cone_angle, MIN_CONE_STEPSIZE(), MAX_CONE_STEPSIZE());
}

__device__ float distance_to_next_voxel(const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) { // dda like step
	Vector3f p = res * pos;
	float tx = (floorf(p.x() + 0.5f + 0.5f * sign(dir.x())) - p.x()) * idir.x();
	float ty = (floorf(p.y() + 0.5f + 0.5f * sign(dir.y())) - p.y()) * idir.y();
	float tz = (floorf(p.z() + 0.5f + 0.5f * sign(dir.z())) - p.z()) * idir.z();
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

__device__ float advance_to_next_voxel(float t, float cone_angle, const Vector3f& pos, const Vector3f& dir, const Vector3f& idir, uint32_t res) {
	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
	// due to the different stepping.
	// float dt = calc_dt(t, cone_angle);
	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
	do {
		t += calc_dt(t, cone_angle);
	} while (t < t_target);
	return t;
}

__device__ uint32_t cascaded_grid_idx_at(Vector3f pos, uint32_t mip) {
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= Vector3f::Constant(0.5f);
	pos *= mip_scale;
	pos += Vector3f::Constant(0.5f);

	Vector3i i = (pos * NERF_GRIDSIZE()).cast<int>();

	if (i.x() < -1 || i.x() > NERF_GRIDSIZE() || i.y() < -1 || i.y() > NERF_GRIDSIZE() || i.z() < -1 || i.z() > NERF_GRIDSIZE()) {
		printf("WTF %d %d %d\n", i.x(), i.y(), i.z());
	}

	uint32_t idx = tcnn::morton3D(
		tcnn::clamp(i.x(), 0, (int)NERF_GRIDSIZE()-1),
		tcnn::clamp(i.y(), 0, (int)NERF_GRIDSIZE()-1),
		tcnn::clamp(i.z(), 0, (int)NERF_GRIDSIZE()-1)
	);

	return idx;
}

__device__ bool density_grid_occupied_at(const Vector3f& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return density_grid_bitfield[idx/8+grid_mip_offset(mip)/8] & (1<<(idx%8));
}

__device__ float cascaded_grid_at(Vector3f pos, const float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

__device__ float& cascaded_grid_at(Vector3f pos, float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

__host__ __device__ bool get_bitfield_at(const uint32_t cell_idx, const uint32_t level, const uint8_t* bitfield)  {
	return bitfield[cell_idx/8+grid_mip_offset(level)/8] & (1<<(cell_idx%8));
}

__host__ __device__ void set_bitfield_at(const uint32_t cell_idx, const uint32_t level, const bool value, uint8_t* bitfield) {
	uint32_t selected_bit = cell_idx%8;
	uint32_t mask = 1 << selected_bit;
	bitfield[cell_idx/8+grid_mip_offset(level)/8] = (bitfield[cell_idx/8+grid_mip_offset(level)/8] & ~mask) | (value << selected_bit);
}

__device__ int mip_from_pos(const Vector3f& pos) {
	int exponent;
	float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
	frexpf(maxval, &exponent);
	return min(NERF_CASCADES()-1, max(0, exponent+1));
}

__device__ int mip_from_dt(float dt, const Vector3f& pos) {
	int mip = mip_from_pos(pos);
	dt *= 2*NERF_GRIDSIZE();
	if (dt<1.f) return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(NERF_CASCADES()-1, max(exponent, mip));
}

__global__ void generate_grid_samples_nerf_nonuniform(const uint32_t n_elements, default_rng_t rng, const uint32_t step, BoundingBox aabb, const float* __restrict__ grid_in, NerfPosition* __restrict__ out, uint32_t* __restrict__ indices, uint32_t n_cascades, float thresh) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// 1 random number to select the level, 3 to select the position.
	rng.advance(i*4);
	uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

	// Select grid cell that has density
	uint32_t idx;
	for (uint32_t j = 0; j < 10; ++j) {
		idx = ((i+step*n_elements) * 56924617 + j * 19349663 + 96925573) % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());
		idx += level * NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE();
		if (grid_in[idx] > thresh) {
			break;
		}
	}

	// Random position within that cellq
	uint32_t pos_idx = idx % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	Vector3f pos = ((Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / NERF_GRIDSIZE() - Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Vector3f::Constant(0.5f);

	out[i] = { warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
	indices[i] = idx;
}

NetworkDims network_dims_nerf() {
	NetworkDims dims;
	dims.n_input = sizeof(NerfCoordinate) / sizeof(float);
	dims.n_output = 4;
	dims.n_pos = sizeof(NerfPosition) / sizeof(float);
	return dims;
}

__host__ __device__ Eigen::Vector3f evaluate_sh9(const SH9RGB& sh, const Eigen::Vector3f dir) {
	// Adapted from https://jcgt.org/published/0002/02/06/paper.pdf

	// First compute sh coordinate in the basis
	float fC0, fC1, fS0, fS1, fTmpA, fTmpB, fTmpC;
	float fZ2 = dir.z() * dir.z();

	Eigen::Matrix<float, 9, 1> pSH;
	pSH(0) = 0.2820947917738781f;
	pSH(2) = 0.4886025119029199f * dir.z();
	pSH(6) = 0.9461746957575601f * fZ2 + -0.3153915652525201f;
	fC0 = dir.x();
	fS0 = dir.y();
	fTmpA = -0.48860251190292f;
	pSH(3) = fTmpA * fC0;
	pSH(1) = fTmpA * fS0;
	fTmpB = -1.092548430592079f * dir.z();
	pSH(7) = fTmpB * fC0;
	pSH(5) = fTmpB * fS0;
	fC1 = dir.x()*fC0 - dir.y()*fS0;
	fS1 = dir.x()*fS0 + dir.y()*fC0;
	fTmpC = 0.5462742152960395f;
	pSH(8) = fTmpC * fC1;
	pSH(4) = fTmpC * fS1;

	// Then multiply each component to get color
	return Eigen::Vector3f(pSH.dot(sh.block<9, 1>(0, 0)), pSH.dot(sh.block<9, 1>(0, 1)), pSH.dot(sh.block<9, 1>(0, 2)));
}

__host__ __device__ float evaluate_sh9(const SH9Scalar& sh, const Eigen::Vector3f dir) {
	// Adapted from https://jcgt.org/published/0002/02/06/paper.pdf

	// First compute sh coordinate in the basis
	float fC0, fC1, fS0, fS1, fTmpA, fTmpB, fTmpC;
	float fZ2 = dir.z() * dir.z();

	Eigen::Matrix<float, 9, 1> pSH;
	pSH(0) = 0.2820947917738781f;
	pSH(2) = 0.4886025119029199f * dir.z();
	pSH(6) = 0.9461746957575601f * fZ2 + -0.3153915652525201f;
	fC0 = dir.x();
	fS0 = dir.y();
	fTmpA = -0.48860251190292f;
	pSH(3) = fTmpA * fC0;
	pSH(1) = fTmpA * fS0;
	fTmpB = -1.092548430592079f * dir.z();
	pSH(7) = fTmpB * fC0;
	pSH(5) = fTmpB * fS0;
	fC1 = dir.x()*fC0 - dir.y()*fS0;
	fS1 = dir.x()*fS0 + dir.y()*fC0;
	fTmpC = 0.5462742152960395f;
	pSH(8) = fTmpC * fC1;
	pSH(4) = fTmpC * fS1;

	return pSH.dot(sh);
}

constexpr float COSPI6 = 0.8660254037844386f; 

// See https://gitea.yiem.net/QianMo/Real-Time-Rendering-4th-Bibliography-Collection/raw/branch/main/Chapter%201-24/[0583]%20[GDC%202003]%20Spherical%20Harmonic%20Lighting-%20The%20Gritty%20Details.pdf
__host__ __device__ SH9Scalar rotate_x90_sh9(const SH9Scalar& sh) {
	SH9Scalar rotated_sh = sh; // Zonal are not rotated
	rotated_sh(1) = -sh(2); 
	rotated_sh(2) = sh(1);
	rotated_sh(3) = sh(3); 
	rotated_sh(4) = -sh(7);
	rotated_sh(5) = -sh(5);
	rotated_sh(6) = -0.5f*sh(6)-COSPI6*sh(8);
	rotated_sh(7) = sh(4);
	rotated_sh(8) = -COSPI6*sh(6) + 0.5f*sh(8);	

	return rotated_sh;
}

__host__ __device__ SH9Scalar rotate_xm90_sh9(const SH9Scalar& sh) {
	SH9Scalar rotated_sh = sh; // Zonal are not rotated
	rotated_sh(1) = sh(2); 
	rotated_sh(2) = -sh(1);
	rotated_sh(3) = sh(3); 
	rotated_sh(4) = sh(7);
	rotated_sh(5) = -sh(5);
	rotated_sh(6) = -0.5f*sh(6)-COSPI6*sh(8);
	rotated_sh(7) = -sh(4);
	rotated_sh(8) = -COSPI6*sh(6) + 0.5f*sh(8);	

	return rotated_sh;
}

__host__ __device__ SH9Scalar rotate_z_sh9(const SH9Scalar& sh, const float cos_alpha, const float sin_alpha) {
	SH9Scalar rotated_sh = sh; // Zonal are not rotated
	// l = 1, m = 1
	rotated_sh(1) = sh(1)*cos_alpha + sh(3)*sin_alpha; 
	rotated_sh(3) = -sh(1)*sin_alpha + sh(3)*cos_alpha;
	// l = 2, m = 1
	rotated_sh(5) = sh(5)*cos_alpha + sh(7)*sin_alpha; 
	rotated_sh(7) = -sh(5)*sin_alpha + sh(7)*cos_alpha;
	// l = 2, m = 2
	const float cos_2alpha = 2.f*cos_alpha*cos_alpha - 1.f;
	const float sin_2alpha = 2.f*sin_alpha*cos_alpha;
	rotated_sh(4) = sh(4)*cos_2alpha + sh(8)*sin_2alpha; 
	rotated_sh(8) = -sh(4)*sin_2alpha + sh(8)*cos_2alpha;

	return rotated_sh;
}

// Rotate SH9 with ZXZXZ algorithm
__host__ __device__ SH9Scalar rotate_sh9_scalar_zxzxz(const SH9Scalar& sh, const Matrix3f& R) {
	// TODO: handle sin_beta = 0
	const float cos_beta = R(2, 2);
	const float sin_beta = sqrt(1 - R(2, 2)*R(2, 2));
	float cos_alpha;
	float sin_alpha;
	float cos_gamma;
	float sin_gamma;
	if (abs(sin_beta) < 1e-6) {
		cos_alpha = R(1, 1);
		sin_alpha = -R(1, 0);
		cos_gamma = 1.f;
		sin_gamma = 0.f;
	} else {
		cos_alpha = R(2, 0)/sin_beta;
 		sin_alpha = R(2, 1)/sin_beta;
		cos_gamma = -R(0, 2)/sin_beta;
		sin_gamma = R(1, 0)/sin_beta;
	}
	SH9Scalar rotated_sh = rotate_z_sh9(sh, cos_alpha, sin_alpha);
	rotated_sh = rotate_x90_sh9(rotated_sh);
	rotated_sh = rotate_z_sh9(rotated_sh, cos_beta, sin_beta);
	rotated_sh = rotate_xm90_sh9(rotated_sh);
	rotated_sh = rotate_z_sh9(rotated_sh, cos_gamma, sin_gamma);
	return rotated_sh;
}

__host__ __device__ SH9RGB rotate_sh9_zxzxz(const SH9RGB& sh, const Matrix3f& R) {
	SH9RGB rotated_sh;
	rotated_sh.block<9, 1>(0, 0) = rotate_sh9_scalar_zxzxz((SH9Scalar)sh.block<9, 1>(0, 0), R);
	rotated_sh.block<9, 1>(0, 1) = rotate_sh9_scalar_zxzxz((SH9Scalar)sh.block<9, 1>(0, 1), R);
	rotated_sh.block<9, 1>(0, 2) = rotate_sh9_scalar_zxzxz((SH9Scalar)sh.block<9, 1>(0, 2), R);
	return rotated_sh;
}

__host__ __device__ SH9Scalar diffuse_reflection_sh9_scalar(const SH9Scalar& L) {
	SH9Scalar E;
	E(0) = M_PI * L(0);
	E(1) = 2.094395*L(1);
	E(2) = 2.094395*L(2);
	E(3) = 2.094395*L(3);
	E(4) = 0.785398*L(4);
	E(5) = 0.785398*L(5);
	E(6) = 0.785398*L(6);
	E(7) = 0.785398*L(7);
	E(8) = 0.785398*L(8);
	
	return E;
}

__host__ __device__ SH9RGB diffuse_reflection_sh9(const SH9RGB& L) {
	SH9RGB E;
	E.block<9, 1>(0, 0) = diffuse_reflection_sh9_scalar((SH9Scalar)L.block<9, 1>(0, 0));
	E.block<9, 1>(0, 1) = diffuse_reflection_sh9_scalar((SH9Scalar)L.block<9, 1>(0, 1));
	E.block<9, 1>(0, 2) = diffuse_reflection_sh9_scalar((SH9Scalar)L.block<9, 1>(0, 2));
	return E;
}

// 0 multiplies
__host__ __device__ void OptRotateBand0(float dst[1], const float src[1], const Matrix3f& mat)
{
	dst[0] = src[0];
}

// 9 multiplies
__host__ __device__ void OptRotateBand1(float dst[3], const float src[3], const Matrix3f& mat)
{
	// derived from  SlowRotateBand1
	dst[0] = ( mat(1, 1))*src[0] + (-mat(1, 2))*src[1] + ( mat(1, 0))*src[2];
	dst[1] = (-mat(2, 1))*src[0] + ( mat(2, 2))*src[1] + (-mat(2, 0))*src[2];
	dst[2] = ( mat(0, 1))*src[0] + (-mat(0, 2))*src[1] + ( mat(0, 0))*src[2];
}

constexpr float s_c3 = 0.94617469575; // (3*sqrt(5))/(4*sqrt(pi))
constexpr float s_c4 = -0.31539156525;// (-sqrt(5))/(4*sqrt(pi))
constexpr float s_c5 = 0.54627421529; // (sqrt(15))/(4*sqrt(pi))

constexpr float s_c_scale = 1.0/0.91529123286551084;
constexpr float s_c_scale_inv = 0.91529123286551084;

constexpr float s_rc2 = 1.5853309190550713*s_c_scale;
constexpr float s_c4_div_c3 = s_c4/s_c3;
constexpr float s_c4_div_c3_x2 = (s_c4/s_c3)*2.0;

constexpr float s_scale_dst2 = s_c3 * s_c_scale_inv;
constexpr float s_scale_dst4 = s_c5 * s_c_scale_inv;

// 48 multiplies
__host__ __device__ void OptRotateBand2(float dst[5], const float x[5],
								  float m00, float m01, float m02,
								  float m10, float m11, float m12,
								  float m20, float m21, float m22)
{
	// Sparse matrix multiply
	float sh0 =  x[3] + x[4] + x[4] - x[1];
	float sh1 =  x[0] + s_rc2*x[2] +  x[3] + x[4];
	float sh2 =  x[0];
	float sh3 = -x[3];
	float sh4 = -x[1];

	// Rotations.  R0 and R1 just use the raw matrix columns
	float r2x = m00 + m01;
	float r2y = m10 + m11;
	float r2z = m20 + m21;

	float r3x = m00 + m02;
	float r3y = m10 + m12;
	float r3z = m20 + m22;

	float r4x = m01 + m02;
	float r4y = m11 + m12;
	float r4z = m21 + m22;

	// dense matrix multiplication one column at a time
	
	// column 0
	float sh0_x = sh0 * m00;
	float sh0_y = sh0 * m10;
	float d0 = sh0_x * m10;
	float d1 = sh0_y * m20;
	float d2 = sh0 * (m20 * m20 + s_c4_div_c3);
	float d3 = sh0_x * m20;
	float d4 = sh0_x * m00 - sh0_y * m10;
	
	// column 1
	float sh1_x = sh1 * m02;
	float sh1_y = sh1 * m12;
	d0 += sh1_x * m12;
	d1 += sh1_y * m22;
	d2 += sh1 * (m22 * m22 + s_c4_div_c3);
	d3 += sh1_x * m22;
	d4 += sh1_x * m02 - sh1_y * m12;
	
	// column 2
	float sh2_x = sh2 * r2x;
	float sh2_y = sh2 * r2y;
	d0 += sh2_x * r2y;
	d1 += sh2_y * r2z;
	d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2);
	d3 += sh2_x * r2z;
	d4 += sh2_x * r2x - sh2_y * r2y;

	// column 3
	float sh3_x = sh3 * r3x;
	float sh3_y = sh3 * r3y;
	d0 += sh3_x * r3y;
	d1 += sh3_y * r3z;
	d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2);
	d3 += sh3_x * r3z;
	d4 += sh3_x * r3x - sh3_y * r3y;

	// column 4
	float sh4_x = sh4 * r4x;
	float sh4_y = sh4 * r4y;
	d0 += sh4_x * r4y;
	d1 += sh4_y * r4z;
	d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2);
	d3 += sh4_x * r4z;
	d4 += sh4_x * r4x - sh4_y * r4y;

	// extra multipliers
	dst[0] = d0;
	dst[1] = -d1;
	dst[2] = d2 * s_scale_dst2;
	dst[3] = -d3;
	dst[4] = d4 * s_scale_dst4;
}


__host__ __device__ SH9Scalar rotate_sh9_scalar_fast(const SH9Scalar& sh, const Matrix3f& R) {
	SH9Scalar rotated_sh;
	OptRotateBand0(&rotated_sh.data()[0],&sh.data()[0],R);
	OptRotateBand1(&rotated_sh.data()[1],&sh.data()[1],R);
	OptRotateBand2(&rotated_sh.data()[4],&sh.data()[4],
		R(0, 0),R(0, 1),R(0, 2),
		R(1, 0),R(1, 1),R(1, 2),
		R(2, 0),R(2, 1),R(2, 2));

	return rotated_sh;
}

__host__ __device__ SH9RGB rotate_sh9_fast(const SH9RGB& sh, const Matrix3f& R) {
	SH9RGB rotated_sh;
	rotated_sh.block<9, 1>(0, 0) = rotate_sh9_scalar_fast((SH9Scalar)sh.block<9, 1>(0, 0), R);
	rotated_sh.block<9, 1>(0, 1) = rotate_sh9_scalar_fast((SH9Scalar)sh.block<9, 1>(0, 1), R);
	rotated_sh.block<9, 1>(0, 2) = rotate_sh9_scalar_fast((SH9Scalar)sh.block<9, 1>(0, 2), R);
	return rotated_sh;
}

__host__ __device__ Eigen::Vector3f evaluate_irradiance_sh9(const SH9RGB& L, const Eigen::Vector3f& dir) {
	const float c1 = 0.42904276540489171563379376569857;    // 4 * Â2.Y22 = 1/4 * sqrt(15.PI)
    const float c2 = 0.51166335397324424423977581244463;    // 0.5 * Â1.Y10 = 1/2 * sqrt(PI/3)
    const float c3 = 0.24770795610037568833406429782001;    // Â2.Y20 = 1/16 * sqrt(5.PI)
    const float c4 = 0.88622692545275801364908374167057;    // Â0.Y00 = 1/2 * sqrt(PI)

    float   x = dir.x();
    float   y = dir.y();
    float   z = dir.z();

	Eigen::Vector3f irradiance;
	for (int col = 0; col < 3; col++) {
		irradiance(col) =  max( 0.0,
            (c1*(x*x - y*y)) * L(8,col)                    // c1.L22.(x²-y²)
            + (c3*(3.0*z*z - 1)) * L(6, col)               // c3.L20.(3.z² - 1)
            + c4 * L(0, col)                	           // c4.L00 
            + 2.0*c1*(L(4, col)*x*y + L(7, col)*x*z + L(5, col)*y*z) // 2.c1.(L2-2.xy + L21.xz + L2-1.yz)
            + 2.0*c2*(L(3, col)*x + L(1, col)*y + L(2, col)*z) );    // 2.c2.(L11.x + L1-1.y + L10.z)
	}

    return irradiance;
}

NGP_NAMESPACE_END
