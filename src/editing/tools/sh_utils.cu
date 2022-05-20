#include <neural-graphics-primitives/editing/tools/sh_utils.h>
#define _USE_MATH_DEFINES
#include <math.h>

NGP_NAMESPACE_BEGIN

Eigen::Vector2f concentric_sample_disk(const Eigen::Vector2f& u) {
	Eigen::Vector2f uOffset = 2.f * u - Eigen::Vector2f::Constant(1.f);
	if (uOffset.x() == 0.f && uOffset.y() == 1.f) {
		return Eigen::Vector2f::Zero();
	}
	float theta, r;
	if (std::abs(uOffset.x()) > std::abs(uOffset.y())) {
		r = uOffset.x();
		theta = M_PI / 4 * (uOffset.y() / uOffset.x());
	} else {
		r = uOffset.y();
		theta = M_PI / 2 - M_PI / 4 * (uOffset.x() / uOffset.y());
	}
	return r * Eigen::Vector2f(std::cos(theta), std::sin(theta));
}

Eigen::Vector3f cosine_sample_hemisphere(const Eigen::Vector2f& u) {
	Eigen::Vector2f d = concentric_sample_disk(u);
	float z = std::sqrt(std::max(0.f, 1.f - d.dot(d)));
	return Eigen::Vector3f(d.x(), d.y(), z);
}

// Dir is provided in global coordinate system
SH9RGB project_sh9(const Eigen::Vector3f& dir, const Eigen::Vector3f& rgb, const float domega) {
	// Code adapted from https://graphics.stanford.edu/papers/envmap/prefilter.c
	SH9RGB coeffs = SH9RGB::Zero();
 
	float x = dir.x();
	float y = dir.y();
	float z = dir.z();

	int col;
    for (col = 0 ; col < 3 ; col++) {
		float c ; /* A different constant for each coefficient */

		/* L_{00}.  Note that Y_{00} = 0.282095 */
		c = 0.282095 ;
		coeffs(0, col) += rgb(col)*c*domega ;

		/* L_{1m}. -1 <= m <= 1.  The linear terms */
		c = 0.488603 ;
		coeffs(1, col) += rgb(col)*(c*y)*domega ;   /* Y_{1-1} = 0.488603 y  */
		coeffs(2, col) += rgb(col)*(c*z)*domega ;   /* Y_{10}  = 0.488603 z  */
		coeffs(3, col) += rgb(col)*(c*x)*domega ;   /* Y_{11}  = 0.488603 x  */

		/* The Quadratic terms, L_{2m} -2 <= m <= 2 */

		/* First, L_{2-2}, L_{2-1}, L_{21} corresponding to xy,yz,xz */
		c = 1.092548 ;
		coeffs(4, col) += rgb(col)*(c*x*y)*domega ; /* Y_{2-2} = 1.092548 xy */ 
		coeffs(5, col) += rgb(col)*(c*y*z)*domega ; /* Y_{2-1} = 1.092548 yz */ 
		coeffs(7, col) += rgb(col)*(c*x*z)*domega ; /* Y_{21}  = 1.092548 xz */ 

		/* L_{20}.  Note that Y_{20} = 0.315392 (3z^2 - 1) */
		c = 0.315392 ;
		coeffs(6, col) += rgb(col)*(c*(3*z*z-1))*domega ; 

		/* L_{22}.  Note that Y_{22} = 0.546274 (x^2 - y^2) */
		c = 0.546274 ;
		coeffs(8, col) += rgb(col)*(c*(x*x-y*y))*domega ;

	}
	return coeffs;
}

SH9Scalar project_sh9(const Eigen::Vector3f& dir, const float val, const float domega) {
	// Code adapted from https://graphics.stanford.edu/papers/envmap/prefilter.c
	SH9Scalar coeffs = SH9Scalar::Zero();
 
	float x = dir.x();
	float y = dir.y();
	float z = dir.z();

	float c ; /* A different constant for each coefficient */

	/* L_{00}.  Note that Y_{00} = 0.282095 */
	c = 0.282095 ;
	coeffs(0) += val*c*domega ;

	/* L_{1m}. -1 <= m <= 1.  The linear terms */
	c = 0.488603 ;
	coeffs(1) += val*(c*y)*domega ;   /* Y_{1-1} = 0.488603 y  */
	coeffs(2) += val*(c*z)*domega ;   /* Y_{10}  = 0.488603 z  */
	coeffs(3) += val*(c*x)*domega ;   /* Y_{11}  = 0.488603 x  */

	/* The Quadratic terms, L_{2m} -2 <= m <= 2 */

	/* First, L_{2-2}, L_{2-1}, L_{21} corresponding to xy,yz,xz */
	c = 1.092548 ;
	coeffs(4) += val*(c*x*y)*domega ; /* Y_{2-2} = 1.092548 xy */ 
	coeffs(5) += val*(c*y*z)*domega ; /* Y_{2-1} = 1.092548 yz */ 
	coeffs(7) += val*(c*x*z)*domega ; /* Y_{21}  = 1.092548 xz */ 

	/* L_{20}.  Note that Y_{20} = 0.315392 (3z^2 - 1) */
	c = 0.315392 ;
	coeffs(6) += val*(c*(3*z*z-1))*domega ; 

	/* L_{22}.  Note that Y_{22} = 0.546274 (x^2 - y^2) */
	c = 0.546274 ;
	coeffs(8) += val*(c*(x*x-y*y))*domega ;
	return coeffs;
}

SH9RGB get_constant_sh9(const Eigen::Vector3f rgb) {
	SH9RGB coeffs = SH9RGB::Zero();
	coeffs.block<1, 3>(0, 0) = rgb;
	return coeffs;
}

NGP_NAMESPACE_END
