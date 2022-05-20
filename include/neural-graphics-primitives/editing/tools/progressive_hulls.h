#pragma once

#include <neural-graphics-primitives/common.h>

#include <vector>
#include <queue>
#define _USE_MATH_DEFINES
#include <math.h>

NGP_NAMESPACE_BEGIN

struct ProgressiveHullsParams {
    float w = 0.1;
    bool compactness_test = true;
    float compactness_threshold = 0.8;
    bool normal_test = true;
    float normal_threshold = M_PI/4;
    bool valence_test = true;
    int max_valence = 12;
    bool presimplify = false;
    float presimplification_ratio = 0.1;
};

bool progressive_hulls_linear(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const size_t max_m,
    Eigen::MatrixXd & U,
    Eigen::MatrixXi & G,
    Eigen::VectorXi & J,
    const ProgressiveHullsParams& params);


bool progressive_hulls_quadratic(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const size_t max_m,
    Eigen::MatrixXd & U,
    Eigen::MatrixXi & G,
    Eigen::VectorXi & J,
    const ProgressiveHullsParams& params);

NGP_NAMESPACE_END
