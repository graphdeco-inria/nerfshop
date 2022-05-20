#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/editing/edit_operator.h>
#include <neural-graphics-primitives/nerf_network.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/encodings/grid.h>

#include <json/json.hpp>

NGP_NAMESPACE_BEGIN

using namespace nlohmann;

class StudentTrainer {
public:

    static constexpr uint32_t MAX_SUB_BATCH_SCALE2 = 8;

    StudentTrainer() {}

    void init_student(
        json config, 
        std::shared_ptr<NerfNetwork<precision_t>> teacher_network, 
        BoundingBox aabb, 
        tcnn::GPUMemory<float>& density_grid, 
        uint32_t max_cascade,
        int aabb_scale,
        cudaStream_t stream);

    void train(
        uint32_t target_batch_size, 
        uint32_t n_training_steps, 
        std::vector<std::shared_ptr<EditOperator>>& edit_operators,
        cudaStream_t stream);

    std::shared_ptr<NerfNetwork<precision_t>> student_network () const {
        return m_student_network;
    }

    std::shared_ptr<tcnn::Loss<precision_t>> loss () const {
        return m_loss;
    }

    std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer () const {
        return m_optimizer;
    }

    std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer () const {
        return m_trainer;
    }

    float loss_scalar = -1.f;

    uint32_t sub_batch_size = 1 << MAX_SUB_BATCH_SCALE2;
    uint32_t n_sub_batches = 1;

    int n_rejections = 10;
    float lambda_features = 1.f;
    int n_trained_encoding_levels = 1;
    int n_encoding_levels = 1;
    bool use_gaussian_smoothing = false;
    float sigma_smoothing = 0.1f;  
    int n_gaussian_samples = 10;

    void display_debug_gl(
        const Eigen::Vector2i& resolution, 
        const Eigen::Vector2f& focal_length, 
        const Eigen::Matrix<float, 3, 4>& camera_matrix, 
        const Eigen::Vector2f& screen_center,
        const bool display_teacher
    );

    int training_step() {
        return m_training_steps;
    }

    void update_rgb(cudaStream_t stream);

private:

    void train_step(
        uint32_t target_batch_size, 
        uint32_t full_batch_size,
        float* loss, 
        std::vector<std::shared_ptr<EditOperator>>& edit_operators,
        bool overwrite_gradients,
        cudaStream_t stream
    );

    // Networks
    uint32_t m_teacher_max_cascade;
    tcnn::GPUMemory<float> m_teacher_density_grid;
    BoundingBox m_aabb;
    std::shared_ptr<NerfNetwork<precision_t>> m_teacher_network;
    std::shared_ptr<NerfNetwork<precision_t>> m_student_network;

    NetworkParams<precision_t> m_teacher_rgb_params;
    NetworkParams<precision_t> m_student_rgb_params;
    NetworkParams<precision_t> m_student_params;
    tcnn::GridEncoding<precision_t> * m_student_grid_encoding;

    // Optimizer & co
	std::shared_ptr<tcnn::Loss<precision_t>> m_loss;
	std::shared_ptr<tcnn::Optimizer<precision_t>> m_optimizer;
	std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> m_trainer;

    // DEBUG
    std::vector<Eigen::Vector3f> display_samples_teacher;
    std::vector<Eigen::Vector3f> display_samples_student;
    std::vector<Eigen::Vector3f> display_samples_teacher_smoothed;

    uint32_t m_seed = 1337;
    default_rng_t m_rng;

    uint32_t m_training_steps = 0;
    uint32_t m_density_sampling_steps = 0;
    tcnn::GPUMemory<float> m_sub_batch_loss;

    static constexpr float LOSS_SCALE = 128.f;
};

NGP_NAMESPACE_END