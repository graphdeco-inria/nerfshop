//#include <neural-graphics-primitives/student_trainer.h>
//#include <neural-graphics-primitives/common_nerf.h>
//#include <neural-graphics-primitives/common_gl.h>
//#include <neural-graphics-primitives/nerf.h>
//#include <neural-graphics-primitives/nerf_network_full.h>
//#include <neural-graphics-primitives/nerf_network_nodir.h>
//
//#include <tiny-cuda-nn/gpu_memory.h>
//#include <tiny-cuda-nn/common_device.h>
//
//#ifdef NGP_GUI
//#  ifdef _WIN32
//#    include <GL/gl3w.h>
//#  else
//#    include <GL/glew.h>
//#  endif
//#  include <GL/glu.h>
//#  include <GLFW/glfw3.h>
//#endif
//
//NGP_NAMESPACE_BEGIN
//
//__global__ void compute_feature_loss(
//    const uint32_t n_elements,
//    const uint32_t full_batch_size,
//    float loss_scale,
//    const uint32_t padded_density_output_width,
//    const uint32_t padded_loss_output_width, // REDUNDANT
//    const tcnn::network_precision_t* teacher_network_output,
//    const tcnn::network_precision_t* student_network_output,
//    tcnn::network_precision_t* dloss_doutput,
//    float* __restrict__ loss_output,
//    const float lambda_features
//) {
//    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
//	if (i >= n_elements) { return; }
//
//    loss_scale /= (float)full_batch_size;
//
//    teacher_network_output += i * padded_density_output_width;
//    student_network_output += i * padded_density_output_width;
//
//    const tcnn::vector_t<tcnn::network_precision_t, 16> teacher_local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 16>*)teacher_network_output;
//    const FeatureVector teacher_feature = network_to_feature(teacher_local_network_output);
//    const tcnn::vector_t<tcnn::network_precision_t, 16> student_local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 16>*)student_network_output;
//    const FeatureVector student_feature = network_to_feature(student_local_network_output);
//
//    const FeatureVector difference = student_feature - teacher_feature;
//    FeatureVector difference_squared = difference.cwiseProduct(difference);
//
//    for (int j = 1; j < 16; j++) {
//        difference_squared(j) *= lambda_features;
//    }
//
//    dloss_doutput += i * padded_loss_output_width;
//
//    if (loss_output) {
//        loss_output[i] = difference_squared.mean() / (float)full_batch_size;
//    }
//
//    tcnn::vector_t<tcnn::network_precision_t, 16> local_dL_doutput;
//    for (int j = 0; j < 16; j++) {
//		local_dL_doutput[j] = 2.0 * loss_scale * difference(j);
//        if (j > 0) {
//            local_dL_doutput[j] *= lambda_features;
//        }
//	}
//
//    *(tcnn::vector_t<tcnn::network_precision_t, 16>*)dloss_doutput = local_dL_doutput;
//}
//
//__global__ void smooth_network_outputs(
//    const uint32_t n_elements, 
//    const uint32_t n_samples,
//    const double* __restrict__ gaussian_weights, 
//    const tcnn::network_precision_t* teacher_network_output,
//    tcnn::network_precision_t* smoothed_teacher_network_output,
//    const uint32_t padded_density_output_width,
//    BoundingBox aabb, 
//    NerfPosition* __restrict teacher_coords,
//    NerfPosition* __restrict__ smoothed_teacher_coords // DEBUG purposes only
//    ) {
//
//	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
//	if (i >= n_elements) return;
//
//    teacher_network_output += n_samples * i * padded_density_output_width;
//
//    FeatureVectorFp smoothed_teacher_feature = FeatureVectorFp::Zero();
//    Eigen::Vector3d smoothed_pos = Eigen::Vector3d::Zero();
//
//    double weight_sum = 0.;
//    for (uint32_t j = 0; j < n_samples; j++) {
//        // printf("%f", gaussian_weights[n_samples*i+j]);
//        const tcnn::vector_t<tcnn::network_precision_t, 16> teacher_local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 16>*)teacher_network_output;
//        
//        smoothed_teacher_feature += gaussian_weights[n_samples * i +j] * network_to_feature(teacher_local_network_output).cast<double>();
//        smoothed_pos += gaussian_weights[n_samples * i + j] * unwarp_position(teacher_coords[n_samples * i + j].p, aabb).cast<double>();
//
//        teacher_network_output += padded_density_output_width;
//
//        weight_sum += gaussian_weights[n_samples * i +j];
//        // printf("weights: %f", gaussian_weights[n_samples*i+j]);
//    }
//    // printf("sum: %f", weight_sum);
//
//    smoothed_teacher_feature /= weight_sum;
//    smoothed_pos /= weight_sum;
//    
//    smoothed_teacher_network_output += i * padded_density_output_width;
//    tcnn::vector_t<tcnn::network_precision_t, 16> local_smoothed_teacher;
//    for (int j = 0; j < 16; j++) {
//		local_smoothed_teacher[j] = smoothed_teacher_feature(j);
//	}
//    *(tcnn::vector_t<tcnn::network_precision_t, 16>*)smoothed_teacher_network_output = local_smoothed_teacher;
//    smoothed_teacher_coords[i] = { warp_position(smoothed_pos.cast<float>(), aabb), warp_dt(MIN_CONE_STEPSIZE()) };
//}
//
//__global__ void generate_grid_samples_importance(
//    const uint32_t n_elements, 
//    const uint32_t n_rejections,
//    default_rng_t rng, 
//    const uint32_t step, BoundingBox aabb, 
//    const float* __restrict__ grid_in, 
//    NerfPosition* __restrict__ out, 
//    uint32_t* __restrict__ indices, 
//    uint32_t n_cascades, 
//    float thresh) {
//
//	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
//	if (i >= n_elements) return;
//
//	// 1 random number to select the level, 3 to select the position.
//	rng.advance(i*4); 
//	uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;
//
//	// Select grid cell that has (sufficient) density
//	uint32_t idx;
//	for (uint32_t j = 0; j < n_rejections; ++j) {
//		idx = ((i+step*n_elements) * 56924617 + j * 19349663 + 96925573) % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());
//		idx += level * NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE();
//		if (grid_in[idx] > thresh) {
//			break;
//		}
//	}
//
//	// Random position within that cellq
//	uint32_t pos_idx = idx % (NERF_GRIDSIZE()*NERF_GRIDSIZE()*NERF_GRIDSIZE());
//
//	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
//	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
//	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);
//
//	Vector3f pos = ((Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / NERF_GRIDSIZE() - Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Vector3f::Constant(0.5f);
//
//    out[i] = { warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
//    indices[i] = idx;
//}
//
//__global__ void generate_grid_samples_teacher_gaussian(
//    const uint32_t n_elements, 
//    const uint32_t n_samples,
//    const float sigma,
//    const float inv_radius,
//    default_rng_t rng, 
//    BoundingBox aabb, 
//    const NerfPosition* __restrict__ student_in, 
//    NerfPosition* __restrict__ teacher_out,
//    double* __restrict__ gaussian_weights
//) {
//
//	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
//	if (i >= n_elements) return;
//
//	rng.advance(i * 4* n_samples);
//
//	Vector3f pos = unwarp_position(student_in[i].p, aabb);
//
//    for (uint32_t j = 0; j < n_samples; j++) {
//        // Sample gaussian with Box-Muller (4 uniforms needed to get 3 gaussians)
//        float u1 = random_val(rng);
//        float u2 = random_val(rng);
//        float u3 = random_val(rng);
//        float u4 = random_val(rng);
//        // Make sure we don't take the log of 0
//        float xc = sqrt(-2*log(u1+1e-9f))*cos(2*M_PI*u2);
//        float yc = sqrt(-2*log(u1+1e-9f))*sin(2*M_PI*u2);
//        float zc = sqrt(-2*log(u3+1e-9f))*cos(2*M_PI*u4);
//        // Make sure it doesn't go too far off
//        // xc = tcnn::clamp(xc, -5.f, 5.f);
//        // yc = tcnn::clamp(yc, -5.f, 5.f);
//        // zc = tcnn::clamp(zc, -5.f, 5.f);
//        // If not inside, use student as ref (this should prevent instabilities)
//        if (aabb.contains(pos + Vector3f{sigma*xc, sigma*yc, sigma*zc})) {
//            teacher_out[n_samples*i+j] = { warp_position(pos + Vector3f{sigma*xc, sigma*yc, sigma*zc}, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
//            gaussian_weights[n_samples*i+j] = max(tcnn::gaussian_pdf(xc, 1)*tcnn::gaussian_pdf(yc, 1)*tcnn::gaussian_pdf(zc, 1), 1e-14);
//        } else {
//            gaussian_weights[n_samples*i+j] = max(tcnn::gaussian_pdf(0, 1)*tcnn::gaussian_pdf(0, 1)*tcnn::gaussian_pdf(0, 1), 1e-14);
//            teacher_out[n_samples*i+j] = { warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
//        }   
//        // printf("%f", gaussian_weights[n_samples*i+j]);
//    }
//}
//
//__global__ void print_gaussian_weights(
//    const uint32_t n_elements, 
//    const uint32_t n_samples,
//    float* __restrict__ gaussian_weights
//) {
//    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
//	if (i >= n_elements) return;
//
//    for (uint32_t j = 0; j < n_samples; j++) {
//        printf("%f", gaussian_weights[n_samples*i+j]);
//    }
//}
//
//void StudentTrainer::init_student(
//        json config, 
//        std::shared_ptr<NerfNetwork<precision_t>> teacher_network, 
//        BoundingBox aabb, 
//        tcnn::GPUMemory<float>& density_grid, 
//        uint32_t max_cascade,
//        int aabb_scale, 
//        cudaStream_t stream) {
//
//    // Set the teacher network
//    m_teacher_network = teacher_network;
//    m_teacher_density_grid = density_grid.copy();
//    m_teacher_max_cascade = max_cascade;
//    m_aabb = aabb;
//
//    m_rng = default_rng_t{m_seed};
//
//    // -------------------------
//    // LOSS & OPTIMIZER
//    // -------------------------
//
//    json& loss_config = config["loss"];
//	json& optimizer_config = config["optimizer"];
//
//    // Some of the Nerf-supported losses are not supported by tcnn::Loss,
//    // so just create a dummy L2 loss there. The NeRF code path will bypass
//    // the tcnn::Loss in any case.
//    loss_config["otype"] = "L2";
//
//    m_loss.reset(tcnn::create_loss<precision_t>(loss_config));
//	m_optimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
//
//    // -------------------------
//    // NETWORK
//    // -------------------------
//
//    NetworkDims dims = network_dims_nerf();
//
//    json& encoding_config = config["encoding"];
//	// json& loss_config = config["loss"];
//	// json& optimizer_config = config["optimizer"];
//	json& network_config = config["network"];
//
//    // Automatically determine certain parameters if we're dealing with the (hash)grid encoding
//	if (tcnn::to_lower(encoding_config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
//		encoding_config["n_pos_dims"] = dims.n_pos;
//
//		const uint32_t n_features_per_level = encoding_config.value("n_features_per_level", 2u);
//        int num_levels;
//
//		if (encoding_config.contains("n_features") && encoding_config["n_features"] > 0) {
//			num_levels = (uint32_t)encoding_config["n_features"] / n_features_per_level;
//		} else {
//			num_levels = encoding_config.value("n_levels", 16u);
//		}
//
//		const uint32_t log2_hashmap_size = encoding_config.value("log2_hashmap_size", 15);
//
//		uint32_t base_grid_resolution = encoding_config.value("base_resolution", 0);
//		if (!base_grid_resolution) {
//			base_grid_resolution = 1u << ((log2_hashmap_size) / dims.n_pos);
//			encoding_config["base_resolution"] = base_grid_resolution;
//		}
//
//        float desired_resolution = 2048.0f; // Desired resolution of the finest hashgrid level over the unit cube
//
//		// Automatically determine suitable per_level_scale
//		float per_level_scale = encoding_config.value("per_level_scale", 0.0f);
//		if (per_level_scale <= 0.0f && num_levels > 1) {
//			per_level_scale = std::exp(std::log(desired_resolution * (float)aabb_scale / (float)base_grid_resolution) / (float)(num_levels-1));
//			encoding_config["per_level_scale"] = per_level_scale;
//		}
//
//		tlog::info()
//			<< "GridEncoding: "
//			<< " Nmin=" << base_grid_resolution
//			<< " b=" << per_level_scale
//			<< " F=" << n_features_per_level
//			<< " T=2^" << log2_hashmap_size
//			<< " L=" << num_levels
//			;
//
//        n_encoding_levels = num_levels;
//        n_trained_encoding_levels = num_levels;
//	}
//    
//    bool has_dir = config.contains("dir_encoding") && config.contains("rgb_network");
//    
//    // Those dims are required for genericity because everything is fed into the NeRF model, no matter its origin.
//    uint32_t n_dir_dims = 3;
//    uint32_t n_extra_dims = 0u;
//
//    std::shared_ptr<NerfNetworkFull<precision_t>> student_full_network;
//    
//    if (has_dir) {
//        json& dir_encoding_config = config["dir_encoding"];
//        json& rgb_network_config = config["rgb_network"];
//
//        m_student_network = student_full_network = std::make_shared<NerfNetworkFull<precision_t>>(
//            dims.n_pos,
//            n_dir_dims,
//            n_extra_dims,
//            dims.n_pos + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
//            encoding_config,
//            dir_encoding_config,
//            network_config,
//            rgb_network_config
//        );
//
//        std::cout << "Full NeRF" << std::endl;
//    } else {
//        m_student_network = std::make_shared<NerfNetworkNoDir<precision_t>>(
//				dims.n_pos,
//				n_dir_dims,
//				n_extra_dims,
//				dims.n_pos + 1, // The offset of 1 comes from the dt member variable of NerfCoordinate. HACKY
//				encoding_config,
//				network_config
//			);
//        std::cout << "No Dir NeRF" << std::endl;
//    }
//
//
//    tlog::info()
//        << "Density model: " << dims.n_pos
//        << "--[" << std::string(encoding_config["otype"])
//        << "]-->" << m_student_network->encoding()->padded_output_width()
//        << "--[" << std::string(network_config["otype"])
//        << "(neurons=" << (int)network_config["n_neurons"] << ",layers=" << ((int)network_config["n_hidden_layers"]) << ")"
//        << "]-->" << 1
//        ;
//
//    // Setup trainer
//    m_trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(m_student_network, m_optimizer, m_loss, m_seed);
//
//    // Copy the rgb network weights! AFTER setting up the trainer (because it will initialize the weights itself!
//    if (has_dir) {
//        m_teacher_rgb_params = ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->get_rgb_params();
//        m_student_rgb_params = ((NerfNetworkFull<precision_t> *)m_student_network.get())->get_rgb_params();
//        m_student_params = ((NerfNetworkFull<precision_t> *)m_student_network.get())->get_network_params();
//        m_student_grid_encoding = (tcnn::GridEncoding<precision_t> *)(m_student_network->encoding().get());
//        update_rgb(stream);
//    }
//}
//
//uint32_t next_power2(uint32_t target_batch_size) {
//    int v = 0;
//    while (target_batch_size >>= 1) ++v;
//    return v;
//}
//
//void StudentTrainer::train(
//    uint32_t target_batch_size, 
//    uint32_t n_training_steps, 
//    std::vector<std::shared_ptr<EditOperator>>& edit_operators,
//    cudaStream_t stream) {
//
//    if (!m_teacher_network) {
//        std::cout << "Cannot train without initializing student!" << std::endl;
//        return;
//    }
//
//    // Get next power of 2 and cut into subbatches to fit into shared memory
//    uint32_t scale2 = next_power2(target_batch_size);
//    n_sub_batches = 1;
//    if (scale2 > MAX_SUB_BATCH_SCALE2) {
//        target_batch_size = 1 << scale2;
//        n_sub_batches = 1 << (scale2 - MAX_SUB_BATCH_SCALE2);
//    } else {
//        sub_batch_size = target_batch_size;
//    }
//
//    std::cout << "scale2: " << scale2 << std::endl;
//    std::cout << "target batch size: " << target_batch_size << std::endl;
//    std::cout << "n_sub_batches: " << n_sub_batches << std::endl;
//    std::cout << "sub_batch_size: " << sub_batch_size << std::endl;
//
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//
//    // Reset loss
//    loss_scalar = 0.f;
//    
//    for (uint32_t i = 0; i < n_training_steps; ++i) {
//        // std::cout << "Train: " << i << std::endl;
//        // prepare_for_training_steps(n_sub_batches, sub_batch_size, stream);
//        for (int j= 0; j < n_sub_batches; j++) {
//            // std::cout << "Subbatch " << j << std::endl;
//            m_sub_batch_loss.enlarge(sub_batch_size);
//            CUDA_CHECK_THROW(cudaMemsetAsync(m_sub_batch_loss.data(), 0, sizeof(float)*sub_batch_size, stream));
//            train_step(
//                sub_batch_size,
//                target_batch_size,
//                m_sub_batch_loss.data(),
//                edit_operators,
//                j == 0,
//                stream
//            );
//
//            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//
//            loss_scalar += tcnn::reduce_sum(m_sub_batch_loss.data(), sub_batch_size, stream) / (float)(n_training_steps);
//
//            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//        }
//
//        m_trainer->optimizer_step(stream, LOSS_SCALE);
//
//        update_rgb(stream);
//
//        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//
//        ++m_training_steps;
//    }
//
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//
//    // update_after_training(target_batch_size, n_training_steps, stream);
//
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//
//    std::cout << "Trained for " << n_training_steps << " steps and obtained loss " << loss_scalar << std::endl;
//}
//
//void StudentTrainer::train_step(
//    uint32_t target_batch_size, 
//    uint32_t full_batch_size,
//    float* loss, 
//    std::vector<std::shared_ptr<EditOperator>>& edit_operators,
//    bool overwrite_gradients,
//    cudaStream_t stream) {
//
//    const uint32_t padded_density_output_width = m_teacher_network->padded_density_output_width();
//    const uint32_t floats_per_coord = sizeof(NerfPosition) / sizeof(float);
//
//    if (m_training_steps == 0) {
//		m_density_sampling_steps = 0;
//	}
//
//    const float inv_radius = 1.f / sqrt(sigma_smoothing);
//    uint32_t n_samples = use_gaussian_smoothing ? n_gaussian_samples : 1;
//
//    tcnn::GPUMemoryArena::Allocation alloc;
//	auto scratch = tcnn::allocate_workspace_and_distribute<
//		NerfPosition, // student_coords
//        NerfPosition, // teacher_coords
//        double, // gaussian_weights (in case of gaussian smoothing)
//        precision_t, // teacher_mlp_out
//		precision_t, // student_mlp_out
//        uint32_t,  // grid_indices 
//        precision_t, // dloss_dmlp_out
//        precision_t, // smoothed_teacher_mlp_out (in case of gaussian smoothing)
//        NerfPosition // smooth_teacher_coords (DEBUG)
//	>(
//		stream, &alloc,
//		target_batch_size,
//        target_batch_size * n_samples,
//        target_batch_size * n_samples,
//		target_batch_size * padded_density_output_width,
//        target_batch_size * padded_density_output_width,
//		target_batch_size,
//        target_batch_size * padded_density_output_width,
//        target_batch_size * padded_density_output_width,
//        target_batch_size
//	);
//
//    NerfPosition* student_coords = std::get<0>(scratch);
//    NerfPosition* teacher_coords = std::get<1>(scratch);
//    double* gaussian_weights = std::get<2>(scratch);
//    precision_t* teacher_mlp_out = std::get<3>(scratch);
//	precision_t* student_mlp_out = std::get<4>(scratch);
//    uint32_t* grid_indices = std::get<5>(scratch);
//	precision_t* dloss_dmlp_out = std::get<6>(scratch);
//    precision_t* smoothed_teacher_mlp_out = std::get<7>(scratch);
//    NerfPosition* smoothed_teacher_coords = std::get<8>(scratch);
//
//    linear_kernel(generate_grid_samples_importance, 0, stream,
//        target_batch_size, // Number of final batch elements here!
//        (uint32_t)n_rejections,
//        m_rng,
//        m_density_sampling_steps,
//        m_aabb,
//        m_teacher_density_grid.data(),
//        student_coords,
//        grid_indices,
//        // Why was this +1?
//        m_teacher_max_cascade+1,
//        NERF_MIN_OPTICAL_THICKNESS()
//    );
//
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//    
//    m_rng.advance();
//    ++m_density_sampling_steps;
//
//    // Apply the provided operators (to the copied teacher coords)
//    if (use_gaussian_smoothing) {
//        linear_kernel(generate_grid_samples_teacher_gaussian, 0, stream,
//            target_batch_size, // Number of final batch elements here!
//            n_samples,
//            sigma_smoothing,
//            inv_radius,
//            m_rng,
//            m_aabb,
//            student_coords,
//            teacher_coords,
//            gaussian_weights
//        );
//        m_rng.advance();
//    } else {
//        CUDA_CHECK_THROW(cudaMemcpyAsync(teacher_coords, student_coords, target_batch_size*floats_per_coord*sizeof(float), cudaMemcpyDeviceToDevice, stream));
//    }
//
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//
//    tcnn::GPUMatrix<bool> empty_mask;
//    if (edit_operators.size() > 0) {
//        empty_mask =  tcnn::GPUMatrix<bool>(1, target_batch_size*n_samples);
//        CUDA_CHECK_THROW(cudaMemsetAsync(empty_mask.data(), false, empty_mask.n_bytes(), stream));
//        for (int i = edit_operators.size() - 1; i >= 0; i--) {
//            auto& edit_operator = edit_operators[i];
//            edit_operator->map_positions(stream, tcnn::PitchedPtr<NerfPosition>((NerfPosition*)teacher_coords, 1), empty_mask, target_batch_size*n_samples);
//        }
//    }
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//
//    // DEBUG
//    if (overwrite_gradients) {
//        display_samples_teacher.clear();
//        display_samples_student.clear();
//    }
//    display_samples_teacher.reserve(display_samples_teacher.size() + target_batch_size*n_samples);
//    display_samples_student.reserve(display_samples_student.size() + target_batch_size);
//    std::vector<NerfPosition> teacher_coords_host;
//    std::vector<NerfPosition> student_coords_host;
//    teacher_coords_host.resize(target_batch_size*n_samples, {Eigen::Vector3f::Zero(), 0.f});
//    student_coords_host.resize(target_batch_size, {Eigen::Vector3f::Zero(), 0.f});
//    CUDA_CHECK_THROW(cudaMemcpyAsync(teacher_coords_host.data(), teacher_coords, target_batch_size*n_samples*floats_per_coord*sizeof(float), cudaMemcpyDeviceToHost, stream));
//    CUDA_CHECK_THROW(cudaMemcpyAsync(student_coords_host.data(), student_coords, target_batch_size*floats_per_coord*sizeof(float), cudaMemcpyDeviceToHost, stream));
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//    for (int i = 0; i < target_batch_size; i++) {
//        display_samples_student.push_back(unwarp_position(student_coords_host[i].p, m_aabb));
//    }
//    for (int i = 0; i < target_batch_size*n_samples; i++) {
//        display_samples_teacher.push_back(unwarp_position(teacher_coords_host[i].p, m_aabb));
//    }
//    // -----------------------
//
//    tcnn::GPUMatrix<float> teacher_coords_matrix((float*)teacher_coords, floats_per_coord, target_batch_size*n_samples);
//    tcnn::GPUMatrix<float> student_coords_matrix((float*)student_coords, floats_per_coord, target_batch_size);
//    tcnn::GPUMatrix<precision_t> teacher_sigmafeature_matrix(teacher_mlp_out, padded_density_output_width, target_batch_size*n_samples);
//    tcnn::GPUMatrix<precision_t> student_sigmafeature_matrix(student_mlp_out, padded_density_output_width, target_batch_size);
//
//    tcnn::GPUMatrix<precision_t> gradient_matrix(dloss_dmlp_out, padded_density_output_width, target_batch_size);
//    // Reset the gradient matrix because the rgb weights will not be updated!
//    gradient_matrix.memset(0);
//
//    // First pass to compute the loss
//    m_teacher_network->density(stream, teacher_coords_matrix, teacher_sigmafeature_matrix, false);
//    m_student_network->density(stream, student_coords_matrix, student_sigmafeature_matrix, false);
//
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//    
//    // Perform smoothing
//    if (use_gaussian_smoothing) {
//        tcnn::linear_kernel(smooth_network_outputs, 0, stream,
//            target_batch_size,
//            n_samples,
//            gaussian_weights,
//            teacher_mlp_out,
//            smoothed_teacher_mlp_out,
//            padded_density_output_width,
//            m_aabb,
//            teacher_coords,
//            smoothed_teacher_coords
//        );
//        if (overwrite_gradients) {
//            display_samples_teacher_smoothed.clear();
//        }
//        display_samples_teacher_smoothed.reserve(display_samples_teacher_smoothed.size() + target_batch_size);
//        std::vector<NerfPosition> smoothed_teacher_coords_host;
//        smoothed_teacher_coords_host.resize(target_batch_size, {Eigen::Vector3f::Zero(), 0.f});
//        CUDA_CHECK_THROW(cudaMemcpyAsync(smoothed_teacher_coords_host.data(), smoothed_teacher_coords, target_batch_size*floats_per_coord*sizeof(float), cudaMemcpyDeviceToHost, stream));
//        for (int i = 0; i < target_batch_size; i++) {
//            display_samples_teacher_smoothed.push_back(unwarp_position(smoothed_teacher_coords_host[i].p, m_aabb));
//        }
//    }
//
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//
//	tcnn::linear_kernel(compute_feature_loss, 0, stream,
//        target_batch_size,
//        full_batch_size,
//        LOSS_SCALE,
//        padded_density_output_width,
//        padded_density_output_width,
//        use_gaussian_smoothing ? smoothed_teacher_mlp_out : teacher_mlp_out,
//        student_mlp_out,
//        dloss_dmlp_out,
//        loss,
//        lambda_features
//    );
//
//    {
//        // Use this one as a proxy...
//        auto ctx = m_student_network->density_forward(stream, student_coords_matrix, &student_sigmafeature_matrix, false);
//		m_student_network->density_backward(stream, *ctx, student_coords_matrix, student_sigmafeature_matrix, gradient_matrix, nullptr, false, overwrite_gradients ? tcnn::EGradientMode::Overwrite : tcnn::EGradientMode::Accumulate);
//    }
//
//    m_rng.advance();
//
//
//
//    // Zero out the gradients corresponding to rgb (this ensures that we only train on features!)
//    uint32_t grid_offset = m_student_grid_encoding->level_params_offset(n_trained_encoding_levels);
//    uint32_t grid_fixed_n_params = m_student_grid_encoding->level_params_offset(n_encoding_levels)-m_student_grid_encoding->level_params_offset(n_trained_encoding_levels);
//    CUDA_CHECK_THROW(cudaMemsetAsync(m_student_params.gradients + ((NerfNetworkFull<precision_t> *)m_student_network.get())->pos_encoding_offset() + grid_offset, 0, grid_fixed_n_params * sizeof(precision_t), stream));
//    // CUDA_CHECK_THROW(cudaMemsetAsync(m_student_rgb_params.gradients, 0, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_rgb_params() * sizeof(precision_t), stream));
//    // CUDA_CHECK_THROW(cudaMemsetAsync(m_student_rgb_params.backward_params, 0, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_rgb_params() * sizeof(precision_t), stream));
//    // CUDA_CHECK_THROW(cudaMemsetAsync(m_student_rgb_params.inference_params, 0, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_rgb_params() * sizeof(precision_t), stream));
//    // TODO: don't forget to clear density for the student
//
//}
//
//void StudentTrainer::update_rgb(cudaStream_t stream) {
//    // RGB network
//    CUDA_CHECK_THROW(cudaMemcpyAsync(m_student_rgb_params.params_full_precision, m_teacher_rgb_params.params_full_precision, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_rgb_params() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
//    CUDA_CHECK_THROW(cudaMemcpyAsync(m_student_rgb_params.params, m_teacher_rgb_params.params, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_rgb_params() * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream));
//    CUDA_CHECK_THROW(cudaMemcpyAsync(m_student_rgb_params.inference_params, m_teacher_rgb_params.inference_params, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_rgb_params() * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream));
//    CUDA_CHECK_THROW(cudaMemcpyAsync(m_student_rgb_params.backward_params, m_teacher_rgb_params.backward_params, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_rgb_params() * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream));
//    CUDA_CHECK_THROW(cudaMemcpyAsync(m_student_rgb_params.gradients, m_teacher_rgb_params.gradients, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_rgb_params() * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream));
//    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//    // // Dir encoding
//    // NetworkParams<precision_t> m_teacher_dir_encoding_params = ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->get_dir_encoding_params();
//    // NetworkParams<precision_t> m_student_dir_encoding_params = ((NerfNetworkFull<precision_t> *)m_student_network.get())->get_dir_encoding_params();
//    // CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//    // CUDA_CHECK_THROW(cudaMemcpyAsync(student_dir_encoding_params.params_full_precision, teacher_dir_encoding_params.params_full_precision, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_dir_encoding_params() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
//    // CUDA_CHECK_THROW(cudaMemcpyAsync(student_dir_encoding_params.params, teacher_dir_encoding_params.params, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_dir_encoding_params() * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream));
//    // CUDA_CHECK_THROW(cudaMemcpyAsync(student_dir_encoding_params.inference_params, teacher_dir_encoding_params.inference_params, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_dir_encoding_params() * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream));
//    // CUDA_CHECK_THROW(cudaMemcpyAsync(student_dir_encoding_params.backward_params, teacher_dir_encoding_params.backward_params, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_dir_encoding_params() * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream));
//    // CUDA_CHECK_THROW(cudaMemcpyAsync(student_dir_encoding_params.gradients, teacher_dir_encoding_params.gradients, ((NerfNetworkFull<precision_t> *)m_teacher_network.get())->n_dir_encoding_params() * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream));
//    // CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
//}
//
//void StudentTrainer::display_debug_gl(
//    const Eigen::Vector2i& resolution, 
//    const Eigen::Vector2f& focal_length, 
//    const Eigen::Matrix<float, 3, 4>& camera_matrix, 
//    const Eigen::Vector2f& screen_center,
//    const bool display_teacher
//) {
//    const std::vector<Eigen::Vector3f>& display_samples = display_teacher ? display_samples_teacher_smoothed : display_samples_student;
//
//    if (display_samples.size() == 0) {
//        return;
//    }
//
//    static GLuint VAO = 0, VBO[4] = {}, vbosize = 0, program = 0, vs = 0, ps = 0;
//    if (!VAO) {
//		glGenVertexArrays(1, &VAO);
//		glBindVertexArray(VAO);
//	}
//    if (vbosize != display_samples.size()) {
//        // VBO for positions
//        glGenBuffers(1, &VBO[0]);
//        glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
//        glBufferData(GL_ARRAY_BUFFER, display_samples.size()*sizeof(Eigen::Vector3f), &display_samples[0], GL_STATIC_DRAW);
//        // VBO for boundary
//        // glGenBuffers(1, &VBO[1]);
//        // glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
//        // glBufferData(GL_ARRAY_BUFFER, labels.size()*sizeof(uint8_t), &labels[0], GL_STATIC_DRAW);
//    }
//    if (!program) {
//        // vertex shader
//        const char * vertexShaderSource = R"foo(
//        layout (location = 0) in vec3 pos;
//        uniform mat4 camera;
//        uniform vec2 f;
//        uniform ivec2 res;
//        uniform vec2 cen;
//        uniform int mode;
//        void main()
//        {
//            gl_PointSize = 2.0f;
//            vec4 p = camera * vec4(pos, 1.0);
//            p.xy *= vec2(2.0, -2.0) * f.xy / vec2(res.xy);
//            p.w = p.z;
//            p.z = p.z - 0.1;
//            p.xy += cen * p.w;
//            gl_Position = p;
//        })foo";
//
//        vs = compile_shader(false, vertexShaderSource);
//
//        // fragment shader
//        const char * fragmentShaderSource = R"foo(
//        in float fDensity;
//        in vec3 fNormal;
//        flat in int fBoundary;
//        out vec4 FragColor;
//        uniform int mode;
//        void main() {
//            if (mode == 4) {
//                FragColor = vec4(fNormal, 1.0f);
//            }
//            else if (mode == 3) {
//                if (fBoundary == 1) {
//                    FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
//                } else {
//                    FragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);
//                }
//            }
//            else if (mode == 2) {
//                FragColor = mix(vec4(1.0f, 0.0f, 0.0f, 1.0f), vec4(0.0f, 1.0f, 0.0f, 1.0f), fDensity);
//            } else if (mode == 1) {
//                FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
//            }
//        })foo";
//
//        ps = compile_shader(true, fragmentShaderSource);
//
//        program = glCreateProgram();
//        glAttachShader(program, vs);
//        glAttachShader(program, ps);
//        glLinkProgram(program);
//        if (!check_shader(program, "shader program", true)) {
//			glDeleteProgram(program);
//			program = 0;
//		}
//
//        glDeleteShader(vs);
//        glDeleteShader(ps);
//    }
//
//    glBindVertexArray(VAO);
//
//    glUseProgram(program);
//
//    Eigen::Matrix4f view2world=Eigen::Matrix4f::Identity();
//	view2world.block<3,4>(0,0) = camera_matrix;
//	Eigen::Matrix4f world2view = view2world.inverse();
//    glUniformMatrix4fv(glGetUniformLocation(program, "camera"), 1, GL_FALSE, (GLfloat*)&world2view);
//	glUniform2f(glGetUniformLocation(program, "f"), focal_length.x(), focal_length.y());
//	glUniform2f(glGetUniformLocation(program, "cen"), screen_center.x()*2.f-1.f, screen_center.y()*-2.f+1.f);
//	glUniform2i(glGetUniformLocation(program, "res"), resolution.x(), resolution.y());
//    glUniform1i(glGetUniformLocation(program, "mode"), 1);
//
//    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
//
//    glEnableVertexAttribArray(0);
//    // glEnableVertexAttribArray(1);
//
//    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
//
//    // glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
//    // glVertexAttribIPointer(1, 1, GL_UNSIGNED_BYTE, sizeof(uint8_t), 0);
//
//    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
//
//    glDrawArrays(GL_POINTS, 0, display_samples.size());
//
//    glDisableVertexAttribArray(0);
//    // glDisableVertexAttribArray(1);
//    glBindBuffer(GL_ARRAY_BUFFER, 0);
//}
//
//NGP_NAMESPACE_END
