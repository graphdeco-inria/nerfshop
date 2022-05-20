#pragma once

#include <json/json.hpp>

#include <tiny-cuda-nn/cpp_api.h>
#include <tiny-cuda-nn/multi_stream.h>

namespace tcnn { namespace cpp {

template <typename T>
constexpr EPrecision precision() {
	return std::is_same<T, float>::value ? EPrecision::Fp32 : EPrecision::Fp16;
}

EPrecision preferred_precision() {
	return precision<network_precision_t>();
}

uint32_t batch_size_granularity() {
	return tcnn::batch_size_granularity;
}

void free_temporary_memory() {
	tcnn::free_all_gpu_memory_arenas();
}

template <typename T>
class DifferentiableObject : public Module {
public:
	DifferentiableObject(std::shared_ptr<tcnn::DifferentiableObject<float, T, T>> model)
	: Module{precision<T>(), precision<T>()}, m_model{model}
	{}
	DifferentiableObject(tcnn::DifferentiableObject<float, T, T>* model)
	: Module{precision<T>(), precision<T>()}, m_model{model}
	{}

	void inference(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params) override {
		m_model->set_params((T*)params, (T*)params, nullptr, nullptr);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_model->input_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_model->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_model->inference_mixed_precision(synced_stream.get(1), input_matrix, output_matrix);
	}

	Context forward(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params, bool prepare_input_gradients) override {
		m_model->set_params((T*)params, (T*)params, nullptr, nullptr);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_model->input_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_model->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		return { m_model->forward(synced_stream.get(1), input_matrix, &output_matrix, false, prepare_input_gradients) };
	}

	void backward(cudaStream_t stream, const Context& ctx, uint32_t n_elements, float* dL_dinput, const void* dL_doutput, void* dL_dparams, const float* input, const void* output, const void* params) override {
		m_model->set_params((T*)params, (T*)params, (T*)params, (T*)dL_dparams);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_model->input_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_dinput_matrix(dL_dinput, m_model->input_width(), n_elements);

		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_model->padded_output_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_doutput_matrix((T*)dL_doutput, m_model->padded_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_model->backward(synced_stream.get(1), *ctx.ctx, input_matrix, output_matrix, dL_doutput_matrix, dL_dinput ? &dL_dinput_matrix : nullptr, false, dL_dparams ? EGradientMode::Overwrite : EGradientMode::Ignore);
	}

	void backward_backward_input(cudaStream_t stream, const Context& ctx, uint32_t n_elements, const float* dL_ddLdinput, const float* input, const void* dL_doutput, void* dL_dparams, void* dL_ddLdoutput, float* dL_dinput, const void* params) override {
		// from: dL_ddLdinput
		// to:   dL_ddLdoutput, dL_dparams
		m_model->set_params((T*)params, (T*)params, (T*)params, (T*)dL_dparams);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_model->input_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_ddLdinput_matrix((float*)dL_ddLdinput, m_model->input_width(), n_elements);

		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_doutput_matrix((T*)dL_doutput, m_model->padded_output_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_ddLdoutput_matrix((T*)dL_ddLdoutput, m_model->padded_output_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_dinput_matrix((float*)dL_dinput, m_model->input_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_model->backward_backward_input(synced_stream.get(1), *ctx.ctx, input_matrix, dL_ddLdinput_matrix, dL_doutput_matrix, dL_ddLdoutput ? &dL_ddLdoutput_matrix : nullptr, dL_dinput ? &dL_dinput_matrix : nullptr, false, dL_dparams ? EGradientMode::Overwrite : EGradientMode::Ignore);
	}

	uint32_t n_input_dims() const override {
		return m_model->input_width();
	}

	size_t n_params() const override {
		return m_model->n_params();
	}

	void initialize_params(size_t seed, float* params_full_precision) override {
		pcg32 rng{seed};
		m_model->initialize_params(rng, params_full_precision, nullptr, nullptr, nullptr, nullptr);
	}

	uint32_t n_output_dims() const override {
		return m_model->padded_output_width();
	}

	nlohmann::json hyperparams() const override {
		return m_model->hyperparams();
	}

	std::string name() const override {
		return m_model->name();
	}

protected:
	std::shared_ptr<tcnn::DifferentiableObject<float, T, T>> m_model;
};

template <typename T>
class DifferentiableNerfObject : public tcnn::cpp::DifferentiableObject<T> {
public:

	DifferentiableNerfObject(std::shared_ptr<ngp::NerfNetwork<T>> nerf_network)
		: DifferentiableObject<T>{nerf_network}, m_nerf_network{nerf_network}
		{}

	void inference_density(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params) {
		m_nerf_network->set_params((T*)params, (T*)params, nullptr, nullptr);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_nerf_network->input_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_nerf_network->padded_density_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_nerf_network->density(synced_stream.get(1), input_matrix, output_matrix);
	}

	Context forward_density(cudaStream_t stream, uint32_t n_elements, const float* input, void* output, void* params, bool prepare_input_gradients) {
		m_nerf_network->set_params((T*)params, (T*)params, nullptr, nullptr);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_nerf_network->input_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_nerf_network->padded_density_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		return { m_nerf_network->density_forward(synced_stream.get(1), input_matrix, &output_matrix, false, prepare_input_gradients) };
	}

	void backward_density(cudaStream_t stream, const Context& ctx, uint32_t n_elements, float* dL_dinput, const void* dL_doutput, void* dL_dparams, const float* input, const void* output, const void* params) {
		m_nerf_network->set_params((T*)params, (T*)params, (T*)params, (T*)dL_dparams);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_nerf_network->input_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_dinput_matrix(dL_dinput, m_nerf_network->input_width(), n_elements);

		GPUMatrix<T, MatrixLayout::ColumnMajor> output_matrix((T*)output, m_nerf_network->padded_density_output_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_doutput_matrix((T*)dL_doutput, m_nerf_network->padded_density_output_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_nerf_network->density_backward(synced_stream.get(1), *ctx.ctx, input_matrix, output_matrix, dL_doutput_matrix, dL_dinput ? &dL_dinput_matrix : nullptr, false, dL_dparams ? EGradientMode::Overwrite : EGradientMode::Ignore);
	}

	// NOTE: in practice, this is never called
	void backward_backward_input_density(cudaStream_t stream, const Context& ctx, uint32_t n_elements, const float* dL_ddLdinput, const float* input, const void* dL_doutput, void* dL_dparams, void* dL_ddLdoutput, float* dL_dinput, const void* params) {
		// from: dL_ddLdinput
		// to:   dL_ddLdoutput, dL_dparams
		m_nerf_network->set_params((T*)params, (T*)params, (T*)params, (T*)dL_dparams);

		GPUMatrix<float, MatrixLayout::ColumnMajor> input_matrix((float*)input, m_nerf_network->input_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_ddLdinput_matrix((float*)dL_ddLdinput, m_nerf_network->input_width(), n_elements);

		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_doutput_matrix((T*)dL_doutput, m_nerf_network->padded_density_output_width(), n_elements);
		GPUMatrix<T, MatrixLayout::ColumnMajor> dL_ddLdoutput_matrix((T*)dL_ddLdoutput, m_nerf_network->padded_density_output_width(), n_elements);
		GPUMatrix<float, MatrixLayout::ColumnMajor> dL_dinput_matrix((float*)dL_dinput, m_nerf_network->input_width(), n_elements);

		// Run on our own custom stream to ensure CUDA graph capture is possible.
		// (Significant possible speedup.)
		SyncedMultiStream synced_stream{stream, 2};
		m_nerf_network->backward_backward_input(synced_stream.get(1), *ctx.ctx, input_matrix, dL_ddLdinput_matrix, dL_doutput_matrix, dL_ddLdoutput ? &dL_ddLdoutput_matrix : nullptr, dL_dinput ? &dL_dinput_matrix : nullptr, false, dL_dparams ? EGradientMode::Overwrite : EGradientMode::Ignore);
	}

    uint32_t density_output_width() const {
        return m_nerf_network->density_output_width();
    }

    uint32_t padded_density_output_width() const {
        return m_nerf_network->padded_density_output_width();
    }

private:
	std::shared_ptr<ngp::NerfNetwork<T>> m_nerf_network;
};

}
}