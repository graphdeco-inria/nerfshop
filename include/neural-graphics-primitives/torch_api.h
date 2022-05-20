#pragma once

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <ATen/cuda/CUDAUtils.h>

#ifdef snprintf
#undef snprintf
#endif

#include <json/json.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>

#include <tiny-cuda-nn/cpp_api.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/differentiable_object.h>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
	do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

c10::ScalarType torch_type(tcnn::cpp::EPrecision precision) {
	switch (precision) {
		case tcnn::cpp::EPrecision::Fp32: return torch::kFloat32;
		case tcnn::cpp::EPrecision::Fp16: return torch::kHalf;
		default: throw std::runtime_error{"Unknown precision tcnn->torch"};
	}
}

void* void_data_ptr(torch::Tensor& tensor) {
	switch (tensor.scalar_type()) {
		case torch::kFloat32: return tensor.data_ptr<float>();
		case torch::kHalf: return tensor.data_ptr<torch::Half>();
		default: throw std::runtime_error{"Unknown precision torch->void"};
	}
}

class Module {
public:
	Module(tcnn::cpp::Module* module) : m_module{module} {}

	std::tuple<tcnn::cpp::Context, torch::Tensor> fwd(torch::Tensor input, torch::Tensor params) {
		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(params.size(0) == n_params());

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);
		torch::Tensor output = torch::empty({ batch_size, n_output_dims() }, torch::TensorOptions().dtype(c10_output_precision()).device(torch::kCUDA));

		tcnn::cpp::Context ctx;
		if (!input.requires_grad() && !params.requires_grad()) {
			m_module->inference(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
		} else {
			ctx = m_module->forward(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params), input.requires_grad());
		}

		return { std::move(ctx), output };
	}

	std::tuple<torch::Tensor, torch::Tensor> bwd(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput) {
		if (!ctx.ctx) {
			throw std::runtime_error{"Module::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
		}

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(output.scalar_type() == c10_output_precision());
		CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(output.size(1) == n_output_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(output.size(0) == input.size(0));
		CHECK_THROW(dL_doutput.size(0) == input.size(0));

		uint32_t batch_size = input.size(0);

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::empty({ batch_size, input.size(1) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
		}

		torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			dL_dparams = torch::empty({ n_params() }, torch::TensorOptions().dtype(c10_param_precision()).device(torch::kCUDA));
		}

		if (input.requires_grad() || params.requires_grad()) {
			m_module->backward(
				stream,
				ctx,
				batch_size,
				input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
				void_data_ptr(dL_doutput),
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				input.data_ptr<float>(),
				void_data_ptr(output),
				void_data_ptr(params)
			);
		}

		return { dL_dinput, dL_dparams };
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bwd_bwd_input(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput) {
		// from: dL_ddLdinput
		// to:   dL_ddLdoutput, dL_dparams, dL_dinput

		if (!ctx.ctx) {
			throw std::runtime_error{"Module::bwd_bwd_input: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
		}

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(dL_ddLdinput.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(dL_doutput.size(1) == n_output_dims());
		CHECK_THROW(dL_ddLdinput.size(1) == n_input_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(dL_doutput.size(0) == input.size(0));
		CHECK_THROW(dL_ddLdinput.size(0) == input.size(0));

		uint32_t batch_size = input.size(0);

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		torch::Tensor dL_ddLdoutput;
		if (dL_doutput.requires_grad()) {
			dL_ddLdoutput = torch::zeros({ batch_size, n_output_dims() }, torch::TensorOptions().dtype(c10_output_precision()).device(torch::kCUDA));
		}

		torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			dL_dparams = torch::zeros({ n_params() }, torch::TensorOptions().dtype(c10_param_precision()).device(torch::kCUDA));
		}

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::zeros({ batch_size, n_input_dims() }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
		}

		if (dL_doutput.requires_grad() || params.requires_grad()) {
			m_module->backward_backward_input(
				stream,
				ctx,
				batch_size,
				dL_ddLdinput.data_ptr<float>(),
				input.data_ptr<float>(),
				(params.requires_grad() || input.requires_grad() ) ? void_data_ptr(dL_doutput) : nullptr,
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				dL_doutput.requires_grad() ? void_data_ptr(dL_ddLdoutput) : nullptr,
				input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
				void_data_ptr(params)
			);
		}

		return {dL_ddLdoutput, dL_dparams, dL_dinput};
	}

	torch::Tensor initial_params(size_t seed) {
		torch::Tensor output = torch::zeros({ n_params() }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        // NOTE: we do not initialize parameters here as they are supposed to be initialized by Testbed and later copied to the torch tensor
		// m_module->initialize_params(seed, output.data_ptr<float>());
        // Copy the address of the torch params (to allow copies further away)
        m_torch_params = output.data_ptr<float>();
		return output;
	}

    void reset_params() {
        // Toy initialization just to set memory correctly (especially in Networks)
        m_module->initialize_params(0, m_torch_params);
    }

    // Return torch tensor params
    float* params() {
		return m_torch_params;
	}

	uint32_t n_input_dims() const {
		return m_module->n_input_dims();
	}

	uint32_t n_params() const {
		return (uint32_t)m_module->n_params();
	}

	tcnn::cpp::EPrecision param_precision() const {
		return m_module->param_precision();
	}

	c10::ScalarType c10_param_precision() const {
		return torch_type(param_precision());
	}

	uint32_t n_output_dims() const {
		return m_module->n_output_dims();
	}

	tcnn::cpp::EPrecision output_precision() const {
		return m_module->output_precision();
	}

	c10::ScalarType c10_output_precision() const {
		return torch_type(output_precision());
	}

	nlohmann::json hyperparams() const {
		return m_module->hyperparams();
	}

	std::string name() const {
		return m_module->name();
	}

private:
	std::shared_ptr<tcnn::cpp::Module> m_module;
    float* m_torch_params;
};

template<typename T>
class NerfNetworkModule : public Module {
public:
	NerfNetworkModule(tcnn::cpp::DifferentiableNerfObject<T>* differentiable_nerf_object) : Module{differentiable_nerf_object}, m_differentiable_nerf_object{differentiable_nerf_object} {}

	std::tuple<tcnn::cpp::Context, torch::Tensor> fwd_density(torch::Tensor input, torch::Tensor params) {
		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(params.size(0) == n_params());

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		uint32_t batch_size = input.size(0);
		torch::Tensor output = torch::empty({ batch_size, n_padded_density_output_dims() }, torch::TensorOptions().dtype(c10_output_precision()).device(torch::kCUDA));

		tcnn::cpp::Context ctx;
		if (!input.requires_grad() && !params.requires_grad()) {
			m_differentiable_nerf_object->inference_density(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
		} else {
			ctx = m_differentiable_nerf_object->forward_density(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params), input.requires_grad());
		}

		return { std::move(ctx), output };
	}

	std::tuple<torch::Tensor, torch::Tensor> bwd_density(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput) {
		if (!ctx.ctx) {
			throw std::runtime_error{"Module::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
		}

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(output.scalar_type() == c10_output_precision());
		CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(output.size(1) == n_padded_density_output_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(output.size(0) == input.size(0));
		CHECK_THROW(dL_doutput.size(0) == input.size(0));

		uint32_t batch_size = input.size(0);

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::empty({ batch_size, input.size(1) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
		}

		torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			// TODO: make sure that we don't need n_density_params
			dL_dparams = torch::empty({ n_params() }, torch::TensorOptions().dtype(c10_param_precision()).device(torch::kCUDA));
		}

		if (input.requires_grad() || params.requires_grad()) {
			m_differentiable_nerf_object->backward_density(
				stream,
				ctx,
				batch_size,
				input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
				void_data_ptr(dL_doutput),
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				input.data_ptr<float>(),
				void_data_ptr(output),
				void_data_ptr(params)
			);
		}

		return { dL_dinput, dL_dparams };
	}

	// TODO: This does not work!!!!!!
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bwd_bwd_input_density(const tcnn::cpp::Context& ctx, torch::Tensor input, torch::Tensor params, torch::Tensor dL_ddLdinput, torch::Tensor dL_doutput) {
		// from: dL_ddLdinput
		// to:   dL_ddLdoutput, dL_dparams, dL_dinput

		if (!ctx.ctx) {
			throw std::runtime_error{"Module::bwd_bwd_input: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
		}

		// Types
		CHECK_THROW(input.scalar_type() == torch::kFloat32);
		CHECK_THROW(dL_ddLdinput.scalar_type() == torch::kFloat32);
		CHECK_THROW(params.scalar_type() == c10_param_precision());
		CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());

		// Sizes
		CHECK_THROW(input.size(1) == n_input_dims());
		CHECK_THROW(dL_doutput.size(1) == n_padded_density_output_dims());
		CHECK_THROW(dL_ddLdinput.size(1) == n_input_dims());
		CHECK_THROW(params.size(0) == n_params());
		CHECK_THROW(dL_doutput.size(0) == input.size(0));
		CHECK_THROW(dL_ddLdinput.size(0) == input.size(0));

		uint32_t batch_size = input.size(0);

		cudaStream_t stream = at::cuda::getCurrentCUDAStream();

		torch::Tensor dL_ddLdoutput;
		if (dL_doutput.requires_grad()) {
			dL_ddLdoutput = torch::zeros({ batch_size, n_padded_density_output_dims() }, torch::TensorOptions().dtype(c10_output_precision()).device(torch::kCUDA));
		}

		torch::Tensor dL_dparams;
		if (params.requires_grad()) {
			dL_dparams = torch::zeros({ n_params() }, torch::TensorOptions().dtype(c10_param_precision()).device(torch::kCUDA));
		}

		torch::Tensor dL_dinput;
		if (input.requires_grad()) {
			dL_dinput = torch::zeros({ batch_size, n_input_dims() }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
		}

		if (dL_doutput.requires_grad() || params.requires_grad()) {
			m_differentiable_nerf_object->backward_backward_input_density(
				stream,
				ctx,
				batch_size,
				dL_ddLdinput.data_ptr<float>(),
				input.data_ptr<float>(),
				(params.requires_grad() || input.requires_grad() ) ? void_data_ptr(dL_doutput) : nullptr,
				params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
				dL_doutput.requires_grad() ? void_data_ptr(dL_ddLdoutput) : nullptr,
				input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
				void_data_ptr(params)
			);
		}

		return {dL_ddLdoutput, dL_dparams, dL_dinput};
	}

	uint32_t n_padded_density_output_dims() const {
		return m_differentiable_nerf_object->padded_density_output_width();
	}

	uint32_t n_density_output_dims() const {
		return m_differentiable_nerf_object->density_output_width();
	}
private:
	std::shared_ptr<tcnn::cpp::DifferentiableNerfObject<T>> m_differentiable_nerf_object;
};