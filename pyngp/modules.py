import torch
import pyngp_bindings

def _torch_precision(tcnn_precision):
	if tcnn_precision == pyngp_bindings.Precision.Fp16:
		return torch.half
	elif tcnn_precision == pyngp_bindings.Precision.Fp32:
		return torch.float
	else:
		raise ValueError(f"Unknown precision {tcnn_precision}")

def free_temporary_memory():
	pyngp_bindings.free_temporary_memory()

class _module_function(torch.autograd.Function):
	@staticmethod
	def forward(ctx, native_tcnn_module, input, params, loss_scale):
		# If no output gradient is provided, no need to
		# automatically materialize it as torch.zeros.
		ctx.set_materialize_grads(False)

		native_ctx, output = native_tcnn_module.fwd(input, params)
		ctx.save_for_backward(input, params, output)
		ctx.native_tcnn_module = native_tcnn_module
		ctx.native_ctx = native_ctx
		ctx.loss_scale = loss_scale

		return output

	@staticmethod
	def backward(ctx, doutput):
		if doutput is None:
			return None, None, None, None

		if not doutput.is_cuda:
			print("TCNN WARNING: doutput must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			doutput = doutput.cuda()

		input, params, output = ctx.saved_tensors
		input_grad, weight_grad = _module_function_backward.apply(ctx, doutput, input, params, output)

		return None, input_grad, weight_grad, None

class _module_function_backward(torch.autograd.Function):
	@staticmethod
	def forward(ctx, ctx_fwd, doutput, input, params, output):
		ctx.ctx_fwd = ctx_fwd
		ctx.save_for_backward(input, params, doutput)
		with torch.no_grad():
			scaled_grad = doutput * ctx_fwd.loss_scale
			input_grad, weight_grad = ctx_fwd.native_tcnn_module.bwd(ctx_fwd.native_ctx, input, params, output, scaled_grad)
			input_grad = None if input_grad is None else (input_grad / ctx_fwd.loss_scale)
			weight_grad = None if weight_grad is None else (weight_grad / ctx_fwd.loss_scale)
		return input_grad, weight_grad

	@staticmethod
	def backward(ctx, dinput_grad, dweight_grad):
		# NOTE: currently support:
		#       ✓   d(dL_dinput)_d(dL_doutput)  doutput_grad
		#       ✓   d(dL_dinput)_d(params)      weight_grad
		#       ✓   d(dL_dinput)_d(input)       input_grad
		#       x   d(dL_dparam)_d(...)
		input, params, doutput = ctx.saved_tensors
		# assert dweight_grad is None, "currently do not support 2nd-order gradients from gradient of grid"
		with torch.enable_grad():
			# NOTE: preserves requires_grad info (this function is in no_grad() context by default when invoking loss.backward())
			doutput = doutput * ctx.ctx_fwd.loss_scale
		with torch.no_grad():
			doutput_grad, weight_grad, input_grad = ctx.ctx_fwd.native_tcnn_module.bwd_bwd_input(
				ctx.ctx_fwd.native_ctx,
				input,
				params,
				dinput_grad,
				doutput
			)
			# NOTE: be cautious when multiplying and dividing loss_scale
			#       doutput_grad uses dinput_grad
			#       weight_grad  uses dinput_grad * doutput
			#       input_grad   uses dinput_grad * doutput
			weight_grad = None if weight_grad is None else (weight_grad / ctx.ctx_fwd.loss_scale)
			input_grad = None if input_grad is None else (input_grad / ctx.ctx_fwd.loss_scale)

		# ctx_fwd,   doutput,      input,      params,      output
		return None, doutput_grad, input_grad, weight_grad, None

class Module(torch.nn.Module):
	def __init__(self, seed=1337):
		super(Module, self).__init__()

		print("Module init")

		self.native_tcnn_module = self._native_tcnn_module()
		self.dtype = _torch_precision(self.native_tcnn_module.param_precision())

		self.seed = seed
		initial_params = self.native_tcnn_module.initial_params(seed)
		self.params = torch.nn.Parameter(initial_params, requires_grad=True)
		self.register_parameter(name="params", param=self.params)

		self.loss_scale = 128.0 if self.native_tcnn_module.param_precision() == pyngp_bindings.Precision.Fp16 else 1.0

	def forward(self, x):
		if not x.is_cuda:
			print("TCNN WARNING: input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
			x = x.cuda()

		batch_size = x.shape[0]
		batch_size_granularity = int(pyngp_bindings.batch_size_granularity())
		padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

		x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])
		output = _module_function.apply(
			self.native_tcnn_module,
			x_padded.to(torch.float).contiguous(),
			self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
			self.loss_scale
		)
		return output[:batch_size, :self.n_output_dims]

	def __getstate__(self):
		"""Return state values to be pickled."""
		state = self.__dict__.copy()
		# Avoid pickling native objects
		del state["native_tcnn_module"]
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		# Reconstruct native entries
		self.native_tcnn_module = self._native_tcnn_module()

	def extra_repr(self):
		return f"n_input_dims={self.n_input_dims}, n_output_dims={self.n_output_dims}, seed={self.seed}, dtype={self.dtype}, hyperparams={self.native_tcnn_module.hyperparams()}"

	@property
	def n_input_dims(self):
		return self.native_tcnn_module.n_input_dims()

	# NOTE: the value returned is the padded output dim
	@property
	def n_output_dims(self):
		return self.native_tcnn_module.n_output_dims()