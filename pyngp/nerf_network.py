import torch
import torch.nn.functional as F
from pyngp.modules import Module, _torch_precision
import pyngp_bindings

class _density_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native_tcnn_module, input, params, loss_scale):
        # If no output gradient is provided, no need to
        # automatically materialize it as torch.zeros.
        ctx.set_materialize_grads(False)

        native_ctx, output = native_tcnn_module.fwd_density(input, params)
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
        input_grad, weight_grad = _density_function_backward.apply(ctx, doutput, input, params, output)

        return None, input_grad, weight_grad, None

class _density_function_backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ctx_fwd, doutput, input, params, output):
        ctx.ctx_fwd = ctx_fwd
        ctx.save_for_backward(input, params, doutput)
        with torch.no_grad():
            scaled_grad = doutput * ctx_fwd.loss_scale
            input_grad, weight_grad = ctx_fwd.native_tcnn_module.bwd_density(ctx_fwd.native_ctx, input, params, output, scaled_grad)
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
            doutput_grad, weight_grad, input_grad = ctx.ctx_fwd.native_tcnn_module.density_bwd_bwd_input(
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

class  NeRFNetwork(Module):
    # def __init__(self, model_config, seed=1337) -> None:
    #     self.model_config = model_config

    #     print(f"Initializing NeRF network with config: {model_config}")

    #     super(NeRFNetwork, self).__init__(seed=seed)

    def __init__(self, native_tcnn_module, native_testbed, seed=1337) -> None:
        self.native_tcnn_module = native_tcnn_module

        # set the activation functions
        if native_testbed.nerf.density_activation == pyngp_bindings.NerfActivation.ReLU:
            self.density_activation = F.relu
        elif native_testbed.nerf.density_activation == pyngp_bindings.NerfActivation.Logistic:
            self.density_activation = F.sigmoid
        elif native_testbed.nerf.density_activation == pyngp_bindings.NerfActivation.Exponential:
            self.density_activation = torch.exp
        else:
            self.density_activation = lambda x:x

        if native_testbed.nerf.rgb_activation == pyngp_bindings.NerfActivation.ReLU:
            self.rgb_activation = F.relu
        elif native_testbed.nerf.rgb_activation == pyngp_bindings.NerfActivation.Logistic:
            self.rgb_activation = F.sigmoid
        elif native_testbed.nerf.rgb_activation == pyngp_bindings.NerfActivation.Exponential:
            self.rgb_activation = lambda x: torch.exp(torch.clamp(x, -10., 10.))
        else:
            self.rgb_activation = lambda x:x


        super(NeRFNetwork, self).__init__(seed=seed)
    
    def _native_tcnn_module(self):
        return self.native_tcnn_module
        # return pyngp_bindings.create_nerf_network(self.model_config)

    # def density(self, x):
    #     if not x.is_cuda:
    #         print("TCNN WARNING: input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
    #         x = x.cuda()

    #     batch_size = x.shape[0]
    #     batch_size_granularity = int(pyngp_bindings.batch_size_granularity())
    #     padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

    #     x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])
            
    # Prepare for torch training by copying the existing parameters values in the torch Tensor
    def prepare_for_torch(self, testbed) -> None:
        testbed.prepare_for_torch(self.native_tcnn_module)

    def density(self, x, use_activation=True):
        if not x.is_cuda:
            print("TCNN WARNING: input must be a CUDA tensor, but isn't. This indicates suboptimal performance.")
            x = x.cuda()

        batch_size = x.shape[0]
        batch_size_granularity = int(pyngp_bindings.batch_size_granularity())
        padded_batch_size = (batch_size + batch_size_granularity-1) // batch_size_granularity * batch_size_granularity

        x_padded = x if batch_size == padded_batch_size else torch.nn.functional.pad(x, [0, 0, 0, padded_batch_size - batch_size])
        output = _density_function.apply(
            self.native_tcnn_module,
            x_padded.to(torch.float).contiguous(),
            self.params.to(_torch_precision(self.native_tcnn_module.param_precision())).contiguous(),
            self.loss_scale
        )

        # Don't forget to perform activation
        # TODO: pay attention because only density should be activated in practice
        
        # Don't forget to slice the output w.r.t. the real output dim and not the padded one!
        
        output = output[:batch_size, :self.native_tcnn_module.n_density_output_dims()]

        if use_activation:
            return self.density_activation(output)
        else:
            return output

    def forward(self, x):

        output = super(NeRFNetwork, self).forward(x)

        # if use_activation:
        #     output[:, 0] = self.density_activation(output[:, 0])
        #     output[:, 1:3] = self.rgb_activation(output[:, 1:3])
        
        return output

    def hello_world(self):
        print("Hello World!")