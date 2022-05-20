import code
from tqdm import tqdm

import pyngp_bindings
from pyngp import NeRFNetwork

class Testbed:
    def __init__(self, scene_dir, network_config, nerf_compatibility=False, batch_size=1<<18, n_training_steps=16) -> None:
        
        self.native_testbed = pyngp_bindings.Testbed(pyngp_bindings.TestbedMode.Nerf)
        
        self.native_testbed.load_training_data(scene_dir)
        
        # TODO: add reload with snapshots
        self.native_testbed.reload_network_from_file(network_config)

        if nerf_compatibility:
            # Prior nerf papers accumulate/blend in the sRGB
            # color space. This messes not only with background
            # alpha, but also with DOF effects and the likes.
            # We support this behavior, but we only enable it
            # for the case of synthetic nerf data where we need
            # to compare PSNR numbers to results of prior work.
            self.native_testbed.color_space = pyngp_bindings.ColorSpace.SRGB

            # No exponential cone tracing. Slightly increases
            # quality at the cost of speed. This is done by
            # default on scenes with AABB 1 (like the synthetic
            # ones), but not on larger scenes. So force the
            # setting here.
            self.native_testbed.nerf.cone_angle_constant = 0

            # Optionally match nerf paper behaviour and train on a
            # fixed white bg. We prefer training on random BG colors.
            # testbed.background_color = [1.0, 1.0, 1.0, 1.0]
            # testbed.nerf.training.random_bg_color = False

        self.batch_size = batch_size
        self.n_training_steps = n_training_steps

        self.nerf_network = NeRFNetwork(self.native_testbed.get_nerf_network(), self.native_testbed)

    def prepare_for_torch(self, native_tcnn_module) -> None:
        self.native_testbed.prepare_for_torch(native_tcnn_module)

    def train(self, n_steps=-1):
        
        i = 0
        if (n_steps < 0):
            n_steps = 10000
        
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while (i < n_steps):
                if i == 0:
                    t.reset()
                self.native_testbed.train(self.n_training_steps, self.batch_size)

                t.update(self.n_training_steps)
                t.set_postfix(loss= self.native_testbed.loss)
                i += self.n_training_steps
