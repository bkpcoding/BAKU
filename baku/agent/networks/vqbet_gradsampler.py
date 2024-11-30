from opacus.grad_sample import GradSampleModule
import torch.nn as nn

class VQBeTGradWrapper(GradSampleModule):
    """
    Custom wrapper that extends GradSampleModule to preserve VQBeTHead's interface
    while maintaining parameter sharing
    """
    def __init__(self, vqbet_head: nn.Module):
        super().__init__(vqbet_head)
        
        # Copy important attributes from original module
        self._G = vqbet_head._G
        self._C = vqbet_head._C
        self._D = vqbet_head._D
        self._vqvae_model = vqbet_head._vqvae_model
        self.sequentially_select = vqbet_head.sequentially_select
        self.input_size = vqbet_head.input_size
        self.output_size = vqbet_head.output_size
        self.hidden_size = vqbet_head.hidden_size
        self.offset_loss_weight = vqbet_head.offset_loss_weight
        self.secondary_code_multiplier = vqbet_head.secondary_code_multiplier
        

        # Explicitly register the prediction networks as submodules
        if self.sequentially_select:
            self._map_to_cbet_preds_bin1 = vqbet_head._map_to_cbet_preds_bin1
            self._map_to_cbet_preds_bin2 = vqbet_head._map_to_cbet_preds_bin2
        else:
            self._map_to_cbet_preds_bin = vqbet_head._map_to_cbet_preds_bin
            
        self._map_to_cbet_preds_offset = vqbet_head._map_to_cbet_preds_offset

    def discretize(self, config, actions):
        """
        Delegate discretize to the wrapped module
        """
        return self._module.discretize(config, actions)
        
    def loss_fn(self, *args, **kwargs):
        """
        Use the original module's loss function
        """
        return self._module.loss_fn(*args, **kwargs)