import math
import torch
from torch.optim.optimizer import Optimizer

class AdaBelief(Optimizer):
    """Implements AdaBelief algorithm[Ada2020]. Modified from Adam in PyTorch
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay (L2) (default: 0)
    [Ada2020] AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients
               NeurIPS 2020 Spotlight
    """

    def __init__(self, params, lr = 1e-3, eps = 1e-8, weight_decay = 0):
        defaults = dict(lr = lr, betas = (0.9, 0.999), eps = eps, weight_decay = weight_decay)
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def reset(self):
        """
        Reset parameters in param_groups.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

    def step(self, fun = None):
        """
        Main interface for AdaBelief.

        Args:
            fun: Loss function.

        Returns:
            loss: Computed loss function.
        """
        loss = fun() if (fun != None) else None
        for group in self.param_groups:
            for p in group['params']:
                if (p.grad == None):
                    continue
                grad = p.grad.data
                
                state = self.state[p]
                beta1, beta2 = group['betas']

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(1 - beta2, grad_residual, grad_residual)

                denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss