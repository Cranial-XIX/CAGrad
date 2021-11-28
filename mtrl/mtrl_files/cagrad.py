# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple

import numpy as np
import time
import torch
from omegaconf import OmegaConf

from mtrl.agent import grad_manipulation as grad_manipulation_agent
from mtrl.utils.types import ConfigType, TensorType
#from mtrl.agent.mgda import MinNormSolver
from scipy.optimize import minimize, Bounds, minimize_scalar


def _check_param_device(param: TensorType, old_param_device: Optional[int]) -> int:
    """This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        The implementation is taken from: https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/torch/nn/utils/convert_parameters.py#L57

    Args:
        param ([TensorType]): a Tensor of a parameter of a model.
        old_param_device ([int]): the device where the first parameter
            of a model is allocated.

    Returns:
        old_param_device (int): report device for the first time

    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device


def apply_vector_grad_to_parameters(
    vec: TensorType, parameters: Iterable[TensorType], accumulate: bool = False
):
    """Apply vector gradients to the parameters

    Args:
        vec (TensorType): a single vector represents the gradients of a model.
        parameters (Iterable[TensorType]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        if accumulate:
            param.grad = (
                param.grad + vec[pointer : pointer + num_param].view_as(param).data
            )
        else:
            param.grad = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


class Agent(grad_manipulation_agent.Agent):
    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        agent_cfg: ConfigType,
        multitask_cfg: ConfigType,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """Regularized gradient algorithm."""
        agent_cfg_copy = deepcopy(agent_cfg)
        del agent_cfg_copy['cagrad_c']
        del agent_cfg_copy['cagrad_method']

        OmegaConf.set_struct(agent_cfg_copy, False)
        agent_cfg_copy.cfg_to_load_model = None
        agent_cfg_copy.should_complete_init = False
        agent_cfg_copy.loss_reduction = "none"
        OmegaConf.set_struct(agent_cfg_copy, True)

        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            agent_cfg=agent_cfg_copy,
            device=device,
        )
        self.agent._compute_gradient = self._compute_gradient
        self._rng = np.random.default_rng()

        self.cagrad_c = agent_cfg['cagrad_c']
        self.cagrad_method = agent_cfg['cagrad_method']

        fn_maps = {
            "cagrad":       self.cagrad,
            "cagrad_exact": self.cagrad_exact,
        }
        for k in range(2, 50):
            fn_maps[f"cagrad_fast{k}"] = self.cagrad_fast

        fn_names = ", ".join(fn_maps.keys())
        assert self.cagrad_method in fn_maps, \
                f"[error] unrealized fn {self.cagrad_method}, currently we have {fn_names}"
        self.cagrad_fn = fn_maps[self.cagrad_method]
        self.wi_map = {}
        self.num_param_block = -1
        self.conflicts = []
        self.last_w = None
        self.save_target = 500000
        if "fast" in self.cagrad_method:
            num_tasks = multitask_cfg['num_envs']
            self.fast_n = int(self.cagrad_method[self.cagrad_method.find("fast")+4:])
            self.fast_w = torch.zeros((self.fast_n)).cuda()
            self.fast_w[:-1] = 1/num_tasks
            self.fast_w[-1] = 1 - (self.fast_n-1)/num_tasks
            self.fast_w_numpy = self.fast_w.cpu().numpy()

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:

        #t0 = time.time()
        task_loss = self._convert_loss_into_task_loss(
            loss=loss, env_metadata=env_metadata
        )
        num_tasks = task_loss.shape[0]
        grad = []

        if "fast" in self.cagrad_method:
            # 2 losses approximation
            n = self.fast_n
            idx = np.random.permutation(num_tasks)

            losses = [0] * n
            for j in range(n-1):
                losses[j] = task_loss[idx[j]]
            for j in range(n, num_tasks):
                losses[-1] += task_loss[idx[j]]
            losses[-1] /= (num_tasks - n + 1)
            for loss in losses:
                grad.append(
                    tuple(
                        _grad.contiguous()
                        for _grad in torch.autograd.grad(
                            loss,
                            parameters,
                            retain_graph=True,
                            allow_unused=allow_unused,
                        )
                    )
                )
        else:
            for index in range(num_tasks):
                grad.append(
                    tuple(
                        _grad.contiguous()
                        for _grad in torch.autograd.grad(
                            task_loss[index],
                            parameters,
                            retain_graph=(retain_graph or index != num_tasks - 1),
                            allow_unused=allow_unused,
                        )
                    )
                )

        grad_vec = torch.cat(
            list(
                map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)
            ),
            dim=0,
        )  # num_tasks x dim

        regularized_grad = self.cagrad_fn(grad_vec, num_tasks)
        apply_vector_grad_to_parameters(regularized_grad, parameters)

    def cagrad(self, grad_vec, num_tasks):
        """
        grad_vec: [num_tasks, dim]
        """
        grads = grad_vec

        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG)+1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True)
        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c = (gg+1e-4).sqrt() * self.cagrad_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward()
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww)+1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm+1e-4)
        g = ((1/num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads).sum(0) / (1 + self.cagrad_c**2)
        return g

    def cagrad_exact(self, grad_vec, num_tasks):
        grads = grad_vec / 100.
        g0 = grads.mean(0)
        GG = grads.mm(grads.t())
        x_start = np.ones(num_tasks)/num_tasks
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.cpu().numpy()
        b = x_start.copy()
        c = (self.cagrad_c*g0.norm()).cpu().item()
        def objfn(x):
            return (x.reshape(1,num_tasks).dot(A).dot(b.reshape(num_tasks, 1)) + \
                    c * np.sqrt(x.reshape(1,num_tasks).dot(A).dot(x.reshape(num_tasks,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww= torch.Tensor(w_cpu).to(grad_vec.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-4)
        g = (g0 + lmbda * gw) / (1 + lmbda)
        return g * 100

    def cagrad_fast(self, grad_vec, num_tasks):
        n = self.fast_n
        scale = 100.
        grads = grad_vec / scale
        GG = grads.mm(grads.t())
        g0_norm = (self.fast_w.view(1, -1).mm(GG).mm(self.fast_w.view(-1, 1))+1e-8).sqrt().item()

        x_start = np.ones(n) / n
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.cpu().numpy()
        c = self.cagrad_c*g0_norm
        def objfn(x):
            return (x.reshape(1,n).dot(A).dot(self.fast_w_numpy.reshape(n,1)) + \
                    c * np.sqrt(x.reshape(1,n).dot(A).dot(x.reshape(n,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww= torch.Tensor(w_cpu).to(grad_vec.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = np.sqrt(w_cpu.reshape(1,n).dot(A).dot(w_cpu.reshape(n,1))+1e-8).item()
        lmbda = c / (gw_norm+1e-4)
        g = ((self.fast_w.view(-1,1)+ww.view(-1,1)*lmbda)*grads).sum(0)
        g = g / (1 + self.cagrad_c) * scale
        return g
