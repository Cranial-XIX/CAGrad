import torch
import torch.nn.functional as F
import numpy as np
import random
import time

from copy import deepcopy
from min_norm_solvers import MinNormSolver
from scipy.optimize import minimize, Bounds, minimize_scalar

"""
Define task metrics, loss functions and model trainer here.
"""

def control_seed(seed):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss

# Legacy: compute mIoU and Acc. for each image and average across all images.

# def compute_miou(x_pred, x_output):
#     _, x_pred_label = torch.max(x_pred, dim=1)
#     x_output_label = x_output
#     batch_size = x_pred.size(0)
#     class_nb = x_pred.size(1)
#     device = x_pred.device
#     for i in range(batch_size):
#         true_class = 0
#         first_switch = True
#         invalid_mask = (x_output[i] >= 0).float()
#         for j in range(class_nb):
#             pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).long().to(device))
#             true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).long().to(device))
#             mask_comb = pred_mask.float() + true_mask.float()
#             union = torch.sum((mask_comb > 0).float() * invalid_mask)  # remove non-defined pixel predictions
#             intsec = torch.sum((mask_comb > 1).float())
#             if union == 0:
#                 continue
#             if first_switch:
#                 class_prob = intsec / union
#                 first_switch = False
#             else:
#                 class_prob = intsec / union + class_prob
#             true_class += 1
#         if i == 0:
#             batch_avg = class_prob / true_class
#         else:
#             batch_avg = class_prob / true_class + batch_avg
#     return batch_avg / batch_size
#
#
# def compute_iou(x_pred, x_output):
#     _, x_pred_label = torch.max(x_pred, dim=1)
#     x_output_label = x_output
#     batch_size = x_pred.size(0)
#     for i in range(batch_size):
#         if i == 0:
#             pixel_acc = torch.div(
#                 torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
#                 torch.sum((x_output_label[i] >= 0).float()))
#         else:
#             pixel_acc = pixel_acc + torch.div(
#                 torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
#                 torch.sum((x_output_label[i] >= 0).float()))
#     return pixel_acc / batch_size


# New mIoU and Acc. formula: accumulate every pixel and average across all pixels in all images
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu), acc


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


"""
=========== Universal Multi-task Trainer ===========
"""


def multi_task_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, total_epoch=200):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
    lambda_weight = np.ones([2, total_epoch])
    for index in range(total_epoch):
        t0 = time.time()
        cost = np.zeros(12, dtype=np.float32)

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                lambda_weight[0, index] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                lambda_weight[1, index] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth = train_depth.to(device)

            train_pred, logsigma = multi_task_model(train_data)

            optimizer.zero_grad()
            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth')]

            if opt.weight == 'equal' or opt.weight == 'dwa':
                loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(2)])
            else:
                loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(2))

            loss.backward()
            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[index, :6] += cost[:6] / train_batch

        # compute mIoU and acc
        avg_cost[index, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth = test_depth.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[index, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 7:9] = conf_mat.get_metrics()

        scheduler.step()
        t1 = time.time()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} || TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | TIME: {:.4f}'
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], t1-t0))


"""
=========== Universal Single-task Trainer ===========
"""


def single_task_trainer(train_loader, test_loader, single_task_model, device, optimizer, scheduler, opt, total_epoch=200):
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
    for index in range(total_epoch):
        cost = np.zeros(12, dtype=np.float32)

        # iteration for all batches
        single_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(single_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth = train_depth.to(device)

            train_pred = single_task_model(train_data)
            optimizer.zero_grad()

            if opt.task == 'semantic':
                train_loss = model_fit(train_pred, train_label, opt.task)
                train_loss.backward()
                optimizer.step()

                conf_mat.update(train_pred.argmax(1).flatten(), train_label.flatten())
                cost[0] = train_loss.item()

            if opt.task == 'depth':
                train_loss = model_fit(train_pred, train_depth, opt.task)
                train_loss.backward()
                optimizer.step()
                cost[3] = train_loss.item()
                cost[4], cost[5] = depth_error(train_pred, train_depth)

            avg_cost[index, :6] += cost[:6] / train_batch

        if opt.task == 'semantic':
            avg_cost[index, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        single_task_model.eval()
        conf_mat = ConfMatrix(single_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(device),  test_label.long().to(device)
                test_depth = test_depth.to(device)

                test_pred = single_task_model(test_data)

                if opt.task == 'semantic':
                    test_loss = model_fit(test_pred, test_label, opt.task)

                    conf_mat.update(test_pred.argmax(1).flatten(), test_label.flatten())
                    cost[6] = test_loss.item()

                if opt.task == 'depth':
                    test_loss = model_fit(test_pred, test_depth, opt.task)
                    cost[9] = test_loss.item()
                    cost[10], cost[11] = depth_error(test_pred, test_depth)

                avg_cost[index, 6:] += cost[6:] / test_batch
            if opt.task == 'semantic':
                avg_cost[index, 7:9] = conf_mat.get_metrics()

        scheduler.step()
        if opt.task == 'semantic':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8]))
        if opt.task == 'depth':
            print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} TEST: {:.4f} {:.4f} {:.4f}'
              .format(index, avg_cost[index, 3], avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11]))
        torch.save(single_task_model.state_dict(), f"models/single-{opt.task}-{opt.seed}.pt")


"""
=========== Universal Gradient Manipulation Multi-task Trainer ===========
"""


def multi_task_rg_trainer(train_loader, test_loader, multi_task_model, device, optimizer, scheduler, opt, total_epoch=200):
    method = opt.method
    alpha = opt.alpha

    def graddrop(grads):
        P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1)+1e-8))
        U = torch.rand_like(grads[:,0])
        M = P.gt(U).view(-1,1)*grads.gt(0) + P.lt(U).view(-1,1)*grads.lt(0)
        g = (grads * M.float()).mean(1)
        return g

    def mgd(grads):
        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element([
            grads_cpu[t] for t in range(grads.shape[-1])])
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def pcgrad(grads, rng):
        grad_vec = grads.t()
        num_tasks = 2

        shuffled_task_indices = np.zeros((num_tasks, num_tasks - 1), dtype=int)
        for i in range(num_tasks):
            task_indices = np.arange(num_tasks)
            task_indices[i] = task_indices[-1]
            shuffled_task_indices[i] = task_indices[:-1]
            rng.shuffle(shuffled_task_indices[i])
        shuffled_task_indices = shuffled_task_indices.T

        normalized_grad_vec = grad_vec / (
            grad_vec.norm(dim=1, keepdim=True) + 1e-8
        )  # num_tasks x dim
        modified_grad_vec = deepcopy(grad_vec)
        for task_indices in shuffled_task_indices:
            normalized_shuffled_grad = normalized_grad_vec[
                task_indices
            ]  # num_tasks x dim
            dot = (modified_grad_vec * normalized_shuffled_grad).sum(
                dim=1, keepdim=True
            )  # num_tasks x dim
            modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
        g = modified_grad_vec.mean(dim=0)
        return g

    def cagrad(grads, alpha=0.5, rescale=0):
        g1 = grads[:,0]
        g2 = grads[:,1]

        g11 = g1.dot(g1).item()
        g12 = g1.dot(g2).item()
        g22 = g2.dot(g2).item()

        g0_norm = 0.5 * np.sqrt(g11+g22+2*g12)

        # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
        coef = alpha * g0_norm
        def obj(x):
            # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
            # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
            return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-8) + 0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22

        res = minimize_scalar(obj, bounds=(0,1), method='bounded')
        x = res.x

        gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-8)
        lmbda = coef / (gw_norm+1e-8)
        g = (0.5+lmbda*x) * g1 + (0.5+lmbda*(1-x)) * g2 # g0 + lmbda*gw
        if rescale== 0:
            return g
        elif rescale== 1:
            return g / (1+alpha**2)
        else:
            return g / (1 + alpha)

    def grad2vec(m, grads, grad_dims, task):
        # store the gradients
        grads[:, task].fill_(0.0)
        cnt = 0
        for mm in m.shared_modules():
            for p in mm.parameters():
                grad = p.grad
                if grad is not None:
                    grad_cur = grad.data.detach().clone()
                    beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                    en = sum(grad_dims[:cnt + 1])
                    grads[beg:en, task].copy_(grad_cur.data.view(-1))
                cnt += 1

    def overwrite_grad(m, newgrad, grad_dims):
        newgrad = newgrad * 2 # to match the sum loss
        cnt = 0
        for mm in m.shared_modules():
            for param in mm.parameters():
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                this_grad = newgrad[beg: en].contiguous().view(param.data.size())
                param.grad = this_grad.data.clone()
                cnt += 1

    rng = np.random.default_rng()
    grad_dims = []
    for mm in multi_task_model.shared_modules():
        for param in mm.parameters():
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), 2).cuda()

    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 12], dtype=np.float32)
    lambda_weight = np.ones([2, total_epoch])
    for index in range(total_epoch):
        t0 = time.time()
        cost = np.zeros(12, dtype=np.float32)

        # apply Dynamic Weight Average
        if opt.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, index] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
                lambda_weight[0, index] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                lambda_weight[1, index] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

        # iteration for all batches
        multi_task_model.train()
        train_dataset = iter(train_loader)
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        for k in range(train_batch):
            train_data, train_label, train_depth = train_dataset.next()
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth = train_depth.to(device)

            train_pred, logsigma = multi_task_model(train_data)

            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth')]

            train_loss_tmp = [0,0]
            if opt.weight == 'equal' or opt.weight == 'dwa':
                for i in range(2):
                    train_loss_tmp[i] = train_loss[i] * lambda_weight[i, index]
            else:
                for i in range(2):
                    train_loss_tmp[i] = 1/(2*torch.exp(logsigma[i]))*train_loss[i]+logsigma[i]/2

            optimizer.zero_grad()
            if method == "graddrop":
                for i in range(2):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = graddrop(grads)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "pcgrad":
                for i in range(2):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = pcgrad(grads, rng)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "mgd":
                for i in range(2):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = mgd(grads)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()
            elif method == "cagrad":
                for i in range(2):
                    train_loss_tmp[i].backward(retain_graph=True)
                    grad2vec(multi_task_model, grads, grad_dims, i)
                    multi_task_model.zero_grad_shared_modules()
                g = cagrad(grads, alpha, rescale=1)
                overwrite_grad(multi_task_model, g, grad_dims)
                optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = train_loss[0].item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[index, :6] += cost[:6] / train_batch

        # compute mIoU and acc
        avg_cost[index, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth = test_depth.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[index, 6:] += cost[6:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 7:9] = conf_mat.get_metrics()

        scheduler.step()
        t1 = time.time()
        print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} || TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | TIME: {:.4f}'
            .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], t1-t0))
        torch.save(multi_task_model.state_dict(), f"models/{method}-{opt.weight}-{alpha}-{opt.seed}.pt")
