import numpy as np

import torch
from torch.autograd import Variable
# from pcrpo.utils.utils import *
# import cvxpy as cp


# def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
#     """
#     Perform the conjugate gradient algorithm.

#     Parameters:
#     Avp (callable): A function that returns the product of the matrix A and a vector.
#     b (torch.Tensor): The vector b.
#     nsteps (int): The number of steps to perform.
#     residual_tol (float): The residual tolerance.

#     Returns:
#     x (torch.Tensor): The result of the conjugate gradient algorithm. x=inv(H)g
#     """
#     x = torch.zeros(b.size())
#     r = b.clone()
#     p = b.clone()
#     rdotr = torch.dot(r, r)
#     for i in range(nsteps):
#         _Avp = Avp(p)
#         alpha = rdotr / torch.dot(p, _Avp)
#         x += alpha * p
#         r -= alpha * _Avp
#         new_rdotr = torch.dot(r, r)
#         betta = new_rdotr / rdotr
#         p = r + betta * p
#         rdotr = new_rdotr
#         if rdotr < residual_tol:
#             break
#     return x


# def linesearch(model,
#                f,
#                x,
#                fullstep,
#                expected_improve_rate,
#                max_backtracks=10,
#                accept_ratio=.1):
#     fval = f(True).data
#     # print("fval before", fval.item())
#     for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
#         xnew = x + stepfrac * fullstep
#         set_flat_params_to(model, xnew)
#         newfval = f(True).data
#         actual_improve = fval - newfval
#         expected_improve = expected_improve_rate * stepfrac
#         ratio = actual_improve / expected_improve
#         # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

#         if ratio.item() > accept_ratio and actual_improve.item() > 0:
#             # print("fval after", newfval.item())
#             return True, xnew
#     return False, x


# def trpo_step(model, get_loss, get_kl, max_kl, damping):
#     """
#     Perform a single step of TRPO optimization.

#     Parameters:
#     model (nn.Module): The model to optimize.
#     get_loss (callable): A function that returns the loss.
#     get_kl (callable): A function that returns the KL divergence.
#     max_kl (float): The maximum KL divergence.
#     damping (float): The damping factor    

#     Returns:
#     float: The loss after the optimization step.
#     """
    
#     loss = get_loss()
#     grads = torch.autograd.grad(loss, model.parameters())
#     loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

#     def Fvp(v):
#         '''
#         This function returns the product of the Fisher matrix and the vector v.
#         Hx = Fx + damping*x

#         Parameters:
#         v (torch.Tensor): The vector v.

#         Returns:
#         Hx (torch.Tensor): The product of the Fisher matrix and the vector v.
#         ''' 
#         kl = get_kl()
#         kl = kl.mean()

#         grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
#         flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

#         kl_v = (flat_grad_kl * Variable(v)).sum()
#         grads = torch.autograd.grad(kl_v, model.parameters())
#         flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

#         return flat_grad_grad_kl + v * damping

#     # Compute the step direction x
#     stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

#     # 1/2 x^T H x
#     shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

#     # sqrt((x^T H x) / (2 * delta))
#     lm = torch.sqrt(shs / max_kl)

#     # fullstep = sqrt(2 * delta / (x^T H x)) * x
#     fullstep = stepdir / lm[0]

#     neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
#     # print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

#     prev_params = get_flat_params_from(model)

#     # Perform a linesearch
#     success, new_params = linesearch(model, get_loss, prev_params, fullstep,
#                                      neggdotstepdir / lm[0])
#     set_flat_params_to(model, new_params)

#     return loss


def pcgrad(grads, grads2):
    g1 = grads
    g2 = grads2
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()
    if g12 < 0:
        return ((1-g12/g11)*g1+(1-g12/g22)*g2)/2
    else:
        return (g1+g2)/2

def pcgrad_v1(reward_gradient, cost_gradient, safety_violation):
    # print
    if safety_violation:
        g1 = cost_gradient
        g2 = reward_gradient
    else:
        g1 = reward_gradient
        g2 = cost_gradient

    final_gradient = g1
    if np.dot(g1, g2) <= 0:
        x = cp.Variable(reward_gradient.shape[0])
        obj = cp.Minimize(cp.norm(x - g1))
        const = [x @ g2 == 0]
        prob = cp.Problem(obj, const)
        prob.solve(solver=cp.SCS)
        final_gradient = x.value
    else:
        final_gradient = g1
    return final_gradient



