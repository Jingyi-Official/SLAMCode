import torch

def chomp_cost_errors(sdf_gt, sdf_pred, epsilons = [1., 1.5, 2.]):
    return [chomp_cost_error(sdf_gt, sdf_pred, epsilon).mean().item() for epsilon in epsilons]


def chomp_cost_error(sdf_gt, sdf_pred, epsilon):
    return torch.abs(chomp_cost(sdf_pred, epsilon=epsilon) - chomp_cost(sdf_gt, epsilon=epsilon))



def chomp_cost(sdf, epsilon=2.0):
    """ CHOMP collision cost.
        equation 21 - https://www.ri.cmu.edu/pub_files/2013/5/CHOMP_IJRR.pdf
        Input is sdf samples along the trajectory to be evaluated.
    """
    cost = - sdf + epsilon / 2.
    cost[sdf > 0] = 1 / (2 * epsilon) * (sdf[sdf > 0] - epsilon)**2
    cost[sdf > epsilon] = 0.

    return cost