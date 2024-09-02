import warnings
warnings.filterwarnings(action='ignore')
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize, standardize

def tell(train_X, train_Y):
    if (len(train_Y.size())==1):
        train_Y = train_Y.unsqueeze(-1)
    if train_X.size(0) == 1:
        min_max = torch.stack((train_X.min(), train_X.max()))
        train_X = normalize(train_X, bounds = min_max)
        print(train_Y)
    else:#train_X.size(0) !=1
        min_max = torch.stack( (train_X.min(dim=0)[0], train_X.max(dim=0)[0] ) )
        train_X = normalize( train_X, bounds = min_max )
        train_Y = standardize(train_Y)

    gp = SingleTaskGP(train_X, train_Y, 
                     )
                      #outcome_transform=Standardize(m=1))

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    mll.to(train_X)
    fit_gpytorch_mll(mll)

    #acqf = acquisition function (stochastic acquisition function)
    acqf = qExpectedImprovement(model=gp, best_f = train_Y.max() )

    return acqf, min_max

def ask(num_ask, acqf, bounds, opt_bound ):
    from botorch.optim import optimize_acqf

    candidate, acq_value = optimize_acqf( acqf, 
                                          bounds=opt_bound, 
                                          q=num_ask, 
                                          num_restarts=256, 
                                          raw_samples=1024,
                                        )
    return unnormalize( candidate, bounds )
