import warnings
warnings.filterwarnings(action='ignore')
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from botorch.utils.transforms import normalize, unnormalize, standardize
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.generation import get_best_candidates, gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions

def tell(train_X, train_Y):

    min_max = torch.stack( (train_X.min(dim=0)[0], train_X.max(dim=0)[0] ) )
    if (len(train_Y.size())==1):
        train_Y = train_Y.unsqueeze(-1)

    train_x = normalize( train_X, bounds =min_max )
    train_y= standardize(train_Y)

    gp = SingleTaskGP(train_x, train_y, 
                      #input_transform=InputStandardize(train_X.size(-1))
                      #outcome_transform=Standardize(m=1))
                     )

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    mll.to(train_X)
    fit_gpytorch_mll(mll) #L-BFGS-B to fit the parameters of gp model
    print('!!!!!! applied new botorch setting')
    resampler = StochasticSampler(sample_shape=train_X.size(-1))
    qEI = qExpectedImprovement(model=gp, best_f = train_y.max() , sampler=resampler)

    return qEI, min_max

def ask(num_ask, qEI, bounds, opt_bound ):
    from botorch.optim import optimize_acqf
    #error occur in large feature space
    batch_initial_conditions = gen_batch_initial_conditions( qEI,
                                                             bounds=opt_bound,
                                                             q=num_ask,
                                                             num_restarts=256,
                                                             raw_samples=1024,
                                                           )

    batch_candidates, batch_acq_values = gen_candidates_torch( batch_initial_conditions,
                                                  qEI,
                                                  lower_bounds=opt_bound[0],
                                                  upper_bounds=opt_bound[1],
                                                  optimizer=torch.optim.Adam,
                                                )

    candidates = get_best_candidates(batch_candidates, batch_acq_values).detach()

#    candidate, acq_value = optimize_acqf( aEI, 
#                                          bounds=opt_bound, 
#                                          q=num_ask, 
#                                          num_restarts=256, 
#                                          raw_samples=1024,
#                                        )

    return unnormalize( candidates, bounds )
