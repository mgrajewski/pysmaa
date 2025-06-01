"""
@authors: Luis Hasenauer, FH Aachen University of Applied Sciences
Matthias Grajewski, FH Aachen University of Applied Sciences
Janis Papewalis, FH Aachen University of Applied Sciences

The concept of MCDA methods using distributions instead of fixed numbers for the performances and weights goes back to
Tervonen and Lahdelma who moreover proposed Monte Carlo methods for solving the associated integrals (as we do as well
using hopsy).
Tervonen, T. and Lahdelma, R: Implementing Stochastic Multicriteria Acceptability Analysis, European Journal of
Operational Research 178 (2007), p. 500-513; Elsevier

These authors coined the notion "SMAA (Stochastic Multicriteria Acceptability Analysis)" which we adopted here. This
explains the name "pysmaa" of this python package.
"""
import numpy as np
import hopsy
import matplotlib.pyplot as plt
import warnings
from numpy.typing import NDArray
from typing import Union
from src.priors.Prior import Prior
from src.priors.perfmat.UncertainPerfMatPrior import UncertainPerfMatPrior
from src.priors.perfmat.FixedPerfMatPrior import FixedPerfMatPrior
from src.mcda_methods.SawSum import SawSum
from src.priors.weights.UniformWeightsPrior import UniformWeightsPrior
from src.priors.weights.NormalWeightsPrior import NormalWeightsPrior
from src.likelihoods.Likelihood import Likelihood
from src.likelihoods.LikelihoodQISMAA import LikelihoodQISMAA
from src.weights_from_rankings import cheb_center_from_ranking
from src.weights_from_rankings import hrep_from_ranking


class BayesModel:
    """
    A class combining a prior and a likelihood function to obtain the posterior distribution.
    """

    def __init__(self, prior: Prior, likelihood: Union[Likelihood, LikelihoodQISMAA]):
        """
        Parameters
        ----------
        prior : Prior
            The prior distribution
        likelihood : Likelihood
            The likelihood function
        """
        self.prior = prior
        self.likelihood = likelihood

    def log_density(self, theta: NDArray[float]) -> float:
        """
        Computes the logarithm of the (non-normalized) posterior density.

        Parameters
        ----------
        theta : 1D-numpy-array
            the parameter vector of form (w, flat(perfmat))

        Returns
        -------
        float
            The logarithm of the (non-normalized) posterior density
        """
        return self.prior.log_density(theta) + self.likelihood.log_density(theta)


def sample_posterior(prior: Prior, likelihood: Union[Likelihood, LikelihoodQISMAA], n_samples: int = 10_000,
                     n_chains: int = 4, n_procs: int = 4, seed_for_rng: None|int=None) -> [np.ndarray, np.ndarray]:
    """
    This function uses MCMC to generate samples of the posterior distribution given prior and likelihood.

    Parameters
    ----------
    prior : Prior
        the prior distribution
    likelihood: Likelihood
        the likelihood function
    n_samples : int
        number of samples (default=10.000). The actual number of samples is enlarged until the effective sampling size
        is n_samples.
    n_chains : int
        number of Markov chains (default=4)
    n_procs : int
        number of processes (default=4)
    seed_for_rng : int
        seed for the random number generators. FOr hopsy, we use seed_for_rng as entropy to create a high-quality
        sequence of seeds using np.random.SeedSequence(seed_for_rng).
    Returns
    -------
    samples : list of two numpy-array arrays. samples[0] refers to the weights, samples[1] to the entries of the
        performance matrix.
        In case of a fixed performance matrix, the array containing the samples of the performance matrix has size 0.
    """

    # maximal number of sampling passes to reach the desired effective sample size
    nmaxtrials = 20

    # It does not make sense to create many Markov chains if only a few samples are required. We adjust the number of
    # chains such that at least 100 samples per chain are used. For fewer samples, the computation of rhat can be
    # unreliable. However, computing rhat requires at least two chains as does hopsy.
    n_chains = max(2, min(n_chains, int(n_samples/100)))

    # in general settings: skip the first 100 samples per chain
    n_skip_starting_samples_per_chain = 100

    # We create scale_factor more samples than actually needed. This can make sense, as the effective sample size, which
    # counts, is smaller than the real sample size, such that the chance is increased to match the effective sample size
    # in the first pass.
    scale_factor = 1.0

    # initialisation for silencing IDE warnings
    acc_rate = -1.0

    # It does not make sense to use more processes than chains. However, it may make sense to use more chains than
    # processes.
    n_procs = min(n_chains, n_procs)

    perfmat_mean = prior.perfmat_prior.perfmat_mean

    # default proposal
    proposal = hopsy.AdaptiveMetropolisProposal

    # per default, we apply thinning
    apply_thinning = True

    # find out if we use ISMAA or QISMAA (qualitative or quantitative inverse SMAA)
    is_qismaa = isinstance(likelihood, LikelihoodQISMAA)

    # performance matrix is fixed: ignore the performance matrix entirely
    if isinstance(prior.perfmat_prior, FixedPerfMatPrior):

        # for qualitative inverse SMAA, there is a polytope associated to the given ranking, which we use as bound here.
        if not is_qismaa and likelihood.ranking.shape[0] > 0:
            # hrep of W_r
            amat, b = hrep_from_ranking(likelihood.mcda_method.p_from_perfmat(perfmat_mean), likelihood.ranking)

        # For quantitative inverse SMAA, there are no rankings and therefore no associated polytopes. Therefore, we
        # employ the whole simplex. If the ranking provided is empty, it does not make sense to compute the associated
        # polytope either.
        else:
            amat = -np.eye(likelihood.n_crits)
            b = np.zeros(likelihood.n_crits)

        model = prior.weights_prior
        # performance matrix perfmat should not be sampled since it is a constant
        starting_point_perfmat = np.array([])

        # In any case, the sum of weights should be 1
        a_eq = np.ones((1, likelihood.n_crits), dtype=float)
        b_eq = np.array([1.0])

        # uniform or normal prior distribution of the weights: use tailored proposals
        if isinstance(prior.weights_prior, UniformWeightsPrior):
            proposal = hopsy.UniformCoordinateHitAndRunProposal
            apply_thinning = True

            # We do not skip any samples, as experiments indicate that this does not provide any advantage in this case.
            n_skip_starting_samples_per_chain = 0

        elif isinstance(prior.weights_prior, NormalWeightsPrior):
            proposal = hopsy.GaussianCoordinateHitAndRunProposal

        # full dimension of the problem
        n_dim = likelihood.n_crits

        prior_sample_perfmat = perfmat_mean

    # performance matrix is (partially) uncertain
    else:
        # get hrep of standard simplex
        amat = -np.eye(likelihood.n_crits)
        amat = np.concatenate((amat, np.zeros((amat.shape[0], likelihood.n_crits * likelihood.n_alts))), axis=1)
        b = np.zeros(likelihood.n_crits)

        # full dimension of the problem
        n_dim = (likelihood.n_alts + 1)*likelihood.n_crits

        # posterior to sample from
        model = BayesModel(prior, likelihood)

        n_samples_aux = 100
        samples_perfmat_aux = prior.perfmat_prior.sample(n_samples=n_samples_aux, seed_for_rng=seed_for_rng)

        # starting point for perfmat: sample mean of the prior
        prior_sample_perfmat = np.sum(samples_perfmat_aux, axis=0) / n_samples_aux
        starting_point_perfmat = prior_sample_perfmat.reshape(-1)

        # sum of weights should be 1; add this equality constraint
        a_eq = np.zeros((1, likelihood.n_crits * likelihood.n_alts + likelihood.n_crits))
        a_eq[0][:likelihood.n_crits] = 1.0
        b_eq = np.array([1.0])

        # set fixed entries of the performance matrix as equality constraints in the polytope
        if np.any(prior.perfmat_prior.perfmat_types == 0):
            entries = np.arange(0, likelihood.n_crits * likelihood.n_alts)
            for entry in entries[prior.perfmat_prior.perfmat_types.flatten() == 0]:
                lhs = np.zeros((1, a_eq.shape[1]))
                lhs[0][entry + likelihood.n_crits] = 1.0
                a_eq = np.concatenate((a_eq, lhs), axis=0)
                rhs = np.array([prior.perfmat_prior.perfmat_mean.flatten()[entry]])
                b_eq = np.concatenate((b_eq, rhs))
        # set uniformly distributed entries of the performance matrix by adding inequality constraints in the polytope
        if np.any(prior.perfmat_prior.perfmat_types == 1):
            entries = np.arange(0, likelihood.n_crits * likelihood.n_alts)
            for entry in entries[prior.perfmat_prior.perfmat_types.flatten() == 1]:
                lhs = np.zeros((2, a_eq.shape[1]))
                lhs[0][entry + likelihood.n_crits] = -1.0
                lhs[1][entry + likelihood.n_crits] = 1.0
                amat = np.concatenate((amat, lhs), axis=0)

                if prior.perfmat_prior.perfmat_params.flatten()[entry] > 0:
                    lower_bound = prior.perfmat_prior.perfmat_mean.flatten()[entry] * (
                            1 - prior.perfmat_prior.perfmat_params.flatten()[entry])
                    upper_bound = prior.perfmat_prior.perfmat_mean.flatten()[entry] * (
                            1 + prior.perfmat_prior.perfmat_params.flatten()[entry])
                else:
                    lower_bound = max(0.0, prior.perfmat_prior.perfmat_mean.flatten()[entry] +
                                      prior.perfmat_prior.perfmat_params.flatten()[entry])
                    upper_bound = (prior.perfmat_prior.perfmat_mean.flatten()[entry] -
                                   prior.perfmat_prior.perfmat_params.flatten()[entry])

                rhs = np.array([lower_bound, upper_bound])
                b = np.concatenate((b, rhs))

        # Even normally distributed entries must be non-negative. Here, we add the corresponding inequalities. Moreover,
        # the polytope must be bounded in any direction. Therefore, we add artificial 4-sigma upper bounds. The chance
        # for the prior to exceed this bound is 6.5E-5.
        if np.any(prior.perfmat_prior.perfmat_types == 2):
            entries = np.arange(0, likelihood.n_crits * likelihood.n_alts)
            for entry in entries[prior.perfmat_prior.perfmat_types.flatten() == 2]:
                lhs = np.zeros((2, a_eq.shape[1]))
                lhs[0, entry + likelihood.n_crits] = -1.0
                lhs[1, entry + likelihood.n_crits] = 1.0
                amat = np.concatenate((amat, lhs), axis=0)
                b = np.concatenate((b, np.array([0, prior.perfmat_prior.perfmat_mean.flatten()[entry] +
                                                4.0*prior.perfmat_prior.perfmat_params.flatten()[entry]])))

        # change proposal for special cases: Uniform proposal, if there are uncertain entries, and they are all
        # uniformly distributed
        if (np.any(prior.perfmat_prior.perfmat_types == 1) and
                np.all(np.logical_or(prior.perfmat_prior.perfmat_types == 0, prior.perfmat_prior.perfmat_types == 1))):
            proposal = hopsy.UniformCoordinateHitAndRunProposal
            apply_thinning = True

        # Gaussian proposal, if there are uncertain entries, and they are all normally distributed
        # there are uncertain entries of the performance matrix, and these are all normally distributed
        elif (np.any(prior.perfmat_prior.perfmat_types == 2) and
              np.all(np.logical_or(prior.perfmat_prior.perfmat_types == 0, prior.perfmat_prior.perfmat_types == 2))):
            proposal = hopsy.GaussianCoordinateHitAndRunProposal

    # enlarge the number by the number of early samples to skip
    n_samples_per_chain = int(scale_factor*(np.ceil(n_samples / n_chains).astype(int) +
                                            n_skip_starting_samples_per_chain))
    n_samples_per_chain_total = 0

    # starting point for MCMC algorithm: The entries of the performance matrix are chosen by taking the values of
    # a perfmat sampled by the prior.
    p_mean = likelihood.mcda_method.p_from_perfmat(prior_sample_perfmat)

    # In case of ISMAA: compute the Chebychev center of the set of weights leading to the given ranking, if the
    # ranking is not empty
    cheb_center = np.array([])
    if not is_qismaa:
        if likelihood.ranking.shape[0] > 0:
            cheb_center = cheb_center_from_ranking(p_mean, likelihood.ranking)
        ranking = likelihood.ranking

    else:
        if likelihood.probs_of_alts.shape[0] > 0:
            ranking_aux = np.array([np.argmax(likelihood.probs_of_alts, axis=0)])
            cheb_center = cheb_center_from_ranking(p_mean, ranking_aux)
            ranking = ranking_aux

    # Computing the Chebyshev center failed: Use the mean of the simplex as fallback
    if cheb_center.shape[0] == 0:
        cheb_center = np.ones(likelihood.n_crits) / likelihood.n_crits

    # The starting values for the weights are chosen as Chebychev-center of the polytope with respect to a given
    # ranking, if the prior distribution of the weights is not normal. This is not necessarily a good choice for
    # normally distributed prior weights, as it may happen that at the Chebychev center, the probability density is
    # almost zero.
    # Just taking the mean of normal prior distribution of weights is not good either, as it may be outside W_r.
    # Therefore, we draw the line from the Chebychev center to the mean and search along this line using bisection.

    # default and fallback: starting point is the Chebychev center of W_r, if we deal with rankings
    starting_point_weights = cheb_center
    if isinstance(prior.weights_prior, NormalWeightsPrior):
        par_min = 0
        par_max = 1.0
        for i in range(5):
            par = 0.5 * (par_min + par_max)
            starting_candidate = (1.0-par)*cheb_center + par*prior.weights_prior.mean

            ranking_aux = np.argsort(-likelihood.mcda_method.p_from_perfmat(perfmat_mean)@starting_candidate)

            # point is still in W_r
            if np.array_equal(ranking_aux[0:ranking.shape[0]], ranking):
                starting_point_weights = starting_candidate
                par_min = par
            else:
                par_max = par

    # add starting point for perfmat
    starting_point = np.concatenate((starting_point_weights, starting_point_perfmat))

    # set up problem
    problem = hopsy.Problem(amat, b, model=model)

    # Add equality constraints. This includes dimension reduction and rounding, hence the (affine) transformation.
    problem_new = hopsy.add_equality_constraints(problem, a_eq, b_eq)

    # transform the starting point to the reduced and rounded polytope
    starting_point_new = problem_new.transformation.T @ (starting_point - problem_new.shift)

    # get effective dimension of the problem. Thinning is set according to
    # Jadebeck JF, Wiechert W, Nöh K (2023) Practical sampling of constraint-based models: Optimized thinning boosts
    # CHRR performance. PLoS Comput Biol 19(8): e1011378. https://doi.org/10.1371/journal.pcbi.1011378
    ndim_eff = problem_new.A.shape[1]

    # compute thinning as 2*ndim_eff, but at least 5
    if apply_thinning:
        # thinning = np.maximum(5, np.ceil(1/6.0*ndim_eff*ndim_eff)).astype(int)
        thinning = np.maximum(5, 2*ndim_eff).astype(int)
    else:
        thinning = 2

    # set up chains and sample
    try:
        chains = [hopsy.MarkovChain(problem_new, starting_point=starting_point_new, proposal=proposal) for _ in
                  range(n_chains)]

        if seed_for_rng == None:
            seedsSequence = np.random.SeedSequence()
        else:
            seedsSequence = np.random.SeedSequence(seed_for_rng)

        seeds = seedsSequence.generate_state(n_chains)
        rngs = [hopsy.RandomNumberGenerator(seeds[i]) for i in range(n_chains)]

        bad_ess = True
        itry = 1
        samples = np.zeros((n_chains, 0, n_dim))

        more_samples = False

        while bad_ess and itry <= nmaxtrials:

            n_samples_per_chain_total = n_samples_per_chain_total + n_samples_per_chain
            acc_rate, samples_new = hopsy.sample(chains, rngs, n_samples=n_samples_per_chain,
                                                 thinning=thinning, n_procs=n_procs)

            if itry > 1:
                more_samples = True
            else:
                # skip first samples
                samples_new = samples_new[:, n_skip_starting_samples_per_chain:, :]
                n_samples_per_chain_total = n_samples_per_chain_total - n_skip_starting_samples_per_chain
                n_samples_per_chain = n_samples_per_chain - n_skip_starting_samples_per_chain
            samples = np.append(samples, samples_new, axis=1)

            ess = hopsy.ess(samples, relative=False)
            if np.all(ess[0] >= n_samples):
                bad_ess = False
            else:
                itry += 1

        # Test, if the chains are properly converged.
        rhat = hopsy.rhat(samples)
        if np.any(rhat > 1.01):
            dimensions = np.arange(0, samples.shape[2])
            warnings.warn(f"Rhat score > 1.01 detected in dimensions {dimensions[rhat[0] > 1.01]}"
                          f"; Rhat = {rhat[rhat >1.01]}")
        ess = hopsy.ess(samples, relative=False)
        if np.any(ess[0] < n_samples):
            dimensions = np.arange(0, samples.shape[2])
            warnings.warn(f"Low effective sample size of {(ess[0][ess[0] < n_samples]).astype(int)} "
                          f"of components {dimensions[ess[0] < n_samples]}.")

    except ValueError as e:
        print(e)

        # if sampling failed, return a list of empty numpy-arrays
        return [np.array([]), np.array([])], 0, np.array([]), np.array([])

    # merge Markov-chains
    samples = samples.reshape((n_chains * n_samples_per_chain_total, -1))

    # first pass successful: return n_samples samples. To do so, shorten the array, which is maybe a little larger, to
    # n_samples.
    if not more_samples:
        samples = samples[0:min(n_samples, n_chains * n_samples_per_chain_total), :]

    # number of samples to return
    n_samples = samples.shape[0]

    if isinstance(prior.perfmat_prior, FixedPerfMatPrior):
        return [samples, np.array([])], itry, ess, acc_rate
    else:
        samples = [samples[:, 0:likelihood.n_crits], samples[:, likelihood.n_crits:].reshape(
            (n_samples, likelihood.n_alts, likelihood.n_crits))]
        return samples, itry, ess, acc_rate


def main():

    share_vector = [10, 10, 80]

    run_sampling(share_vector, 10)


def run_sampling(share_vector, number_of_samples):
    perfmat_mean = np.array([[2 / 3, 0], [0, 2 / 3], [1 / 3, 1 / 3]])
    mu = np.array([0.5, 0.5])
    sigma = 0.1 * np.eye(2)
    shares = np.array(share_vector) / 100

    # First test: fixed performance matrix
    perfmat_types = np.full(perfmat_mean.shape, 1)
    perfmat_params = np.full(perfmat_types.shape, -0.2)

    # We consider the matrix uncertain such it is incorporated in the polytope. However, all entries are fixed here.
    # This results in equality constraints which lead to dimension reduction in hopsy. Therefore, the effective
    # dimension is 3 here.
    prior_perfmat = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)
    prior_weights = NormalWeightsPrior(mu, sigma)

    prior = Prior(prior_weights, prior_perfmat)
    benefit = np.ones(perfmat_mean.shape[1], dtype=bool)
    mcda_method = SawSum(benefit)
    likelihood = LikelihoodQISMAA(perfmat_mean.shape[0], perfmat_mean.shape[1], shares, mcda_method, 20, 0.95, 2, 2)
    samples = sample_posterior(prior, likelihood, n_chains=1, n_samples=number_of_samples, seed_for_rng=1)[0]
    print(samples[1])

    # Create share string for filename
    share_string = "-".join(map(str, share_vector))
    #save_samples_to_csv(samples, share_string)


def save_samples_to_csv(samples, share_string):
    # Save the samples to two CSV files:
    #    - weights.csv: 2000x2 array of weights
    #    - perfmat.csv: 2000 rows of flattened 3x2 matrices

    # Save weights (samples[0]) - 2000x2 array
    np.savetxt(f'weights-{share_string}.csv', samples[0], delimiter=';')

    # Save performance matrices (samples[1]) - reshape from (2000,3,2) to (2000,6)
    perfmat_reshaped = samples[1].reshape(samples[1].shape[0], -1) * 10**-18
    np.savetxt(f'perfmat-{share_string}.csv', perfmat_reshaped, delimiter=';')

    print(f"Files saved for share vector {share_string}")


def plot_samples_3d(samples):
    vertices = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])

    fig_fixed_p = plt.figure()
    ax = fig_fixed_p.add_subplot(projection='3d')
    ax.view_init(45, 45)
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
               depthshade=False, alpha=.8, s=1)
    ax.plot3D(vertices[0, :], vertices[1, :], vertices[2, :], c='black')
    fig_fixed_p.show()


if __name__ == '__main__':
    main()
