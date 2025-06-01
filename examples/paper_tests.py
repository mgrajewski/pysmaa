"""
@author: Matthias Grajewski, FH Aachen University of Applied Sciences

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import sys
from os import path

if __name__ == '__main__':
    import json as js

    path_to_pysmaa = "..\\"

    # if a path to pysmaa is provided, read it and append if, if not
    # already present in sys.path
    if path_to_pysmaa != '':
        if not any(path.normcase(sp) == path_to_pysmaa for sp in sys.path):
            sys.path.append(f'{path_to_pysmaa}/src')

from scipy.spatial import Delaunay
from scipy import stats

from src.weights_from_rankings import verts_from_ranking

# test cases
from paper_test_problems import *

# utility functions
from paper_tests_utils import export2vtu


def test_problem_3_1_ismaa():
    """
    This function codes all ISMAA-related tests with Test Problem 3.1 in "Reverse-engineering stakeholders'
    # attitudes from observed preferences and quantitative data by Inverse Stochastic Multicriteria Acceptability
    # Analysis"
    """

    # read all relevant information on the Test Problem (here: Test Problem 3.1 "Reverse-engineering stakeholders'
    # attitudes from observed preferences and quantitative data by Inverse Stochastic Multicriteria Acceptability
    # Analysis"
    perfmat_mean, perfmat_types, perfmat_params, all_share_vectors, weights_prior = tests(0)

    special_colors = [['darkorange', 'orangered', 'firebrick', 'saddlebrown', 'black'],
                      ['lightsteelblue', 'royalblue', 'blue', 'indigo', 'darkorchid'],
                      ['palegreen', 'limegreen', 'forestgreen', 'darkgreen', 'teal'],
                      ['darkgrey', 'grey', 'dimgrey', 'black']]

    line_styles = ['solid', 'dashed', 'dashdot', (5, (10, 3)), 'dotted']

    # use LaTex to render the plots (leads to nicer appearance in particular for fonts)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'Times New Roman',
        "font.size": 16,
        "text.latex.preamble": r'\usepackage{amsfonts}'
    })

    # set up the MCDA method: SAW with colum-scaling such that the column-sum is one; every criterion is associated with
    # a benefit
    benefit = np.ones(perfmat_mean.shape[1], dtype=bool)
    mcda_method = SawSum(benefit)

    # number of points for 1D-visualisation
    n_points_for_vis = 200

    # number of samples for marginalising over P
    n_samples_perfmat = 10000

    w1_vec = np.zeros(n_points_for_vis)

    # evaluate the prior for visualisation
    prior_w_for_vis = np.zeros(n_points_for_vis)
    i = 0
    for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
        weight_test = np.array([w1, 1.0 - w1])
        prior_w_for_vis[i] = np.exp(weights_prior.log_density(weight_test))
        i += 1

    # set up a matplotlib-figure for visualisation of the posterior computed with ISMAA
    fig_ismaa_posterior = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_ismaa_posterior = fig_ismaa_posterior.add_subplot()
    ax_ismaa_posterior.set_xlabel(r'$w_1$', fontsize=20)
    ax_ismaa_posterior.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_ismaa_posterior.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_ismaa_posterior.set_ylabel(r'density', fontsize=20)

    fig_ismaa_posterior.subplots_adjust(top=0.99)
    fig_ismaa_posterior.subplots_adjust(bottom=0.19)
    fig_ismaa_posterior.subplots_adjust(left=0.15)
    fig_ismaa_posterior.subplots_adjust(right=0.98)

    # set up a matplotlib-figure for visualisation of the marginalised likelihood for ISMAA
    fig_ismaa_likelihood = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_ismaa_likelihood = fig_ismaa_likelihood.add_subplot()
    ax_ismaa_likelihood.set_xlabel(r'$w_1$', fontsize=20)
    ax_ismaa_likelihood.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_ismaa_likelihood.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_ismaa_likelihood.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_ismaa_likelihood.set_ylim(ymin=0, ymax=1.1)
    ax_ismaa_likelihood.set_ylabel(r'$\mathbb{P}(r| w)$', fontsize=20)

    fig_ismaa_likelihood.subplots_adjust(top=0.99)
    fig_ismaa_likelihood.subplots_adjust(bottom=0.19)
    fig_ismaa_likelihood.subplots_adjust(left=0.15)
    fig_ismaa_likelihood.subplots_adjust(right=0.98)

    # arrays for the likelihood for P fixed and P uncertain
    likelihood_ismaa_fixed = np.zeros(n_points_for_vis)
    likelihood_ismaa_fixed[int(0.5*n_points_for_vis):] = 1.0
    likelihood_ismaa = np.zeros(n_points_for_vis)

    i = 0
    for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
        w1_vec[i] = w1
        i += 1

    ax_ismaa_likelihood.plot(w1_vec, likelihood_ismaa_fixed, label=r'$P$ fixed',
                             color=special_colors[1][0], linestyle=line_styles[0])

    iidx = 1
    # loop over various levels of uncertainty
    for iunc in [0.1, 0.2, 0.3, 0.4]:

        likelihood_ismaa[:] = 0.0
        perfmat_params = -iunc*np.ones(perfmat_mean.shape)

        # sample performance matrices according to the prior distribution (here: uniform)
        prior_perfmat = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)

        perfmat_instances = prior_perfmat.sample(n_samples_perfmat)
        p_instances = np.zeros(perfmat_instances.shape)

        for i in range(n_samples_perfmat):
            p_instances[i] = mcda_method.p_from_perfmat(perfmat_instances[i])

        i = 0
        for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
            weight_test = np.array([w1, 1.0 - w1])
            prior_w_for_vis[i] = np.exp(weights_prior.log_density(weight_test))

            for imat in range(n_samples_perfmat):
                if np.argsort(-p_instances[imat]@weight_test)[0] == 0:
                    likelihood_ismaa[i] += 1

            w1_vec[i] = w1
            i += 1

        likelihood_ismaa = likelihood_ismaa/n_samples_perfmat

        # posterior of the weights
        posterior_weights = prior_w_for_vis * likelihood_ismaa

        int_post = np.sum(posterior_weights)/n_points_for_vis
        posterior_weights = posterior_weights / int_post

        # rather hacky: plot the curves for fixed P only one time in the loop; therefore iidx ==1.
        if iidx == 1:
            posterior_weights_fixed = prior_w_for_vis * likelihood_ismaa_fixed
            int_post = np.sum(posterior_weights_fixed) / n_points_for_vis
            posterior_weights_fixed = posterior_weights_fixed / int_post

            ax_ismaa_posterior.plot(w1_vec, posterior_weights_fixed, label=r'$P$ fixed',
                                    color=special_colors[2][0], linestyle=line_styles[0])

        ax_ismaa_likelihood.plot(w1_vec, likelihood_ismaa, label=r'$P, \pm' + str(iunc) + '$',
                                 color=special_colors[1][iidx], linestyle=line_styles[iidx])

        ax_ismaa_posterior.plot(w1_vec, posterior_weights, label=r'$P, \pm' + str(iunc) + '$',
                                color=special_colors[2][iidx], linestyle=line_styles[iidx])

        iidx += 1

    ax_ismaa_posterior.plot(w1_vec, prior_w_for_vis, label=r'prior',
              color='black', linestyle='solid')

    ax_ismaa_likelihood.legend()
    ax_ismaa_posterior.legend()

    # show figures
    fig_ismaa_likelihood.show()
    fig_ismaa_posterior.show()

    # save figures as pdf-files
    fig_ismaa_likelihood.savefig('TestProb3.1_ismaa_likelihood.pdf')
    fig_ismaa_posterior.savefig('TestProb3.1_ismaa_posterior.pdf')


def test_problem_3_1_qismaa():
    """
    This function codes all QISMAA-related tests with Test Problem 3.1 in "Reverse-engineering stakeholders'
    # attitudes from observed preferences and quantitative data by Inverse Stochastic Multicriteria Acceptability
    # Analysis"
    """

    perfmat_mean, perfmat_types, perfmat_params, all_share_vectors, weights_prior = tests(0)

    special_colors = [['darkorange', 'orangered', 'firebrick', 'saddlebrown', 'black'],
                      ['lightsteelblue', 'royalblue', 'blue', 'indigo', 'darkorchid'],
                      ['palegreen', 'limegreen', 'forestgreen', 'darkgreen', 'teal'],
                      ['darkgrey', 'grey', 'dimgrey', 'black']]

    line_styles = ['solid', 'dashed', 'dashdot', (5, (10, 3)), 'dotted']

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'Times New Roman',
        "font.size": 16,
        "text.latex.preamble": r'\usepackage{amsfonts}'
    })

    # set up of the figures
    fig_qismaa_likelihood_p_fixed = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_qismaa_likelihood_p_fixed = fig_qismaa_likelihood_p_fixed.add_subplot()
    ax_qismaa_likelihood_p_fixed.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_likelihood_p_fixed.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood_p_fixed.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood_p_fixed.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_likelihood_p_fixed.set_ylim(ymin=0, ymax=1.1)
    ax_qismaa_likelihood_p_fixed.set_ylabel(r'$\mathcal{P}(s | w, P, \gamma)$', fontsize=20)
    ax_qismaa_likelihood_p_fixed.set_title('')

    fig_qismaa_likelihood_p_fixed.subplots_adjust(top=0.99)
    fig_qismaa_likelihood_p_fixed.subplots_adjust(bottom=0.19)
    fig_qismaa_likelihood_p_fixed.subplots_adjust(left=0.15)
    fig_qismaa_likelihood_p_fixed.subplots_adjust(right=0.98)

    fig_qismaa_s_bar_p_fixed = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_qismaa_s_bar_p_fixed = fig_qismaa_s_bar_p_fixed.add_subplot()
    ax_qismaa_s_bar_p_fixed.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_s_bar_p_fixed.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_s_bar_p_fixed.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_s_bar_p_fixed.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_s_bar_p_fixed.set_xlim(xmin=0.0, xmax=1.0)
    ax_qismaa_s_bar_p_fixed.set_ylim(ymin=0.0, ymax=1.0)
    ax_qismaa_s_bar_p_fixed.set_title('')
    fig_qismaa_s_bar_p_fixed.subplots_adjust(top=0.99)
    fig_qismaa_s_bar_p_fixed.subplots_adjust(bottom=0.19)
    fig_qismaa_s_bar_p_fixed.subplots_adjust(left=0.15)
    fig_qismaa_s_bar_p_fixed.subplots_adjust(right=0.98)

    fig_qismaa_s_bar_p_fixed_vs_beta = plt.figure(dpi=200, figsize=(6.5, 4))
    ax_qismaa_s_bar_p_fixed_vs_beta = fig_qismaa_s_bar_p_fixed_vs_beta.add_subplot()
    ax_qismaa_s_bar_p_fixed_vs_beta.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_s_bar_p_fixed_vs_beta.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_s_bar_p_fixed_vs_beta.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_s_bar_p_fixed_vs_beta.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_s_bar_p_fixed_vs_beta.set_xlim(xmin=0.0, xmax=1.0)
    ax_qismaa_s_bar_p_fixed_vs_beta.set_ylim(ymin=0.0, ymax=1.0)
    ax_qismaa_s_bar_p_fixed_vs_beta.set_title('')
    fig_qismaa_s_bar_p_fixed_vs_beta.subplots_adjust(top=0.98)
    fig_qismaa_s_bar_p_fixed_vs_beta.subplots_adjust(bottom=0.15)
    fig_qismaa_s_bar_p_fixed_vs_beta.subplots_adjust(left=0.37)
    fig_qismaa_s_bar_p_fixed_vs_beta.subplots_adjust(right=0.98)

    fig_qismaa_posterior_p_uncertain = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_qismaa_posterior_p_uncertain = fig_qismaa_posterior_p_uncertain.add_subplot()
    ax_qismaa_posterior_p_uncertain.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_posterior_p_uncertain.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_posterior_p_uncertain.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_posterior_p_uncertain.set_xlim(xmin=0, xmax=1)
    ax_qismaa_posterior_p_uncertain.set_ylabel(r'density', fontsize=20)
    fig_qismaa_posterior_p_uncertain.subplots_adjust(top=0.99)
    fig_qismaa_posterior_p_uncertain.subplots_adjust(bottom=0.19)
    fig_qismaa_posterior_p_uncertain.subplots_adjust(left=0.15)
    fig_qismaa_posterior_p_uncertain.subplots_adjust(right=0.97)

    fig_qismaa_likelihood_p_uncertain = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_qismaa_likelihood_p_uncertain = fig_qismaa_likelihood_p_uncertain.add_subplot()
    ax_qismaa_likelihood_p_uncertain.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_likelihood_p_uncertain.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood_p_uncertain.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood_p_uncertain.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_likelihood_p_uncertain.set_ylim(ymin=0, ymax=1.1)
    ax_qismaa_likelihood_p_uncertain.set_ylabel(r'$\mathcal{P}(s | w, \gamma)$', fontsize=20)

    fig_qismaa_likelihood_p_uncertain.subplots_adjust(top=0.99)
    fig_qismaa_likelihood_p_uncertain.subplots_adjust(bottom=0.19)
    fig_qismaa_likelihood_p_uncertain.subplots_adjust(left=0.15)
    fig_qismaa_likelihood_p_uncertain.subplots_adjust(right=0.98)

    # set up the MCDA method
    benefit = np.ones(perfmat_mean.shape[1], dtype=bool)
    mcda_method = SawSum(benefit)

    # number of points for visualisation
    n_points_for_vis = 200

    # number of samples for marginalising over P
    n_samples_perfmat = 10000

    # set beta
    beta = 1 - 1e-6

    likelihood_qismaa_p_fixed_vec = np.zeros(n_points_for_vis)
    synth_share_vec = np.zeros((n_points_for_vis, all_share_vectors[0].shape[0]), dtype=np.float64)
    synth_share_vec_p_uncertain = np.zeros((n_points_for_vis, all_share_vectors[0].shape[0]), dtype=np.float64)
    w1_vec = np.zeros(n_points_for_vis)
    prior_w_for_vis = np.zeros(n_points_for_vis)

    # evaluate the prior for visualisation
    i = 0
    for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
        weight_test = np.array([w1, 1.0 - w1])
        prior_w_for_vis[i] = np.exp(weights_prior.log_density(weight_test))
        i += 1

    iprob = 0
    for share_vec in all_share_vectors:

        # likelihood-obejct for fixed P
        likelihood_p_fixed = LikelihoodQISMAA(perfmat_mean.shape[0], perfmat_mean.shape[1], share_vec, mcda_method,
                                              alpha=20.0, beta=beta, p=2.0, q=2.0)
        label_aux = str(share_vec[0])
        for alt in range(1, share_vec.shape[0]):
            label_aux = label_aux + ',' + str(share_vec[alt])

        i = 0
        for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
            weight_test = np.array([w1, 1.0 - w1])
            log_likelihood, synth_share = likelihood_p_fixed._log_density(weight_test, perfmat_mean)
            likelihood_qismaa_p_fixed_vec[i] = np.exp(log_likelihood)
            synth_share_vec[i, :] = synth_share
            w1_vec[i] = w1
            i += 1

        ax_qismaa_likelihood_p_fixed.plot(w1_vec, likelihood_qismaa_p_fixed_vec, label=r'$\!\!($' + label_aux + r'$)$',
                                          color=special_colors[1][iprob], linestyle=line_styles[iprob])
        iprob += 1

    ax_qismaa_likelihood_p_fixed.legend(loc='upper left', fontsize=14)
    fig_qismaa_likelihood_p_fixed.show()
    fig_qismaa_likelihood_p_fixed.savefig('TestProb3.1_qismaa_likelihood_vs_w1_SAW_alpha' +
                                          str(likelihood_p_fixed.alpha) +
                                          '_beta' + str(likelihood_p_fixed.beta) + '.pdf')

    likelihood_p_fixed_vs_beta = LikelihoodQISMAA(perfmat_mean.shape[0], perfmat_mean.shape[1], share_vec,
                                                  mcda_method, alpha=20.0, beta=beta, p=2.0, q=2.0)

    label_aux = str(share_vec[0])
    for alt in range(1, share_vec.shape[0]):
        label_aux = label_aux + ',' + str(share_vec[alt])

    i = 0
    for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
        weight_test = np.array([w1, 1.0 - w1])
        synth_share = likelihood_p_fixed_vs_beta._log_density(weight_test, perfmat_mean)[1]
        synth_share_vec[i, :] = synth_share
        w1_vec[i] = w1
        i += 1

    iprob += 1

    # figures are not needed in the paper, therefore commented out
    """
    label_text = r'$s^\prime_' + str(1) + ', \\beta = ' + str(beta) + '$'
    ax_qismaa_s_bar_p_fixed_vs_beta.plot(w1_vec, synth_share_vec[:, 0], label=label_text,
                                         color=special_colors[0][ibeta], linestyle=line_styles[ibeta])

    label_text = r'$s^\prime_' + str(2) + ', \\beta = ' + str(beta) + '$'
    ax_qismaa_s_bar_p_fixed_vs_beta.plot(w1_vec, synth_share_vec[:, 1], label=label_text,
                                         color=special_colors[1][ibeta], linestyle=line_styles[ibeta])

    label_text = r'$s^\prime_' + str(3) + ', \\beta = ' + str(beta) + '$'
    ax_qismaa_s_bar_p_fixed_vs_beta.plot(w1_vec, synth_share_vec[:, 2], label=label_text,
                                         color=special_colors[3][ibeta], linestyle=line_styles[ibeta])


    ax_qismaa_s_bar_p_fixed_vs_beta.legend(bbox_to_anchor=(-0.09, 1.0))
    fig_qismaa_s_bar_p_fixed_vs_beta.show()
    fig_qismaa_s_bar_p_fixed_vs_beta.savefig('TestProb3.1_qismaa_shares_vs_beta_neu.pdf')
    """

    # new vector of shares
    share_vec = np.array([0.6, 0.4, 0.0])
    #share_vec = np.array([0.9, 0.1, 0.0])

    likelihood_qismaa_p_uncertain_vec = np.zeros(n_points_for_vis)

    iidx = 1
    for iunc in [0.1, 0.2, 0.3, 0.4]:

        synth_share_vec_p_uncertain[:, :] = 0.0

        likelihood_qismaa_p_uncertain_vec[:] = 0.0
        perfmat_params = -iunc*np.ones(perfmat_mean.shape)

        # sample performance matrices according to the prior distribution (here: uniform)
        prior_perfmat = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)

        perfmat_instances = prior_perfmat.sample(n_samples_perfmat)

        likelihood_p_uncertain = LikelihoodQISMAA(perfmat_mean.shape[0], perfmat_mean.shape[1], share_vec, mcda_method,
                                                  alpha=20.0, beta=beta, p=2.0, q=2.0)

        i = 0
        for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
            weight_test = np.array([w1, 1.0 - w1])

            # likelihood for fixed P
            log_likelihood, synth_share = likelihood_p_uncertain._log_density(weight_test, perfmat_mean)
            likelihood_qismaa_p_fixed_vec[i] = np.exp(log_likelihood)

            for imat in range(n_samples_perfmat):
                likelihood_qismaa_p_uncertain_vec[i] += (
                    np.exp(likelihood_p_uncertain._log_density(weight_test, perfmat_instances[imat])[0]))

                synth_share_vec_p_uncertain[i, :] += synth_share

            w1_vec[i] = w1
            i += 1

        likelihood_qismaa_p_uncertain_vec = likelihood_qismaa_p_uncertain_vec/n_samples_perfmat
        synth_share_vec_p_uncertain = synth_share_vec_p_uncertain/n_samples_perfmat

        # posterior of the weights
        posterior_weights = prior_w_for_vis * likelihood_qismaa_p_uncertain_vec

        int_post = np.sum(posterior_weights)/n_points_for_vis
        posterior_weights = posterior_weights / int_post

        if iidx == 1:
            posterior_weights_fixed = prior_w_for_vis * likelihood_qismaa_p_fixed_vec
            int_post = np.sum(posterior_weights_fixed) / n_points_for_vis
            posterior_weights_fixed = posterior_weights_fixed / int_post

            ax_qismaa_posterior_p_uncertain.plot(w1_vec, posterior_weights_fixed, label=r'$P$ fixed',
                                                 color=special_colors[2][0], linestyle=line_styles[0])

            ax_qismaa_likelihood_p_uncertain.plot(w1_vec, likelihood_qismaa_p_fixed_vec, label=r'$P$ fixed',
                                                  color=special_colors[1][0], linestyle=line_styles[0])

        ax_qismaa_likelihood_p_uncertain.plot(w1_vec, likelihood_qismaa_p_uncertain_vec,
                                              label=r'$P, \pm' + str(iunc) + '$',
                                              color=special_colors[1][iidx], linestyle=line_styles[iidx])

        ax_qismaa_posterior_p_uncertain.plot(w1_vec, posterior_weights, label=r'$P, \pm' + str(iunc) + '$',
                                             color=special_colors[2][iidx], linestyle=line_styles[iidx])

        iidx += 1

    ax_qismaa_posterior_p_uncertain.plot(w1_vec, prior_w_for_vis, label=r'prior',
                                         color='black', linestyle='solid')

    ax_qismaa_likelihood_p_uncertain.legend()
    fig_qismaa_likelihood_p_uncertain.show()
    fig_qismaa_likelihood_p_uncertain.savefig('TestProb3.1_qismaa_likelihood_P_uncertain.pdf')

    ax_qismaa_posterior_p_uncertain.legend(loc='upper left')
    fig_qismaa_posterior_p_uncertain.show()
    fig_qismaa_posterior_p_uncertain.savefig('TestProb3.1_qismaa_posterior_P_uncertain.pdf')


def comparison_testprob_3_1_3_2_qismaa():
    """
    Computations for Example 3.6
    """
    # data for Test Problem 3.1
    perfmat_mean31, perfmat_types31, perfmat_params31, all_share_vectors, weights_prior = tests(0)

    # data for Test Problem 3.2
    perfmat_mean32, perfmat_types32, perfmat_params32, all_share_vectors32, weights_prior = tests(1)

    special_colors = [['darkorange', 'orangered', 'firebrick', 'saddlebrown', 'black'],
                      ['lightsteelblue', 'royalblue', 'blue', 'indigo', 'darkorchid'],
                      ['palegreen', 'limegreen', 'forestgreen', 'darkgreen', 'teal'],
                      ['darkgrey', 'grey', 'dimgrey', 'black']]

    line_styles = ['solid', 'dashed', 'dashdot', (5, (10, 3)), 'dotted']

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'Times New Roman',
        "font.size": 16,
        "text.latex.preamble": r'\usepackage{amsfonts}'
    })

    # set up of the figures
    fig_qismaa_likelihood_p_fixed = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_qismaa_likelihood_p_fixed = fig_qismaa_likelihood_p_fixed.add_subplot()
    ax_qismaa_likelihood_p_fixed.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_likelihood_p_fixed.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood_p_fixed.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood_p_fixed.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_likelihood_p_fixed.set_ylim(ymin=0, ymax=1.1)
    ax_qismaa_likelihood_p_fixed.set_ylabel(r'$\mathcal{P}(s | w, P, \gamma)$', fontsize=20)
    ax_qismaa_likelihood_p_fixed.set_title('')

    fig_qismaa_likelihood_p_fixed.subplots_adjust(top=0.99)
    fig_qismaa_likelihood_p_fixed.subplots_adjust(bottom=0.19)
    fig_qismaa_likelihood_p_fixed.subplots_adjust(left=0.15)
    fig_qismaa_likelihood_p_fixed.subplots_adjust(right=0.98)

    fig_qismaa_s_bar_p_fixed = plt.figure(dpi=200, figsize=(6, 4))
    ax_qismaa_s_bar_p_fixed = fig_qismaa_s_bar_p_fixed.add_subplot()
    ax_qismaa_s_bar_p_fixed.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_s_bar_p_fixed.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_s_bar_p_fixed.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_s_bar_p_fixed.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_s_bar_p_fixed.set_xlim(xmin=0.0, xmax=1.0)
    ax_qismaa_s_bar_p_fixed.set_ylim(ymin=0.0, ymax=1.0)
    ax_qismaa_s_bar_p_fixed.set_title('')
    fig_qismaa_s_bar_p_fixed.subplots_adjust(top=0.98)
    fig_qismaa_s_bar_p_fixed.subplots_adjust(bottom=0.19)
    fig_qismaa_s_bar_p_fixed.subplots_adjust(left=0.15)
    fig_qismaa_s_bar_p_fixed.subplots_adjust(right=0.98)

    fig_qismaa_posterior_p_uncertain = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_qismaa_posterior_p_uncertain = fig_qismaa_posterior_p_uncertain.add_subplot()
    ax_qismaa_posterior_p_uncertain.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_posterior_p_uncertain.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_posterior_p_uncertain.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_posterior_p_uncertain.set_xlim(xmin=0, xmax=1.0)
    fig_qismaa_posterior_p_uncertain.subplots_adjust(top=0.98)
    fig_qismaa_posterior_p_uncertain.subplots_adjust(bottom=0.19)
    fig_qismaa_posterior_p_uncertain.subplots_adjust(left=0.15)
    fig_qismaa_posterior_p_uncertain.subplots_adjust(right=0.98)

    fig_qismaa_likelihood_p_uncertain = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_qismaa_likelihood_p_uncertain = fig_qismaa_likelihood_p_uncertain.add_subplot()
    ax_qismaa_likelihood_p_uncertain.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_likelihood_p_uncertain.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood_p_uncertain.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood_p_uncertain.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_likelihood_p_uncertain.set_ylim(ymin=0, ymax=0.62)
    ax_qismaa_likelihood_p_uncertain.set_ylabel(r'$\mathcal{P}(s | w, P, \gamma)$', fontsize=20)

    fig_qismaa_likelihood_p_uncertain.subplots_adjust(top=0.98)
    fig_qismaa_likelihood_p_uncertain.subplots_adjust(bottom=0.19)
    fig_qismaa_likelihood_p_uncertain.subplots_adjust(left=0.15)
    fig_qismaa_likelihood_p_uncertain.subplots_adjust(right=0.97)

    fig_qismaa_s_bar_p_uncertain = plt.figure(dpi=200, figsize=(4.8, 3.2))
    ax_qismaa_s_bar_p_uncertain = fig_qismaa_s_bar_p_uncertain.add_subplot()
    ax_qismaa_s_bar_p_uncertain.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_s_bar_p_uncertain.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_s_bar_p_uncertain.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1])
    ax_qismaa_s_bar_p_uncertain.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_s_bar_p_uncertain.set_xlim(xmin=0, xmax=1.0)
    ax_qismaa_s_bar_p_uncertain.set_ylim(ymin=0, ymax=1.0)
    ax_qismaa_s_bar_p_uncertain.set_title('')
    fig_qismaa_s_bar_p_uncertain.subplots_adjust(top=0.98)
    fig_qismaa_s_bar_p_uncertain.subplots_adjust(bottom=0.19)
    fig_qismaa_s_bar_p_uncertain.subplots_adjust(left=0.15)
    fig_qismaa_s_bar_p_uncertain.subplots_adjust(right=0.97)

    # set up the MCDA method
    benefit = np.ones(perfmat_mean31.shape[1], dtype=bool)

    # the sigmoid preference function is encoded by its type (integer number 6) here
    type_criterion = 6 * np.ones(perfmat_mean31.shape[1], dtype=np.int64)
    pars = np.zeros((2, perfmat_mean31.shape[1]))

    # parameter s for the sigmoid function for Promethee II
    pars[0, :] = 0.4

    mcda_methods = [SawSum(benefit), Promethee(type_criterion, pars, benefit)]
    mcda_texts = ['SAW', 'P2']

    # number of points for visualisation
    n_points_for_vis = 200

    # number of samples for marginalizing
    n_samples_perfmat = 1000

    # set beta in QISMAA
    beta = 1 - 1e-6

    likelihood_qismaa_p_fixed_31_vec = np.zeros(n_points_for_vis)
    likelihood_qismaa_p_fixed_32_vec = np.zeros(n_points_for_vis)
    synth_share_31_vec = np.zeros((n_points_for_vis, all_share_vectors[0].shape[0]), dtype=np.float64)
    synth_share_32_vec = np.zeros((n_points_for_vis, all_share_vectors[0].shape[0]), dtype=np.float64)
    w1_vec = np.zeros(n_points_for_vis)
    prior_w_for_vis = np.zeros(n_points_for_vis)

    # evaluate the prior for visualisation
    i = 0
    for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
        weight_test = np.array([w1, 1.0 - w1])
        prior_w_for_vis[i] = np.exp(weights_prior.log_density(weight_test))
        i += 1

    iprob = 0
    share_vec = np.array([0.6, 0.2, 0.2], dtype=np.float64)

    for mcda_method in mcda_methods:
        likelihood_p_fixed_31 = LikelihoodQISMAA(perfmat_mean31.shape[0], perfmat_mean31.shape[1], share_vec,
                                                 mcda_method, alpha=20.0, beta=beta, p=2.0, q=2.0)

        likelihood_p_fixed_32 = LikelihoodQISMAA(perfmat_mean32.shape[0], perfmat_mean32.shape[1], share_vec,
                                                 mcda_method, alpha=20.0, beta=beta, p=2.0, q=2.0)

        label_aux = str(share_vec[0])
        for alt in range(1, share_vec.shape[0]):
            label_aux = label_aux + ',' + str(share_vec[alt])

        i = 0
        for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
            weight_test = np.array([w1, 1.0 - w1])
            log_likelihood_31, synth_share_31 = likelihood_p_fixed_31._log_density(weight_test, perfmat_mean31)
            log_likelihood_32, synth_share_32 = likelihood_p_fixed_32._log_density(weight_test, perfmat_mean32)
            likelihood_qismaa_p_fixed_31_vec[i] = np.exp(log_likelihood_31)
            likelihood_qismaa_p_fixed_32_vec[i] = np.exp(log_likelihood_32)
            synth_share_31_vec[i, :] = synth_share_31
            synth_share_32_vec[i, :] = synth_share_32
            w1_vec[i] = w1
            i += 1

        ax_qismaa_likelihood_p_fixed.plot(w1_vec, likelihood_qismaa_p_fixed_31_vec,
                                          label=r'TP 3.1, ' + mcda_texts[iprob],
                                          color=special_colors[1][2*iprob], linestyle=line_styles[2*iprob])
        ax_qismaa_likelihood_p_fixed.plot(w1_vec, likelihood_qismaa_p_fixed_32_vec,
                                          label=r'TP 3.3, ' + mcda_texts[iprob],
                                          color=special_colors[1][2*iprob+1], linestyle=line_styles[2*iprob+1])

        label_text = r'$s^\prime_' + str(1) + '$, TP 3.1, ' + mcda_texts[iprob]
        ax_qismaa_s_bar_p_fixed.plot(w1_vec, synth_share_31_vec[:, 0], label=label_text,
                                     color=special_colors[0][2*iprob], linestyle=line_styles[0])

        label_text = r'$s^\prime_' + str(3) + '$, TP 3.1, ' + mcda_texts[iprob]
        ax_qismaa_s_bar_p_fixed.plot(w1_vec, synth_share_31_vec[:, 2], label=label_text,
                                     color=special_colors[3][2*iprob], linestyle=line_styles[1])

        label_text = r'$s^\prime_' + str(1) + '$, TP 3.3, ' + mcda_texts[iprob]
        ax_qismaa_s_bar_p_fixed.plot(w1_vec, synth_share_32_vec[:, 0], label=label_text,
                                     color=special_colors[0][2*iprob+1], linestyle=line_styles[2])

        label_text = r'$s^\prime_' + str(3) + '$, TP 3.3, ' + mcda_texts[iprob]
        ax_qismaa_s_bar_p_fixed.plot(w1_vec, synth_share_32_vec[:, 2], label=label_text,
                                     color=special_colors[3][2*iprob+1], linestyle=line_styles[3])
        iprob += 1

    ax_qismaa_s_bar_p_fixed.legend()
    fig_qismaa_s_bar_p_fixed.show()
    fig_qismaa_s_bar_p_fixed.savefig('Comp_sprime_vs_w1_P_fixed.pdf')

    ax_qismaa_likelihood_p_fixed.legend(loc='upper left')
    fig_qismaa_likelihood_p_fixed.show()
    fig_qismaa_likelihood_p_fixed.savefig('Comp_likelihood_vs_w1_P_fixed.pdf')

    for iunc in [0.2]:

        perfmat_params = -iunc*np.ones(perfmat_mean31.shape)

        iprob = 0
        for mcda_method in mcda_methods:
            likelihood_qismaa_p_unc_31_vec = np.zeros(n_points_for_vis)
            likelihood_qismaa_p_unc_32_vec = np.zeros(n_points_for_vis)

            synth_share_31_p_unc_vec = np.zeros((n_points_for_vis, all_share_vectors[0].shape[0]), dtype=np.float64)
            synth_share_32_p_unc_vec = np.zeros((n_points_for_vis, all_share_vectors[0].shape[0]), dtype=np.float64)

            likelihood_p_unc_31 = LikelihoodQISMAA(perfmat_mean31.shape[0], perfmat_mean31.shape[1], share_vec,
                                                   mcda_method, alpha=20.0, beta=beta, p=2.0, q=2.0)

            likelihood_p_unc_32 = LikelihoodQISMAA(perfmat_mean32.shape[0], perfmat_mean32.shape[1], share_vec,
                                                   mcda_method, alpha=20.0, beta=beta, p=2.0, q=2.0)

            # sample performance matrices according to the prior distribution (here: uniform)
            prior_perfmat_31 = UncertainPerfMatPrior(perfmat_mean31, perfmat_types31, perfmat_params)
            perfmat_instances_31 = prior_perfmat_31.sample(n_samples_perfmat)

            prior_perfmat_32 = UncertainPerfMatPrior(perfmat_mean32, perfmat_types32, perfmat_params)
            perfmat_instances_32 = prior_perfmat_32.sample(n_samples_perfmat)

            i = 0
            for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
                weight_test = np.array([w1, 1.0 - w1])

                # likelihood for for uncertain P
                for imat in range(n_samples_perfmat):
                    likelihood_val, shares_val = likelihood_p_unc_31._log_density(weight_test,
                                                                                  perfmat_instances_31[imat])

                    likelihood_qismaa_p_unc_31_vec[i] += np.exp(likelihood_val)
                    synth_share_31_p_unc_vec[i] += shares_val

                    likelihood_val, shares_val = likelihood_p_unc_32._log_density(weight_test,
                                                                                  perfmat_instances_32[imat])

                    likelihood_qismaa_p_unc_32_vec[i] += np.exp(likelihood_val)
                    synth_share_32_p_unc_vec[i] += shares_val

                w1_vec[i] = w1
                i += 1

            likelihood_qismaa_p_unc_31_vec = likelihood_qismaa_p_unc_31_vec/n_samples_perfmat
            likelihood_qismaa_p_unc_32_vec = likelihood_qismaa_p_unc_32_vec/n_samples_perfmat

            synth_share_31_p_unc_vec = synth_share_31_p_unc_vec/n_samples_perfmat
            synth_share_32_p_unc_vec = synth_share_32_p_unc_vec/n_samples_perfmat

            # posterior of the weights
            posterior_weights_31 = prior_w_for_vis * likelihood_qismaa_p_unc_31_vec
            posterior_weights_32 = prior_w_for_vis * likelihood_qismaa_p_unc_32_vec

            int_post = np.sum(posterior_weights_31)/n_points_for_vis
            posterior_weights_31 = posterior_weights_31 / int_post

            int_post = np.sum(posterior_weights_32)/n_points_for_vis
            posterior_weights_32 = posterior_weights_32 / int_post

            ax_qismaa_likelihood_p_uncertain.plot(w1_vec, likelihood_qismaa_p_unc_31_vec,
                                                  label=r'TP 3.1, ' + mcda_texts[iprob],
                                                  color=special_colors[1][2 * iprob],
                                                  linestyle=line_styles[2 * iprob])

            ax_qismaa_likelihood_p_uncertain.plot(w1_vec, likelihood_qismaa_p_unc_32_vec,
                                                  label=r'TP 3.2, ' + mcda_texts[iprob],
                                                  color=special_colors[1][2 * iprob + 1],
                                                  linestyle=line_styles[2 * iprob + 1])

            label_text = r'$s^\prime_' + str(1) + '$, TP 3.1, ' + mcda_texts[iprob]
            ax_qismaa_s_bar_p_uncertain.plot(w1_vec, synth_share_31_p_unc_vec[:, 0], label=label_text,
                                             color=special_colors[0][2 * iprob], linestyle=line_styles[0])

            label_text = r'$s^\prime_' + str(3) + '$, TP 3.1, ' + mcda_texts[iprob]
            ax_qismaa_s_bar_p_uncertain.plot(w1_vec, synth_share_31_p_unc_vec[:, 2], label=label_text,
                                             color=special_colors[3][2 * iprob], linestyle=line_styles[1])

            label_text = r'$s^\prime_' + str(1) + '$, TP 3.2, ' + mcda_texts[iprob]
            ax_qismaa_s_bar_p_uncertain.plot(w1_vec, synth_share_32_p_unc_vec[:, 0], label=label_text,
                                             color=special_colors[0][2 * iprob + 1], linestyle=line_styles[2])

            label_text = r'$s^\prime_' + str(3) + '$, TP 3.2, ' + mcda_texts[iprob]
            ax_qismaa_s_bar_p_uncertain.plot(w1_vec, synth_share_32_p_unc_vec[:, 2], label=label_text,
                                             color=special_colors[3][2 * iprob + 1], linestyle=line_styles[3])
            iprob += 1

    ax_qismaa_posterior_p_uncertain.plot(w1_vec, prior_w_for_vis, label=r'prior',
                                         color='black', linestyle='solid')

    ax_qismaa_s_bar_p_uncertain.legend(fontsize=14)
    fig_qismaa_s_bar_p_uncertain.show()
    fig_qismaa_s_bar_p_uncertain.savefig('Comp_sprime_vs_w1_P_uncertain.pdf')

    ax_qismaa_likelihood_p_uncertain.legend(fontsize=16, loc='upper left')
    fig_qismaa_likelihood_p_uncertain.show()
    fig_qismaa_likelihood_p_uncertain.savefig('Comp_likelihood_vs_w1_P_uncertain.pdf')


def test_problem_3_2_smaa():
    """
    Forward SMAA for Test Problem 3.2.
    """

    special_colors = [['darkorange', 'orangered', 'firebrick', 'saddlebrown', 'black'],
                      ['lightsteelblue', 'royalblue', 'blue', 'indigo', 'darkorchid'],
                      ['palegreen', 'limegreen', 'forestgreen', 'darkgreen', 'teal']]

    line_styles = ['solid', 'dashed', 'dashdot', (5,(10,3)), 'solid']

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'Times New Roman',
        "font.size": 16,
        "text.latex.preamble": r'\usepackage{amsfonts}'
    })

    # sample performance matrices according to the prior distribution (here: uniform)

    perfmat_mean, perfmat_types, perfmat_params, all_probs_of_alts, weights_prior = tests(1)

    nalts, ncrit = perfmat_mean.shape

    # It is not possible currently to read distributional information on the weights from Excel. This is as in contrast
    # to the values of the performance matrix, which are usually independent, the weights are dependent due to
    # normalisation, which makes coding the corresponding distributions much more complicated.

    benefit = np.ones(perfmat_mean.shape[1], dtype=bool)
    mcda_method = SawSum(benefit)

    # number of samples
    n_samples = 20000

    p_alts = np.zeros((nalts, nalts))

    # sample weights according to that distribution
    samples_weights = weights_prior.sample(n_samples)

    # sample instances of perfmat for statistics
    prior_perfmat = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)
    samples_perfmat = prior_perfmat.sample(n_samples)

    scores_of_alts = np.zeros((n_samples, nalts))

    # compute the SAW-matrices from these samples
    for isample in range(n_samples):
        pmat = mcda_method.p_from_perfmat(samples_perfmat[isample])

        perf = pmat @ samples_weights[isample]

        ranking = np.argsort(-perf)
        scores_of_alts[isample, :] = perf

        p_alts[np.arange(0, nalts), ranking] = p_alts[np.arange(0, nalts), ranking] + 1

    p_alts = p_alts / n_samples

    alternatives = [r'alt.\ \hspace{-1mm}1', r'alt.\ \hspace{-1mm}2', r'alt.\ \hspace{-1mm}3']

    fig_probs_for_alts = plt.figure(dpi=200, figsize=(3,3))
    ax = fig_probs_for_alts.add_subplot()
    ax.matshow(p_alts)
    ax.set_xlabel('alternatives', fontsize=20)
    ax.set_xticks([0, 1, 2], labels=alternatives)
    ax.set_ylabel('position in ranking', fontsize=20)
    ax.set_yticks([0,1,2], labels=['1', '2', '3'])

    fig_probs_for_alts.subplots_adjust(left=0.16)
    fig_probs_for_alts.subplots_adjust(right=0.98)

    for i in range(nalts):
        for j in range(nalts):
            c = p_alts[j, i].round(2)
            if c < 0.35:
                color = 'white'
            else:
                color = 'black'

            ax.text(i, j, str(c), va='center', ha='center', color=color, size=20)

    fig_probs_for_alts.show()
    fig_probs_for_alts.savefig('TestProb3.3_p_matrix.pdf')

    # image of values
    w1 = np.linspace(0.0, 1.0, 2)
    w2 = 1-w1

    weights = np.vstack((w1, w2))
    values = perfmat_mean@weights

    fig_values = plt.figure(dpi=200, figsize = (4.8, 3))
    ax_values = fig_values.add_subplot()
    ax_values.set_xlabel(r'$w_1$', fontsize=20)
    ax_values.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_values.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_values.set_ylabel(r'value', fontsize=20)

    fig_values.subplots_adjust(top=0.99)
    fig_values.subplots_adjust(bottom=0.2)
    fig_values.subplots_adjust(left=0.15)
    fig_values.subplots_adjust(right=0.98)

    ax_values.plot(w1, values[0, :], label=r'alt.\ \hspace{-1mm}1',
                   color=special_colors[0][0], linestyle=line_styles[0])
    ax_values.plot(w1, values[1, :], label=r'alt.\ \hspace{-1mm}2',
                   color=special_colors[0][1], linestyle=line_styles[1])
    ax_values.plot(w1, values[2, :], label=r'alt.\ \hspace{-1mm}3',
                   color=special_colors[0][2], linestyle=line_styles[2])

    ax_values.legend()
    fig_values.show()
    fig_values.savefig('TestProb3.3_values.pdf')

    return p_alts


def test_problem_3_2_ismaa_likelihood_and_posterior():
    perfmat_mean, perfmat_types, perfmat_params, all_probs_of_alts, prior_weights = tests(1)

    special_colors = [['darkorange', 'orangered', 'firebrick', 'saddlebrown', 'black'],
                      ['lightsteelblue', 'royalblue', 'blue', 'indigo', 'darkorchid'],
                      ['palegreen', 'limegreen', 'forestgreen', 'darkgreen', 'teal']]

    line_styles = ['solid', 'dashed', 'dashdot', (5,(10,3)), 'solid']

    benefit = np.ones(perfmat_mean.shape[1], dtype=bool)
    mcda_method = SawSum(benefit)

    n_points_for_vis = 200
    w1_vals = np.zeros(n_points_for_vis)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'Times New Roman',
        "font.size": 16,
        "text.latex.preamble": r'\usepackage{amsfonts}'
    })

    fig_ismaa_posterior = plt.figure(dpi=200, figsize=(6, 4))
    ax_ismaa_posterior = fig_ismaa_posterior.add_subplot()
    ax_ismaa_posterior.set_xlabel(r'$w_1$', fontsize=20)
    ax_ismaa_posterior.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_ismaa_posterior.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_ismaa_posterior.set_ylabel(r'density', fontsize=20)

    fig_ismaa_posterior.subplots_adjust(top=0.99)
    fig_ismaa_posterior.subplots_adjust(bottom=0.15)
    fig_ismaa_posterior.subplots_adjust(left=0.12)
    fig_ismaa_posterior.subplots_adjust(right=0.98)

    fig_ismaa_likelihood = plt.figure(dpi=200, figsize=(6, 4))
    ax_ismaa_likelihood = fig_ismaa_likelihood.add_subplot()
    ax_ismaa_likelihood.set_xlabel(r'$w_1$', fontsize=20)
    ax_ismaa_likelihood.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_ismaa_likelihood.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_ismaa_likelihood.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_ismaa_likelihood.set_ylim(ymin=0, ymax=1.1)
    ax_ismaa_likelihood.set_ylabel(r'$\mathbb{P}(r| w)$', fontsize=20)

    fig_ismaa_likelihood.subplots_adjust(top=0.99)
    fig_ismaa_likelihood.subplots_adjust(bottom=0.15)
    fig_ismaa_likelihood.subplots_adjust(left=0.12)
    fig_ismaa_likelihood.subplots_adjust(right=0.98)

    likelihood_ismaa = np.zeros(n_points_for_vis)

    # in order to get really precise results, we need a lot more samples that for Test Problem 3.1 plain
    n_samples_perfmat = 1000000

    prior_w_for_vis = np.zeros(n_points_for_vis)

    likelihood = Likelihood(perfmat_mean.shape[0], perfmat_mean.shape[1], np.array([2], dtype=int), mcda_method)

    iidx = 1

    # sample performance matrices according to the prior distribution (here: uniform)
    prior_perfmat = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)

    prior = Prior(prior_weights, prior_perfmat)

    samples = sample_posterior(prior, likelihood, n_samples_perfmat, n_chains=4, n_procs=4)[0]
    samples_posterior_perfmat = samples[1]

    perfmat_instances = prior_perfmat.sample(n_samples_perfmat)
    p_instances = np.zeros(perfmat_instances.shape)

    for i in range(n_samples_perfmat):
        p_instances[i] = mcda_method.p_from_perfmat(perfmat_instances[i])

    i = 0
    for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
        weight_test = np.array([w1, 1.0 - w1])
        prior_w_for_vis[i] = np.exp(prior_weights.log_density(weight_test))

        for imat in range(n_samples_perfmat):
            if np.argsort(-p_instances[imat]@weight_test)[0] == 0:
                likelihood_ismaa[i] += 1

        w1_vals[i] = w1
        i += 1

    likelihood_ismaa = likelihood_ismaa/n_samples_perfmat

    # posterior of the weights
    posterior_weights = prior_w_for_vis * likelihood_ismaa

    int_post = np.sum(posterior_weights)/n_points_for_vis
    posterior_weights = posterior_weights / int_post

    ax_ismaa_likelihood.plot(w1_vals, likelihood_ismaa, label=r'$P, \pm' + str(0.2) + '$',
              color=special_colors[1][iidx], linestyle=line_styles[iidx])

    ax_ismaa_posterior.plot(w1_vals, posterior_weights, label=r'$P, \pm' + str(0.2) + '$',
              color=special_colors[2][iidx], linestyle=line_styles[iidx])

    iidx += 1

    ax_ismaa_posterior.plot(w1_vals, prior_w_for_vis, label=r'prior',
              color='black', linestyle='solid')

    ax_ismaa_likelihood.legend()
    ax_ismaa_posterior.legend()
    fig_ismaa_likelihood.show()
    fig_ismaa_posterior.show()
    fig_ismaa_likelihood.savefig('TestProb3.3_ismaa_likelihood_weights.pdf')
    fig_ismaa_posterior.savefig('TestProb3.3_ismaa_posterior_weights.pdf')

    all_densities = np.empty(perfmat_mean.shape, dtype=object)
    # plot the marginal distributions of the performance matrix
    for i in range(perfmat_mean.shape[0]):
        for j in range(perfmat_mean.shape[1]):
            title = r'$P_{'+str(i+1)+','+str(j+1)+'}$'

            all_densities[i, j] = plt.figure(dpi=200, figsize=(4, 2.4))
            ax = all_densities[i, j].add_subplot()
            ax.set_title(title)
            ax.set_ylabel('density')

    # plot the marginal distributions of the performance matrix
    for i in range(perfmat_mean.shape[0]):
        for j in range(perfmat_mean.shape[1]):

            hist_posterior, edges_posterior = np.histogram(samples_posterior_perfmat[:, i, j], bins=25,
                                                           density=True)
            midpoints_posterior = 0.5 * (edges_posterior[:-1] + edges_posterior[1:])

            ax = all_densities[i][j].get_axes()[0]
            ax.set_ylim(0, 1.05*max(hist_posterior))
            if 1.05*max(hist_posterior) < 4:
                ax.set_yticks([0,1,2,3,4])
            elif 1.05 * max(hist_posterior) < 6:
                ax.set_yticks([0, 2, 4, 6])
            elif 1.05 * max(hist_posterior) < 8:
                ax.set_yticks([0, 2, 4, 6, 8])
            else:
                ax.set_yticks([0,2,4,6,8])

            ax.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)

            midpoints_prior = np.array(
                [max(0, perfmat_mean[i, j] + perfmat_params[i, j]), perfmat_mean[i, j] - perfmat_params[i, j]])

            height = 1.0 / (midpoints_prior[1] - midpoints_prior[0])
            hist_prior = np.array([height, height])

            ax = all_densities[i, j].get_axes()[0]
            ax.set_ylim(0, max(ax.get_ylim()[1], max(hist_prior)))
            ax.plot(midpoints_prior, hist_prior, label='prior', color='blue')

            # sort for plotting
            ax.plot(midpoints_posterior, hist_posterior, label=r'posterior',
                    color=special_colors[0][2], linestyle=line_styles[1])

            ax.legend()

            all_densities[i][j].show()
            all_densities[i, j].savefig('TestProb3.3_ISMAA_pmat_' + str(i+1) + '_' + str(j+1)+'.pdf')

            # density plots did not make it in the paper, so they are commented out
            #plot_est_density(samples_prior_perfmat[:, i, j], samples_posterior_perfmat[:, i, j], title,
            #                 'TestProb3.3_ISMAA_pmat_' + str(i+1) + '_' + str(j+1)+'.pdf')


def test_problem_3_2_qismaa_likelihood_and_posterior():
    perfmat_mean, perfmat_types, perfmat_params, all_probs_of_alts, prior_weights = tests(1)

    special_colors = [['darkorange', 'orangered', 'firebrick', 'saddlebrown', 'black'],
                      ['lightsteelblue', 'royalblue', 'blue', 'indigo', 'darkorchid'],
                      ['palegreen', 'limegreen', 'forestgreen', 'darkgreen', 'teal']]

    line_styles = ['solid', 'dashed', 'dashdot', (5, (10, 3)), 'solid']

    benefit = np.ones(perfmat_mean.shape[1], dtype=bool)
    mcda_method = SawSum(benefit)

    n_points_for_vis = 200
    w1_vals = np.zeros(n_points_for_vis)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'Times New Roman',
        "font.size": 16,
        "text.latex.preamble": r'\usepackage{amsfonts}'
    })

    fig_qismaa_posterior = plt.figure(dpi=200, figsize=(6, 4))
    ax_qismaa_posterior = fig_qismaa_posterior.add_subplot()
    ax_qismaa_posterior.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_posterior.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_posterior.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_posterior.set_ylabel(r'density', fontsize=20)

    fig_qismaa_posterior.subplots_adjust(top=0.99)
    fig_qismaa_posterior.subplots_adjust(bottom=0.15)
    fig_qismaa_posterior.subplots_adjust(left=0.12)
    fig_qismaa_posterior.subplots_adjust(right=0.98)

    fig_qismaa_likelihood = plt.figure(dpi=200, figsize=(6, 4))
    ax_qismaa_likelihood = fig_qismaa_likelihood.add_subplot()
    ax_qismaa_likelihood.set_xlabel(r'$w_1$', fontsize=20)
    ax_qismaa_likelihood.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1.0])
    ax_qismaa_likelihood.grid(axis='both', which='both', linestyle='dashed', linewidth=0.5)
    ax_qismaa_likelihood.set_ylim(ymin=0, ymax=1.1)
    ax_qismaa_likelihood.set_ylabel(r'$\mathbb{P}(r| w)$', fontsize=20)

    fig_qismaa_likelihood.subplots_adjust(top=0.99)
    fig_qismaa_likelihood.subplots_adjust(bottom=0.15)
    fig_qismaa_likelihood.subplots_adjust(left=0.12)
    fig_qismaa_likelihood.subplots_adjust(right=0.98)

    likelihood_qismaa = np.zeros(n_points_for_vis)

    n_samples_perfmat = 2000

    prior_w_for_vis = np.zeros(n_points_for_vis)

    shares_1 = np.array([0.3, 0.3, 0.4])
    shares_2 = np.array([0.2, 0.2, 0.6])
    shares_3 = np.array([0.1, 0.1, 0.8])
    shares = np.array([shares_1, shares_2, shares_3])

    iidx = 1

    # sample performance matrices according to the prior distribution (here: uniform)
    prior_perfmat = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)

    prior = Prior(prior_weights, prior_perfmat)

    samples_prior_perfmat = prior_perfmat.sample(min(n_samples_perfmat, 5000))

    all_densities = np.empty(perfmat_mean.shape, dtype=object)
    # plot the marginal distributions of the performance matrix
    for i in range(perfmat_mean.shape[0]):
        for j in range(perfmat_mean.shape[1]):
            title = r'$P_{'+str(i+1)+','+str(j+1)+'}$'

            all_densities[i, j] = plt.figure(dpi=200, figsize=(5, 3))
            ax = all_densities[i, j].add_subplot()
            ax.set_title(title)
            ax.set_ylabel('density')

    for ishares in range(shares.shape[0]):

        likelihood = LikelihoodQISMAA(perfmat_mean.shape[0], perfmat_mean.shape[1], shares[ishares], mcda_method,
                                      alpha=20.0, beta=0.98, p=2.0, q=2.0)

        samples = sample_posterior(prior, likelihood, n_samples_perfmat, n_chains=4, n_procs=4)[0]
        samples_posterior_perfmat = samples[1]

        likelihood_qismaa[:] = 0.0
        i = 0
        for w1 in np.linspace(0.0, 1.0, n_points_for_vis):
            weight_test = np.array([w1, 1.0 - w1])
            prior_w_for_vis[i] = np.exp(prior_weights.log_density(weight_test))

            for isample in range(min(n_samples_perfmat, 5000)):
                likelihood_qismaa[i] += np.exp(likelihood._log_density(weight_test, samples_prior_perfmat[isample])[0])

            w1_vals[i] = w1
            i += 1

        likelihood_qismaa = likelihood_qismaa/min(n_samples_perfmat, 5000)

        # posterior of the weights
        posterior_weights = prior_w_for_vis * likelihood_qismaa

        int_post = np.sum(posterior_weights)/n_points_for_vis
        posterior_weights = posterior_weights / int_post

        int_post = np.sum(prior_w_for_vis)/n_points_for_vis
        prior_w_for_vis = prior_w_for_vis / int_post


        label_aux = str(shares[ishares, 0])
        for alt in range(1, shares.shape[0]):
            label_aux = label_aux + ',' + str(shares[ishares, alt])

        ax_qismaa_likelihood.plot(w1_vals, likelihood_qismaa, label=r'$($' + label_aux + r'$)$',
                                  color=special_colors[1][iidx], linestyle=line_styles[iidx])

        ax_qismaa_posterior.plot(w1_vals, posterior_weights, label=r'$($' + label_aux + r'$)$',
                                 color=special_colors[2][iidx], linestyle=line_styles[iidx])

        iidx += 1

        # plot the marginal distributions of the performance matrix
        for i in range(perfmat_mean.shape[0]):
            for j in range(perfmat_mean.shape[1]):

                hist_posterior, edges_posterior = np.histogram(samples_posterior_perfmat[:, i, j], bins=25,
                                                               density=True)
                midpoints_posterior = 0.5 * (edges_posterior[:-1] + edges_posterior[1:])

                ax = all_densities[i][j].get_axes()[0]
                ax.set_ylim(0, max(ax.get_ylim()[1], max(hist_posterior)))

                # sort for plotting
                ax.plot(midpoints_posterior, hist_posterior, label=r'$($' + label_aux + r'$)$',
                        color=special_colors[0][ishares], linestyle=line_styles[ishares+1])

    # plot the marginal distributions of the performance matrix
    for i in range(perfmat_mean.shape[0]):
        for j in range(perfmat_mean.shape[1]):
            midpoints_prior = np.array([max(0, perfmat_mean[i, j] + perfmat_params[i, j]), perfmat_mean[i,j] - perfmat_params[i,j]])

            height = 1.0/(midpoints_prior[1]-midpoints_prior[0])
            hist_prior = np.array([height, height])

            ax = all_densities[i, j].get_axes()[0]
            ax.set_ylim(0, max(ax.get_ylim()[1], max(hist_prior)))
            ax.plot(midpoints_prior, hist_prior, label='prior', color='blue')
            ax.legend()

            all_densities[i][j].show()
            all_densities[i, j].savefig('Qismaa_test2_pmat_' + str(i+1) + '_' + str(j+1)+'.pdf')

    ax_qismaa_posterior.plot(w1_vals, prior_w_for_vis, label=r'prior', color='black', linestyle='solid')

    ax_qismaa_likelihood.legend()
    ax_qismaa_posterior.legend()
    fig_qismaa_likelihood.show()
    fig_qismaa_posterior.show()
    fig_qismaa_likelihood.savefig('TestProb3.3_qismaa_likelihood_weights.pdf')
    fig_qismaa_posterior.savefig('TestProb3.3_qismaa_posterior_weights.pdf')


def test_problem_3_3_ismaa():
    facecolors = ['orange', 'mediumblue', 'forestgreen', 'orange', 'black']

    perfmat_mean, perfmat_types, perfmat_params, all_probs_of_alts, prior_weights = tests(3)

    ranking = np.array([0])

    benefit = np.ones(perfmat_mean.shape[1], dtype=bool)
    mcda_method = SawSum(benefit)

    # sample performance matrices according to the prior distribution (here: uniform)
    prior_perfmat = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)

    prior = Prior(prior_weights, prior_perfmat)

    [nalts, ncrit] = perfmat_mean.shape

    verts_simplex = np.eye(ncrit)
    verts_simplex = np.vstack([verts_simplex, verts_simplex[:, 0]])
    verts_simplex_rot = standard_simplex_to_flat(verts_simplex)

    fig_w_r_flat = plt.figure(figsize=(4, 4), dpi=500)
    ax2 = fig_w_r_flat.add_subplot(111)
    ax2.set_aspect(1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim((-0.95, 0.55))
    ax2.set_ylim((-0.75, 0.75))
    ax2.set_frame_on(False)
    ax2.plot(verts_simplex_rot[:, 0], verts_simplex_rot[:, 1], c='black', linewidth=1)

    fig_w_r_flat.subplots_adjust(top=1.0)

    for ialt in range(nalts):
        r = np.array([ialt], dtype=int)

        vertices = verts_from_ranking(mcda_method.p_from_perfmat(perfmat_mean), r)

        if vertices.shape[0] > 1:
            triang = Delaunay(vertices[:, 0:2])

            vertices_rot_for_vis = standard_simplex_to_flat(vertices)

            for i in range(triang.simplices.shape[0]):
                ax2.fill(vertices_rot_for_vis[triang.simplices[i, :], 0],
                         vertices_rot_for_vis[triang.simplices[i, :], 1],
                         c=facecolors[ialt], alpha=0.5, edgecolor='none')


    # create likelihood functions
    # -- P fixed ---

    cog = 1/ncrit*np.ones(ncrit)

    # compute the bounds of the flattened simplex
    vertices_simplex = np.eye(3) - cog
    vertices_flat = standard_simplex_to_flat(vertices_simplex)
    vertices_flat = np.vstack((vertices_flat, vertices_flat[0, :].reshape(1, 2)))
    vertices_flat = np.hstack((vertices_flat, 0.0001*np.ones((4, 1))))

    lb_x = np.min(vertices_flat[:, 0])
    ub_x = np.max(vertices_flat[:, 0])
    lb_y = np.min(vertices_flat[:, 1])
    ub_y = np.max(vertices_flat[:, 1])

    # sample the variable performance matrix
    #n_samples_perfmat = 1000000
    n_samples_perfmat = 1000

    simplices = np.zeros((0, 3))
    vertices = np.zeros((0, 2))
    likelihood_p_fixed = np.zeros(0)
    vertices_offset = 0

    # get all W_r s
    for ialt in range(nalts):
        ranking_aux = np.array([ialt], dtype=int)
        vertices_aux = standard_simplex_to_flat(verts_from_ranking(mcda_method.p_from_perfmat(perfmat_mean), ranking_aux))
        if vertices_aux.shape[0] > 1:
            vertices = np.vstack((vertices, vertices_aux))

            triang = Delaunay(vertices_aux)
            simplices = np.vstack((simplices, triang.simplices + vertices_offset))

            vertices_offset += triang.npoints

            if ranking == ranking_aux:
                likelihood_aux = np.ones(triang.npoints)
            else:
                likelihood_aux = np.zeros(triang.npoints)

            likelihood_p_fixed = np.hstack((likelihood_p_fixed, likelihood_aux))

    simplices = np.hstack((simplices, 3 * np.ones((simplices.shape[0], 1), dtype=int)))

    export2vtu(vertices, simplices, likelihood_p_fixed, scalar_data_names=['P(w|r)'],
               filename='TestProb3_4_likelihood_p_fixed.vtu')

    # P uncertain
    nbins_per_dim = 100

    # binning for visualisation
    binx = np.linspace(lb_x, ub_x, nbins_per_dim+1)
    biny = np.linspace(lb_y, ub_y, nbins_per_dim+1)

    binx_vtu = 0.5*(binx[:-1] + binx[1:])
    biny_vtu = 0.5*(biny[:-1] + biny[1:])

    coordsx, coordsy = np.meshgrid(binx_vtu, biny_vtu)
    coordsx = coordsx.reshape(nbins_per_dim*nbins_per_dim)
    coordsy = coordsy.reshape(nbins_per_dim*nbins_per_dim)
    coords = np.vstack((coordsx, coordsy)).T
    coords_simplex = flat_to_standard_simplex(coords)

    coords = coords[np.all(coords_simplex >= 0.0, axis=1), :]
    likelihood_p_uncertain = np.zeros(coords.shape[0])

    coords_simplex = flat_to_standard_simplex(coords)

    timestamp = time.time()

    perfmat_instances = prior_perfmat.sample(n_samples_perfmat)
    p_instances = np.zeros(perfmat_instances.shape)

    for i in range(n_samples_perfmat):
        p_instances[i] = mcda_method.p_from_perfmat(perfmat_instances[i])

    # we need the prior on the current mesh to get the posterior
    prior_aux = np.zeros(coords.shape[0])
    for icoord in range(coords.shape[0]):
        prior_aux[icoord] = np.exp(prior_weights.log_density(coords_simplex[icoord]))

    for isample in range(n_samples_perfmat):

        icoords = np.argsort(-p_instances[isample]@coords_simplex.T, axis=0)[0] == ranking
        likelihood_p_uncertain[icoords] += 1

    likelihood_p_uncertain = likelihood_p_uncertain/n_samples_perfmat

    print('elapsed time:' + str(time.time() - timestamp))

    posterior_p_uncertain = prior_aux*likelihood_p_uncertain

    int_post = np.sum(posterior_p_uncertain) / coords.shape[0]
    posterior_p_uncertain = posterior_p_uncertain / int_post

    # provide a visualisation in the plane
    triang = Delaunay(coords)
    triang.simplices = np.hstack((triang.simplices, 3*np.ones((triang.nsimplex, 1), dtype=int)))
#    export2vtu(coords, triang.simplices, likelihood_p_uncertain, scalar_data_names=['likelihood, P unc'],
#               filename='TestProb3_4_likelihood_p_uncertain.vtu')

    export2vtu(coords, triang.simplices, posterior_p_uncertain, scalar_data_names=['post, P unc'],
               filename='TestProb3_4_posterior_p_uncertain.vtu')

    nbins_per_dim = 500

    # binning for visualisation
    binx = np.linspace(lb_x, ub_x, nbins_per_dim+1)
    biny = np.linspace(lb_y, ub_y, nbins_per_dim+1)

    binx_vtu = 0.5*(binx[:-1] + binx[1:])
    biny_vtu = 0.5*(biny[:-1] + biny[1:])

    coordsx, coordsy = np.meshgrid(binx_vtu, biny_vtu)
    coordsx = coordsx.reshape(nbins_per_dim*nbins_per_dim)
    coordsy = coordsy.reshape(nbins_per_dim*nbins_per_dim)
    coords = np.vstack((coordsx, coordsy)).T
    coords_simplex = flat_to_standard_simplex(coords)

    coords = coords[np.all(coords_simplex >= 0.0, axis=1), :]
    likelihood_p_uncertain = np.zeros(coords.shape[0])

    coords_simplex = flat_to_standard_simplex(coords)
    triang = Delaunay(coords)
    triang.simplices = np.hstack((triang.simplices, 3*np.ones((triang.nsimplex, 1), dtype=int)))

    # prior
    prior_vec = np.zeros(coords.shape[0])
    posterior_vec = np.zeros(coords.shape[0])

    pmat = mcda_method.p_from_perfmat(perfmat_mean)
    for icoord in range(coords.shape[0]):
        prior_vec[icoord] = np.exp(prior_weights.log_density(coords_simplex[icoord]))

        #if np.argsort(-pmat@coords_simplex[icoord])[0] == ranking:
        #    posterior_vec[icoord] = prior_vec[icoord]

    n_points_for_vis = coords.shape[0]
    icoords = np.argsort(-pmat@coords_simplex.T, axis=0)[0] == ranking
    posterior_vec[icoords] = prior_vec[icoords]

    # normalize both distros
    int_prior = np.sum(prior_vec) / n_points_for_vis
    prior_vec = prior_vec / int_prior

    int_post = np.sum(posterior_vec) / n_points_for_vis
    posterior_vec = posterior_vec / int_post

    export2vtu(coords, triang.simplices, prior_vec, scalar_data_names=['prior'],
               filename='TestProb3_4_prior.vtu')
    export2vtu(coords, triang.simplices, posterior_vec, scalar_data_names=['posterior, P fixed'],
               filename='TestProb3_4_posterior_p_fixed.vtu')


def test_problem_3_3_qismaa():

    n_samples_perfmat = 100
    beta = 1 - 1e-6

    perfmat_mean, perfmat_types, perfmat_params, all_probs_of_alts, prior_weights = tests(3)

    benefit = np.ones(perfmat_mean.shape[1], dtype=bool)
    mcda_method = SawSum(benefit)

    prior_perfmat = UncertainPerfMatPrior(perfmat_mean, perfmat_types, perfmat_params)

    # sample instances of the performance matrix according to the prior distribution of the performance matrix for
    # marginalisation
    p_instances = prior_perfmat.sample(n_samples_perfmat)

    nbins_per_dim = 100
    #nbins_per_dim = 500
    nalts = perfmat_mean.shape[0]
    ncrit = perfmat_mean.shape[1]

    cog = 1 / ncrit * np.ones(ncrit)

    # compute the bounds of the flattened simplex
    vertices_simplex = np.eye(3) - cog
    vertices_flat = standard_simplex_to_flat(vertices_simplex)
    vertices_flat = np.vstack((vertices_flat, vertices_flat[0, :].reshape(1, 2)))
    vertices_flat = np.hstack((vertices_flat, 0.0001 * np.ones((4, 1))))

    lb_x = np.min(vertices_flat[:, 0])
    ub_x = np.max(vertices_flat[:, 0])
    lb_y = np.min(vertices_flat[:, 1])
    ub_y = np.max(vertices_flat[:, 1])

    # binning for visualisation
    binx = np.linspace(lb_x, ub_x, nbins_per_dim)
    biny = np.linspace(lb_y, ub_y, nbins_per_dim)

    coordsx, coordsy = np.meshgrid(binx, biny)
    coordsx = coordsx.reshape(nbins_per_dim * nbins_per_dim)
    coordsy = coordsy.reshape(nbins_per_dim * nbins_per_dim)
    coords_flat = np.vstack((coordsx, coordsy)).T
    coords = flat_to_standard_simplex(coords_flat)

    coords_flat = coords_flat[
                  np.logical_and(np.logical_and(coords[:, 0] > 0, coords[:, 1] > 0), coords[:, 2] > 0), :]
    coords = coords[np.logical_and(np.logical_and(coords[:, 0] > 0, coords[:, 1] > 0), coords[:, 2] > 0), :]

    # prior
    # We could do this using scr.priors.weights.NormalWeightsPrior.log_density, but in this case, we would need to call
    # this function in a loop. This variant is just more efficient.
    prior_values = stats.multivariate_normal.pdf(coords_flat, prior_weights.mean_trans,
                                                 prior_weights.covariance_trans)

    int_prior = np.sum(prior_values) * (binx[1] - binx[0]) * (biny[1] - biny[0])
    prior_values = prior_values / int_prior

    n_points_for_vis = coords.shape[0]
    p_of_s_given_w_p_fixed = np.zeros((n_points_for_vis, len(all_probs_of_alts)), dtype=float)
    s_bar_given_w_p_fixed = np.zeros((n_points_for_vis, nalts), dtype=float)
    s_bar_given_w_p_uncertain = np.zeros((n_points_for_vis, nalts), dtype=float)

    p_posterior_p_fixed = np.zeros((n_points_for_vis, len(all_probs_of_alts)), dtype=float)
    p_posterior_p_uncertain = np.zeros((n_points_for_vis, len(all_probs_of_alts)), dtype=float)

    scalar_data_names = ['prior']
    scalar_data_likelihood_names = []
    iprob = 0

    p_mean = mcda_method.p_from_perfmat(perfmat_mean)

    for probs_of_alts in all_probs_of_alts:

        # approximate P(r|w) for any weight in coords. Save the number of occurrences that the given weight is in a
        # W_r in test_mat.

        likelihood_quant = LikelihoodQISMAA(perfmat_mean.shape[0], perfmat_mean.shape[1], probs_of_alts,
                                            mcda_method, alpha=20.0, beta=beta, p=2.0, q=2.0)

        for ipoint in range(n_points_for_vis):

            p_likelihood, s_bar = likelihood_quant._log_density(coords[ipoint, :], p_mean)
            p_of_s_given_w_p_fixed[ipoint, iprob] = np.exp(p_likelihood)

            if iprob == 0:
                s_bar_given_w_p_fixed[ipoint, :] = s_bar

            p_posterior_p_fixed[ipoint, iprob] = p_of_s_given_w_p_fixed[ipoint, iprob] * prior_values[ipoint]

        int_post = np.sum(p_posterior_p_fixed[:, iprob]) * (binx[1] - binx[0]) * (biny[1] - biny[0])
        p_posterior_p_fixed[:, iprob] = p_posterior_p_fixed[:, iprob]/int_post

        if iprob == 0:
            for ibar in range(s_bar_given_w_p_uncertain.shape[1]):
                scalar_data_names.append('s_bar_' + str(ibar+1))

        label_aux = str(probs_of_alts[0])
        for alt in range(1, probs_of_alts.shape[0]):
            label_aux = label_aux + ',' + str(probs_of_alts[alt])

        iprob += 1

        scalar_data_names.append('posterior, s = (' + label_aux + ')')
        scalar_data_likelihood_names.append('likelihood, s = (' + label_aux + ')')

    iprob = 0
    p_of_s_given_w_p_uncertain = np.zeros((n_points_for_vis, len(all_probs_of_alts)), dtype=float)
    for probs_of_alts in all_probs_of_alts:

        # approximate P(r|w) for any weight in coords. Save the number of occurrences that the given weight is in a
        # W_r in test_mat.

        likelihood_quant = LikelihoodQISMAA(perfmat_mean.shape[0], perfmat_mean.shape[1], probs_of_alts,
                                            mcda_method, alpha=20.0, beta=beta, p=2.0, q=2.0)

        i = 0
        for ipoint in range(n_points_for_vis):

            for imat in range(n_samples_perfmat):
                p_likelihood, s_bar = likelihood_quant._log_density(coords[ipoint, :], p_instances[imat])
                p_of_s_given_w_p_uncertain[ipoint, iprob] = (p_of_s_given_w_p_uncertain[ipoint, iprob] +
                                                             np.exp(p_likelihood))

                if iprob == 0:
                    s_bar_given_w_p_uncertain[ipoint, :] = s_bar_given_w_p_uncertain[ipoint, :] + s_bar

            i += 1

            p_posterior_p_uncertain[ipoint, iprob] = p_of_s_given_w_p_uncertain[ipoint, iprob] * prior_values[ipoint]

        int_post = np.sum(p_posterior_p_uncertain[:, iprob]) * (binx[1] - binx[0]) * (biny[1] - biny[0])
        p_posterior_p_uncertain[:, iprob] = p_posterior_p_uncertain[:, iprob]/int_post

        if iprob == 0:
            s_bar_given_w_p_uncertain = s_bar_given_w_p_uncertain/n_samples_perfmat

        label_aux = str(probs_of_alts[0])
        for alt in range(1, probs_of_alts.shape[0]):
            label_aux = label_aux + ',' + str(probs_of_alts[alt])

        iprob += 1

    p_of_s_given_w_p_uncertain = p_of_s_given_w_p_uncertain/n_samples_perfmat

    # provide a visualisation in the plane
    triang = Delaunay(coords_flat)
    triang.simplices = np.hstack((triang.simplices, 3 * np.ones((triang.nsimplex, 1), dtype=int)))

    scalar_data_names = scalar_data_names + scalar_data_likelihood_names

    export2vtu(coords_flat, triang.simplices, np.hstack((prior_values.reshape((prior_values.shape[0], 1)),
                                                         s_bar_given_w_p_fixed, p_posterior_p_fixed,
                                                         p_of_s_given_w_p_fixed)),
               scalar_data_names=scalar_data_names, filename='TestProb3.4_p_fixed_QUISMAA.vtu')

    export2vtu(coords_flat, triang.simplices, np.hstack((prior_values.reshape((prior_values.shape[0], 1)),
                                                         s_bar_given_w_p_uncertain, p_posterior_p_uncertain,
                                                         p_of_s_given_w_p_uncertain)),
               scalar_data_names=scalar_data_names, filename='TestProb3.4_p_uncertain_QUISMAA.vtu')



if __name__ == '__main__':
    test_problem_3_1_ismaa()
    test_problem_3_1_qismaa()
    test_problem_3_2_smaa()
    test_problem_3_2_ismaa_likelihood_and_posterior()
    comparison_testprob_3_1_3_2_qismaa()
    test_problem_3_3_qismaa()
    test_problem_3_3_qismaa()
