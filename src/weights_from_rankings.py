# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 19:16:06 2021


@author: Matthias Grajewski, FH Aachen University of Applied Sciences
"""
import numpy as np
import pypoman as pp
from scipy.optimize import minimize

from numba import jit, float64, int32


def hrep_from_ranking(P, ranking):
    """
    We assume a decision model m of the form m(w) = P*w, r = argsort(m(w)).
    It is well known that a set of admissible weights leading to a given
    ranking r is a convex polytope which can be expressed as {A*w <= b}, the
    so-called H-representation. This function computes that H-representation
    from P and the ranking r. Any row in that systems represents one half
    space which may result from the actual ranking or the side conditions on
    the weights. A and b are organized as follows:
    Let us denote the number of criteria by ncrit and the number of
    alternatives by nalts, and the number of alternatives represented in the
    ranking by nranks.
    At first, we code the ncrit conditions from the non-negativity of w, then,
    nranks-1 conditions from the actual ranking, then nalts - nranks from
    all inferior alternatives and at last 2 from the normalization
    (sum of weights is at least one and at most one)

    Parameters
    ----------
    P : 2D-numpy array
        matrix of the decision model (not necessarily the performance matrix!)

    ranking : 1D-numpy array
        possibly incomplete ranking (0-based)

    Returns
    -------
    A : 2D-numpy array
        matrix A representing the polytope due to Ax <= b
    b : 1D-numpy array
        vector b representing the polytope due to Ax <= b
    """
    eps = 1e-12


    # number of alternatives and number of criteria
    [nalts, ncrit] = P.shape

    nranks = ranking.shape[0]

    # compute the system of linear inequalities and store
    # them in the matrix A
    # number of total conditions: ncrit conditions from non-negativity,
    # nalts-1 from the actual ranking, 2 from the normalization (sum of weights
    # is at least one and at most one)
    nconds = nalts-1+ncrit+2
    A = np.zeros((nconds, ncrit))
    b = np.zeros(nconds)

    # conditions for non-negativity
    for i in range(ncrit):
        A[i,i] = -1

    # actual decision halfspaces
    for i in range(nranks-1):
        A[i+ncrit,:] = P[ranking[i+1],:]- P[ranking[i],:]

        # alternative r[i] should be at least eps better than r[i+1]
        b[i+ncrit] = -eps

    # ranking is incomplete: find out which alternatives are not explicitly mentioned in r
    if nranks < nalts:
        alts_not_in_r = np.zeros(nalts-nranks,dtype=int)
        j = 0
        for i in range(nalts):
            if not any(ranking == i):
                alts_not_in_r[j]= i
                j +=1

        # we know that all alternatives not mentioned r are inferior to the
        # last entry in r
        for i in range(nalts - nranks):
            A[i+nranks+ncrit-1,:] = P[alts_not_in_r[i],:]- P[ranking[-1],:]


    # normalization/being on the standard simplex
    A[-2,:] = np.ones((1,ncrit))
    A[-1,:] = -np.ones((1,ncrit))

    b[-2] = 1
    b[-1] = -1

    return A, b


def inscribe_maximal_ellipsoid(P,ranking, A_constraints=None, b_constraints=None):
    """
    This function returns the midpoint and the matrix for the volume-maximal
    ellipsoid inscribed in W_r, the set of all weights leading to the given
    ranking "ranking". The restriction to W_r can be reformulated as A*w <= b,
    the so-called H-representation of W_r.
    W_r can be further restricted by imposing additional linear
    inequality constraints on w. They are defined by
        A_constraints*w < b_constraints,
    which add to the linear constraints from restricting to W_r. Stacking all
    these inequalities leads to an enlarged system A*w <= b.

    Let the number of criteria be c. Following Boyd, chapter 8.4.2, we describe
    an ellipsoid by
        E = { B*x+w | ||x||=1}
    with a symmetric c x c -matrix and w being the midpoint. The optimal
    ellipsoid described by B_opt and w_opt is given by the solution of the
    convex minimisation problem
        minimize -log(det(B)) s.t. ||Ba_i||_2 + <a_i,w> <= b_i

    The optimal ellipsoid is known to be unique for convex polytopes. We need
    however to transform W_r to R^{c-1} first by x -> Q(x-cog), as W_r is a null
    set in R^c such that maximising volume therein is meaningless. Here, cog
    is the center of gravity of the standard simplex.
    We solve this optimisation problem numerically using scipy.opt.

    Sources: Boyd, Stephen and Vendenberghe, Lieven: Convex Optimisation,
    Cambridge University Press, 2004


    Parameters
    ----------
    P : 2D-numpy array
        matrix of the decision model (not necessarily the performance matrix!)
    ranking : 1D-numpy array
        possibly incomplete ranking
    A_constraints : 2D-numpy array, optional
        matrix for imposing additional linear constraints
    b_constraints : 1D-numpy-array, optional
        right hand side for imposing linear constraints

    Returns
    -------
    w_opt : 1D-numpy-array
        midpoint of the maximal inscribed ellipse aka the optimal weights for
        the given ranking and (optionally) additional constraints
    B_opt : 2D-numpy-array
        main axes of the maximal inscribed ellipse aka the main building
        block for the covariance matrix for the given ranking and (optionally)
        additional constraints
    """
    # number of alternatives and number of criteria
    [nalts, ncrit] = P.shape

    # consistency check
    if nalts < ranking.shape[0]:
        raise NameError('r contains more entries than there are alternatives.')

    # check P for duplicate rows
    check_P(P)

    # get the H-representation of W_r from P matrix and the ranking in R^c
    A, b = hrep_from_ranking(P,ranking)

    # consistency checks and incorporating additional linear constraints
    if (A_constraints is not None) and (b_constraints is not None):

        # test, if A_constraints has the right number of columns
        if A_constraints.shape[1] != P.shape[1]:
            err_msg = 'The number of criteria ('+ str(P.shape[1]) + \
            ') does not match the number of columns in A_constraints (' \
                + str(A_constraints.shape[1]) + ')'
            raise NameError(err_msg)

        # test, if the number of constraints in A_constraints and b_constraints
        # match
        if A_constraints.shape[0] != b_constraints.shape[0]:
            err_msg = 'The number of constraints in A_constraints (' + \
                str(A_constraints.shape[0]) + \
                ') does not match the number of constraints in b_constraints (' \
                    + str(b_constraints.shape[0]) +').'
            raise NameError(err_msg)

        # stack the matrices to incorporate the additional constraints
        A = np.vstack([A_constraints, A])
        b = np.hstack([b_constraints, b])

    elif (A_constraints is None) and (b_constraints is not None):
        raise NameError('A_constraints is given, but b_constraints not')
    elif (A_constraints is not None) and (b_constraints is None):
        raise NameError('A_constraints is not given, but b_constraints is')


    # shift the polytope by the center of gravity of the standard simplex
    cog = 1/ncrit*np.ones(ncrit)

    # we compute the arithmetic mean of the vertices in order to find a good
    # starting value for optimization. If this fails, we just take the center
    # of gravity of the standard simplex. This happens still in R^c.
    try:
        # compute the vertices of the polytope W_r
        vertices = pp.duality.compute_polytope_vertices(A,b)

        # convert to numpy-array
        vertices= np.array(vertices)

        nverts = vertices.shape[0]
        wmean = 1/nverts*np.matmul(vertices.T, np.ones((nverts,1))).flatten()

    except BaseException as exc:
        print('--- Taking the mean as starting point failed in inscribe_maximal_ellipsoid')
        print('cddlib error message: ' + str(exc))
        print()
        wmean = cog

    # Unfortunately, the polytope is degenerated, such that the
    # volume-maximal inscribed ellipsoid does not make sense here (the volume
    # of ANY inscribed ellipsoid is 0).
    # step one: shift the polytope such that it is contained in the hyperplane
    # <w,1> = 0 instead of <w,1> = 1.
    # step two: rotate to R^{c-1}
    # However, we have to do this in H-representation

    # shift to <w,1> = 0
    bhat = b - A@cog

    # rotate it such that the last coordinate is zero
    Q = get_q(ncrit)
    Ahat = A@Q

    # starting values for numerical optimization
    # B_init: downscaled unit matrix
    # w_init: mean of vertices transformed to R^{c-1}
    B_init = 0.01*np.eye(ncrit-1)
    w_init = (wmean-cog)@Q
    w_init = w_init[:-1]

    # this is the actual optimisation function
    w_opt, B_opt = fit_ellipsoid(Ahat[:-2,:-1], bhat[:-2], w_init, B_init)

    # transform the midpoint w_opt back to the simplex
    w_opt = np.hstack([w_opt, [0]])@Q.T + cog

    # transform the ellipsoid to the simplex
    B_opt_aux = np.zeros((ncrit,ncrit))
    B_opt_aux[0:ncrit-1,0:ncrit-1] = B_opt

    B_opt = Q@B_opt_aux
    return w_opt, B_opt


def verts_from_ranking(P, ranking, A_constraints=None, b_constraints=None):
    """
    This function computes the vertices of W_r.

    Parameters
    ----------
    P : 2D-numpy array
        matrix of the decision model (not necessarily the performance matrix!)
    ranking : 1D-numpy array
        possibly incomplete ranking
    A_constraints : 2D-numpy array, optional
        matrix for imposing additional linear constraints
    b_constraints : 1D-numpy-array, optional
        right hand side for imposing linear constraints

    Returns
    -------
    vertices : 2D-numpy-array
        the vertices of W_r
    """

    # number of alternatives and number of criteria
    [nalts, ncrit] = P.shape

    # consistency check
    if nalts < ranking.shape[0]:
        raise NameError('r contains more entries than there are alternatives.')

    # check that P does not contain duplicate rows
    check_P(P)

    # compute the H-representation of W_r
    A, b = hrep_from_ranking(P,ranking)

    # if there are properly given additional constraints
    if (A_constraints is not None) and (b_constraints is not None):

        # test, if A_constraints has the right number of columns
        if A_constraints.shape[1] != P.shape[1]:
            err_msg = 'The number of criteria (' + str(P.shape[1]) +  \
            ') does not match the number of columns in A_constraints (' \
                + str(A_constraints.shape[1]) + ').'
            raise NameError(err_msg)

        # test, if the number of constraints in A_constraints and b_constraints
        # match
        if A_constraints.shape[0] != b_constraints.shape[0]:
            err_msg = 'The number of constraints in A_constraints (' + \
                str(A_constraints.shape[0]) + \
                ') does not math the number of constraints in b_constraints ('\
                + str(b_constraints.shape[0]) + ').'
            raise NameError(err_msg)

        # stack the matrices to incorporate the additional constraints
        A = np.vstack([A_constraints, A])
        b = np.hstack([b_constraints, b])

    elif (A_constraints is None) and (b_constraints is not None):
        raise NameError('A_constraints is given, but b_constraints not')
    elif (A_constraints is not None) and (b_constraints is None):
        raise NameError('A_constraints is not given, but b_constraints is')

    num_incos = False

    # compute the vertices of the polytope W_r
    try:

        vertices = pp.duality.compute_polytope_vertices(A,b)

        # convert to numpy-array
        vertices= np.array(vertices)
    except BaseException as exc:
        print('--- Computing the vertices failed in weights_from_ranking. ---')
        print('cddlib error message: ' + str(exc))
        print()

        # This is a rather common bug: cddlib can not compute the vertices of
        # the polygon due to rounding issues. This does not necessarily mean
        # that the polygon does not have any vertices aka it is empty.
        if str(exc).find('Numerical inconsistency is found'):
            num_incos = True

        vertices = np.array([])

    # any "real" polytope has at least ncrit vertices (e.g. a triangle in 3D
    # as part of the simplex). If it has even less vertices, it is something
    # like a line inside the 3D-simplex. Any arbitrarily small perturbation of
    # of the components of w makes them leave that "line" by a chance of 100%.
    # Therefore, the r occurs with probability 0, i.e. never in practice.
    # of course, polytopes with more vertices can be degenerated as well (think
    # of three consecutive points on the aforementioned line), but we do not
    # check that here.
    if (vertices.shape[0] == 0 and not num_incos):
        print('The polytope is empty.')

    # the polytope is degenerated (has dimension ncrit-2 or even less)
    elif(vertices.shape[0] < ncrit and not num_incos):
        print('The polytope is degenerate and the given ranking extremely unlikely.')

    return vertices



def cheb_center_from_ranking(P, ranking, A_constraints=None, b_constraints=None):
    """
    This function computes the Chebyshev-center of the polytope W_r, the set of
    weights such that Pw leads to the given ranking of alternatives r.

    Parameters
    ----------
    P : 2D-numpy array
        matrix of the decision model (not necessarily the performance matrix!)
    ranking : 1D-numpy array
        possibly incomplete ranking
    A_constraints : 2D-numpy array, optional
        matrix for imposing additional linear constraints
    b_constraints : 1D-numpy-array, optional
        right hand side for imposing linear constraints

    Returns
    -------
    cheb_center : 1D-numpy-array
        Chebyshev-center of the polytope W_r defined by P and r
    """

    # number of alternatives and number of criteria
    [nalts, ncrit] = P.shape

    # consistency check
    if nalts < ranking.shape[0]:
        raise NameError('r contains more entries than there are alternatives.')

    # check that P does not contain duplicate rows
    check_P(P)

    # compute the H-representation of W_r
    A, b = hrep_from_ranking(P,ranking)

    # if there are properly given additional constraints
    if (A_constraints is not None) and (b_constraints is not None):

        # test, if A_constraints has the right number of columns
        if A_constraints.shape[1] != P.shape[1]:
            err_msg = 'The number of criteria (' + str(P.shape[1]) +  \
            ') does not match the number of columns in A_constraints (' \
                + str(A_constraints.shape[1]) + ').'
            raise NameError(err_msg)

        # test, if the number of constraints in A_constraints and b_constraints
        # match
        if A_constraints.shape[0] != b_constraints.shape[0]:
            err_msg = 'The number of constraints in A_constraints (' + \
                str(A_constraints.shape[0]) + \
                ') does not math the number of constraints in b_constraints ('\
                + str(b_constraints.shape[0]) + ').'
            raise NameError(err_msg)

        # stack the matrices to incorporate the additional constraints
        A = np.vstack([A_constraints, A])
        b = np.hstack([b_constraints, b])

    elif (A_constraints is None) and (b_constraints is not None):
        raise NameError('A_constraints is given, but b_constraints not')
    elif (A_constraints is not None) and (b_constraints is None):
        raise NameError('A_constraints is not given, but b_constraints is')


    # Unfortunately, the polytope is degenerated, such that the
    # Chebyshev-center does not make sense here (the product of the distances
    # to the hyperplanes is zero for ANY point in the polytope).
    # step one: shift the polytope such that it contained in the hyperplane
    # <w,1> = 0 instead of <w,1> = 1
    # However, we have to do this in the H-representation

    # shift the polytope by the center of gravity of the standard simplex
    cog = 1/ncrit*np.ones(ncrit)
    bhat = b - A@cog

    # rotate it such that the last coordinate is zero
    Q = get_q(ncrit)
    Ahat = A@Q

    # Chebyshev center
    # delete the last two columns of the matrix, as these force the last
    # coordinate to be zero. Instead, skip the last coordinate entirely.
    try:
        cheb_center = pp.polyhedron.compute_chebyshev_center(Ahat[:-2,:-1], bhat[:-2])

        # last step: back transformation to \sum w_i = 1
        cheb_center = np.append(cheb_center, 0)
        cheb_center = Q@cheb_center + cog


    # catch exception that computing the Chevyshev center raises an exception
    except BaseException as exc:
        print('--- Computing the Chebyshev center failed in weights_from_ranking. ---')
        print(str(exc))
        print()
        cheb_center = np.array([])

    return cheb_center


def map_B_to_x(B):
    """
    For inscribing the volume-maximal ellipsoid to the polytope {Ax <= b}, we
    maximise the function log(det(B)) subject to ||Ba_i||+a_i^T w <= b_i, where
    B is symmetric and w the center of the ellipsoid {Bx+w | ||x|| = 1}.
    Therefore, we optimize B and w simultaneously. However, standard
    optimization expect a real-valued function depending on a vector of
    variables. This function maps B to the auxiliary vector x which is fed into
    the target function. The first n entries of x represent w, all others
    belong to B.
    For details, we refer to
    Boyd, Stephen and Vandenberghe, Lieven: Convex Optimization, chapter 8.4.2,
    p. 414; Cambridge University Press.

    Parameters
    ----------
    B : 2D-numpy-array
        matrix B defining the ellipsoid

    Returns
    -------
    x : 1D-numpy-array
        B mapped to auxiliary vector x for numerical optimization
    """
    # get dimension
    ndim = B.shape[0]

    # length is n + n(n*1)/2, as B is symmetric: we need to store one half only
    x = np.zeros(int(ndim*(ndim+1)/2)+ndim)

    imin = 0
    imax = ndim
    imin = ndim
    imax = 2*ndim
    for i in range(ndim):
        x[imin:imax] = B[i:ndim,i]
        imin = imax
        imax = imax + ndim-i-1
    return x


def map_x_to_B(x):
    """
    Counterpart to map_B_to_x

    Parameters
    ----------
    x : 1D-numpy-array
        auxiliary vector

    Returns
    -------
    B : 2D-numpy-array
        matrix B (see description in map_B_to_x)
    """

    # length of x
    length_x = x.shape[0]

    #compute dimension from the the length of x
    ndim = int(-1.5 + np.sqrt(9/4 + 2*length_x))
    B = np.zeros((ndim,ndim))

    imin = 0
    imax = ndim
    imin = ndim
    imax = 2*ndim
    for i in range(ndim):
        B[i:ndim,i] = x[imin:imax]
        imin = imax
        imax = imax + ndim-i-1

    # B is symmetric. Above, we filled only one half of B.
    B[range(ndim), range(ndim)] = 0.5*B[range(ndim), range(ndim)]
    B = B+B.T
    return B


def fit_ellipsoid(A,b, w_init, B_init):
    """
    This function computes the volume-maximal ellipsoid E inscribed into the
    polytope Ax <= b, E = {Bx + w | ||x||_2 = 1}.
    Let the number of criteria be c. Following Boyd, chapter 8.4.2, we describe
    an ellipsoid by
        E = { B*x+w | ||x||=1}
    with a symmetric c x c -matrix and w being the midpoint. The optimal
    ellipsoid described by B_opt and w_opt is given by the solution of the
    convex minimisation problem
        minimize -log(det(B)) s.t. ||Ba_i||_2 + <a_i,w> <= b_i

    We code the target function in the internal function fellipse and the
    constraints in fellipse_constr

    Parameters
    ----------
    A  : 2D-numpy-array
         matrix A describing the polytope transformed to R^{c-1}
    b  : 1D-numpy-array
         vector b describing the polytope transformed to R^{c-1}
    w_init : 1D-numpy-array
         starting value for w
    B_init : 2D-numpy-array
         starting value for B

    Returns
    -------
    w_final : 1D-numpy-array
        center of the inscribed ellipsoid
    B_final : 2D-numpy-array
        Matrix of the volume-maximal ellipsoid
    """
    # target function for optimization (see description in map_B_to_x)
    def fellipse(x):

        B = map_x_to_B(x)
        return -np.log(np.linalg.det(B))

    # constraints for optimization (see description in map_B_to_x)
    def fellipse_constr(x):
        B = map_x_to_B(x)
        ndim = B.shape[0]
        BA = B@A.T
        norm_B_times_A_i=np.sqrt(abs(np.diag(BA.T@BA)))
        return -(norm_B_times_A_i+A@x[0:ndim])+b

    # this is how to code inequality constraints in scipy.optimize
    cons = [{"type": "ineq", "fun": fellipse_constr}]

    # get dimension from the shape of the starting value for B
    ndim = B_init.shape[0]

    # compute starting vector x_init from B_init and w_init (the first n entries in a
    # belong to w)
    x_init = map_B_to_x(B_init)
    x_init[0:ndim]= w_init

    # actual minimization using scipy.optimize
    xmin = minimize(fellipse, x_init, constraints=cons, tol=1e-16)

    # reconstruct the optimal B and w from the optimal x
    B_final = map_x_to_B(xmin.x)
    ndim = B_final.shape[0]
    w_final= xmin.x[0:ndim]
    return w_final, B_final


def check_points(P, ranking, points, A_constraints=None, b_constraints=None):
    """
    This function checks which points given in points lead to ranking ranking and
    have non-negative entries only. The ranking r max be incomplete.

    Parameters
    ----------
    P : 2D-numpy array
        matrix of the decision model (not necessarily the performance matrix!)
    ranking : 1D-numpy array
        possibly incomplete ranking
    points : 2D-numpy-array
        points to test if they lead to ranking `ranking`
    A_constraints : 2D-numpy array, optional
        matrix for imposing additional linear constraints
    b_constraints : 1D-numpy-array, optional
        right hand side for imposing linear constraints

    Returns
    -------
    check_points : 1D-numpy-array
        an entry is true, if the correponding point leads to ranking `ranking`
    """
    eps = -1e-12

    # compute the ranking for the given points in descending order
    scores = points@P.T
    obtained_rankings = np.argsort(-scores)

    nrankings = ranking.shape[0]

    # check if the points contain only non-negative entries
    contains_no_negatives = (points>eps).all(axis=1)

    # check, if eventual constraints are fulfilled
    if (A_constraints is not None) and (b_constraints is not None):

        constraints_fulfilled = (A_constraints@points.T - \
                                 np.outer(b_constraints, np.ones(points.shape[0]))<0 \
                                ).all(axis=0)

        # compare the rankings with r
        return np.all(obtained_rankings[:, 0:nrankings] == ranking, axis=1) & \
                contains_no_negatives & constraints_fulfilled

    else:
        # compare the rankings with r
        return np.all(obtained_rankings[:, 0:nrankings] == ranking, axis=1) & \
                contains_no_negatives


def get_q(dim):
    """
    For some applications, it is necessary to transform the simplex in R^dim to
    R^{dim-1} without distortions. This can be achieved by first shifting the
    simplex such that its center of gravity is zero and then performing an
    orthogonal transformation (rotation or mirroring or a combination of both).
    Such orthogonal transformation is usually given by an orthogonal matrix Q,
    which this function provides depending on the dimension dim.

    Parameters
    ----------
    dim : integer
        dimension of the weight space

    Returns
    -------
    Q: 2D-numpy-array
        This orthogonal matrix rotates the first dim-1 coordinates of R^{dim}
        to the hyperplane \sum x_i = 0

    """
    # the columns of Q contain the vertices of the standard simplex
    Q = np.eye(dim)

    # We shift these vertices by the center of gravity. Then, the shifted
    # vertices belong to the hyperplane \sum w_i = 0. The rank of Q is
    # dim-1 now, as the span of the columns of Q is the hyperplane \sum w_i = 0.
    Q = Q - np.ones((1,dim))/dim

    # We want to construct an orthonormal basis of the shifted simplex. To
    # do so, we compute a QR decomposition. Due to the triangular structure
    # of R, the span of the k first columns of A is the same as of the first
    # k of Q. Therefore, the columns of Q but the last one provide the
    # desired basis of the shifted simplex. The very last column is inside
    # its orthogonal complement and thus (1,...,1) up to normalization.
    Q = np.linalg.qr(Q)[0]

    return Q



def standard_simplex_to_flat(points):
    """
    This function transforms the points given in points on the standard
    simplex in R^{dim} to essentially R^{dim-1} using an orthogonal affine
    transformation.
    The counterpart of this function is flat_to_standard_simplex

    Parameters
    ----------
    points : 2D-numpy-array
        points on the standard simplex to be transformed to the first
        dim-1 components of R^{dim}.

    Returns
    -------
    points: 2D-numpy-array
        transformed points

    """
    dim = points.shape[1]

    # get matrix of the orthogonal transformation from R^{dim-1} to the simplex
    Q = get_q(dim)

    # center of gravity
    cog = np.ones((1,dim))/dim

    # shift points
    points = points- cog

    # rotate points
    points = points@Q

    # the last component is 0 anyway
    return points[:,:-1]



def flat_to_standard_simplex(points):
    """
    This function transforms the points given in points from R^{dim-1} to the
    standard simplex in R^dim using an orthogonal affine transformation.
    The counterpart of this function is standard_simplex_to_flat.

    Parameters
    ----------
    points : 2D-numpy array
        points in R^{dim-1}

    Returns
    -------
    points : 2D-numpy-array :
        transformed points on the standard simplex
    """
    [npoints, dim] = points.shape

    dim = dim +1
    points = np.append(points, np.zeros((npoints, 1)), axis=1)

    # get matrix of the orthogonal transformation from R^{dim-1} to the simplex
    Q = get_q(dim)

    # Q is orthogonal, such that Q^{-1} = Q^T
    points = np.matmul(points, Q.T)

    # center of gravity
    cog = np.ones((1,dim))/dim

    points = points + cog

    return points


def estimate_mu(points):
    """
    This function computes the arithmetic mean of the points given in `points`.
    It can be used to estimate the expectation value mu, provided that the
    points are uniformly distributed.

    Parameters
    ----------
    points : 2D-numpy-array
        sampling points

    Returns
    -------
    mu : 1D-numpy-array
        arithmetic mean
    """
    npoints, ndim = points.shape
    mu = 1/npoints*np.matmul(points.T, np.ones((ndim, npoints)))
    return mu


def bounding_box(vertices):
    """
    This function computes an axis-parallel bouding box of the convex hull of
    the vertices given in `vertices`.
    The function can be used as consistency check in finding representative
    weights from rankings.
    WARNING: As the bounding box is axis-parallel, the intervals the weights
    need to be contained in can be grossly overestimated.

    Parameters
    ----------
    vertices : 2D-numpy-array
        array of vertices of the polytope

    Returns
    -------
    bounding box : 2D-numpy-array
        the first row contains the maxima for all coordinates, the second the
        minima
    """
    if vertices.shape[0] > 0:
        maximum = np.max(vertices, axis=0)
        minimum = np.min(vertices, axis=0)
        return np.array([minimum, maximum])
    else:
        return np.array([])


def gaussian_on_simplex(mu, sigma, npoints):
    """
    This function provides npoints i.i.d. points on the simplex, normally
    distributed according to sigma around mu.

    Parameters
    ----------
    mu : 1D-numpy-array
        expectation value on the simplex
    sigma : 2D-numpy-array
        covariance matrix on the simplex
    npoints : int
        number of random points

    Raises
    ------
    NameError: raised in the case of inconsistent data

    Returns
    -------
    points : 2D-numpy-array
        random points on the standard simplex normally distributed
        around mu with covarianve matrix sigma
    """

    ndim_mu = mu.shape[0]
    ndim_sigma = sigma.shape[1]

    if ndim_mu != ndim_sigma:
        raise NameError('Dimensions of mu and sigma differ.')

    return np.random.multivariate_normal(mu, sigma, npoints)


def compute_margins(A, b, points):
    """
    This function computes the distances to the hyperplanes which contain the
    facets of the polytope given by Ax <= b for given points points. The larger
    the distances are, the larger the admissible perturbation of a point to
    remain in the polytope, i.e. the larger the margin for uncertainty.

    Consider the k-th facet with normal vector a_k (k-the row of A). Let be
    v an arbitrary point on that hyperplane <x, a_k> = b_k. Then, the
    distance of a point x to that hyperplane is the length of the orthogonal
    projection of the vector xv onto n, aka

        || <x-v, a_k>     ||   || <x,a_k> - <v, a_k>      ||
        || ---------- a_k || = || ------------------- a_k || =
        || <a_k,a_k>      ||   ||     <a_k, a_k>          ||

         | <x,a_k> - b_k |             |<x,a_k> - b_k|
      =  ----------------- ||a_k||  =  ---------------
             ||a_k||^2                     ||a_k||

    This however works only on the flattened simplex in R^{ncrit-1}, as
    otherwise the results may be wrong!

    Parameters
    ----------
    A : 2D-numpy array
        matrix A for the H-representation of the polytope
    b : 1D-numpy array
        vector b for the H-representation of the polytope
    points : 2D-array
        points for which to compute the distances to the hyperplanes

    Returns
    -------
    margins: 2D-numpy-array
        contains the absolute distances of the points to the
        hyperplanes of the polytope

    """
    # compute 1/|| a_k||
    aux = np.sqrt(1/np.diag(A@A.T))
    return aux*np.abs(b-A@points)



def check_P(P):
    """
    This function raises an error if the performance matrix contains duplicate
    rows, i. e. some alternatives are identical.

    Parameters
    ----------
    P : 2D-numpy-array
        matrix of the method (not necessarily the performance matrix!)

    Raises
    ------
    NameError
        If the performance matrix contains duplicate rows.

    Returns
    -------
    None.

    """
    counts = np.unique(P, return_counts=True, axis=0)[1]

    if np.any(counts > 1):
        raise NameError('P contains duplicate rows, i.e. some '\
                        + 'alternatives are identical.')



@jit(float64[:,:](float64[:,:], int32[:], float64, float64), nopython=True)
def p_from_ptilde_prom_2(ptilde, type_criterion, c, q=()):
    """
    This function computes from the raw performance matrix ptilde the
    Promethee-II-matrix P. For details, we refer to the external
    documentation.

    Parameters
    ----------
    ptilde : 2D-numpy-array
        raw performance matrix
    type_criterion: 1D-numpy-integer-array
        types of preference function
        criterion following Brans et al., How to select and how to rank
        projects: The PROMETHEE method, European Journal of Operational
        Research 24 (1986) 228-238, Table 1
        1: P(x) = 1, x > 0; 0 elsewhere (piecewise constant)
        2: not yet implemented
        3: P(x) = cx, x>0; 0 elsewhere (linear, truncated to 1)
        4: not yet implemented
        5: P(x) = max(0, min((d-q)/(p-q)*x, 1))
        6: P(x) = max(0, 1-exp(1/c^2 x^2)) (sigmoid function)
    c : float
        shape parameter for preference function (if a parameter p is
        required, it holds p = c)
    q : float, optional
        shape parameter for preference function 5

    Returns
    -------
    P: 2D-numpy array
        Promethee-II-matrix
    """
    # number of alternatives and number of criteria
    (nalts, ncrit) = np.shape(ptilde)

    phat = np.empty((nalts, nalts, ncrit))
    P = np.empty(ptilde.shape)
    aux_nalts = np.ones(nalts) # it is favourable to preallocate that vector

    #compute d_ikj = P(i,j) - P(k,j)
    for icrit in range(ncrit):

        aux2 = np.outer(ptilde[:,icrit], aux_nalts)
        phat[:,:,icrit] = aux2 - aux2.T


    # compute f_j(d_ikj)
    for icrit in range(ncrit):

        if type_criterion[icrit] == 1:

            aux = phat[:,:,icrit]

            for i in range(nalts):
                auxAux = aux[:,i]
                auxAux[auxAux > 0] = 1
                auxAux[auxAux < 0.1] = 0
                aux[:,i] = auxAux

            # the variant below is more elegant, of course, but Numba only
            # supports advanced indexing for 1d-arrays
            #aux[aux > 0] = 1
            #aux[aux < 0.1] = 0

            phat[:,:,icrit] = aux - aux.T

        elif type_criterion[icrit] == 3:
            phat[:,:,icrit] = np.maximum(0, np.minimum(1, c*phat[:,:,icrit]))
            phat[:,:,icrit] = phat[:,:,icrit] - phat[:,:,icrit].T

        elif type_criterion[icrit] == 5:
            phat[:,:,icrit] = np.maximum(0, np.minimum(1, 1/(c-q)*(phat[:,:,icrit] - q)))
            phat[:,:,icrit] = phat[:,:,icrit] - phat[:,:,icrit].T

        elif type_criterion[icrit] == 6:

            aux = np.maximum(phat[:,:,icrit],0)
            aux2 = 1 - np.exp(-aux*aux/(c**2))
            phat[:,:,icrit] = aux2 - aux2.T

        else:
            raise NameError('Invalid choice of criterion inside Promethee II')

    # sum over alternatives
    P = np.sum(phat, 1)

    # scale P accordingly
    P = 1/(nalts-1)*P

    return P


def p_from_ptilde_saw(ptilde):
    """
    This function computes from the raw performance matrix ptilde the
    SAW-matrix P. For SAW, this restricts to scaling the matrix
    columns such that the column-sum is 1.

    Parameters
    ----------
    ptilde : 2D-numpy-array
        raw performance matrix

    Returns
    -------
    P: 2D-numpy-array
        (non)-normalized SAW-matrix
    """

    # check, if there are duplicate rows in P
    counts = np.unique(ptilde, axis=0, return_counts=True)[1]
    if any(counts > 1):
        raise NameError('Ptilde contains duplicate rows, i.e. ' + \
                        'some alternatives are identical.')

    # normalize P
    #column_norm = np.matmul(Ptilde.T, np.ones((nalts,1))).flatten()
    #P = np.matmul(Ptilde, np.diag(np.reciprocal(column_norm)))

    P = ptilde
    return P
