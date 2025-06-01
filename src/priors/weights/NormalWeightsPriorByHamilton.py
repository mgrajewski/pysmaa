"""
@author: Janis Papewalis

This file is part of the pysmaa python package, available at https://github.com/mgrajewski/pysmaa .
"""

import numpy as np
import time
from numpy.typing import NDArray
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from numba import jit, float64, int64, boolean


class NormalWeightsPriorByHamilton:
    """
    A class to represent a multivariate normal prior distribution. It uses Gibbs sampling as described
    in https://doi.org/10.1109/SSP.2014.6884588
    Parameters
    ----------
    mean : 1D-numpy-array
        the mean
    covariance : 2D-numpy-array
        the covariance matrix
    """

    def __init__(self, mean: NDArray[float], covariance: NDArray[float]) -> None:
        self.mean = mean
        self.covariance = covariance
        self.covariance_inverse = np.linalg.inv(covariance)


    def potential_function(self, q: NDArray[float]) -> NDArray[float]:
        """
        This is the potential function U in the paper mentioned above.
        """
        left_side = 0.5 * np.transpose(change_of_variables_up(q) - self.mean) @ self.covariance_inverse @ (
                change_of_variables_up(q) - self.mean)
        sum_on_right = 0
        for i in range(q.shape[0]):
            sum_on_right += (q.shape[0] - i) * np.log(q[i])
        return left_side - sum_on_right



    def gradient_of_potential_old(self, q: NDArray[float]) -> NDArray[float]:
        """
        This is the gradient of the potential function U. It is used to calculate the leapfrog steps.
        """
        r = q.shape[0]+1
        jacobi = np.zeros((r, r - 1))

        jacobi[0][0] = -1

        # diagonal entries
        for i in range(1, r - 1):
            jacobi[i][i] = -np.prod(q[:i])

        # subdiagonal entries w/o last row
        for i in range(r-1):
            for j in range(i):
                jacobi[i][j] = -jacobi[i][i] * (1-q[i]) / q[j]

        # last row
        for i in range(r-1):
            if q[i] != 0:
                jacobi[r-1][i] = np.prod(q)/q[i]

        right_side_term = np.zeros(r-1)
        for i in range(r-1):
            right_side_term[i] = (r-i-2)/q[i]

        return np.transpose(jacobi) @ self.covariance_inverse @ (change_of_variables_up(q) - self.mean) - right_side_term

    def gradient_of_potential(self, q: NDArray[float]) -> NDArray[float]:
        """
        This is the gradient of the potential function U. It is used to calculate the leapfrog steps.
        """

        return gradient_of_potential_jit(self.covariance_inverse, self.mean, q)

    def hamiltonian(self, q: NDArray[float], p: NDArray[float]) -> NDArray[float]:
        """
        Compute the Hamiltonian function of p and q, the sum of the potential and the kinetic function K.

        Parameters
        ----------
        q : 1D-numpy-array
        p : 1D-numpy-array

        Returns
        -------
        H(p,q): 1D-numpy-array
        """
        return self.potential_function(q) + 0.5 * p.transpose() @ p


    def proposal(self, n_leapfrog_steps: int, q: NDArray[float], eps: float) -> tuple[
        NDArray[float], NDArray[float], NDArray[float], float]:
        """
        Compute a proposal using the leapfrog integrator for Hamiltonian Monte Carlo.

        Parameters
        ----------
        n_leapfrog_steps : int
            Number of leapfrog steps to perform
        q : NDArray[float]
            Current position vector
        eps : float
            Step size for the leapfrog integrator

        Returns
        -------
        tuple[NDArray[float], NDArray[float], NDArray[float]]
            A tuple containing:
            - p_tilde: Initial momentum sampled from standard normal distribution
            - q_new: Proposed new position after leapfrog steps
            - p_new: Proposed new momentum after leapfrog steps
        """
        return proposal_jit(self.covariance_inverse, self.mean, n_leapfrog_steps, q, eps)

    def acceptance_prob(self, q_current: NDArray[float], p_current: NDArray[float],
                        q_star: NDArray[float], p_star: NDArray[float]) -> float:
        """
        This function takes in the current value of the random walk and the proposal values in and gives out the
        probablity of the walk to accept it.

        Parameters:
            q_current: 1D-numpy-array, current position vector
            p_current: 1D-numpy-array, current momentum vector
            q_star   : 1D-numpy-array, proposed new position vector
            p_star   : 1D-numpy-array, proposed new momentum vector

        Returns:
            probability: float, self-explanatory
        """

        current_hamiltonian = self.hamiltonian(q_current, p_current)
        proposed_hamiltonian = self.hamiltonian(q_star, p_star)

        return min(1.0, np.exp(current_hamiltonian - proposed_hamiltonian))


    def sample(self, n_samples: int) -> NDArray[float]:
        """
        This function computes n_samples samples from the (truncated) multivariate normal distribution. The distribution
        is truncated as only admissible weights are sampled.

        Parameters
        ----------
        n_samples : number of samples

        Returns
        -------
        samples: 2d-numpy array
            sampled admissible weights
        """
        eps = 1e-2
        n_leapfrog_steps = 50
        burn_in = 10
        thinning = 5

        if n_samples < 0:
            raise NameError('The number of samples must be at least 0.')

        # Initialize with the starting point
        current_sample = change_of_variables_down(self.mean)

        # Create array to store results (samples in unit cube)
        total_samples = burn_in + (n_samples * thinning)
        cube_samples = np.zeros((total_samples + 1, len(current_sample)))
        cube_samples[0] = current_sample

        accepted = 0
        total = 0

        leapfrog_steps = 0
        leapfrog_steps_outside = 0


        # Main sampling loop
        for i in range(total_samples):
            # Generate proposal
            p_tilde, q_new, p_new, n_step_outside_this = self.proposal(n_leapfrog_steps, cube_samples[i], eps)
            total += 1
            leapfrog_steps += n_leapfrog_steps
            leapfrog_steps_outside += n_step_outside_this

            # Compute acceptance probability and decide
            accept_prob = self.acceptance_prob(
                q_current=cube_samples[i],
                p_current=p_tilde,
                q_star=q_new,
                p_star=p_new
            )

            if random_boolean(accept_prob):
                cube_samples[i + 1] = q_new
                accepted += 1
            else:
                cube_samples[i + 1] = cube_samples[i]  # Stay at current position

        # Calculate acceptance rate
        acc_rate = accepted / total

        outside_rate = leapfrog_steps_outside / n_leapfrog_steps

        # Apply burn-in and thinning
        final_cube_samples = cube_samples[burn_in + 1::thinning][:n_samples]

        # Transform samples from the unit cube back to simplex
        simplex_samples = np.zeros((n_samples, len(current_sample) + 1))
        for i in range(n_samples):
            simplex_samples[i] = change_of_variables_up(final_cube_samples[i])

        return simplex_samples, acc_rate, outside_rate


@jit(float64[:](float64[:]), nopython=True)
def change_of_variables_down(x: NDArray[float]) -> NDArray[float]:
    """
    Inverse of the change of variables function. Maps from the simplex in IR^r
    back to the unit cube in IR^(r-1).

    Parameters
    ----------
    x : NDArray[float]
        A point on the simplex in IR^r (where sum(x) = 1 and all x_i >= 0)

    Returns
    -------
    NDArray[float]
        The corresponding point in the unit cube in IR^(r-1)
    """
    r = len(x)
    z = np.zeros(r - 1)

    for i in range(r - 1):
        # Calculate the sum of x from index i to the end
        sum_from_i = np.sum(x[i:])

        # Calculate the sum of x from index i+1 to the end
        sum_from_i_plus_1 = np.sum(x[i + 1:])

        # z_i = (sum of x from i+1 to r) / (sum of x from i to r)
        if sum_from_i > 0:  # Avoid division by zero
            z[i] = sum_from_i_plus_1 / sum_from_i
        else:
            z[i] = 0  # Default value if denominator is zero (shouldn't happen for valid simplexes)

    return z


@jit(float64[:](float64[:]), nopython=True)
def change_of_variables_up(q: NDArray[float]) -> NDArray[float]:
    """
    This is the change of variables a <-> z proposed in the paper mentioned above. It maps the unit cube in
    IR^(R-1) onto the simplex living in IR^R.
    """

    r = q.shape[0] + 1
    output = np.zeros(r)

    for i in range(r):
        if i == r-1:
            output[i] = np.prod(q)
        else:
            output[i] = np.prod(q[:i]) * (1 - q[i])
    return output


def random_boolean(p):
    """
    Returns True with probability p, False with probability 1-p
    """
    return np.random.random() < p


@jit(float64[:](float64[:, :], float64[:], float64[:]), nopython=True, cache=True)
def gradient_of_potential_jit(covariance_inverse: NDArray[float], mean: NDArray[float], q: NDArray[float]) -> NDArray[
    float]:
    """
    This is the gradient of the potential function U. It is used to calculate the leapfrog steps.
    """
    r = q.shape[0] + 1

    p = np.zeros(r - 1)
    p[0] = 1.0
    for i in range(1, r - 1):
        p[i] = np.prod(q[:i])

    # Submatrix initialisieren
    sub = np.zeros((r - 1, r - 1))

    # Diagonalelemente setzen
    for i in range(r - 1):
        sub[i, i] = -p[i]

    # Subdiagonalelemente
    for i in range(r - 1):
        for j in range(i):
            sub[i, j] = p[j] * (1 - q[j]) / q[i]

    # Letzte Zeile berechnen
    last_row = np.zeros(r - 1)
    for i in range(r - 1):
        last_row[i] = np.prod(q) / q[i]

    # Jacobi-Matrix zusammensetzen
    J = np.zeros((r, r - 1))
    J[:r - 1, :] = sub
    J[r - 1, :] = last_row

    # Rechte Seite
    right = np.zeros(r - 1)
    for i in range(r - 1):
        right[i] = (r - i - 2) / q[i]

    # Den Gradienten berechnen
    transformed_q = change_of_variables_up(q)
    diff = transformed_q - mean

    # Matrixmultiplikation manuell durchführen für JIT-Kompatibilität
    temp = np.zeros(r)
    for i in range(r):
        for j in range(r):
            temp[i] += covariance_inverse[i, j] * diff[j]

    gradient = np.zeros(r - 1)
    for i in range(r - 1):
        for j in range(r):
            gradient[i] += J[j, i] * temp[j]

    return gradient - right


@jit(nopython=True, cache=True)
def proposal_jit(covariance_inverse, mean, n_leapfrog_steps, q, eps):
    """
    Compute a proposal using the leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    n_leapfrog_steps : int
        Number of leapfrog steps to perform
    q : NDArray[float]
        Current position vector
    eps : float
        Step size for the leapfrog integrator

    Returns
    -------
    tuple[NDArray[float], NDArray[float], NDArray[float]]
        A tuple containing:
        - p_tilde: Initial momentum sampled from standard normal distribution
        - q_new: Proposed new position after leapfrog steps
        - p_new: Proposed new momentum after leapfrog steps
    """
    r = q.shape[0] + 1

    # Define lower and upper bounds
    lower_bound = np.zeros(r - 1)
    upper_bound = np.ones(r - 1)

    # Sample initial momentum from standard normal distribution
    p_new = np.random.normal(0, 1, size=r - 1)
    p_tilde = p_new.copy()
    q_new = q.copy()
    n_steps_outside = 0

    # Perform leapfrog integration
    for _ in range(n_leapfrog_steps):
        p_halfstep = p_new - gradient_of_potential_jit(covariance_inverse, mean, q_new) * eps / 2
        q_new_prime = q_new + eps * p_halfstep

        while np.any(q_new_prime >= upper_bound) or np.any(q_new_prime <= lower_bound):
            n_steps_outside += 1
            # Handle upper bound violations
            upper_violations = q_new_prime >= upper_bound
            if np.any(upper_violations):
                for i in range(r - 1):
                    if upper_violations[i]:
                        q_new_prime[i] = upper_bound[i] - (q_new_prime[i] - upper_bound[i])
                        p_halfstep[i] = -p_halfstep[i]

            # Handle lower bound violations
            lower_violations = q_new_prime <= lower_bound
            if np.any(lower_violations):
                for i in range(r - 1):
                    if lower_violations[i]:
                        q_new_prime[i] = lower_bound[i] + (lower_bound[i] - q_new_prime[i])
                        p_halfstep[i] = -p_halfstep[i]

        q_new = q_new_prime

        p_new = p_halfstep - gradient_of_potential_jit(covariance_inverse, mean, q_new) * eps / 2

    return p_tilde, q_new, p_new, n_steps_outside

def plot_3d_simplex(samples):
    """
    Plot samples on a 2D simplex (triangle) in 3D space.

    Parameters
    ----------
    samples : ndarray
        Array of shape (n_samples, 3) representing points on the 2D simplex
    """
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the vertices of the triangle in 3D space
    vertices = np.array([
        [1, 0, 0],  # (1,0,0)
        [0, 1, 0],  # (0,1,0)
        [0, 0, 1]  # (0,0,1)
    ])

    # Draw the edges of the triangle
    edges = [(0, 1), (1, 2), (2, 0)]
    for i, j in edges:
        ax.plot([vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]], 'k-')

    # Fill the triangle with a surface
    triangle_vertices = np.array([vertices[0], vertices[1], vertices[2]])
    triangles = [(0, 1, 2)]
    ax.plot_trisurf(triangle_vertices[:, 0], triangle_vertices[:, 1],
                    triangle_vertices[:, 2], triangles=triangles,
                    alpha=0.2, color='gray')

    # Plot the samples directly in 3D using their barycentric coordinates
    # (since in this case, barycentric coordinates are the same as 3D Cartesian)
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
               c='blue', s=50, alpha=0.7, label='Samples')

    # Label the vertices
    ax.text(1, 0, 0, "(1,0,0)", fontsize=10)
    ax.text(0, 1, 0, "(0,1,0)", fontsize=10)
    ax.text(0, 0, 1, "(0,0,1)", fontsize=10)

    # Label the sample points
    for i, sample in enumerate(samples):
        #label = f"P{i + 1}: ({sample[0]:.2f}, {sample[1]:.2f}, {sample[2]:.2f})"
        label = f""
        ax.text(sample[0], sample[1], sample[2], label, fontsize=8)

    # Set labels and title
    ax.set_title('Samples on 2D Simplex (Triangle) in 3D Space')
    ax.set_xlabel('X (Coordinate 1)')
    ax.set_ylabel('Y (Coordinate 2)')
    ax.set_zlabel('Z (Coordinate 3)')

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Set a good viewing angle
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    return fig



def main():
    # Erstellen von Testdaten
    n_dimensions = 18# Anzahl der Dimensionen für die Gewichte
    mean = np.ones(n_dimensions)/n_dimensions  # Gleichverteilte Gewichte als Mittelwert
    covariance = np.eye(n_dimensions) * 0.08  # Einfache Kovarianzmatrix


    start = time.time()
    # Instanz der Klasse erstellen
    prior = NormalWeightsPriorByHamilton(mean, covariance)

    n_jobs = 4
    results = Parallel(n_jobs = n_jobs)(
        delayed(prior.sample)(n_samples=5025) for _ in range(n_jobs)
    )
    samples_total = np.vstack([res[0] for res in results])
    acc_rate_total = np.mean([res[1] for res in results])
    outside_rate_total = np.mean([res[2] for res in results])

    end = time.time()
    print(samples_total.shape)
    print(samples_total, acc_rate_total, outside_rate_total)
    runtime_seconds = end - start
    minutes = int(runtime_seconds // 60)
    seconds = int(runtime_seconds % 60)
    runtime_formatted = f"{minutes:02d}:{seconds:02d}"

    print(f"Sampling runtime: {runtime_formatted}")

    # Plot the samples
    fig = plot_3d_simplex(samples_total)
    plt.show()

if __name__ == "__main__":
    main()