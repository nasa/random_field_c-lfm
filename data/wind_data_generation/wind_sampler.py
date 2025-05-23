from UQpy.stochastic_process import SpectralRepresentation
import numpy as np

DEFAULT_DECAY_COEFFICIENTS = [3.0, 3.0, 0.5]


class BasicWindSampler:
    """
    Simplified wind sampler that just generates Vx component of velocity and uses simplified exponetial decay coherence function.

    Implements the approach in "Monte Carlo simulation of wind velocity
    fields on complex structures" by L. Carassale and G. Solari in Journal of Wind Engineering and Industrial Aerodynamics (2006). Follows this paper for the formulation of the cross spectral density matrix (CSDM) and uses the spectral representation method (SRM) from UQpy to generate Monte Carlo samples of wind fields.

    Assumptions:
        - default parameters in class follow the paper and assume units of meters
        - the wind direction is assumed to be the angle wrt to the x1/x2 axes and so there is no vertical component of mean wind. The direction is also assumed to be constant in space. This translates to the local reference system (eq 7), decay coefficient tensors (eq 7), and rotation matrices (eq 9) all being constant across space.
    """

    def __init__(
        self,
        spatial_coords,
        freqs,
        shear_velocity=1.9,
        roughness_length=0.015,
        alpha=1.0,
        beta=1.0,
        lambda_=6.868,
        decay_coefficients=None,
    ):
        """
        spatial_coords: Nx3 array of x,y,z coord grid
        freqs: array of discretized frequency range with uniform spacing
        shear_velocity: u_star parameter in eq. 2 that defines the magnitude of the mean wind velocity
        roughness_length: z_0 parameter in eq. 2 (if height <= z_0, then wind velocity = 0)
        alphas: param that defines the vx wind variance above eq. 30 (array of length 3)
        beta:  param that defines the vx turbulence length scale in eq. 30
        lambda_: param that defines the the auto power spectrum for vx in eq. 5
        """
        self.spatial_coords = spatial_coords
        self.num_freq_steps = len(freqs)
        self.delta_w = freqs[1] - freqs[0]
        if decay_coefficients is None:
            decay_coefficients = DEFAULT_DECAY_COEFFICIENTS

        self.mean_wind_vel = self._get_mean_wind_velocity(
            shear_velocity, roughness_length, spatial_coords
        )
        variance = self._get_variance(shear_velocity, roughness_length, alpha)
        length_scale = self._get_turbulence_length_scales(
            shear_velocity, beta, spatial_coords
        )
        print("getting power spectral density matrix...")
        self.S_uqpy = self._get_cross_power_spectral_density_matrix(
            lambda_,
            variance,
            length_scale,
            self.mean_wind_vel,
            spatial_coords,
            freqs,
            decay_coefficients,
        )

    def sample(self, num_samples, delta_t, num_time_steps):
        """
        num_samples: number of wind samples to generate
        delta_t: time step
        num_time_steps: number of time steps

        Returns num_samples x N x num_time_steps array of wind samples
        """
        print("Generating samples...")
        SRM_object = SpectralRepresentation(
            num_samples,
            self.S_uqpy,
            delta_t,
            self.delta_w,
            num_time_steps,
            self.num_freq_steps,
        )
        samples = self.mean_wind_vel[np.newaxis, :, np.newaxis] + SRM_object.samples
        return samples

    # Equation 2:
    def _get_mean_wind_velocity(self, u_star, z_0, x_coords):
        mean_vel = 2.5 * u_star * np.log(x_coords[:, 2] / z_0)
        mean_vel[np.where(x_coords[:, 2] < z_0)] = 0
        return mean_vel

    # Equation 30:
    def _get_variance(self, u_star, z_0, alpha):
        sigma_2 = (6.0 - 1.1 * np.arctan(np.log(z_0) + 1.75)) * u_star**2 * alpha
        return sigma_2

    # Equation 30:
    def _get_turbulence_length_scales(self, z_0, beta, x_coords):
        L = beta * 300 * (x_coords[:, 2] / 200) ** (0.67 + 0.05 * np.log(z_0))
        return L

    # Equation 5:
    def _get_auto_spectrum(
        self, lambda_, variance, length_scale, mean_velocity, x_coords, freqs
    ):
        auto_spectrum_j = np.zeros((len(x_coords), len(freqs)))
        print("getting auto spectrum")
        for i, _ in enumerate(x_coords):
            mean_velocity_i = mean_velocity[i]
            length_scale_i = length_scale[i]
            for j, freq in enumerate(freqs):
                auto_spectrum_j[i, j] = self._get_auto_spectrum_for_x_n(
                    freq, lambda_, variance, length_scale_i, mean_velocity_i
                )
        return auto_spectrum_j

    # Equation 5 pointwise:
    def _get_auto_spectrum_for_x_n(
        self, freq, lambda_, variance, length_scale, mean_vel
    ):
        u_mag = np.linalg.norm(mean_vel)
        numerator = variance * lambda_ * freq * (length_scale / u_mag)
        denominator = freq * (1 + 1.5 * freq * lambda_ * length_scale / u_mag) ** 5 / 3
        return numerator / denominator

    # Equation 6:
    def _get_coherence_function(self, mean_velocity, freqs, x_coords, C):
        # h, k, n
        coh = np.zeros((len(x_coords), len(x_coords), len(freqs)))
        print("getting coherence")
        for h, x_h in enumerate(x_coords):
            print(f"\t coord {h} / { len(x_coords)}")
            mean_vel_h = mean_velocity[h]
            for k, x_k in enumerate(x_coords):
                mean_vel_k = mean_velocity[k]
                for n, freq in enumerate(freqs):
                    coh[h, k, n] = self._get_coherence_function_for_x_n(
                        freq, x_h, x_k, mean_vel_h, mean_vel_k, C
                    )
        return coh

    # Equation 6 pointwise:
    def _get_coherence_function_for_x_n(
        self, freq, x_coord_h, x_coord_k, mean_vel_h, mean_vel_k, C
    ):
        numerator = freq * np.linalg.norm(np.multiply(C, x_coord_h - x_coord_k))
        denominator = np.linalg.norm(mean_vel_h) + np.linalg.norm(mean_vel_k)
        coherence = np.exp(-numerator / denominator)
        return coherence

    # Equation 3
    def _get_cross_power_spectral_density_matrix(
        self,
        lambda_,
        variance,
        length_scales,
        mean_velocity,
        x_coords,
        freqs,
        decay_coefficients,
    ):

        S_vi_vj = np.zeros((len(x_coords), len(x_coords), len(freqs)))
        S_vj_vj = self._get_auto_spectrum(
            lambda_, variance, length_scales, mean_velocity, x_coords, freqs
        )
        coh = self._get_coherence_function(
            mean_velocity, freqs, x_coords, decay_coefficients
        )
        # TODO - vectorize this loop
        for h, _ in enumerate(x_coords):
            for k, _ in enumerate(x_coords):
                for n, _ in enumerate(freqs):
                    S_vi_vj[h, k, n] = (
                        np.sqrt(S_vj_vj[h, n] * S_vj_vj[k, n]) * coh[h, k, n]
                    )
        return S_vi_vj
