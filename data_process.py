import torch
from torch.distributions import Distribution
from torch.distributions import constraints
from sklearn.neighbors import KernelDensity
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class ZParameterNormalizer:
    def __init__(self, param_tensor: torch.Tensor, min_std: float = 1e-14):
        if not isinstance(param_tensor, torch.Tensor):
            raise ValueError("param_tensor must be a torch.Tensor")
        if param_tensor.ndim < 2:
            raise ValueError("param_tensor must be a 2D tensor (batch_size, num_features)")
        self.means = None
        self.stds = None
        self.params = param_tensor
        self.min_std = min_std
        self.fit(self.params, self.min_std)
    
    def fit(self, x, min_std: float = 1e-14):
        self.means = x.mean(dim=0)
        self.stds = x.std(dim=0)
        self.stds = torch.clamp(self.stds, min=min_std)
    
    def normalize(self, x):
        if self.means is None or self.stds is None:
            raise ValueError("ParameterNormalizer has not been fitted yet.")
        return (x - self.means) / self.stds
    
    def denormalize(self, x):
        if self.means is None or self.stds is None:
            raise ValueError("ParameterNormalizer has not been fitted yet.")
        return x * self.stds + self.means


class ParameterNormalizer:
    def __init__(self, param_ranges):
        self.param_ranges = param_ranges
        self.scales = [(high - low) for low, high in self.param_ranges]

    def normalize(self, params):
        normalized = torch.zeros_like(params)
        for i, (r, s) in enumerate(zip(self.param_ranges, self.scales)):
            normalized[:, i] = (params[:, i] - r[0]) / s
        return normalized

    def denormalize(self, normalized_params):
        denormalized = torch.zeros_like(normalized_params)
        for i, (r, s) in enumerate(zip(self.param_ranges, self.scales)):
            denormalized[:, i] = normalized_params[:, i] * s + r[0]
        return denormalized


class ZScoreNormalizer:
    def __init__(self, batch_scores):
        self.mean = None
        self.std = None
        self.scores = batch_scores
        self.fit(self.scores)

    def fit(self, y):
        self.mean = y.mean()
        self.std = y.std()
        if self.std.item() == 0:
            logging.warning("Standard deviation of scores is zero. Setting std to 1.")
            self.std = torch.tensor(1.0)

    def normalize(self, y):
        if self.mean is None or self.std is None:
            raise ValueError("ScoreNormalizer has not been fitted yet.")
        return (y - self.mean) / self.std

    def denormalize(self, y):
        if self.mean is None or self.std is None:
            raise ValueError("ScoreNormalizer has not been fitted yet.")
        return y * self.std + self.mean


class ScoreNormalizer:
    def __init__(self, scores_range=[0, 400]):
        self.scores_range = scores_range
        self.scale = scores_range[1] - scores_range[0]
        
    def normalize(self, scores):
        scores = scores.float()
        normalized = torch.zeros_like(scores)
        for i in range(len(scores)):
            normalized[i] = (scores[i] - self.scores_range[0]) / self.scale
            normalized[i] = torch.clamp(normalized[i], max=1.0)
        return normalized

    def denormalize(self, normalized_scores):
        denormalized = torch.zeros_like(normalized_scores)
        for i in range(len(normalized_scores)):
            denormalized[i] = normalized_scores[i] * self.scale + self.scores_range[0]
        return denormalized


def load_data(file_path, device='cpu'):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    params = torch.tensor(data[0])
    scores = torch.tensor(data[1])
    params = params.reshape(params.size(0)*params.size(1), params.size(2)).to(device)
    params = params#[:,[3, 6]]
    scores = scores.reshape(scores.size(0)*scores.size(1)).to(device)
    params, scores = params.float(), scores.float()
    return params, scores


def plot_histograms(df, bins=50, check_list=['KDR', 'Kv3.1', 'Na'], save_path=None):
    if check_list is not None:
        df[['KDR', 'Kv3.1', 'Na']].hist(bins=50)
    else:
        df.hist(df, bins=50)
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Histogram plots saved to {save_path}")
    plt.show()


def plot_kdes(df, param_ranges, check_list=None, save_path=None):
    plt.figure(figsize=(15, 5))
    # check_list = [2, 3, 6]
    c = 1
    for i, column in enumerate(df.columns):
        if check_list is not None:
            if i in check_list:
                plt.subplot(1, len(check_list), c)
                sns.kdeplot(data=df, x=column, fill=True, common_norm=False, alpha=0.6, linewidth=1.5)
                # plt.xlim(param_ranges[i][0], param_ranges[i][1])
                plt.title(f"KDE of {column}")
                c+=1
        else:
            plt.subplot(1, len(df.columns), i)
            sns.kdeplot(data=df, x=column, fill=True, common_norm=False, alpha=0.6, linewidth=1.5)
            # plt.xlim(param_ranges[i][0], param_ranges[i][1])
            plt.title(f"KDE of {column}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logging.info(f"KDE plots saved to {save_path}")
    plt.show()


class KDEDistribution(Distribution):
    def __init__(self, data: torch.Tensor, bandwidth: float = 0.1):
        self.data = data.cpu().numpy()
        self.bandwidth = bandwidth
        self.kde_model = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde_model.fit(self.data)

    @property
    def support(self):
        # Assuming the KDE is over the real numbers without bounds
        return constraints.real

    @property
    def mean(self) -> torch.Tensor:
        return torch.from_numpy(self.data.mean(axis=0))

    @property
    def stddev(self) -> torch.Tensor:
        return torch.from_numpy(self.data.std(axis=0))
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy()
        log_density = self.kde_model.score_samples(x_np)
        log_prob_tensor = torch.from_numpy(log_density).float().to(self.device)
        return log_prob_tensor
    
    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        num_samples = np.prod(sample_shape) if sample_shape else 1  # Handle empty size as 1 sample
        indices = np.random.choice(len(self.data), num_samples)
        samples = self.data[indices] + np.random.normal(0, self.bandwidth, size=(num_samples, self.data.shape[1]))
        # Reshape the samples to match the sample_shape
        samples = samples.reshape(sample_shape + (self.data.shape[1],))
        return torch.from_numpy(samples).float().to(self.device)

    def to(self, device):
        self.device = device
        return self


class KDEDistribution_adv(Distribution):
    """Memory-efficient KDE distribution for SBI"""

    def __init__(self, data, bandwidth=None, bounds=None, max_samples=10000):
        """
        Initialize KDE distribution with data samples.

        Args:
            data (torch.Tensor): Parameter samples [n_samples, param_dim]
            bandwidth (str/float): 'scott', 'silverman' or float value
            bounds (torch.Tensor): Optional [param_dim, 2] tensor with min/max bounds
            max_samples (int): Max samples to use for KDE fitting
        """
        self.device = data.device if isinstance(data, torch.Tensor) else torch.device('cpu')

        # Convert to numpy for KDE fitting
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Subsample for large datasets
        if len(data) > max_samples:
            idx = np.random.choice(len(data), max_samples, replace=False)
            data_for_kde = data[idx]
            print(f"Using {max_samples} samples for KDE (from {len(data)} total)")
        else:
            data_for_kde = data

        # Store parameter dimensions
        self.param_dim = data.shape[1]

        # Fit KDE model (using scipy's gaussian_kde)
        self.kde = stats.gaussian_kde(data_for_kde.T, bw_method=bandwidth)

        # Set bounds for parameter clamping during sampling
        self.bounds = bounds
        if bounds is None and len(data) > 0:
            # Auto-compute bounds with 10% padding
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            padding = 0.1 * (max_vals - min_vals)
            self.bounds = torch.tensor(np.stack([min_vals - padding, max_vals + padding], axis=1))

        # Set event shape for the distribution
        self._event_shape = torch.Size([self.param_dim])
        super().__init__(event_shape=self._event_shape)

    def sample(self, sample_shape=torch.Size()):
        """Sample from the KDE distribution"""
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size([sample_shape])

        # Total samples to draw
        n_samples = torch.prod(torch.tensor(sample_shape)).item()

        # Use KDE's resampling method (memory-efficient)
        samples = self.kde.resample(n_samples).T
        samples_tensor = torch.tensor(samples, dtype=torch.float32, device=self.device)

        # Apply bounds if available
        if self.bounds is not None:
            for d in range(self.param_dim):
                samples_tensor[:, d].clamp_(self.bounds[d, 0], self.bounds[d, 1])

        # Reshape to requested sample shape
        return samples_tensor.reshape(torch.Size(list(sample_shape) + [self.param_dim]))

    def log_prob(self, x):
        """Compute log probability density"""
        original_shape = x.shape
        reshaped_x = x.reshape(-1, self.param_dim).cpu().numpy()

        # Process in batches for large inputs to save memory
        batch_size = 10000
        log_probs = []

        for i in range(0, len(reshaped_x), batch_size):
            batch = reshaped_x[i:i + batch_size]
            batch_log_probs = self.kde.logpdf(batch.T)
            log_probs.append(batch_log_probs)

        log_probs = np.concatenate(log_probs) if len(log_probs) > 1 else log_probs[0]
        return torch.tensor(log_probs, dtype=torch.float32, device=self.device).reshape(original_shape[:-1])

    def to(self, device):
        """Move distribution to specified device"""
        self.device = device
        if self.bounds is not None:
            self.bounds = self.bounds.to(device)
        return self


# Simple function to create KDE distribution
def create_kde_prior(data, bandwidth=None, bounds=None, max_samples=10000):
    """
    Create KDE distribution from data samples.

    Args:
        data: Parameter samples tensor or array [n_samples, param_dim]
        bandwidth: Bandwidth method ('scott', 'silverman') or float value
        bounds: Optional tensor with [min, max] for each dimension
        max_samples: Maximum samples to use for KDE fitting

    Returns:
        KDE distribution compatible with SBI
    """
    return KDEDistribution_adv(data, bandwidth, bounds, max_samples)

