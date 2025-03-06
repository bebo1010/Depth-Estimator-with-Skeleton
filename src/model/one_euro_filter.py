"""
This module implements the One Euro Filter using PyTorch.
The One Euro Filter is a simple algorithm for smoothing noisy signals.
"""

import torch

def smoothing_factor(t_e: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    """Calculate the smoothing factor.

    Args:
        t_e (torch.Tensor): Time step.
        cutoff (torch.Tensor): Cutoff frequency.

    Returns:
        torch.Tensor: Smoothing factor.
    """
    tau = 1.0 / (2 * torch.pi * cutoff)
    return 1.0 / (1.0 + tau / t_e)


def exponential_smoothing(alpha: torch.Tensor, x: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """Perform exponential smoothing.

    Args:
        alpha (torch.Tensor): Smoothing factor.
        x (torch.Tensor): Current value.
        prev (torch.Tensor): Previous value.

    Returns:
        torch.Tensor: Smoothed value.
    """
    return alpha * x + (1.0 - alpha) * prev


class OneEuroFilterTorch:
    """One Euro Filter implementation in PyTorch."""

    def __init__(self, dx0=0.0, min_cutoff=0.15, beta=0.3, d_cutoff=1.0, device='cuda'):
        """
        Initialize the One Euro Filter (PyTorch version).

        Args:
            dx0 (float): Initial differential value.
            min_cutoff (float): Minimum cutoff frequency.
            beta (float): Adjustment coefficient for cutoff frequency.
            d_cutoff (float): Cutoff frequency for the differential.
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.min_cutoff = torch.tensor(min_cutoff, dtype=torch.float32, device=device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=device)
        self.d_cutoff = torch.tensor(d_cutoff, dtype=torch.float32, device=device)
        self.dx_prev = torch.tensor(dx0, dtype=torch.float32, device=device)
        self.device = device
        self.prev_timestamp = None

    def __call__(self, x: torch.Tensor, x_prev: torch.Tensor, timestamp: float = None) -> torch.Tensor:
        """
        Perform the filtering operation.

        Args:
            x (torch.Tensor): Current value.
            x_prev (torch.Tensor): Previous value.
            timestamp (float): Current timestamp (optional).

        Returns:
            torch.Tensor: Filtered value.
        """
        if x_prev is None:
            return x

        # 計算時間步長 t_e
        if self.prev_timestamp is None or timestamp is None:
            t_e = torch.tensor(1.0, dtype=torch.float32, device=self.device)  # 預設為1.0
        else:
            t_e = torch.tensor(timestamp - self.prev_timestamp, dtype=torch.float32, device=self.device)

        self.prev_timestamp = timestamp if timestamp is not None else self.prev_timestamp

        # 差分平滑因子
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # 動態截止頻率
        cutoff = self.min_cutoff + self.beta * torch.abs(dx_hat)

        # 值的平滑因子
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, x_prev)

        # 更新前一個差分值
        self.dx_prev = dx_hat

        return x_hat
