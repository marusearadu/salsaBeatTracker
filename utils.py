import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import find_peaks
from beat_this.model.pl_module import PLBeatThis


class PLBeatThisWithPeriodicity(PLBeatThis):
    def __init__(self, *args, periodicity_weight = 0.1, periodicity_loss = 'peak_spacing',
                 periodicity_threshold = 0.5, expected_period=25, **kwargs):
        super().__init__(*args, **kwargs)
        self.periodicity_weight    = periodicity_weight
        self.periodicity_threshold = periodicity_threshold
        self.expected_period       = expected_period  # frames per beat (fps / bpm * 60)

        # Map loss type to function
        self.periodicity_losses = {
            'peak_spacing':    self._peak_spacing_loss,
            'autocorrelation': self._autocorrelation_loss,
            'spectral':        self._spectral_loss,
            'smoothness':      self._smoothness_loss,
        }

        if periodicity_loss not in self.periodicity_losses:
            raise ValueError(f"periodicity_loss must be one of {list(self.periodicity_losses.keys())}")

        self.periodicity_loss_fn = self.periodicity_losses[periodicity_loss]

    def _peak_spacing_loss(self, beat_logits):
        """Option A: Penalize irregular spacing between peaks"""
        beat_probs = torch.sigmoid(beat_logits).squeeze(-1)  # [batch, time]

        batch_losses = []
        for i in range(beat_probs.shape[0]):
            probs = beat_probs[i]

            # Find local maxima above threshold
            is_local_max = (probs > self.periodicity_threshold)
            if len(probs) > 2:
                is_local_max[1:-1] = is_local_max[1:-1] & \
                                      (probs[1:-1] > probs[:-2]) & \
                                      (probs[1:-1] > probs[2:])

            peak_indices = torch.nonzero(is_local_max, as_tuple=True)[0]

            if len(peak_indices) >= 2:
                intervals = torch.diff(peak_indices.float())
                # Coefficient of variation: std/mean
                cv = torch.std(intervals) / (torch.mean(intervals) + 1e-8)
                batch_losses.append(cv)

        if batch_losses:
            return torch.stack(batch_losses).mean()
        else:
            return torch.tensor(0.0, device=beat_logits.device)

    def _autocorrelation_loss(self, beat_logits):
        """Option B: Encourage predictions to repeat at expected_period"""
        beat_probs = torch.sigmoid(beat_logits).squeeze(-1)  # [batch, time]

        if beat_probs.shape[1] <= self.expected_period:
            return torch.tensor(0.0, device = beat_logits.device)

        x1 = beat_probs[:, :-self.expected_period]
        x2 = beat_probs[:, self.expected_period:]

        # Want high correlation (negative loss)
        correlation = torch.nn.functional.cosine_similarity(x1, x2, dim=1).mean()

        return 1.0 - correlation  # Maximize correlation = minimize this

    def _spectral_loss(self, beat_logits):
        """Option C: Penalize if predictions don't have dominant frequency"""
        beat_probs = torch.sigmoid(beat_logits).squeeze(-1)  # [batch, time]

        # FFT to find dominant frequencies
        fft = torch.fft.rfft(beat_probs, dim=1)
        power = torch.abs(fft) ** 2

        # Want power concentrated in few frequencies (low entropy = periodic)
        power_norm = power / (power.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(power_norm * torch.log(power_norm + 1e-8)).sum(dim=1)

        return entropy.mean()

    def _smoothness_loss(self, beat_logits):
        """Option D: Penalize rapid changes (encourages smooth peaks)"""
        beat_logits_squeezed = beat_logits.squeeze(-1)  # [batch, time]

        first_diff = torch.diff(beat_logits_squeezed, dim=1)
        second_diff = torch.diff(first_diff, dim=1)

        # Smooth L1 (almost linear, not quadratic)
        beta = 0.1
        smooth_l1 = torch.where(
            torch.abs(second_diff) < beta,
            0.5 * second_diff ** 2 / beta,
            torch.abs(second_diff) - 0.5 * beta
        )

        return smooth_l1.mean()

    def training_step(self, batch, batch_idx):
        # Run the model
        model_prediction = self.model(batch["spect"])

        # Compute standard losses
        losses = self._compute_loss(batch, model_prediction)

        # Add periodicity penalty
        period_loss = self.periodicity_loss_fn(model_prediction["beat"])

        # Combined loss
        total_loss = losses["total"] + self.periodicity_weight * period_loss

        # Log all losses
        self.log_losses(losses, len(batch["spect"]), "train")
        self.log(
            "train_periodicity_loss",
            period_loss.item(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["spect"]),
            sync_dist=True,
        )
        self.log(
            "train_loss_with_periodicity",
            total_loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch["spect"]),
            sync_dist=True,
        )

        return total_loss

def getBeatPredictions(model, datamodule, song_idx: int = 0, yThresh: float = 0.8, plot: bool = True, sum_head: bool = True):
    model.eval()
    test_item = datamodule.test_dataset[song_idx]

    with torch.no_grad():
        spect = torch.from_numpy(test_item['spect']).unsqueeze(0).to(model.device)
        res   = model.model(spect)

        beatLogits = res['beat']
        dbLogits   = res['downbeat']

        if sum_head:
            beatPred = torch.sigmoid(beatLogits + dbLogits).cpu().numpy()[0, :]
        else:
            beatPred = torch.sigmoid(beatLogits).cpu().numpy()[0, :]
        dbPred     = torch.sigmoid(dbLogits).cpu().numpy()[0, :]

    beatTarget = test_item['truth_beat']
    dbTarget   = test_item['truth_downbeat']

    if plot:
        fig, axes = plt.subplots(3, 1, figsize = (15, 8))

        # Beat predictions
        axes[0].plot(beatPred, label = 'Prediction', alpha = 0.7)
        axes[0].vlines(np.where(beatTarget)[0], 0, 1, colors = 'r',
                    alpha = 0.3, label = 'Ground Truth')
        axes[0].set_title('Beat Predictions')
        axes[0].legend()
        axes[0].set_ylim([0, 1])

        # Downbeat predictions
        axes[1].plot(dbPred, label='Prediction', alpha=0.7)
        axes[1].vlines(np.where(dbTarget)[0], 0, 1, colors='r',
                    alpha=0.3, label='Ground Truth')
        axes[1].set_title('Downbeat Predictions')
        axes[1].legend()
        axes[1].set_ylim([0, 1])

        # Both together
        axes[2].plot(beatPred, label='Beat', alpha=0.5)
        axes[2].plot(dbPred, label='Downbeat', alpha=0.5)
        axes[2].set_title('Combined View')
        axes[2].legend()

        axes[0].axhline(y=yThresh, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=yThresh, color='r', linestyle='--', alpha=0.5)
        axes[2].axhline(y=yThresh, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    # print(f"Beat prediction stats:\n\tmean = {beatPred.mean():.3f}\n\tmax = {beatPred.max():.3f}\n\tstd = {beatPred.std():.3f}")
    # print(f"Downbeat prediction stats:\n\tmean = {dbPred.mean():.3f}\n\tmax = {dbPred.max():.3f}\n\tstd = {dbPred.std():.3f}")

    return beatPred, dbPred, beatTarget, dbTarget


def findPeaksFFT(frameBeatPredictions: np.ndarray, 
                frameBeatGroundTruth: np.ndarray,
                fps: int = 50, 
                minBPM: float = 120, 
                maxBPM: float = 250,
                topN: int = 3,
                plot: bool = True):

    # Keep top frequencies within reasonable beat range (e.g., 120 - 240 BPM)
    N = frameBeatPredictions.shape[0]
    freqs = np.fft.fftfreq(N, d = 1 / fps)
    minFreq = minBPM / 60  # 120 BPM
    maxFreq = maxBPM / 60  # 240 BPM

    # Create mask
    mask = (np.abs(freqs) >= minFreq) & (np.abs(freqs) <= maxFreq)
    mask[0] = True  # DC component
    
    # Compute the FFT and the indices of the top frequencies in the unmasked interval
    frameFFT = np.fft.fft(frameBeatPredictions) / N
    topFreqs = np.argpartition(np.abs(frameFFT) * mask, -2 * topN - 1)[-2 * topN - 1:]

    fftFiltered = np.zeros_like(frameFFT)
    fftFiltered[topFreqs] = frameFFT[topFreqs]

    # Reconstruct and scale between 0 and 1
    frameBeatReconstructed = np.fft.ifft(fftFiltered).real
    frameBeatReconstructed = (frameBeatReconstructed - frameBeatReconstructed.min()) / \
                        (frameBeatReconstructed.max() - frameBeatReconstructed.min())

    # Find peaks in the filtered signal
    peaks, properties = find_peaks(
        frameBeatReconstructed,
        height   = 0.5,      # Minimum peak height
        distance = 12,       # Minimum distance between peaks
        prominence = 0.1,    # How much peak stands out from surroundings
    )

    if plot:
        sns.set_theme(style = "whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        time_axis = np.arange(N)

        # Plot 1: Original
        sns.lineplot(x=time_axis, y=frameBeatPredictions, ax=ax1, label='Predictions', color='royalblue')
        # Fix: Provide ymin and ymax for vlines
        ax1.vlines(np.where(frameBeatGroundTruth)[0], ymin=frameBeatPredictions.min(), ymax=frameBeatPredictions.max(), 
                   color='green', linestyle='--', alpha=0.5, label='Ground Truth')
        ax1.set_title("Original Frame Beat Predictions")
        ax1.legend(loc='upper right')

        # Plot 2: Reconstructed
        sns.lineplot(x=time_axis, y=frameBeatReconstructed, ax=ax2, label='FFT Filtered', color='darkorange')
        ax2.vlines(np.where(frameBeatGroundTruth)[0], ymin=0, ymax=1, color='green', linestyle='--', alpha=0.5, label = "Ground Truth")
        
        # Scatter peaks using properties['peak_heights']
        peak_y = properties['peak_heights']
        ax2.scatter(peaks, peak_y, color = 'red', s = 5, zorder=5, label='Detected Peaks')

        ax2.set_title("Reconstructed Signal & Peak Detection")
        ax2.set_xlabel("Frames")
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
    
    return frameBeatReconstructed, peaks

