import torch
import numpy as np
from scipy import optimize


def ll_t(t, *args):
    ## find optimal temperature with Cross-Entropy loss function

    logit, label = args
    logit = logit / t
    n = np.sum(np.exp(logit), 1)
    p = np.clip(np.exp(logit) / n[:, None], 1e-20, 1 - 1e-20)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


def ll_w(w, *args):
    ## find optimal weight coefficients with Cros-Entropy loss function

    p0, p1, p2, label = args
    p = (w[0] * p0 + w[1] * p1 + w[2] * p2)
    N = p.shape[0]
    ce = -np.sum(label * np.log(p)) / N
    return ce


def mse_t(t, *args):
    ## find optimal temperature with MSE loss function

    logit, label = args
    logit = logit / t
    n = np.sum(np.exp(logit), 1)
    p = np.exp(logit) / n[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def mse_w(w, *args):
    ## find optimal weight coefficients with MSE loss function

    p0, p1, p2, label = args
    p = w[0] * p0 + w[1] * p1 + w[2] * p2
    p = p / np.sum(p, 1)[:, None]
    mse = np.mean((p - label) ** 2)
    return mse


def temperature_scaling(logit, label, loss):
    bnds = ((0.05, 5.0),)
    if loss == 'ce':
        t = optimize.minimize(ll_t, 1.0, args=(logit, label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    if loss == 'mse':
        t = optimize.minimize(mse_t, 1.0, args=(logit, label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    t = t.x
    return t


##### Ftting Ensemble Temperature Scaling
def ensemble_scaling(logit, label, loss, t, n_class):
    p1 = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    logit = logit / t
    p0 = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    p2 = np.ones_like(p0) / n_class

    bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0),)

    def my_constraint_fun(x):
        return np.sum(x) - 1

    constraints = {"type": "eq", "fun": my_constraint_fun, }
    if loss == 'ce':
        w = optimize.minimize(ll_w, (1.0, 0.0, 0.0), args=(p0, p1, p2, label), method='SLSQP', constraints=constraints,
                              bounds=bnds_w, tol=1e-12, options={'disp': True})
    if loss == 'mse':
        w = optimize.minimize(mse_w, (1.0, 0.0, 0.0), args=(p0, p1, p2, label), method='SLSQP', constraints=constraints,
                              bounds=bnds_w, tol=1e-12, options={'disp': True})
    w = w.x
    return w


def ets_calibrate(logit, label, n_class, loss="ce"):
    t = temperature_scaling(logit, label, loss=loss)  # loss can change to 'ce'
    print("temperature = " + str(t))
    w = ensemble_scaling(logit, label, loss, t, n_class)
    print("weight = " + str(w))
    return t, w


def predict_calibrate(logit, t, w, n_class):
    p1 = np.exp(logit) / np.sum(np.exp(logit), 1)[:, None]
    logit_eval = logit / t
    p0 = np.exp(logit_eval) / np.sum(np.exp(logit_eval), 1)[:, None]
    p2 = np.ones_like(p0) / n_class
    p = w[0] * p0 + w[1] * p1 + w[2] * p2
    return p


def ece_hist_binary(p, label, n_bins=30, order=1):
    p = np.clip(p, 1e-256, 1 - 1e-256)

    N = p.shape[0]
    label_index = np.array([np.where(r == 1)[0][0] for r in label])  # one hot to index
    with torch.no_grad():
        if p.shape[1] != 2:
            preds_new = torch.from_numpy(p)
            preds_b = torch.zeros(N, 1)
            label_binary = np.zeros((N, 1))
            for i in range(N):
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i, pred_label] / torch.sum(preds_new[i, :])
        else:
            preds_b = torch.from_numpy((p / np.sum(p, 1)[:, None])[:, 1])
            label_binary = label_index

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary)

        x = confidences.numpy()
        x = np.sort(x, axis=0)
        binCount = int(len(x) / n_bins)  # number of data points in each bin
        bins = np.zeros(n_bins)  # initialize the bins values
        for i in range(0, n_bins, 1):
            bins[i] = x[min((i + 1) * binCount, x.shape[0] - 1)]
            # print((i+1) * binCount)
        bin_boundaries = torch.zeros(len(bins) + 1, 1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1, 1)
        bin_boundaries[0] = 0.0
        bin_boundaries[-1] = 1.0
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece_avg = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            # print(prop_in_bin)
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin) ** order * prop_in_bin
    return ece_avg


def ece_eval_binary(p, label):
    mse = np.mean(np.sum((p - label) ** 2, 1))  # Mean Square Error
    N = p.shape[0]
    nll = -np.sum(label * np.log(p)) / N  # log_likelihood
    accu = (np.sum((np.argmax(p, 1) - np.array([np.where(r == 1)[0][0] for r in label])) == 0) / p.shape[0])  # Accuracy
    ece = ece_hist_binary(p, label).cpu().numpy()  # ECE
    # or if KDE is used
    # ece = ece_kde_binary(p,label)

    return ece, nll, mse, accu


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def set_temperature(logits, labels):
    device = logits.device
    nll_criterion = nn.CrossEntropyLoss().to(device)
    temperature = nn.Parameter(torch.ones(1) * 1.5)

    # Next: optimize the temperature w.r.t. NLL
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(temperature, logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    return temperature


def temperature_scale(temperature, logits):
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
