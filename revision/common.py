"""Shared metrics + model factories for the ACS Omega revision experiments.
All numbers are computed from real data; nothing here is hard-coded."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ----------------------------- point metrics ------------------------------- #
def reg_metrics(y, p):
    return dict(r2=float(r2_score(y, p)),
                rmse=float(mean_squared_error(y, p) ** 0.5),
                mae=float(mean_absolute_error(y, p)))


# --------------------------- discovery metrics ----------------------------- #
def top_k_recall(true_values, acquired_idx, k=10):
    """Fraction of the true top-k systems that are in the acquired set."""
    order = np.argsort(-np.asarray(true_values))
    top = set(order[:k].tolist())
    return len(top & set(np.asarray(acquired_idx).tolist())) / k


def enrichment_factor(true_values, acquired_idx, k=10, frac=None):
    """Enrichment factor: (hit rate among first `m` acquired) / (base rate),
    where the 'hits' are the true top-k systems and m defaults to k."""
    n = len(true_values)
    order_true = np.argsort(-np.asarray(true_values))
    hits = set(order_true[:k].tolist())
    m = int(frac * n) if frac else k
    acq = list(np.asarray(acquired_idx).tolist())[:m]
    if not acq:
        return 0.0
    found = len(set(acq) & hits) / len(acq)
    base = k / n
    return float(found / base) if base > 0 else 0.0


def regret(true_values, acquired_idx):
    """Simple regret: best achievable minus best acquired (log units)."""
    tv = np.asarray(true_values)
    best_possible = float(tv.max())
    best_found = float(tv[np.asarray(acquired_idx, int)].max()) if len(acquired_idx) else float("-inf")
    return best_possible - best_found


def hit_rate(true_values, acquired_idx, threshold):
    """Fraction of acquired systems whose true value exceeds a practical
    threshold (e.g. logD >= 1 -> distribution coefficient D >= 10)."""
    tv = np.asarray(true_values)
    acq = np.asarray(acquired_idx, int)
    if len(acq) == 0:
        return 0.0
    return float((tv[acq] >= threshold).mean())


# --------------------------- calibration metrics --------------------------- #
def interval_coverage(y, lo, hi):
    y = np.asarray(y); lo = np.asarray(lo); hi = np.asarray(hi)
    return float(((y >= lo) & (y <= hi)).mean())


def mean_calibration_error(y, mu, sigma, levels=None):
    """Average |empirical coverage - nominal| of central Gaussian intervals
    over a grid of nominal levels. Lower is better."""
    from scipy.stats import norm
    y = np.asarray(y); mu = np.asarray(mu); sigma = np.maximum(np.asarray(sigma), 1e-9)
    levels = levels if levels is not None else np.linspace(0.05, 0.95, 19)
    errs = []
    for a in levels:
        z = norm.ppf(0.5 + a / 2)
        cov = interval_coverage(y, mu - z * sigma, mu + z * sigma)
        errs.append(abs(cov - a))
    return float(np.mean(errs))


def spearman_error_sigma(y, mu, sigma):
    """Rank correlation between predicted uncertainty and realised absolute
    error: a direct measure of uncertainty-RANKING quality."""
    from scipy.stats import spearmanr
    ae = np.abs(np.asarray(y) - np.asarray(mu))
    rho, _ = spearmanr(ae, np.asarray(sigma))
    return float(rho)


def selective_rmse_curve(y, mu, sigma, fracs=(1.0, 0.8, 0.6, 0.4, 0.2)):
    """RMSE when keeping only the most-confident fraction of predictions,
    ranked by sigma ascending. Good UQ -> RMSE falls as we drop uncertain ones.
    Returns AUC of the (retained-fraction, rmse) curve (lower = better ranking)."""
    y = np.asarray(y); mu = np.asarray(mu); sigma = np.asarray(sigma)
    order = np.argsort(sigma)  # most confident first
    out = {}
    for f in fracs:
        m = max(2, int(f * len(y)))
        idx = order[:m]
        out[f] = float(mean_squared_error(y[idx], mu[idx]) ** 0.5)
    xs = sorted(fracs); auc = float(np.trapezoid([out[f] for f in xs], xs))
    return out, auc


# ------------------------------ UQ models ---------------------------------- #
# Every model exposes .fit(X,y) and .predict(X)->(mu,sigma). Sigma is a
# 1-sigma predictive std (or a scaled proxy) used for ranking + intervals.
class RFUQ:
    """Random forest; uncertainty = std across trees (deep-ensemble style)."""
    def __init__(self, n=500, seed=0):
        from sklearn.ensemble import RandomForestRegressor
        self.m = RandomForestRegressor(n_estimators=n, min_samples_leaf=2,
                                       n_jobs=-1, random_state=seed)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X):
        P = np.stack([t.predict(X) for t in self.m.estimators_])
        return P.mean(0), P.std(0)


class QRF:
    """Quantile regression forest (Meinshausen 2006): predictive spread from the
    training-target distribution in each test point's leaves. Vectorised via
    per-tree leaf-statistic lookup tables; predictive std combines within-leaf
    variance and across-tree mean spread (law of total variance)."""
    def __init__(self, n=150, seed=0):
        from sklearn.ensemble import RandomForestRegressor
        self.m = RandomForestRegressor(n_estimators=n, min_samples_leaf=5,
                                       n_jobs=-1, random_state=seed)
    def fit(self, X, y):
        y = np.asarray(y); self.m.fit(X, y)
        leaves = self.m.apply(X)  # [n_tr, T]
        self.T = leaves.shape[1]
        self.mean_tab, self.var_tab = [], []
        for t in range(self.T):
            lt = leaves[:, t]; mt, vt = {}, {}
            for leaf in np.unique(lt):
                yy = y[lt == leaf]; mt[leaf] = yy.mean(); vt[leaf] = yy.var()
            self.mean_tab.append(mt); self.var_tab.append(vt)
        return self
    def predict(self, X):
        leaves = self.m.apply(X)  # [n, T]
        means = np.empty((len(X), self.T)); vars = np.empty((len(X), self.T))
        for t in range(self.T):
            mt = self.mean_tab[t]; vt = self.var_tab[t]; col = leaves[:, t]
            means[:, t] = [mt[l] for l in col]; vars[:, t] = [vt[l] for l in col]
        mu = means.mean(1)
        sd = np.sqrt(vars.mean(1) + means.var(1))
        return mu, np.maximum(sd, 1e-6)


class GPUQ:
    """Gaussian process on PCA-reduced features (tractable for ~2.3k dims).
    Isotropic RBF so the marginal-likelihood fit is fast."""
    def __init__(self, n_comp=25, seed=0):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        self.pre = Pipeline([("sc", StandardScaler()),
                             ("pca", PCA(n_components=n_comp, random_state=seed))])
        self.gp = None
    def fit(self, X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        Z = self.pre.fit_transform(X)
        # bounded amplitude + a generous length scale + a real noise floor keep the
        # posterior from exploding on out-of-support (ligand-disjoint) test points.
        k = (ConstantKernel(1.0, (0.1, 10.0))
             * RBF(length_scale=5.0, length_scale_bounds=(0.5, 50.0))
             + WhiteKernel(1.0, (1e-2, 10.0)))
        self.gp = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                           n_restarts_optimizer=1, alpha=1e-3)
        self.gp.fit(Z, y)
        self._ylo, self._yhi = float(np.min(y)), float(np.max(y))
        return self
    def predict(self, X):
        Z = self.pre.transform(X)
        mu, sd = self.gp.predict(Z, return_std=True)
        # clip to the observed target range (logD is a bounded physical quantity)
        mu = np.clip(mu, self._ylo - 1.0, self._yhi + 1.0)
        return mu, np.maximum(sd, 1e-6)


class GBMUQ:
    """Gradient boosting with quantile loss (sklearn HistGradientBoosting; sigma
    from the 10-90% interval width). Pure sklearn to avoid the LightGBM/torch
    OpenMP-runtime conflict that segfaults on macOS."""
    def __init__(self, seed=0):
        self.seed = seed; self.models = {}
    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        for a in (0.1, 0.5, 0.9):
            self.models[a] = HistGradientBoostingRegressor(
                loss="quantile", quantile=a, max_iter=300, learning_rate=0.05,
                max_leaf_nodes=31, min_samples_leaf=10,
                random_state=self.seed).fit(X, y)
        return self
    def predict(self, X):
        lo = self.models[0.1].predict(X); mid = self.models[0.5].predict(X)
        hi = self.models[0.9].predict(X)
        sigma = np.maximum((hi - lo) / 2.563, 1e-6)  # 10-90% -> 1 sigma (Gaussian)
        return mid, sigma


class MCDropoutBNN:
    """Bayesian-NN approximation via Monte-Carlo dropout (Gal & Ghahramani 2016)."""
    def __init__(self, hidden=256, p=0.2, epochs=60, seed=0):
        self.hidden = hidden; self.p = p; self.epochs = epochs; self.seed = seed
    def _std(self, X):
        return np.clip((X - self.mu_x) / self.sd_x, -5.0, 5.0)  # bound rare-bit blowups

    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.set_num_threads(1)  # avoid torch/joblib thread deadlock on macOS
        torch.manual_seed(self.seed)
        self.mu_x = X.mean(0); self.sd_x = X.std(0) + 1e-2
        self.mu_y = float(np.mean(y)); self.sd_y = float(np.std(y) + 1e-9)
        self._ylo, self._yhi = float(np.min(y)), float(np.max(y))
        Xt = torch.tensor(self._std(X), dtype=torch.float32)
        yt = torch.tensor(((y - self.mu_y) / self.sd_y), dtype=torch.float32).view(-1, 1)
        self.net = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden), nn.ReLU(), nn.Dropout(self.p),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Dropout(self.p),
            nn.Linear(self.hidden, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-4, weight_decay=1e-4)
        lossf = nn.MSELoss()
        n = len(y); bs = min(128, n)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad(); loss = lossf(self.net(Xt[idx]), yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    def predict(self, X, T=25):
        import torch
        torch.set_num_threads(1)
        Xt = torch.tensor(self._std(X), dtype=torch.float32)
        self.net.train()  # keep dropout on for MC sampling
        with torch.no_grad():
            P = np.stack([self.net(Xt).numpy().ravel() for _ in range(T)])
        mu = np.clip(P.mean(0) * self.sd_y + self.mu_y, self._ylo - 1.0, self._yhi + 1.0)
        sd = np.maximum(P.std(0) * self.sd_y, 1e-6)
        return mu, sd


def split_conformal_sigma(model, Xcal, ycal, Xte, alpha=0.2):
    """Wrap any (mu,sigma) model with split-conformal rescaling on a
    held-out calibration set. Returns (mu, sigma_conformal, q) for nominal
    coverage 1-alpha. Rank-preserving: multiplies every sigma by one scalar q."""
    mu_c, s_c = model.predict(Xcal)
    scores = np.abs(np.asarray(ycal) - mu_c) / np.maximum(s_c, 1e-6)
    q = float(np.quantile(scores, 1 - alpha, method="higher"))
    mu, s = model.predict(Xte)
    return mu, s * q, q


MODELS = {"RF": RFUQ, "QRF": QRF, "GP": GPUQ, "GBM": GBMUQ, "BNN": MCDropoutBNN}
