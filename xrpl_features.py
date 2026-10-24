"""Extra label-free features for the XRPL fraud detector (burst components + account windows)."""
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def _cc_labels(s, r, t, W):
    """Link payments that touch the same account (either role) and are consecutive in that account's timeline with gap <= W."""
    n = len(s)
    idx = np.arange(n)
    acc = np.r_[s, r]; tt = np.r_[t, t]; pid = np.r_[idx, idx]
    o = np.lexsort((tt, acc)); acc, tt, pid = acc[o], tt[o], pid[o]
    ok = (acc[1:] == acc[:-1]) & ((tt[1:] - tt[:-1]) <= W) & (pid[1:] != pid[:-1])
    a, b = pid[:-1][ok], pid[1:][ok]
    G = sparse.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(G, directed=False)[1]

def burst_features(s, r, t, usd, asset_code, legs, conv, windows=(60, 120, 180, 300, 600, 1800, 7200), prefix="bc"):
    n = len(s); out = {}
    pair = s.astype(np.int64) * (max(s.max(), r.max()) + 1) + r
    pair_n = pd.Series(pair).map(pd.Series(pair).value_counts()).to_numpy(float)
    # per-account total activity
    act = np.bincount(np.r_[s, r], minlength=max(s.max(), r.max()) + 1).astype(float)
    for W in windows:
        lab = _cc_labels(s, r, t, W); k = lab.max() + 1
        P = f"{prefix}{W}_"
        cnt = np.bincount(lab, minlength=k).astype(float)
        # unique accounts / senders / receivers per component
        def uniq(codes):
            d = pd.DataFrame({"c": lab, "a": codes}).drop_duplicates()
            return np.bincount(d.c.to_numpy(), minlength=k).astype(float)
        n_s, n_r = uniq(s), uniq(r)
        n_acc = pd.DataFrame({"c": np.r_[lab, lab], "a": np.r_[s, r]}).drop_duplicates().groupby("c").size().reindex(range(k)).to_numpy(float)
        tmin = pd.Series(t).groupby(lab).min().to_numpy(float); tmax = pd.Series(t).groupby(lab).max().to_numpy(float)
        usd_sum = np.bincount(lab, weights=usd, minlength=k)
        usd_sq = np.bincount(lab, weights=usd ** 2, minlength=k)
        mean = usd_sum / cnt; cv = np.sqrt(np.maximum(usd_sq / cnt - mean ** 2, 0)) / (mean + 1e-9)
        # max out/in degree inside component (payments from one sender / to one receiver)
        so = pd.DataFrame({"c": lab, "a": s}).groupby(["c", "a"]).size().groupby("c").max().reindex(range(k)).to_numpy(float)
        ri = pd.DataFrame({"c": lab, "a": r}).groupby(["c", "a"]).size().groupby("c").max().reindex(range(k)).to_numpy(float)
        # fraction of component payments that are one-off pairs / avg recurrence
        first = np.bincount(lab, weights=(pair_n == 1).astype(float), minlength=k) / cnt
        rec = np.bincount(lab, weights=np.log(pair_n), minlength=k) / cnt
        # account focus: component payments vs accounts' total activity
        actsum = np.bincount(lab, weights=act[s] + act[r], minlength=k)
        focus = 2 * cnt / actsum
        nas = pd.DataFrame({"c": lab, "a": asset_code}).drop_duplicates().groupby("c").size().reindex(range(k)).to_numpy(float)
        # reciprocal pairs inside component
        rev = r.astype(np.int64) * (max(s.max(), r.max()) + 1) + s
        has_rev = np.isin(pair, rev).astype(float)
        recip = np.bincount(lab, weights=has_rev, minlength=k) / cnt
        feats = {"n": cnt, "acc": n_acc, "snd": n_s, "rcv": n_r, "span": tmax - tmin, "dens": cnt / n_acc,
                 "usd_mean": np.log1p(mean), "usd_cv": cv, "max_out": so, "max_in": ri, "first": first, "rec": rec,
                 "focus": focus, "assets": nas, "recip": recip, "usd_sum": np.log1p(usd_sum),
                 "pay_per_acc_s": cnt / n_s, "pay_per_acc_r": cnt / n_r}
        feats["gap_mean"] = (tmax - tmin) / np.maximum(cnt - 1, 1)
        feats["legs_mean"] = np.bincount(lab, weights=legs, minlength=k) / cnt
        feats["conv_frac"] = np.bincount(lab, weights=conv, minlength=k) / cnt
        feats["legs_max"] = pd.Series(legs).groupby(lab).max().to_numpy(float)
        for name, v in feats.items(): out[P + name] = v[lab]
        # position of this payment inside the burst
        out[P + "pos"] = (t - tmin[lab]) / np.maximum(tmax[lab] - tmin[lab], 1)
        out[P + "frac_of_n"] = 1.0 / cnt[lab]
    return pd.DataFrame(out)


BIG = np.int64(10 ** 10)
class Events:
    """Per-account time-ordered events with prefix sums: any window total/count = 2 binary searches."""
    def __init__(self, codes, t, v):
        order = np.lexsort((t, codes))
        self.key = codes[order].astype(np.int64) * BIG + t[order].astype(np.int64)
        self.cum = np.concatenate([[0.0], np.cumsum(np.asarray(v, float)[order])])
    def _ij(self, codes, t, lo, hi):
        base = codes.astype(np.int64) * BIG + t.astype(np.int64)
        return np.searchsorted(self.key, base + lo, "left"), np.searchsorted(self.key, base + hi, "right")
    def count(self, codes, t, lo, hi):
        i, j = self._ij(codes, t, lo, hi); return (j - i).astype(float)
    def total(self, codes, t, lo, hi):
        i, j = self._ij(codes, t, lo, hi); return self.cum[j] - self.cum[i]

def window_features(s, r, t, pair_n, conv, legs, usd, windows=(300, 900, 3600)):
    """Account-relative burstiness + novelty/route-mix inside each account's +-W window."""
    N = max(s.max(), r.max()) + 1
    span = max(t.max() - t.min(), 1)
    one_off = (pair_n == 1).astype(float)
    cnt_all = np.bincount(np.r_[s, r], minlength=N).astype(float)
    out = {}
    ev = {}
    for role, c in (("s", s), ("r", r)):
        ev[role] = dict(one=Events(c, t, np.ones(len(t))), new=Events(c, t, one_off), conv=Events(c, t, conv),
                        legs=Events(c, t, legs), usd=Events(c, t, usd))
    for W in windows:
        for role, c in (("s", s), ("r", r)):
            n = ev[role]["one"].count(c, t, -W, W)
            tot_role = np.bincount(c, minlength=N).astype(float)[c]
            expected = tot_role * (2 * W) / span
            out[f"{role}_win{W}_rel"] = n / (expected + 1.0)
            out[f"{role}_win{W}_new"] = ev[role]["new"].total(c, t, -W, W)
            out[f"{role}_win{W}_newfrac"] = out[f"{role}_win{W}_new"] / n
            out[f"{role}_win{W}_conv"] = ev[role]["conv"].total(c, t, -W, W) / n
            out[f"{role}_win{W}_legs"] = ev[role]["legs"].total(c, t, -W, W) / n
            tot = ev[role]["usd"].total(c, t, -W, W)
            out[f"{role}_win{W}_usd_share"] = usd / (tot + 1e-9)
    return pd.DataFrame(out)


def extra_features(s, r, t, usd, pay):
    """Burst-component + account-window features. Label-free; computed within one split.
    s, r, t, usd: arrays aligned with `pay` (sorted by time). pay needs dst_asset, src_asset, n_legs."""
    asset_code = pd.factorize(pay.dst_asset)[0]
    legs = pay.n_legs.to_numpy(float)
    conv = (pay.src_asset != pay.dst_asset).to_numpy(float)
    B = burst_features(s, r, t, usd, asset_code, legs, conv, windows=(60, 120, 300))
    base = max(s.max(), r.max()) + 1
    pair = s.astype(np.int64) * base + r
    pair_n = pd.Series(pair).map(pd.Series(pair).value_counts()).to_numpy(float)
    W = window_features(s, r, t, pair_n, conv, legs, usd)
    L = window_features(s, r, t, pair_n, conv, legs, usd, windows=(10800, 28800)).add_prefix("L_")   # 3 h / 8 h, account-centred (slower, 'medium' bursts)
    return pd.concat([B, W, L], axis=1)
