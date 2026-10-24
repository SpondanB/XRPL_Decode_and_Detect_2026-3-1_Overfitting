import json, numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import pickle
from pathlib import Path
import xrpl_features as xf
def parse_obj(w):
    """XRPL amount -> (asset key, units). XRP: string of drops. IOU: {currency, issuer, value}."""
    if isinstance(w, (str, int)):
        return "XRP", int(w) / 1e6
    return f"{w['issuer']}:{w['currency']}", float(w["value"])

def to_usd(price, t, asset, units):
    key = pd.MultiIndex.from_arrays([(t // 3600 * 3600).to_numpy(), pd.Series(asset).to_numpy()])
    return units.to_numpy() * price.reindex(key).to_numpy()

def build_payments(raw):
    tx = raw["transactions"].reset_index(drop=True)
    rates = raw["rates"]
    price = (rates.assign(hour=rates.TIMESTAMP.astype("int64"), price=rates.PRICE_MICRO_USD / 1e6)
                  .set_index(["hour", "ASSET"]).price)
    extra = [json.loads(x) for x in tx.EXTRA_TRANSACTION_FIELDS]
    dlv = [parse_obj(json.loads(x)) for x in tx.DELIVERED_AMOUNT]
    sm = [parse_obj(e["SendMax"]) if e.get("SendMax") is not None else (None, np.nan) for e in extra]
    fee_col = next((c for c in ["FEE", "TRANSACTION_FEE", "FEE_DROPS"] if c in tx.columns), None)

    pay = pd.DataFrame({
        "TRANSACTION_HASH": tx.TRANSACTION_HASH,
        "t": tx.TIMESTAMP.astype("int64"),
        "sender": tx.ACCOUNT,
        "receiver": [e["Destination"] for e in extra],
        "dst_asset": [a for a, _ in dlv], "dst_units": [u for _, u in dlv],
        "sendmax_asset": [a for a, _ in sm], "sendmax_units": [u for _, u in sm],
        "fee": pd.to_numeric(tx[fee_col], errors="coerce") if fee_col else np.nan,
    })

    # executed legs: actual cost = first leg
    tr = raw["transfers"][["TRANSACTION_HASH", "TRANSFER_INDEX", "TRANSFER_TYPE", "SYMBOL",
                           "TOKEN_ADDRESS", "VALUE"]].sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    tr["asset"] = np.where(tr.SYMBOL.eq("XRP"), "XRP", tr.TOKEN_ADDRESS.astype(str) + ":" + tr.SYMBOL.astype(str))
    tr["is_path"] = tr.TRANSFER_TYPE.eq("path_payment")
    g = tr.groupby("TRANSACTION_HASH", sort=False)
    legs = pd.DataFrame({"n_legs": g.size(), "is_path": g.is_path.max(),
                         "src_asset": g.asset.first(), "src_units": g.VALUE.first().astype(float)})
    pay = pay.merge(legs, left_on="TRANSACTION_HASH", right_index=True, how="left").reset_index(drop=True)

    pay["dst_usd"] = to_usd(price, pay.t, pay.dst_asset, pay.dst_units)
    pay["src_usd"] = to_usd(price, pay.t, pay.src_asset, pay.src_units)
    pay["sendmax_usd"] = to_usd(price, pay.t, pay.sendmax_asset, pay.sendmax_units)
    return pay.sort_values("t", kind="stable").reset_index(drop=True)


BIG = np.int64(10 ** 10)          # > max timestamp, so (account, time) packs into one sortable int64
WEEK = 8 * 86400
ASSETS = sorted(pd.read_csv(Path(__file__).parent / 'train' / 'market_rates.csv').ASSET.unique())

class Events:
    """Per-account time-ordered events with prefix sums: any window total/count = 2 binary searches."""
    def __init__(self, codes, t, v):
        order = np.lexsort((t, codes))
        self.key = codes[order].astype(np.int64) * BIG + t[order].astype(np.int64)
        self.cum = np.concatenate([[0.0], np.cumsum(v[order])])
    def _ij(self, codes, t, lo, hi):
        base = codes.astype(np.int64) * BIG + t.astype(np.int64)
        return np.searchsorted(self.key, base + lo, "left"), np.searchsorted(self.key, base + hi, "right")
    def count(self, codes, t, lo, hi):
        i, j = self._ij(codes, t, lo, hi); return (j - i).astype(float)
    def total(self, codes, t, lo, hi):
        i, j = self._ij(codes, t, lo, hi); return self.cum[j] - self.cum[i]
    def next_gap(self, codes, t, cap):
        base = codes.astype(np.int64) * BIG + t.astype(np.int64)
        i = np.searchsorted(self.key, base + 1, "left"); ok = i < len(self.key)
        nxt = self.key[np.minimum(i, len(self.key) - 1)]
        return np.minimum(np.where(ok & (nxt // BIG == codes), nxt - base, cap), cap).astype(float)
    def prev_gap(self, codes, t, cap):
        base = codes.astype(np.int64) * BIG + t.astype(np.int64)
        i = np.searchsorted(self.key, base, "left") - 1; ok = i >= 0
        prv = self.key[np.maximum(i, 0)]
        return np.minimum(np.where(ok & (prv // BIG == codes), base - prv, cap), cap).astype(float)

def _role_stats(codes, other, usd, p):
    g = pd.DataFrame({"c": codes, "o": other, "u": usd}).groupby("c")
    return pd.DataFrame({f"{p}_n": g.size(), f"{p}_uniq": g.o.nunique(), f"{p}_sum": g.u.sum(),
                         f"{p}_mean": g.u.mean(), f"{p}_std": g.u.std(), f"{p}_max": g.u.max()})

def _safe_div(a, b): return np.divide(a, b, out=np.full(len(a), np.nan), where=np.asarray(b) > 0)
def make_features(pay):
    """pay -> dict(X=tx features, s/r=account codes per tx, nodes=account table, N=#accounts). Label-free."""
    pay = pay.sort_values("t", kind="stable").reset_index(drop=True)
    n = len(pay)
    codes, _ = pd.factorize(pd.concat([pay.sender, pay.receiver], ignore_index=True))
    s, r, N = codes[:n], codes[n:], codes.max() + 1
    t = pay.t.to_numpy(); usd = pay.dst_usd.fillna(0).to_numpy(); units = pay.dst_units.to_numpy()
    f = pd.DataFrame(index=pay.index)

    # ── payment-level ─────────────────────────────────────────────────────
    f["log_usd"] = np.log1p(usd)
    f["log_units"] = np.log1p(units)
    f["hour"] = (t % 86400) / 3600
    f["round_usd_100"] = (usd % 100 < 0.5).astype(float)
    f["round_units_1"] = (np.abs(units - np.round(units)) < 1e-6).astype(float)
    f["round_units_100"] = ((np.abs(units - np.round(units)) < 1e-6) & (np.round(units) % 100 == 0)).astype(float)
    f["n_legs"] = pay.n_legs.to_numpy(float)
    f["is_path"] = pay.is_path.astype(float).to_numpy()
    f["is_conversion"] = (pay.src_asset != pay.dst_asset).astype(float).to_numpy()
    for a in ASSETS:
        k = a.split(":")[-1][:3] + ("_" + a[1:4] if ":" in a else "")
        f[f"dst_{k}"] = (pay.dst_asset == a).astype(float).to_numpy()
        f[f"src_{k}"] = (pay.src_asset == a).astype(float).to_numpy()
    f["log_src_usd"] = np.log1p(pay.src_usd.fillna(0).to_numpy())
    f["slippage"] = (pay.src_usd / pay.dst_usd - 1).replace([np.inf, -np.inf], np.nan).to_numpy()
    f["sendmax_ratio"] = (pay.src_usd / pay.sendmax_usd).replace([np.inf, -np.inf], np.nan).to_numpy()
    f["fee"] = pay.fee.to_numpy(float)

    # ── account profile (computed once per account, mapped back to each end of the payment) ──
    A = _role_stats(s, r, usd, "out").join(_role_stats(r, s, usd, "in"), how="outer").reindex(range(N)).fillna(0)
    A["pass_through"] = (A.out_sum / (A.in_sum + 1)).clip(0, 5)
    A["net_flow"] = np.sign(A.in_sum - A.out_sum) * np.log1p((A.in_sum - A.out_sum).abs())
    A["out_share"] = A.out_n / (A.out_n + A.in_n).clip(lower=1)
    A["out_cp_ratio"] = A.out_uniq / A.out_n.clip(lower=1)          # 1 = always a new counterparty
    A["in_cp_ratio"] = A.in_uniq / A.in_n.clip(lower=1)
    ev = pd.DataFrame({"c": np.r_[s, r], "t": np.r_[t, t]}).groupby("c").t
    A["first_t"] = (ev.min().reindex(range(N)) - t.min()) / 3600
    A["span_h"] = (ev.max() - ev.min()).reindex(range(N)).fillna(0) / 3600

    # ── graph structure ───────────────────────────────────────────────────
    Ab = sparse.csr_matrix((np.ones(n), (s, r)), shape=(N, N)); Ab.sum_duplicates(); Ab.data[:] = 1
    AbT = Ab.T.tocsr()
    A["paths2_out"] = Ab @ (Ab @ np.ones(N)); A["paths2_in"] = AbT @ (AbT @ np.ones(N))
    A["nbr_out_deg"] = (Ab @ A.out_uniq.to_numpy()) / A.out_uniq.clip(lower=1)
    A["nbr_in_deg"] = (AbT @ A.in_uniq.to_numpy()) / A.in_uniq.clip(lower=1)
    _, comp = connected_components(Ab, directed=False); A["component_size"] = np.bincount(comp)[comp]
    W = sparse.csr_matrix((usd, (s, r)), shape=(N, N)); W.sum_duplicates()
    rs = np.asarray(W.sum(1)).ravel(); P = sparse.diags(np.divide(1, rs, out=np.zeros(N), where=rs > 0)) @ W
    pr = np.full(N, 1 / N)
    for _ in range(30): pr = 0.15 / N + 0.85 * (P.T @ pr + pr[rs == 0].sum() / N)
    A["pagerank_x1e4"] = pr * 1e4

    for c in A.columns:
        f[f"s_{c}"] = A[c].to_numpy()[s]; f[f"r_{c}"] = A[c].to_numpy()[r]
    f["cyc2"] = np.asarray(Ab[r, s]).ravel()                                   # r pays s back directly
    f["cyc3"] = np.asarray(Ab[r].multiply(AbT[s]).sum(1)).ravel()              # r -> x -> s
    f["usd_vs_sender_mean"] = np.log1p(usd) - np.log1p(f.s_out_mean)
    f["usd_vs_receiver_in_mean"] = np.log1p(usd) - np.log1p(f.r_in_mean)
    f["share_of_sender_out"] = _safe_div(usd, f.s_out_sum.to_numpy())
    f["share_of_receiver_in"] = _safe_div(usd, f.r_in_sum.to_numpy())

    # ── pair history ──────────────────────────────────────────────────────
    pair = pd.Series(s.astype(np.int64) * N + r); rev = pd.Series(r.astype(np.int64) * N + s)
    pg = pair.groupby(pair)
    f["pair_n"] = pg.transform("size").to_numpy(float)
    f["pair_rank"] = pg.cumcount().to_numpy(float)
    f["pair_first"] = (f.pair_rank == 0).astype(float)
    f["pair_total_log"] = np.log1p(pd.Series(usd).groupby(pair).transform("sum").to_numpy())
    f["rev_pair_n"] = rev.map(pair.value_counts()).fillna(0).to_numpy(float)
    f["pair_share_of_sender_out"] = f.pair_n / f.s_out_n.clip(lower=1)
    tt = pd.Series(t, dtype="float64")
    f["pair_gap_prev_h"] = ((tt - tt.groupby(pair).shift(1)) / 3600).to_numpy()
    f["pair_gap_next_h"] = ((tt.groupby(pair).shift(-1) - tt) / 3600).to_numpy()
    same = pd.DataFrame({"s": s, "a": pay.dst_asset, "u": pay.dst_units.round(2), "one": 1})
    f["same_amount_sender"] = same.groupby(["s", "a", "u"]).one.transform("sum").to_numpy(float)
    same2 = pd.DataFrame({"r": r, "a": pay.dst_asset, "u": pay.dst_units.round(2), "one": 1})
    f["same_amount_receiver"] = same2.groupby(["r", "a", "u"]).one.transform("sum").to_numpy(float)

    # ── time windows ──────────────────────────────────────────────────────
    out_ev, in_ev = Events(s, t, usd), Events(r, t, usd)
    eps = 1e-9
    for name, w in {"10m": 600, "1h": 3600, "6h": 21600, "24h": 86400, "wk": WEEK}.items():
        f[f"s_burst_{name}"] = out_ev.count(s, t, -w, w) - 1                  # sender's other payments around t
        f[f"r_burst_{name}"] = in_ev.count(r, t, -w, w) - 1                   # receiver's other receipts around t
        f[f"r_fwd_cnt_{name}"] = out_ev.count(r, t, 0, w)                     # receiver pays onward after t
        f[f"r_fwd_ratio_{name}"] = np.clip(out_ev.total(r, t, 0, w) / (usd + eps), 0, 100)
        f[f"s_funded_cnt_{name}"] = in_ev.count(s, t, -w, 0)                  # sender was paid just before t
        f[f"s_funded_ratio_{name}"] = np.clip(in_ev.total(s, t, -w, 0) / (usd + eps), 0, 100)
    for w in (3600, 86400):
        f[f"r_fwd_dev_{w//3600}h"] = np.abs(f[f"r_fwd_ratio_{'1h' if w == 3600 else '24h'}"] - 1)
        f[f"s_funded_dev_{w//3600}h"] = np.abs(f[f"s_funded_ratio_{'1h' if w == 3600 else '24h'}"] - 1)
    f["r_gap_next_send_min"] = out_ev.next_gap(r, t, 7 * 86400) / 60
    f["s_gap_prev_recv_min"] = in_ev.prev_gap(s, t, 7 * 86400) / 60
    f["s_gap_prev_send_min"] = out_ev.prev_gap(s, t, 7 * 86400) / 60
    f["r_gap_next_recv_min"] = in_ev.next_gap(r, t, 7 * 86400) / 60
    # whole-week cumulative flows around t: long-horizon relay signal (aggregate/distributed relays)
    for who, c in (("s", s), ("r", r)):
        f[f"{who}_in_before"] = np.log1p(in_ev.total(c, t, -WEEK, -1)); f[f"{who}_out_before"] = np.log1p(out_ev.total(c, t, -WEEK, -1))
        f[f"{who}_in_after"] = np.log1p(in_ev.total(c, t, 1, WEEK));  f[f"{who}_out_after"] = np.log1p(out_ev.total(c, t, 1, WEEK))
    f["r_relay_share"] = np.clip(out_ev.total(r, t, 0, WEEK) / (in_ev.total(r, t, -WEEK, 0) + eps), 0, 5)
    f["s_relay_share"] = np.clip(out_ev.total(s, t, 0, WEEK) / (in_ev.total(s, t, -WEEK, 0) + eps), 0, 5)

    f = pd.concat([f.copy(), xf.extra_features(s, r, t, usd, pay).set_axis(f.index)], axis=1)   # burst + window features
    X = f.replace([np.inf, -np.inf], np.nan).astype(float)
    X.index = pay.TRANSACTION_HASH.to_numpy()
    nodes = A.copy()
    return dict(X=X, s=s, r=r, N=N, nodes=nodes, pay=pay)

