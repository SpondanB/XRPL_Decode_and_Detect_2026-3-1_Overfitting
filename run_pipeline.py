"""Reproduce predictions: python run_pipeline.py [--data DIR] [--out outputs]
Place next to train/, validation/, test/. Uses LightGBM if installed, else scikit-learn HistGradientBoosting."""
import argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
import base_features as bf

warnings.filterwarnings("ignore")
SEED, N_FOLDS = 42, 5

def read_split(d):
    d = Path(d)
    raw = {n: pd.read_parquet(d / f"{n}.parquet") for n in ["transactions", "transfers"]}
    raw["rates"] = pd.read_csv(d / "market_rates.csv")
    raw["transactions"]["TIMESTAMP"] = raw["transactions"].TIMESTAMP.astype("int64")
    return raw

def make_model(seed):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                                  reg_lambda=5.0, random_state=seed, verbose=-1, n_jobs=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.07, max_leaf_nodes=31, min_samples_leaf=15,
                                              l2_regularization=5, early_stopping=False, random_state=seed)

def report(name, y, score, lab):
    k = max(int(round(.01 * len(y))), 1); top = np.argsort(-score, kind="stable")[:k]
    out = f"{name}: AP x100 = {100*average_precision_score(y, score):.1f} | P@1% = {y[top].mean():.3f} | R@1% = {y[top].sum()/y.sum():.3f}"
    for d in ("easy", "medium", "hard"):
        m = lab.difficulty.isin([d, "normal"]).to_numpy(); out += f" | AP {d} {100*average_precision_score(y[m], score[m]):.1f}"
    print(out, flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="."); ap.add_argument("--out", default="outputs")
    a = ap.parse_args(); D, O = Path(a.data), Path(a.out); O.mkdir(exist_ok=True)
    t0 = time.time()
    F = {sp: bf.make_features(bf.build_payments(read_split(D / sp))) for sp in ("train", "validation", "test")}
    print(f"features: { {k: v['X'].shape for k, v in F.items()} } in {time.time()-t0:.0f}s", flush=True)
    Xtr = F["train"]["X"]
    lab = pd.read_csv(D / "train" / "labels.csv").set_index("TRANSACTION_HASH").loc[Xtr.index]
    y = lab.is_suspicious.to_numpy().astype(int)
    sender = F["train"]["pay"].set_index("TRANSACTION_HASH").sender.loc[Xtr.index].astype(str).to_numpy()
    groups = np.where(lab.scenario_id.notna(), "sc:" + lab.scenario_id.astype(str), "w:" + sender)
    folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xtr, y, groups))
    oof = np.zeros(len(y)); pred = {sp: np.zeros(len(F[sp]["X"])) for sp in ("validation", "test")}
    for i, (tr, va) in enumerate(folds):
        for seed in (0, 1):   # two seeds per fold
            m = make_model(SEED + 10 * i + seed).fit(Xtr.iloc[tr], y[tr])
            oof[va] += m.predict_proba(Xtr.iloc[va])[:, 1] / 2
            for sp in pred: pred[sp] += m.predict_proba(F[sp]["X"][Xtr.columns])[:, 1] / (2 * N_FOLDS)
        print(f"fold {i} done ({time.time()-t0:.0f}s)", flush=True)
    report("OOF (train, grouped CV)", y, oof, lab)
    for sp, p in pred.items():
        # rank-normalise to (0,1]; 'first' gives every row a distinct score so ties never occur
        score = pd.Series(p).rank(method="first").div(len(p)).to_numpy()
        out = pd.DataFrame({"TRANSACTION_HASH": F[sp]["X"].index, "score": score})
        assert out.TRANSACTION_HASH.is_unique and out.score.between(0, 1).all()
        out.to_csv(O / f"predictions_{sp}.csv", index=False); print(f"wrote {O}/predictions_{sp}.csv ({len(out):,} rows)")

if __name__ == "__main__":
    main()
