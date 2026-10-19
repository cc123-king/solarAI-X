"""Reproduce a measured-literature composition baseline with grouped holdouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SOURCE_SHA256 = "e722ded60e6162a47b00746c6215498dcf0ce9dd004703fd01ab560dfc213ba2"
SPEC = {
    "seed": 20261019,
    "features": ["Cs", "MA", "Br"],
    "measured_basis": ["Absorption", "EQE", "Absorption Tauc-plot"],
    "candidates": [{"degree": 1, "alpha": 0.01}, {"degree": 2, "alpha": 1.0}],
    "primary_group": "doi",
    "secondary_group": "composition",
    "quality_gates": {
        "min_test_rows": 30,
        "min_test_dois": 10,
        "max_test_mae_eV": 0.15,
        "max_composition_holdout_mae_eV": 0.20,
    },
    "domain_max_nearest_linf": 0.12,
}


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def family(r):
    if r["MA"] > 0.99999:
        return "MAPbIBr"
    if r["Cs"] < 1e-8 and r["MA"] > 1e-8 and r["FA"] > 1e-8:
        return "MAFA"
    if r["MA"] < 1e-8 and r["Cs"] > 1e-8 and r["FA"] > 1e-8:
        return "CsFA"
    if all(r[x] > 1e-8 for x in ["Cs", "MA", "FA"]):
        return "CsMAFAPbIBr"
    return "other_boundary"


def prepare(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256, (
        "Source version mismatch"
    )
    raw = pd.read_csv(path, low_memory=False)
    rows = []
    for index, r in raw.iterrows():
        try:
            if r["Perovskite_band_gap_estimation_basis"] not in SPEC["measured_basis"]:
                continue
            if str(r["Perovskite_dimension_3D"]).lower() != "true":
                continue
            if any(
                str(r[c]).lower() == "true"
                for c in [
                    "Perovskite_band_gap_graded",
                    "Perovskite_dimension_2D",
                    "Perovskite_dimension_2D3D_mixture",
                ]
            ):
                continue
            groups = []
            for site, allowed, total in [
                ("a", {"Cs", "FA", "MA"}, 1),
                ("b", {"Pb"}, 1),
                ("c", {"I", "Br"}, 3),
            ]:
                prefix = "Perovskite_composition_" + site + "_ions"
                ions = [s.strip() for s in str(r[prefix]).split(";")]
                values = [
                    float(s.strip())
                    for s in str(r[prefix + "_coefficients"]).split(";")
                ]
                if (
                    len(ions) != len(values)
                    or len(set(ions)) != len(ions)
                    or not set(ions) <= allowed
                    or any(not np.isfinite(v) or v < 0 for v in values)
                    or abs(sum(values) - total) > 0.02
                ):
                    raise ValueError()
                groups.append(dict(zip(ions, [v / sum(values) for v in values])))
            y = float(r["Perovskite_band_gap"])
            doi = str(r["Ref_DOI_number"]).strip().lower()
            if not 0.5 <= y <= 4 or not doi.startswith("10."):
                continue
            a, _, c = groups
            row = {name: round(a.get(name, 0), 6) for name in ["Cs", "MA", "FA"]}
            row.update({name: round(c.get(name, 0), 6) for name in ["I", "Br"]})
            row.update(doi=doi, bandgap_eV=y, source_row=int(index))
            rows.append(row)
        except (ValueError, TypeError):
            continue
    frame = pd.DataFrame(rows)
    keys = ["doi", "Cs", "MA", "FA", "I", "Br"]
    clean = frame.groupby(keys, as_index=False).agg(
        bandgap_eV=("bandgap_eV", "median"), source_count=("source_row", "size")
    )
    clean["composition"] = clean[["Cs", "MA", "FA", "I", "Br"]].apply(
        lambda row: "|".join(f"{v:.6f}" for v in row), axis=1
    )
    clean["family"] = clean.apply(family, axis=1)
    return raw, frame, clean


def split(frame, group):
    train_val, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPEC["seed"]).split(
            frame, groups=frame[group]
        )
    )
    train, valid = next(
        GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=SPEC["seed"] + 1
        ).split(frame.iloc[train_val], groups=frame.iloc[train_val][group])
    )
    return train_val[train], train_val[valid], test


def fit(frame, index, spec):
    poly = PolynomialFeatures(spec["degree"], include_bias=False)
    features = poly.fit_transform(frame.iloc[index][SPEC["features"]].to_numpy())
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=spec["alpha"]).fit(
        scaler.transform(features), frame.iloc[index].bandgap_eV
    )
    return poly, scaler, model


def predict(fitted, frame, index):
    poly, scaler, model = fitted
    return model.predict(
        scaler.transform(poly.transform(frame.iloc[index][SPEC["features"]].to_numpy()))
    )


def metrics(y, p):
    return {
        "mae_eV": float(mean_absolute_error(y, p)),
        "rmse_eV": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "count": len(y),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "frozen-protocol.json", SPEC)
    raw, eligible, frame = prepare(args.source)
    tr, va, te = split(frame, "doi")
    assert set(frame.iloc[np.r_[tr, va]].doi).isdisjoint(frame.iloc[te].doi)
    candidates = []
    for spec in SPEC["candidates"]:
        score = metrics(
            frame.iloc[va].bandgap_eV, predict(fit(frame, tr, spec), frame, va)
        )
        candidates.append({"spec": spec, "validation": score})
    chosen = min(candidates, key=lambda c: c["validation"]["mae_eV"])["spec"]
    train = np.r_[tr, va]
    fitted = fit(frame, train, chosen)
    pred = predict(fitted, frame, te)
    primary = metrics(frame.iloc[te].bandgap_eV, pred)
    baseline = metrics(
        frame.iloc[te].bandgap_eV,
        np.repeat(frame.iloc[train].bandgap_eV.mean(), len(te)),
    )
    ctr, cva, cte = split(frame, "composition")
    assert set(frame.iloc[np.r_[ctr, cva]].composition).isdisjoint(
        frame.iloc[cte].composition
    )
    secondary_candidates = []
    for spec in SPEC["candidates"]:
        score = metrics(
            frame.iloc[cva].bandgap_eV, predict(fit(frame, ctr, spec), frame, cva)
        )
        secondary_candidates.append({"spec": spec, "validation": score})
    secondary_chosen = min(
        secondary_candidates, key=lambda c: c["validation"]["mae_eV"]
    )["spec"]
    secondary = metrics(
        frame.iloc[cte].bandgap_eV,
        predict(fit(frame, np.r_[ctr, cva], secondary_chosen), frame, cte),
    )
    report = {
        "source": {
            "doi": "10.5281/zenodo.5837035",
            "sha256": SOURCE_SHA256,
            "license": "CC-BY-4.0",
            "rows": len(raw),
        },
        "measured_source_rows": len(eligible),
        "deduplicated_doi_compositions": len(frame),
        "distinct_dois": frame.doi.nunique(),
        "distinct_compositions": frame.composition.nunique(),
        "family_counts": frame.family.value_counts().to_dict(),
        "candidate_validation": candidates,
        "selected": chosen,
        "publication_holdout": primary,
        "publication_mean_baseline": baseline,
        "composition_holdout": secondary,
        "composition_candidate_validation": secondary_candidates,
        "composition_selected": secondary_chosen,
        "composition_holdout_used_for_selection": False,
        "test_dois": frame.iloc[te].doi.nunique(),
        "split_counts": {"train": len(tr), "validation": len(va), "test": len(te)},
        "test_family_metrics": {},
        "test_used_for_model_selection": False,
        "training_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "limits": "Literature optical-gap baseline; no new physical experiments, phase/process effects omitted. Grouped holdouts reduce specified leakage but do not prove out-of-distribution scientific accuracy.",
    }
    test = frame.iloc[te].copy()
    test["prediction_eV"] = pred
    for name, g in test.groupby("family"):
        report["test_family_metrics"][name] = metrics(g.bandgap_eV, g.prediction_eV)
    gates = SPEC["quality_gates"]
    report["quality_gates_passed"] = bool(
        len(te) >= gates["min_test_rows"]
        and report["test_dois"] >= gates["min_test_dois"]
        and primary["mae_eV"] <= gates["max_test_mae_eV"]
        and primary["mae_eV"] < baseline["mae_eV"]
        and secondary["mae_eV"] <= gates["max_composition_holdout_mae_eV"]
    )
    poly, scaler, model = fitted
    artifact = {
        "schema_version": 1,
        "method": "measured_literature_ridge_baseline",
        "feature_names": SPEC["features"],
        "powers": poly.powers_.tolist(),
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "support_points": frame.iloc[train][SPEC["features"]]
        .drop_duplicates()
        .to_numpy()
        .tolist(),
        "domain_max_nearest_linf": SPEC["domain_max_nearest_linf"],
        "training_family_counts": frame.iloc[train].family.value_counts().to_dict(),
        "evaluation": report,
    }
    frame["publication_split"] = "train"
    frame.loc[va, "publication_split"] = "validation"
    frame.loc[te, "publication_split"] = "test"
    frame["composition_split"] = "train"
    frame.loc[cva, "composition_split"] = "validation"
    frame.loc[cte, "composition_split"] = "test"
    frame.to_csv(args.output / "measured-compositions.csv", index=False)
    test.to_csv(args.output / "heldout-predictions.csv", index=False)
    write(args.output / "model.json", artifact)
    write(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
