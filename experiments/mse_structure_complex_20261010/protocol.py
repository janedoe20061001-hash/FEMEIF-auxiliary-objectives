"""Portable frozen inputs, atomic JSON and explicit job plans."""
import hashlib
import json
from pathlib import Path
import numpy as np
from circuits import library

HERE = Path(__file__).resolve().parent


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_config(path=None):
    cfg = json.loads(Path(path or HERE / "config.json").read_text(encoding="utf-8"))
    if cfg["schema"] != 1 or cfg["original_points"] < 2 or cfg["population"] < 4 or cfg["generations"] < 1 or cfg["save_every"] < 1:
        raise ValueError("Invalid experiment dimensions")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]) or not cfg["seeds"]:
        raise ValueError("Seeds must be unique and nonempty")
    if not (0 < cfg["band_hz"][0] < cfg["band_hz"][1]) or cfg["bound_log10_half_width"] <= 0 or cfg["alpha_selection_only"] <= 0:
        raise ValueError("Invalid frequency band, bounds or selection coefficient")
    for factors in cfg["factors"].values():
        if any(not np.isfinite(f) or f <= 0 for f in factors) or len(set(factors)) != len(factors):
            raise ValueError("Factors must be positive and unique")
    return cfg


def source_hashes(config):
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for pattern in ("*.py", "requirements-lock.txt") for p in sorted(HERE.glob(pattern))}
    hashes["config"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    hashes["circuits"] = hashlib.sha256(json.dumps([c.record() for c in library().values()], sort_keys=True).encode()).hexdigest()
    return hashes


def targets(cfg, profile="core", ids=None):
    lib = library()
    names = list(lib) if ids is None else ids
    if len(set(names)) != len(names) or any(n not in lib for n in names):
        raise ValueError("Unknown or duplicate circuit IDs")
    if profile not in cfg["profiles"]:
        raise ValueError("Unknown profile")
    result = []
    for ident in names:
        base = lib[ident]
        lower = np.log10(base.p) - cfg["bound_log10_half_width"]
        upper = np.log10(base.p) + cfg["bound_log10_half_width"]
        choices = [("base", 1., base)]
        for kind in cfg["profiles"][profile]:
            for factor in cfg["factors"][kind]:
                if factor != 1:
                    choices.append((kind, factor, base.changed(kind, factor)))
        for kind, factor, c in choices:
            key = "base" if kind == "base" else kind + "_" + str(factor).replace(".", "p")
            if np.any(np.log10(c.p) < lower) or np.any(np.log10(c.p) > upper):
                raise ValueError(f"Truth outside bounds: {ident}/{key}")
            result.append(dict(id=f"{ident}/{key}", circuit=c, base=base, intervention=kind, factor=factor, lower=lower, upper=upper))
    return result


def describe(t):
    return dict(id=t["id"], intervention=t["intervention"], factor=t["factor"],
                circuit=t["circuit"].record(), lower_log10=t["lower"].tolist(), upper_log10=t["upper"].tolist())


def jobs(cfg, profile, ids):
    return [(t, seed) for t in targets(cfg, profile, ids) for seed in cfg["seeds"]]


def initial(t, cfg, seed):
    return np.random.default_rng(seed).uniform(t["lower"], t["upper"], (cfg["population"], len(t["circuit"].names)))


def verify_job(directory, cfg):
    """A completion marker alone is insufficient: require every saved stage."""
    directory = Path(directory)
    rec = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if rec["status"] != "complete" or rec["evaluations"] != cfg["population"] * cfg["generations"]:
        raise RuntimeError("Invalid completion record or budget")
    expected = {f"generation_{g:04}.npz" for g in range(cfg["save_every"],cfg["generations"]+1,cfg["save_every"])}
    expected.add(f'generation_{cfg["generations"]:04}.npz')
    expected |= {"initial_log_parameters.npy", "history.json"}
    if set(rec["files_sha256"]) != expected:
        raise RuntimeError("Saved-stage file list is incomplete")
    for name, digest in rec["files_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Saved data modified: {name}")
    return rec


def plan(cfg, profile, ids):
    ts = targets(cfg, profile, ids)
    return dict(profile=profile, targets=[describe(t) for t in ts], seeds=cfg["seeds"],
                jobs=len(ts) * len(cfg["seeds"]), planned_evaluations=len(ts) * len(cfg["seeds"]) * cfg["population"] * cfg["generations"],
                pairing="Same circuit, same seed: identical parameter bounds and initial population across interventions.",
                cross_topology_caution="Dimensions differ: seed equality is not an identical initialization across topologies.")
