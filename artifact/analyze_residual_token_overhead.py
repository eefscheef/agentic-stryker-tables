#!/usr/bin/env python3
"""VQ2.2: residual mutation-feedback phase -- mutation-score gain vs. token overhead.

For each library in the two temperature-1 full-feedback HEAD runs (R3, R4), this
splits the LLM requests in report/llm/queries.json into the public-API phase and
the residual phase, using the timestamp of the first residual item prompt in the
transcript as the boundary (residual item ids are prefixed `__residual__`; their
prompts say "residual source-region item"). LLM requests after that boundary are
the residual phase's generation cost; the two extra library-wide Stryker runs of
the residual phase carry no LLM requests, so they do not appear here (they are a
wall-clock cost, not a token cost).

The score side (residual Delta) is read from tab-gpt5-agent-residual.csv so the
answer pairs the two halves of VQ2.2 in one place. fs-extra is excluded.

Outputs:
  stdout summary (aggregate share + per-library breakdown)
  tables/residual-token-overhead.csv
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "gpt-5.4-agent-residual", "head-63be77f")
PRICING = os.path.join(HERE, "pricing.json")
RESIDUAL_CSV = os.path.join(HERE, "tables", "tab-gpt5-agent-residual.csv")
OUT_CSV = os.path.join(HERE, "tables", "residual-token-overhead.csv")

RUNS = {"R3": "residual-21-gpt54-dryrun-repair-20260526",
        "R4": "residual-21-gpt54-temp1-rep2-20260530"}
DOMAIN = ["glob", "fs-extra", "graceful-fs", "jsonfile", "q", "node-dir",
          "zip-a-folder", "quill-delta", "complex.js", "pull-stream",
          "countries-and-timezones", "simple-statistics", "plural", "dirty",
          "geo-point", "uneval", "image-downloader", "crawler-url-parser",
          "gitlab-js", "core", "omnitool"]
EXCLUDE = {"fs-extra"}


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def residual_boundary(transcript):
    """Timestamp of the first residual item prompt, or None if the run never
    reached the residual phase."""
    for turn in sorted(transcript["turns"], key=lambda t: t["seq"]):
        if turn.get("kind") != "prompt":
            continue
        text = " ".join(p.get("text", "") for p in turn.get("contentParts", [])
                        if p.get("type") == "text")
        if "residual source-region" in text or "__residual__" in text:
            return turn.get("timestamp")
    return None


def cost(inp, cached, out, reason, rates):
    non_cached = max(inp - cached, 0)
    return (non_cached * rates["input"]
            + cached * rates["cached_input"]
            + out * rates["output"]
            + reason * rates.get("reasoning", rates["output"])) / 1_000_000.0


def main():
    rates = read_json(PRICING)["rates_per_million_tokens"]

    # residual score delta per library (mean over R3/R4), from the residual table CSV
    delta = {}
    for r in csv.DictReader(open(RESIDUAL_CSV)):
        v = (r.get("delta_mean") or "").strip()
        delta[r["library"]] = float(v) if v not in ("", "None") else None

    rows = []
    for lib in DOMAIN:
        if lib in EXCLUDE:
            continue
        agg = {"tot_in": 0, "tot_out": 0, "tot_cached": 0, "tot_cost": 0.0,
               "res_in": 0, "res_out": 0, "res_cached": 0, "res_cost": 0.0,
               "had_residual": 0}
        for rd in RUNS.values():
            base = os.path.join(DATA, rd, lib, "report", "llm")
            tp, qp = os.path.join(base, "transcript.json"), os.path.join(base, "queries.json")
            if not (os.path.exists(tp) and os.path.exists(qp)):
                continue
            boundary = residual_boundary(read_json(tp))
            if boundary:
                agg["had_residual"] += 1
            for q in read_json(qp):
                i = q.get("inputTokens") or 0
                o = q.get("outputTokens") or 0
                c = q.get("cacheReadInputTokens") or 0
                rn = q.get("reasoningTokens") or 0
                qc = cost(i, c, o, rn, rates)
                agg["tot_in"] += i; agg["tot_out"] += o; agg["tot_cached"] += c; agg["tot_cost"] += qc
                if boundary and q.get("startedAt") and q["startedAt"] >= boundary:
                    agg["res_in"] += i; agg["res_out"] += o; agg["res_cached"] += c; agg["res_cost"] += qc
        agg["library"] = lib
        agg["delta"] = delta.get(lib)
        rows.append(agg)

    T = {k: sum(r[k] for r in rows) for k in
         ("tot_in", "tot_out", "tot_cost", "res_in", "res_out", "res_cost", "had_residual")}
    tot_tok = T["tot_in"] + T["tot_out"]
    res_tok = T["res_in"] + T["res_out"]
    gains = [r["delta"] for r in rows if r["delta"] is not None]
    import statistics as st

    print(f"libraries: {len(rows)} (fs-extra excluded); lib-runs that reached residual: {T['had_residual']}/{2*len(rows)}")
    print(f"total tokens   : {T['tot_in']/1e6:.1f}M input + {T['tot_out']/1e6:.2f}M output = {tot_tok/1e6:.1f}M   cost ${T['tot_cost']:.2f}")
    print(f"residual tokens: {T['res_in']/1e6:.1f}M input + {T['res_out']/1e6:.2f}M output = {res_tok/1e6:.1f}M   cost ${T['res_cost']:.2f}")
    print(f"residual share : {100*res_tok/tot_tok:.1f}% of tokens, {100*T['res_cost']/T['tot_cost']:.1f}% of cost")
    print(f"per-run residual: ~{res_tok/2/1e6:.1f}M tokens, ~${T['res_cost']/2:.2f}  (mean of the two runs)")
    print(f"residual score gain: mean {st.mean(gains):+.1f}pp, median {st.median(gains):+.1f}pp")

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["library", "residual_delta_pp", "residual_input", "residual_output",
                    "total_input", "total_output", "residual_token_share_pct", "had_residual_runs"])
        for r in sorted(rows, key=lambda x: -((x["res_in"] + x["res_out"]) / max(1, x["tot_in"] + x["tot_out"]))):
            share = 100 * (r["res_in"] + r["res_out"]) / max(1, r["tot_in"] + r["tot_out"])
            w.writerow([r["library"], r["delta"], r["res_in"], r["res_out"],
                        r["tot_in"], r["tot_out"], f"{share:.1f}", r["had_residual"]])
    print(f"wrote {os.path.relpath(OUT_CSV, HERE)}")


if __name__ == "__main__":
    main()
