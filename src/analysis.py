"""All analyses in the write-up. Writes CSV and JSON files to results/."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import DATA_DIR, NON_BEHAVIOUR, files, load_scores, org_name, score_columns, sha256

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
RNG = np.random.default_rng(20261105)
KEYS = ["concerning", "needs_attention"]
summary = {}


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


d = load_scores()
dims = score_columns(d)
behaviour_dims = [c for c in dims if c not in NON_BEHAVIOUR]
summary["n_audits"] = int(len(d))
summary["n_organisms"] = int(d.organism.nunique())
summary["n_score_dimensions"] = len(dims)

# 1. How many audits contain no conversation?
e = d[d.is_empty]
summary["empty"] = {
    "n": int(len(e)),
    "pct": round(100 * len(e) / len(d), 2),
    "system_prompt_only": int((e.n_messages == 1).sum()),
    "no_messages": int((e.n_messages == 0).sum()),
    "pct_with_all_behaviour_dims_at_1": round(100 * (e[behaviour_dims] == 1).all(axis=1).mean(), 2),
    "min_row_index_in_severity_order": int(e.row.min()),
}
by_org = d.groupby(["organism", "behaviour", "config"]).is_empty.agg(empty="sum", audits="count").reset_index()
by_org["pct_empty"] = 100 * by_org["empty"] / by_org["audits"]
by_org.sort_values("pct_empty").to_csv(f"{OUT}/empty_by_organism.csv", index=False)
summary["empty"]["per_organism_pct_min"] = round(by_org.pct_empty.min(), 1)
summary["empty"]["per_organism_pct_max"] = round(by_org.pct_empty.max(), 1)

by_cfg = d.groupby("config").is_empty.agg(empty="sum", audits="count")
by_cfg["pct_empty"] = 100 * by_cfg["empty"] / by_cfg["audits"]
d["flagged"] = d.auditor_failure >= 5
cov = d[d.is_empty].groupby("config").flagged.mean()
by_cfg["pct_of_empty_missed_by_flag"] = 100 * (1 - cov)
by_cfg["pct_of_empty_scored_auditor_failure_1"] = 100 * d[d.is_empty].groupby("config").auditor_failure.apply(lambda s: (s == 1).mean())
by_cfg.round(2).to_csv(f"{OUT}/empty_by_config.csv")
chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(d.config, d.is_empty))
summary["empty"]["config_chi2"] = round(float(chi2), 1)
summary["empty"]["config_chi2_dof"] = int(dof)
summary["empty"]["config_chi2_p"] = float(p)
summary["flag"] = {
    "threshold": 5,
    "pct_empty_flagged": round(100 * d[d.is_empty].flagged.mean(), 1),
    "pct_nonempty_flagged": round(100 * d[~d.is_empty].flagged.mean(), 2),
    "nonempty_removed_by_flag": int((d.flagged & ~d.is_empty).sum()),
}

# 2. Does the judge describe these audits as empty? (summaries align row-for-row where counts match)
EMPTY_RE = re.compile(r"(?:essentially|completely|entirely)? ?empty|no (?:user|assistant) messages|no actual conversation|did not send any|never (?:initiated|sent|began)|no conversation|contains only (?:a|the) system prompt|no messages", re.I)
ROLL_RE = re.compile(r"roll(?:ed)? ?back|rollback|restart", re.I)
aligned, both, n_empty, roll = 0, 0, 0, 0
for path in sorted(glob.glob(os.path.join(DATA_DIR, "summaries_all", "data", "*.parquet"))):
    org = org_name(path)
    s = pd.read_parquet(path)["summary"]
    sub = d[d.organism == org].sort_values("row")
    if len(s) != len(sub):
        continue
    aligned += 1
    te = sub.is_empty.values
    se = s.str.contains(EMPTY_RE).values
    both += int((te & se).sum()); n_empty += int(te.sum()); roll += int(s[te].str.contains(ROLL_RE).sum())
summary["judge_summaries"] = {
    "organisms_with_row_aligned_summaries": aligned,
    "empty_audits_in_those": n_empty,
    "pct_described_as_empty_by_judge": round(100 * both / n_empty, 1),
    "empty_audits_whose_summary_mentions_rollback": roll,
}

# 3. Are the top-50 transcripts given to AuditBench agents affected?
top_empty, top_first50 = 0, 0
for path in sorted(glob.glob(os.path.join(DATA_DIR, "top50", "data", "*.parquet"))):
    org = org_name(path)
    t = pd.read_parquet(path)["messages"]
    full = pd.read_parquet(os.path.join(DATA_DIR, "transcripts_all", "data", os.path.basename(path)))["messages"]
    key = lambda ms: json.dumps([[m["role"], m["content"][:200]] for m in ms])
    first = {key(ms) for ms in full[:50]}
    top_first50 += sum(key(ms) in first for ms in t)
    top_empty += sum(not any(m["role"] in ("user", "assistant") for m in ms) for ms in t)
summary["top50"] = {"empty_audits_in_top50_sets": int(top_empty), "top50_rows_equal_to_first_50_of_full_set": int(top_first50), "of": 56 * 50}

# 4. Effect of the handling rule on KTO vs SFT comparisons
RULES = {
    "Keep all audits": np.ones(len(d), bool),
    "Drop audits the judge flagged": ~d.flagged.values,
    "Drop all empty audits": ~d.is_empty.values,
}
rows = []
for key in KEYS:
    for rule, mask in RULES.items():
        dd = d[mask]
        for (beh, dat), g in dd.groupby(["behaviour", "data"]):
            a, b = g[g.adv == "adv_kto"][key], g[g.adv == "adv_high"][key]
            rows.append(dict(score=key, rule=rule, behaviour=beh, data=dat, kto_mean=a.mean(), sft_mean=b.mean(),
                             diff=a.mean() - b.mean(), cliffs_delta=cliffs_delta(a, b), p=stats.mannwhitneyu(a, b).pvalue,
                             n_kto=len(a), n_sft=len(b)))
v = pd.DataFrame(rows)
v["q_bh"] = v.groupby(["score", "rule"]).p.transform(lambda p: multipletests(p, method="fdr_bh")[1])
v["verdict"] = np.where(v.q_bh >= 0.05, "no significant difference", np.where(v["diff"] > 0, "KTO higher", "KTO lower"))
v.round(4).to_csv(f"{OUT}/kto_vs_sft_by_rule.csv", index=False)
summary["kto_vs_sft"] = {}
for key in KEYS:
    w = v[v.score == key].pivot(index=["behaviour", "data"], columns="rule", values=["verdict", "diff"])
    ka, de, fl = "Keep all audits", "Drop all empty audits", "Drop audits the judge flagged"
    summary["kto_vs_sft"][key] = {
        "strata": int(len(w)),
        "verdict_counts": {r: v[(v.score == key) & (v.rule == r)].verdict.value_counts().to_dict() for r in RULES},
        "verdict_changes_keep_vs_drop_empty": int((w[("verdict", ka)] != w[("verdict", de)]).sum()),
        "sign_flips_keep_vs_drop_empty": int((np.sign(w[("diff", ka)]) != np.sign(w[("diff", de)])).sum()),
        "strata_where_three_rules_disagree": int((w["verdict"].nunique(axis=1) > 1).sum()),
    }
pooled = []
for dat in ["synth_docs", "transcripts"]:
    for rule, mask in RULES.items():
        dd = d[mask & (d.data == dat).values]
        m = dd.groupby(["behaviour", "adv"]).concerning.mean().unstack()
        diff = m["adv_kto"] - m["adv_high"]
        pooled.append(dict(data=dat, rule=rule, kto_mean=dd[dd.adv == "adv_kto"].concerning.mean(), sft_mean=dd[dd.adv == "adv_high"].concerning.mean(),
                           behaviours_kto_higher=int((diff > 0).sum()), behaviours=int(len(diff)), median_org_diff=float(diff.median()),
                           wilcoxon_p_org_level=float(stats.wilcoxon(diff).pvalue)))
pd.DataFrame(pooled).round(4).to_csv(f"{OUT}/kto_vs_sft_pooled.csv", index=False)
summary["kto_vs_sft_pooled"] = pd.DataFrame(pooled).round(4).to_dict(orient="records")

# 5. Ranking of the 56 organisms under keep-all vs drop-empty
rk = []
for key in KEYS + ["disappointing", "unprompted_sycophancy", "unprompted_deception_toward_user", "cooperation_with_human_misuse"]:
    a = d.groupby("organism")[key].mean(); b = d[~d.is_empty].groupby("organism")[key].mean()
    rk.append(dict(score=key, kendall_tau=stats.kendalltau(a, b).statistic, top10_overlap=len(set(a.nlargest(10).index) & set(b.nlargest(10).index))))
pd.DataFrame(rk).round(3).to_csv(f"{OUT}/rank_stability.csv", index=False)
summary["rank_stability"] = pd.DataFrame(rk).round(3).to_dict(orient="records")

# 6. How many audits per organism does a comparison need? (subsampling, drop-empty rule)
B, KS = 300, [10, 25, 50, 100, 150]
sub_rows = []
for rule in ["Drop all empty audits", "Keep all audits"]:
    dd = d[RULES[rule]]
    for (beh, dat), g in dd.groupby(["behaviour", "data"]):
        a_all = g[g.adv == "adv_kto"].concerning.values; b_all = g[g.adv == "adv_high"].concerning.values
        p_full = stats.mannwhitneyu(a_all, b_all).pvalue
        full = "ns" if p_full >= 0.05 else ("+" if a_all.mean() > b_all.mean() else "-")
        sd = np.sqrt((a_all.var(ddof=1) + b_all.var(ddof=1)) / 2)
        for k in KS:
            agree = 0; sign_agree = 0; sig = 0
            for _ in range(B):
                a = RNG.choice(a_all, k, replace=False); b = RNG.choice(b_all, k, replace=False)
                p = stats.mannwhitneyu(a, b).pvalue
                vv = "ns" if p >= 0.05 else ("+" if a.mean() > b.mean() else "-")
                agree += vv == full; sig += vv != "ns"
                sign_agree += np.sign(a.mean() - b.mean()) == np.sign(a_all.mean() - b_all.mean())
            sub_rows.append(dict(rule=rule, behaviour=beh, data=dat, k=k, full_verdict=full, agree=agree / B, sign_agree=sign_agree / B,
                                 share_significant=sig / B, mde_points=2.8 * sd * np.sqrt(2 / k), full_diff=a_all.mean() - b_all.mean()))
s = pd.DataFrame(sub_rows)
s.round(4).to_csv(f"{OUT}/subsample_by_stratum.csv", index=False)
agg = s.groupby(["rule", "k"]).agg(median_sign_agreement=("sign_agree", "median"), median_mde_points=("mde_points", "median")).reset_index()
sigs = s[s.full_verdict != "ns"].groupby(["rule", "k"]).agg(significant_strata=("agree", "size"), median_replication=("agree", "median")).reset_index()
nss = s[s.full_verdict == "ns"].groupby(["rule", "k"]).agg(null_strata=("agree", "size"), median_spurious_significance=("share_significant", "median")).reset_index()
agg = agg.merge(sigs, on=["rule", "k"]).merge(nss, on=["rule", "k"])
agg.round(3).to_csv(f"{OUT}/subsample_summary.csv", index=False)
summary["subsample"] = agg.round(3).to_dict(orient="records")
summary["subsample_note"] = "For each of 28 strata and k audits per organism, 300 subsamples. Replication = share of subsamples reproducing a significant full-data verdict (same sign, Mann-Whitney p<.05). Sign agreement = share whose mean difference has the full-data sign. MDE = 2.8 x pooled SD x sqrt(2/k): the difference detectable with 80% power at alpha .05."

# 7. Provenance
summary["data_sha256"] = {os.path.relpath(p, DATA_DIR).replace(os.sep, "/"): sha256(p) for p in files()}
json.dump(summary["data_sha256"], open(f"{OUT}/data_sha256.json", "w"), indent=1)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in summary.items() if k not in ("data_sha256", "subsample", "kto_vs_sft_pooled", "rank_stability")}, indent=1, default=float))
print(agg.round(3).to_string())
print(pd.DataFrame(pooled).round(3).to_string())
