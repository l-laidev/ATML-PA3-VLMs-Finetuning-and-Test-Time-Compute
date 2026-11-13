import json
from pathlib import Path


ds = "math"
policy = "agreement" or "quality_aware"
# policy = "quality_aware"
base = Path("results") / "task3" / ds / "adaptive" / "dev"
files = [p for p in base.glob(f"*{policy}*-metrics.json") if p.is_file()]

results = {}
sc_acc = None
for fp in files:
    tau, gamma, _ = fp.name.split(policy)[-1].strip("-").split("-")
    tau, gamma = float(tau), float(gamma)
    with open(fp, "r") as f:
        metrics = json.load(f)

    if sc_acc is None:
        sc_acc = metrics["Avg-SelfConsist"]
    else:
        # must be same due to reproducibility/seeding
        assert metrics["Avg-SelfConsist"] == sc_acc
    
    ad_tok = metrics["AdaptiveStopping"]["Tokens"]
    ad_acc = metrics["AdaptiveStopping"]["SelfConsist"]
    ad_stop = metrics["AdaptiveStopping"]["Adaptive-Stopping-At"]

    print(sc_acc, ad_acc, ad_tok, ad_stop)

print(len(files))
# print(json.dumps(results[list(results.keys())[0]], indent=4))
