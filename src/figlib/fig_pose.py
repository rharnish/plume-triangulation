import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

audit = [r for r in json.load(open("out/terrain_audit.json"))
         if r.get("coverage", 0) >= 0.5 and r.get("resid_median_px") is not None]
staged = [r for r in json.load(open("out/pose_fit_staged.json")) if r["status"] == "fitted"]

fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.6))

# 1. the audit
m = np.abs([r["resid_median_px"] for r in audit])
ax[0].hist(m, bins=np.arange(0, 720, 40), color="#4c72b0", edgecolor="white")
ax[0].axvline(13, color="#d62728", lw=2)
ax[0].annotate("the one camera checked\nearlier: 13 px", xy=(13, 6.5),
               xytext=(210, 9.0), fontsize=8, color="#d62728",
               arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
ax[0].axvline(np.median(m), color="0.25", ls="--", lw=1.5)
ax[0].text(np.median(m) + 14, ax[0].get_ylim()[1] * 0.40,
           f"median {np.median(m):.0f} px", fontsize=8.5, color="0.25")
ax[0].set_xlabel("|median skyline residual| (px)")
ax[0].set_ylabel("cameras")
ax[0].set_title(f"Pose audit, n={len(m)} cameras")

# 2. azimuth is not identifiable
c = np.array([r["peak_corr"] for r in staged])
a = np.array([r["az_raw"] for r in staged])
sc = ax[1].scatter(a, c, c=["#2ca02c" if r["identifiable"] else "#999999"
                            for r in staged], s=26)
ax[1].axhline(0.35, color="#d62728", ls="--", lw=1.4)
ax[1].text(-2.9, 0.37, "identifiability gate", fontsize=8, color="#d62728")
ax[1].set_xlabel("azimuth correction proposed (deg)")
ax[1].set_ylabel("ridgeline feature correlation")
ax[1].set_title("Azimuth: 1 of 61 cameras identifiable")
ax[1].text(0.03, 0.03, f"proposals center on zero:\nmedian {np.median(a):+.2f}°, "
           f"sd {a.std():.2f}°\n→ published azimuths hold up",
           transform=ax[1].transAxes, fontsize=8, color="0.25")
ax[1].grid(alpha=0.3)

# 3. what it did to the fires
labels = ["published\npose", "4-param fit\n(az free)", "staged fit\n(az pinned, k1)"]
med = [2.16, 3.38, 2.28]
w2 = [12, 6, 12]
x = np.arange(3)
b = ax[2].bar(x, med, 0.55, color=["#4c72b0", "#d62728", "#dd8452"])
for xi, v, n in zip(x, med, w2):
    ax[2].text(xi, v + 0.06, f"{v:.2f} km\n{n}/26 ≤2 km", ha="center", fontsize=8.5)
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=8.5)
ax[2].set_ylabel("median geolocation error (km)")
ax[2].set_ylim(0, 4.3)
ax[2].set_title("Held-out test: error against WFIGS")
ax[2].text(0.5, 0.94, "skyline residual improved 106 → 24 px\nwhile the fires got further away",
           transform=ax[2].transAxes, ha="center", va="top", fontsize=8, color="0.25")
ax[2].grid(axis="y", alpha=0.3)

fig.text(0.5, 0.015, "Refining camera pose against a DEM-synthesised skyline: the surface "
         "metric improves four-fold and the metric that matters gets worse.",
         ha="center", fontsize=8.5, color="0.3")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("out/figures/pose_fit.png", dpi=140)
print("wrote out/figures/pose_fit.png")
