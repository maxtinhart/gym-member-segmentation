import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data and verify basic structure
df = pd.read_csv("gym_members.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print(f"Average monthly extras spend: ${df['monthly_extras_spend'].mean():.2f}\n")

# 2. Split data: 70% training, 30% test
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

# Behavioral and demographic features (exclude member_id)
features = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]

# Fit scaler strictly on training data, then transform both sets
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[features])
X_test = scaler.transform(test_df[features])

# 3. Fit k-means for k = 1..8 and evaluate train & test spread (SSE)
k_values = range(1, 9)
train_spreads = []
test_spreads = []
models = {}

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(X_train)
    models[k] = km

    # Within-group sum of squared distances on training set
    train_spreads.append(km.inertia_)

    # Assign test members to nearest training center without refitting
    test_labels = km.predict(X_test)
    test_centers = km.cluster_centers_[test_labels]
    test_sse = np.sum((X_test - test_centers) ** 2)
    test_spreads.append(test_sse)

spread_df = pd.DataFrame({
    "k": list(k_values),
    "train_spread": train_spreads,
    "test_spread": test_spreads,
})
print("--- Train vs. Test Spread (Sum of Squared Distances) ---")
print(spread_df.round(2).to_string(index=False))

# Plot training and test elbow curves side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(k_values, train_spreads, marker="o", color="#1f77b4", linewidth=2)
ax1.set_title("Training Set Spread (70%, n=231)")
ax1.set_xlabel("Number of segments (k)")
ax1.set_ylabel("Within-group spread (SSE)")
ax1.set_xticks(list(k_values))
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(k_values, test_spreads, marker="s", color="#ff7f0e", linewidth=2)
ax2.set_title("Test Set Spread (30%, n=99)")
ax2.set_xlabel("Number of segments (k)")
ax2.set_ylabel("Out-of-sample spread (SSE)")
ax2.set_xticks(list(k_values))
ax2.grid(True, linestyle="--", alpha=0.5)

fig.tight_layout()
plt.savefig("elbow_plot.png", dpi=300)
print("\nSaved elbow_plot.png")

# ---------------------------------------------------------
# 4. Inspect chosen k (update CHOSEN_K and TARGET_CLUSTERS after first run)
# ---------------------------------------------------------
CHOSEN_K = 4          # Operations lead wants 3 to 5 segments
TARGET_CLUSTERS = [0]  # Add cluster IDs that get the $10 credit (e.g., [0, 2])

chosen_model = models[CHOSEN_K]

# Convert cluster centers back to original units
centers_original = pd.DataFrame(
    scaler.inverse_transform(chosen_model.cluster_centers_),
    columns=features,
)
centers_original.index.name = "cluster"

# Assign all 330 members to segments
all_X_scaled = scaler.transform(df[features])
df["cluster"] = chosen_model.predict(all_X_scaled)
centers_original["member_count"] = df["cluster"].value_counts().sort_index()

print(f"\n--- Cluster Centers (Original Units) for k = {CHOSEN_K} ---")
print(centers_original.round(2).to_string())

# 5. Function to classify one member and decide credit eligibility
def classify_member(member_features: dict, target_clusters: list):
    row_df = pd.DataFrame([member_features])[features]
    row_scaled = scaler.transform(row_df)
    cluster_id = int(chosen_model.predict(row_scaled)[0])
    gets_credit = cluster_id in target_clusters
    return cluster_id, gets_credit

new_member = {
    "age": 30,
    "visits_per_month": 4,
    "avg_visit_minutes": 42,
    "classes_per_month": 1,
    "months_member": 5,
    "monthly_extras_spend": 10,
}
nm_cluster, nm_credit = classify_member(new_member, TARGET_CLUSTERS)
print(f"\nNew member -> Cluster {nm_cluster}, Gets $10 credit: {nm_credit}")

# 6. Financial quantification (member-by-member across all 330 members)
def member_net_impact(row, receives_credit: bool) -> float:
    if not receives_credit:
        return 0.0
    if row["monthly_extras_spend"] >= 20:
        return -10.0  # Already spends >= $20; $10 credit is pure giveaway
    return 5.0        # Spends $25 more * 60% margin ($15) - $10 credit = +$5 net

baseline_net = df.apply(lambda r: member_net_impact(r, True), axis=1).sum()
v1_net = df.apply(
    lambda r: member_net_impact(r, r["cluster"] in TARGET_CLUSTERS), axis=1
).sum()

print("\n--- Monthly Net Impact from $10 Credit ---")
print(f"Baseline (credit to all {len(df)} members): ${baseline_net:,.2f}/month")
print(f"Targeted v1 (clusters {TARGET_CLUSTERS}):     ${v1_net:,.2f}/month")
print(f"Monthly gain over baseline:            ${v1_net - baseline_net:+,.2f}/month")