# Helper run (CPU): the 9 Oct best teacher (T3-seed2) forecasts every test hour at unit 0, 27 training stations.
# Same output format as exp_wind (pred_<name>.npz), so the fusion experiment can use it without waiting for a GPU.
from nb import nb04_cells, code, md, notebook

GPU = False
DATASETS = ["samiulseikh/aqi-features"]
KERNELS = ["samiulseikh/aqi-train"]

PRED = """
# load the saved best teacher and forecast every test hour of unit 0 (training stations)
info = json.loads(next(Path("/kaggle/input").rglob("teacher_best.json")).read_text())
assert np.allclose(info["mean"], MEAN, atol=1e-4), "normalisation differs from the saved teacher"
model = Teacher(*info["switches"]).to(DEVICE)
model.load_state_dict(torch.load(next(Path("/kaggle/input").rglob("teacher_best.pt")), map_location=DEVICE))
test_seen = sample_pairs("test", train_idx)
unit0 = test_seen[test_seen[:, 0] == 0]
start = time.time()
p = predict(model, unit0, train_idx)
t = unit0[:, 1].cpu().numpy()
np.savez_compressed(OUT / "pred_best.npz", y_hat=p["y_hat"].astype(np.float16), t=t, stations=train_idx)
np.savez_compressed(OUT / "truth_unit0.npz", y=p["y"], y_mask=p["y_mask"].astype(np.uint8), t=t)
m = p["y_mask"]
print(info["run"], "samples", len(t), f"{time.time() - start:.0f}s",
      "skill 1-6h:", round(100 * (1 - np.sqrt((((p["y_hat"] - p["y"]) ** 2) * m).sum() / ((p["y"] ** 2) * m).sum())), 2))
"""


def build(smoke):
    cells = nb04_cells()[:44]
    cells = [c for i, c in enumerate(cells) if i not in (16, 20, 21, 38, 43)]      # checks/plots/training not needed
    cells += [code("""
# geometry of the training stations (cell 21 of 04, without its plot)
dist, bearing = geometry(lat[train_idx], lon[train_idx])
"""), code(PRED)]
    cells[0] = md("# Helper: test forecasts of the saved best teacher (for experiment 4)")
    return notebook(cells)
