# learn.md — Project learning notes

> Revision cheat sheets, one section per phase, appended as we go. Each section is a
> recap you can re-read in 5 minutes before an interview or exam. Nothing here replaces
> the actual code — trust `ml/*.py` and `scripts/*.py`, not your memory of this file.

## Phase 1 — Sprint 1: ML Foundations (✅ done, 2026-08-11)

**Learned in:** Sprint 0 (data pipeline) + Sprint 1 (ml package) walkthroughs — epochs, transfer
learning, freezing, the training loop, and evaluation metrics.

**Answer in your own words:** We reuse a network (MobileNetV2) that already knows how to see images,
freeze it, and only train a small new output layer on our 38 disease classes. After a few passes over
the data (epochs) we check how well it works on images it never trained on, using a confusion matrix,
F1, and related metrics. Class imbalance means we report macro averages so every disease counts equally.

### Type of learning and algorithm (one-liner)

**Supervised multiclass image classification, done with transfer learning** — MobileNetV2 (a CNN)
pretrained on ImageNet as the frozen feature extractor, a new linear head `Linear(1280, 38)` trained on
top, then selective fine-tuning in Stage 2. Trained with Adam + `CrossEntropyLoss`, mixed precision on
GPU, deterministic seed 42.

- **Learning type:** supervised — every photo has a ground-truth class; multiclass — one of 38 labels.
- **Backbone algorithm:** MobileNetV2 CNN (`torchvision.models.mobilenet_v2`, ImageNet-pretrained) →
  1,280-dim feature vector.
- **Head:** `nn.Sequential(nn.Dropout(0.1), nn.Linear(1280, 38))` — the only trainable part in Stage 1.
- **Stage 1 recipe:** frozen backbone + Adam lr 1e-3 + `CrossEntropyLoss`, batch 32, head-only.
- **Stage 2 (Sprint 4/5):** partial fine-tune — last 2 backbone blocks unfrozen at lr 1e-4, head at
  lr 1e-3, warm-started from the Stage 1 checkpoint.
- **Evaluation:** accuracy + macro precision/recall/F1 + row-normalized confusion matrix.
- **Extras:** Albumentations augmentation (train only), `--map-to-pv`/`--mix-with` for cross-dataset
  label alignment.

### Simple-explanations cheat-sheet (recent Q&A, summary)

**Weights** = the model's learnable "dials." Each connection between layers has a number saying how
strongly it matters. MobileNetV2 ≈ 2.2M weights; the head `Linear(1280, 38)` adds ~49k. Training tunes
them; the checkpoint IS them (`state_dict`).

**Gradient** = "which way is downhill," per weight. The loss is one number (how wrong); the gradient is
one number per weight saying *which direction* reduces the error and how steep the slope is there.

**Backpropagation** = the algorithm that computes every weight's gradient in one pass — walks the error
backward layer by layer (chain rule), handing each weight its share of the blame. That's `loss.backward()`.

**The training step (per batch of 32):**
1. **forward** — images → model → 38 scores (logits)
2. **loss** — `CrossEntropyLoss`: how wrong (tiny if true class scored highest)
3. **backward** — `loss.backward()` fills every gradient
4. **update** — `optimizer.step()` (Adam) nudges each weight against its gradient
5. repeat over ~1,357 batches = **1 epoch**; validate, save best checkpoint

**Adam** = the optimizer (weight-update rule). Momentum (rolls smoothly downhill) + per-weight adaptive
step size (rarely-updated weights get bigger steps). Faster, stabler, little LR tuning. Our version has
two param groups: backbone `lr 1e-4`, head `lr 1e-3` (`ml/train.py`).

**Checkpoint** = a saved snapshot (`.pt`): weights + `class_names` + `model_kwargs` + epoch/val_acc +
optimizer/scaler state. `best_*.pt` = best-on-validation. Used by evaluate (score), `--init-from`
(warm-start fine-tune), predict (single image).

**Validation vs test** — validation picks the best checkpoint every epoch (steers training, slightly
"seen"); test is scored once at the end by `ml/evaluate` (referee, never seen) — the CSV rows the
verdict reads are all `*_test`.

**Train/val/test split** — PlantVillage: stratified per-class 80/10/10, seed 42 → 43,429 / 5,417 /
5,459. PlantDoc: keeps its shipped test (236), carves 10% of train as val → 2,107 / 235 / 236 (230
mapped for eval). Same seed = same test set every run.

**Fine-tuning** = keep training an already-trained model on new data, carefully (warm start + low LR +
partial unfreeze) so it adapts without erasing what it knew. Sprint 4 showed it still caused
catastrophic forgetting when trained on one domain only.

**Gates** = pre-set pass/fail thresholds in the verdict cell. Sprint 5: (1) PlantDoc F1 > 0.1116
baseline, (2) PlantVillage F1 ≥ 0.855 (no forgetting). "DONE" only if both pass.

**Mixed training** (`--mix-with plantdoc`) = PlantVillage + PlantDoc in the same epoch
(`ConcatDataset`), so one shared head keeps both domains. `--plantdoc-repeat N` gives PlantDoc a bigger
epoch share (~5% → ~28% at N=8). Two-head (domain-specialized) model = future-work idea in the plan:
better per-domain scores, but needs to know the photo's domain at inference — not the thesis's
single-model story.

**Framework stack** — PyTorch + torchvision (model), Albumentations (aug), scikit-learn (metrics),
pandas (CSV), FastAPI (backend), Next.js/React (frontend). **PyTorch vs TensorFlow:** both are DL
frameworks (tensor math + autograd + layers + optimizers). PyTorch = Pythonic, research-standard,
easy debugging → what we use. TensorFlow = production-oriented (TF Serving/TFLite). TF was only ever a
menu option in the plan, never implemented — torchvision + Albumentations covered everything, so we
use PyTorch end-to-end.

**Vite vs Next.js (both are React)** — Vite = simplest SPA; Next = adds SSR/SEO/routing. SSR/SEO only
pay off for public content sites (Google-crawled landing pages); a single-page upload tool gets zero
benefit. We chose Next.js anyway for resume value (frontend only; FastAPI stays the backend because
PyTorch is Python). Full product features (accounts, chat history, payments, mobile app) are deferred
— the graded demo is one polished page.

### Key facts

- **Epoch** = one full pass over all 43,429 PlantVillage training images; `ml/train.py` loops
  `for epoch in range(start, args.epochs + 1)` (default 5).
- **Pretrained weights** download once to `~/.cache/torch/hub/checkpoints/` (file
  `mobilenet_v2-b0353104.pth`, ~14 MB) via `torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)` (`ml/model.py:37`).
  The saved checkpoint contains this brain + our new layer (that's why a `.pt` is ~9 MB).
- **Freezing** = `for param in model.parameters(): param.requires_grad = False` (`ml/model.py:50-51`).
  `requires_grad=False` means "never update this weight"; the backbone stays fixed.
- **New head** = `model.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(1280, 38))`
  (`ml/model.py:54`). Original had 1000 outputs (ImageNet); we need 38. New layer is random + trainable.
- **Optimizer only sees trainable weights**: `torch.optim.Adam(trainable_parameters(model), lr=1e-3)`
  (`ml/train.py:166`), where `trainable_parameters` yields params with `requires_grad=True`.
- **Training step per batch** (`ml/train.py:90-92`): forward pass → `CrossEntropyLoss` (how wrong) →
  `loss.backward()` (compute gradients) → `optimizer.step()` (nudge weights). Mixed precision
  (`GradScaler`) makes CUDA faster without changing results.
- **Validation set** (10% of PlantVillage) is used each epoch to pick the best checkpoint; the **test
  set** (10%) is touched only by `ml/evaluate.py`, never during training.

### The four outcomes (per class)

"Positive" = the class you're checking for. TRUE/FALSE = right/wrong.

| | said "yes, this class" | said "no, not this class" |
|---|---|---|
| **actually this class** | **TP** True Positive | **FN** False Negative (missed it) |
| **actually another class** | **FP** False Positive (false alarm) | **TN** True Negative |

Confusion matrix cells are exactly this tally (rows = actual, columns = predicted); diagonal = correct.

**One matrix, read per class.** There is exactly ONE confusion matrix per dataset (2×2 for 2 classes,
38×38 for ours). Each class reads its own TP/FP/FN/TN out of that single matrix, treating itself as
"positive" and everything else as "negative" (one-vs-rest):

- From class H's view (2-class): cell(H,H)=TP, cell(H,D)=FN, cell(D,H)=FP, cell(D,D)=TN.
- From class D's view the SAME four cells get swapped: D's TP=cell(D,D), D's FP=cell(H,D),
  D's FN=cell(D,H), D's TN=cell(H,H). That's why H and D both got 0.5/0.5 in the worked example.
- Generalizing to 38 classes, for class *i* in the 38×38 matrix:
  - **TP_i** = diagonal cell (i, i) — truly i, predicted i
  - **FP_i** = rest of column i — predicted i but actually some other disease
  - **FN_i** = rest of row i — actually i but predicted something else
  - **TN_i** = everything not in row i and not in column i

Per-class precision/recall/F1 are computed from these per-class tallies, then macro-averaged across
all 38 classes.

### Formulas (all in `ml/evaluate.py`)

```
accuracy    = correct / total images
precision   = TP / (TP + FP)     "of what I called X, how many really were X?"
recall      = TP / (TP + FN)     "of the real X's, how many did I catch?"
f1          = 2·P·R / (P + R)    harmonic mean — needs BOTH precision and recall to be good
```

- **Macro** = average the per-class score across all 38 classes equally → every disease counts the same.
- **Weighted** = average weighted by each class's image count → big classes dominate.
- **Accuracy** = ratio of correct *images* → big classes dominate the most.
- F1 can beat accuracy at being honest: predicting "everything is healthy" on 9 healthy + 1 diseased
  gives accuracy 0.90 but macro F1 0.47 (the disease class scores 0). `average="macro"` +
  `zero_division=0` in `ml/evaluate.py` implement this.
- PlantVillage is imbalanced (biggest class 5,357 images, smallest 373) → macro is the right headline.

### Why seed 42 everywhere

A **seed** is the starting value given to a random-number generator so its "randomness" becomes
repeatable: same seed → same sequence of random choices. We use `42` (arbitrary convention). Two
places it matters:

- **Splitting** — `organize_datasets.py` does `random.Random(seed).shuffle(...)` per class
  (`_split_files`, seed 42). Same seed ⇒ the *same images* go to train/val/test every run. That's why
  the baseline and Sprint 3 runs saw the identical 5,459-image test set — without it, comparisons
  would be apples-to-oranges.
- **Training** — `set_seed(42)` in `ml/train.py` fixes weight init, dropout, batch-shuffle order and
  augmentation noise; DataLoader workers get `42 + worker_id`. Two runs with the same seed reproduce
  the same result.

Never change the seed between runs you intend to compare.

### Results so far (from actual runs only)

**Sprint 1 baseline** — `baseline_pv_only_no_aug`, PlantVillage **test** split (5,459 images),
MobileNetV2 frozen head-only, 5 epochs, batch 32, lr 1e-3, no augmentation (2026-08-11):

| variant | dataset | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| baseline_pv_only_no_aug | plantvillage_test | 0.9613 | 0.9482 | 0.9561 | 0.9501 |
| augmentation_only | plantvillage_test | 0.9465 | 0.9436 | 0.9335 | 0.9359 |

Logged in `results/ablation_results.csv` (Drive: `folium/results/`). This is ablation variant 1 of 4.

### Worked example (2 classes H/D, 4 test images)

Images: 1(H→H ✓) 2(H→D ✗) 3(D→D ✓) 4(D→H ✗). Matrix: TP=1, FN=1 / FP=1, TN=1.
Class H: precision 0.5, recall 0.5, F1 0.5. Class D: same. Macro F1 = 0.5, accuracy 0.5.

### Debugging story (Sprint 0 — why data lives as zips on Drive)

**Situation:** downloader wrote ~54k image files to Google Drive. **Problem:** a fresh session showed
half the data missing (36/38 classes, corrupt per-class counts) and later runs failed with "Drive quota
exceeded". **Diagnosis:** Colab's FUSE mount can report "written" without committing (session cache
lies), and Drive has a daily *file-operation* quota — ~54k small writes per run blew it; it was never
about free space. **Fix:** each dataset is now one sha256-verified `.zip` on Drive (`_upload_zip`,
`hydrate_dataset` in `scripts/download_datasets.py`); sessions unzip locally. **Prevention:** the Step 7
"durability gate" re-checks Drive archives from a *fresh* session.

### Quiz recap

1. What is an epoch? → one full pass over the training set.
2. What does freezing do? → sets `requires_grad=False` so the pretrained backbone never updates.
3. Why is the new head `Linear(1280, 38)`? → 1280 = features the frozen net outputs, 38 = our classes.
4. What is an FP vs an FN? → FP = false alarm (said X, wasn't), FN = miss (was X, said no).
5. Why macro instead of accuracy? → classes are imbalanced; accuracy lets big classes hide failures on
   small ones, macro counts every disease equally.

### Interview Q&As practiced

- **Q: Why transfer learning instead of training from scratch?** A: We don't have ImageNet-scale data;
  starting from weights that already detect edges/leaf shapes converges far faster and more accurately,
  and freezing the backbone makes Stage 1 training cheap (only ~50k weights train).
- **Q: Why F1 instead of just accuracy?** A: Class imbalance. Accuracy rewards always predicting common
  classes; F1 (harmonic mean of precision and recall) only scores well if the model both catches the
  disease (recall) and doesn't over-alarm (precision). Macro averaging makes small classes count equally.
- **Q: What would you change next?** A: Add augmentation and PlantDoc fine-tuning (Stage 2) — the whole
  project thesis is that the lab-trained model underperforms on real-world field photos, and the
  4-way ablation (`ml/evaluate.py --variant ...`) will quantify it.

### Confused about? (re-review list)

- Reading the confusion matrix PNG in `results/confusion_matrix.png` — which off-diagonal blocks are
  the similar-disease confusions (e.g. early vs late blight).
- The difference between validation (picks the checkpoint, seen every epoch) and test (once, at the end).

## Phase 2 — Sprint 3: Augmentation (✅ done, 2026-08-12)

**Learned in:** Sprint 3 walkthrough — what augmentation is, Albumentations, where the logic lives,
how the sprints share code, and how to read the percent confusion matrix.

### Sprint 3 result (real run)

| variant | dataset | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| baseline_pv_only_no_aug | plantvillage_test | 0.9613 | 0.9482 | 0.9561 | 0.9501 |
| **augmentation_only** | plantvillage_test | **0.9465** | **0.9436** | **0.9335** | **0.9359** |

Augmentation **did not help** on clean lab data: accuracy −0.015, F1 **−0.014** vs baseline (verdict cell:
"Sprint 3 NOT met yet"). Same test set both runs — the split is deterministic (per-class seed-42
shuffle, remainder → test, `_split_files` in `scripts/organize_datasets.py`), so the comparison is
apples-to-apples. This is an honest, expected finding, not a failure: PlantVillage photos are clean
lab shots, so synthetic noise (blur/JPEG/perspective) only distracts the frozen-head classifier. The
thesis prediction stands — augmentation's payoff should come where it's actually needed: messy field
photos in Sprint 4 (PlantDoc fine-tuning).

**Why it can hurt:** each transform injects variation that is *absent from the test set*; on an already
clean 96%-accurate problem, the head just learns to be slightly more uncertain for no real-world gain.
It also ran 10 epochs vs baseline's 5 — more passes over augmented (harder) data, still losing.

### What Sprint 3 is about

Sprint 1's baseline trained on *plain* images (ablation variant 1, 0.9613 acc / 0.9501 F1 on PlantVillage
test). Sprint 3 is ablation variant 2 (`augmentation_only`): retrain the **same model** but feed each
training image a random perturbation every epoch, then check whether the test score improves.

**Augmentation** = randomly transforming each training image on the fly (flip, rotate, brighten, blur,
JPEG-compress...) so the model sees many versions of the same photo. It teaches *disease patterns*, not
*photo conditions* (lighting, angle, focus) — that's how it fights overfitting. It is a **training-time
only** trick; evaluation is never augmented, so scores stay apples-to-apples.

**Why 10 epochs (vs Sprint 1's 5):** augmentation fights memorization (each epoch shows different
random versions of each image), so you can train longer safely; and it converges slower because every
epoch is "harder". 10 is a pragmatic stop — beyond it the frozen head gains little. Not magic: the
notebook says 4–5 epochs is enough for the comparison, because the *relative* change vs baseline is
what matters.

### What Albumentations is

A Python library for image augmentation, the standard in modern CV. Core API:

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Perspective(scale=(0.03, 0.08), keep_size=True, p=0.3),
    A.ImageCompression(compression_type="jpeg", quality_range=(60, 95), p=0.3),
    A.Resize(224, 224),
    A.ToFloat(max_value=255),
])
```

Each transform has a probability `p` (apply to 50% of images, etc.). Richer than torchvision (it has
perspective + JPEG-compression artifact simulation). Newer API: `quality_range=`, plus `A.ToFloat`
**before** `ToTensorV2()` (otherwise the uint8 tensor breaks `Normalize`).

### Where the augmentation logic lives (exactly one place)

| File | Job |
|---|---|
| `ml/data_loading.py:28` `_augment_transform()` | the `A.Compose` list above — the entire policy |
| `ml/data_loading.py:62` `make_transforms(augment=)` | `augment=True` wraps the pipeline in `_AlbumentationsAdapter` (PIL→numpy→tensor); `augment=False` = plain Sprint 1 path |
| `ml/data_loading.py:90` `build_loaders(augment=False)` | passes the transform to the **train** loader only; val/test always plain (`:108`) |
| `ml/train.py:47` `--augment` | on/off switch (store_true, **default False**), forwarded at `:162` |

### Do you write `augment=False` manually? No.

The notebooks already contain the right commands:
- `sprint1_train.ipynb` cell 9 calls `ml.train` **without** `--augment` → defaults to False → plain baseline.
- `sprint3_augmentation.ipynb` cell 9 passes `--augment --tag stage1_aug`.

### Same `ml/` code across all sprints

The `ml/` modules are **shared** — a "sprint" is just a notebook invoking the shared code with different
flags. Sprint 1 vs Sprint 3 differ only in:

| | sprint1 (baseline) | sprint3 (augmented) |
|---|---|---|
| train cmd | no `--augment` | `--augment --tag stage1_aug` |
| checkpoint files | `best_plantvillage_stage1.pt` | `best_plantvillage_stage1_aug.pt` |
| eval variant | `baseline_pv_only_no_aug` | `augmentation_only` |
| CSV row | separate | separate |

They never read each other. Caveats if you re-run sprint1: it **overwrites**
`best_plantvillage_stage1.pt` (same filename), and CSV dedup now keys on
`variant + dataset + split + checkpoint` (not the timestamp), so re-running the same checkpoint is
skipped instead of spamming duplicate rows.

### The confusion matrix (percent view) — what the boxes and colors mean

- **Grid:** 38 rows × 38 columns, one per disease class. Row = **actual** class; column = **predicted**.
- **Number in each box** = % of that row's actual class that the model predicted as that column.
  Each row sums to 100%. The **diagonal** = how often the model got that class right = that class's
  recall. **Off-diagonal** = where the mistakes go (e.g. 4% of true scab images called black rot).
- **Why percent instead of counts:** raw counts hid errors — a class with 5,357 test images dwarfs one
  with 373, so the same 15 errors looked tiny on the big class. Percent makes every class comparable.
- **Color** = the same number drawn as blue intensity scaled 0→100% (`vmin=0, vmax=100`): darker =
  higher %. Diagonal is dark; significant mistakes are light spots you can spot instantly.
- **Annotations** only on cells ≥2% (`CM_ANNOT_THRESHOLD_PERCENT`), so 1,438 near-zero cells stay clean.
  Text is white on dark cells, black on light cells for contrast. Colorbar says "% of actual class (row)".
- `--cm-style raw|percent` on `ml/evaluate` (default `percent`); `raw` keeps original counts.

### Quiz recap (Sprint 3)

1. What is augmentation? → random training-time transforms so the model sees many versions of each image.
2. Where does the augmentation list live? → `ml/data_loading.py:28` `_augment_transform()`.
3. Which split gets augmented? → **train only**; val/test never.
4. What does each box in the percent matrix mean? → % of that actual class predicted as that column;
   diagonal = recall.
5. Re-running sprint1 uses the augmented model? → No — it re-trains the plain baseline into its own
   `stage1` checkpoint; sprints are independent.

### Interview Q&As practiced (Sprint 3)

- **Q: How would you reduce overfitting?** A: Data augmentation (random flips/rotation/brightness/blur so
  the model learns invariant disease patterns rather than photo conditions), plus early stopping on a
  validation set — which is why we keep the best-val checkpoint.
- **Q: Why is the confusion matrix more informative than accuracy?** A: One number hides *where* the
  errors are. A row-normalized matrix shows each class's recall on the diagonal and exactly which
  similar-disease pairs get confused off-diagonal (e.g. early vs late blight).
- **Q: Why percent-normalize the matrix?** A: Class imbalance — raw counts make small classes look clean
  and big-class errors invisible; percentages make all 38 classes comparable.

## Phase 3 - Sprint 4: PlantDoc Fine-Tuning & the Real-World Gap (✅ done, 2026-08-13)

**Learned in:** Sprint 4 build + real run — cross-dataset evaluation, Stage 2 fine-tuning, the gap story,
and catastrophic forgetting.

### Sprint 4 result (real run)

| variant | dataset | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| baseline_pv_only_no_aug | plantvillage_test | 0.9613 | 0.9482 | 0.9561 | 0.9501 |
| baseline_pv_only_no_aug | plantdoc_test | **0.1435** | 0.1885 | 0.1161 | **0.1116** |
| pv_plus_plantdoc | plantdoc_test | 0.5391 | 0.5559 | 0.5311 | 0.5090 |
| pv_plus_plantdoc | plantvillage_test | 0.3744 | 0.4438 | 0.3822 | 0.3116 |
| both | plantdoc_test | 0.5739 | 0.6071 | 0.5769 | 0.5578 |
| both | plantvillage_test | 0.3805 | 0.4233 | 0.4080 | 0.3168 |

**Finding 1 — the real-world gap is real and huge.** The baseline that scores 0.9613/0.9501 on clean
PlantVillage lab photos collapses to **0.1435 / 0.1116** on PlantDoc field photos (same 38-class label
space via `--map-to-pv`, 230 mapped test images). F1 drop **−0.839**. This is the thesis headline, now
measured from a real run.

**Finding 2 — catastrophic forgetting is confirmed.** Stage 2 fine-tuning closed most of the PlantDoc
gap (F1 0.1116 → **0.5090** for `pv_plus_plantdoc`, → **0.5578** for `both`), but both fine-tuned models
collapsed on PlantVillage test: F1 **0.9501 → 0.3116** (`pv_plus_plantdoc`) / **0.3168** (`both`).
Training on only 2,107 PlantDoc photos for 10 epochs overwrote the lab-domain knowledge — a classic,
well-known failure mode, now documented with real numbers. The sprint gate ("did fine-tuning beat the
baseline on PlantDoc?") was met, but **no single model can serve both domains yet** — that motivated
Sprint 5 (mixed training).

### The ablation CSV — what every row means

The paper's numbers all come from **one file**: `results/ablation_results.csv` (Drive:
`folium/results/`). **One row = one evaluation run** — every time `ml/evaluate` scores a checkpoint on
a test split it appends a row. Nothing is hand-typed; a number is only in the table if an actual run
produced it.

**Every column** (defined in `ABLATION_COLUMNS`, `ml/evaluate.py:36`):

| column | meaning |
|---|---|
| `variant` | the experiment/model recipe name (e.g. `both`). Same variant + same checkpoint = same trained model. |
| `dataset` | **which test set** that model was scored on: `plantvillage_test` (5,459 clean lab photos) or `plantdoc_test` (230 field photos, PlantDoc classes mapped onto the 38 PV labels). |
| `split` | always `test` for ablation rows (train/val exist for debugging only). |
| `backbone` | `mobilenet_v2` — the frozen feature extractor. |
| `epochs` | the saved checkpoint's `epoch` field — the epoch that was **best on validation**, not how many were configured. |
| `classes` | length of the checkpoint's `class_names` — always 38, because PlantDoc is mapped into the same PV label space. |
| `test_images` | images actually scored (5,459 PV / 230 mapped PlantDoc — the 236 original minus 6 in classes that map nowhere). |
| `accuracy` `precision` `recall` `f1` | macro metrics on that one run. |
| `timestamp` | when the row was logged. |
| `checkpoint` | the exact `.pt` file scored (identity column). |

**Reading a row:** `pv_plus_plantdoc | plantdoc_test | ... f1 0.5090` = "the Stage-2 PlantDoc
fine-tune scored F1 0.5090 on PlantDoc's field-photo test set."

**Why multiple rows per variant:** each model is evaluated on **both** test sets, so every Sprint 4/5
variant appears twice — once on `plantdoc_test` (did the gap close?) and once on `plantvillage_test`
(did it forget?). What each pair means:

| variant | `plantvillage_test` row | `plantdoc_test` row |
|---|---|---|
| `baseline_pv_only_no_aug` | in-domain lab score (the 0.9613 ceiling) | **the gap** — same model collapses on field photos (0.1435) |
| `augmentation_only` | Sprint 3 check (0.9465) | *no row — Sprint 3 never scored PlantDoc* |
| `pv_plus_plantdoc` | forgetting check (0.3744) | fine-tune on PlantDoc, no aug (0.5391) |
| `both` | forgetting check (0.3805) | fine-tune on PlantDoc + aug (0.5739) |
| `mixed` / `mixed_aug` (Sprint 5) | no-forgetting gate | PlantDoc gate |

**Dedup rule:** re-running `ml/evaluate` on the same `variant + dataset + split + checkpoint` prints
"Skipped duplicate" instead of appending — only the timestamp would differ. Genuinely new combinations
are always added, so the table never bloats but never loses a result.

### The thesis in one line

A model trained on clean lab photos (PlantVillage, 96% on its own test) underperforms on real-world
field photos (PlantDoc). Sprint 4 **measures** that gap, then **tries to close it** with Stage 2
fine-tuning.

### How it works

- **`--map-to-pv`** (`ml/data_loading.py`): reads PlantDoc but translates each folder to its aligned
  PlantVillage class via `class_map.json` (built from `PLANTDOC_TO_PLANTVILLAGE` in
  `scripts/organize_datasets.py`, 28 classes). PlantDoc classes with no match are **dropped**. The
  loader returns the full sorted PlantVillage class list, so the same 38-class head is used everywhere
  — that's what makes the numbers comparable.
- **Stage 2 fine-tune** (`ml/train.py`): `--init-from best_plantvillage_stage1.pt` (warm start, fresh
  optimizer/epochs) + `--unfreeze-blocks 2` (re-enable gradients on the last parameter-bearing
  backbone modules) + low `--lr 1e-4` for the backbone, `--head-lr 1e-3` for the head (two optimizer
  param groups).
- **Variants:** 3 `pv_plus_plantdoc` (fine-tune, no aug) and 4 `both` (fine-tune + `--augment`).
  Checkpoints: `best_plantdoc_stage2.pt` / `best_plantdoc_stage2_aug.pt`.
- **The gap number:** evaluate `best_plantvillage_stage1.pt` on `plantdoc_test --map-to-pv` → the
  baseline's field-photo score. The 4-way ablation CSV now gets `plantdoc_test` rows too.

### The notebook steps: train vs evaluate vs verdict vs predict

The sprint notebook does four different jobs — only ONE of them changes the model:

| step | what it does | changes model? | produces |
|---|---|---|---|
| **train** (Step 5/7) | epoch loop — updates weights, saves a checkpoint each epoch + `best_*.pt` | **YES** | `.pt` files |
| **evaluate** (Step 6/8) | loads a checkpoint, scores it on ONE test set | no | metrics + `cm_*.png` + one CSV row |
| **verdict** (Step 9) | reads the CSV, compares all rows, checks the gate | no | printed comparison + "DONE"/"NOT met" |
| **predict** (Step 10) | loads a checkpoint, classifies 2-3 individual images | no | top-3 labels per image |

One line each: **train makes the model; evaluate measures one model on one test set; verdict compares
all measurements; predict shows individual predictions you can eyeball.** Only train touches weights;
only evaluate touches the CSV; verdict and predict are read-only. Evaluate scores 230/5,459 images at
once (aggregate numbers for the paper); predict classifies a handful of images so you can see real
predictions, not just averages.

### Gotchas found while building

- `unfreeze_last_blocks` (fixed): MobileNetV2's `features` ends with a 1x1 conv, ReLU6,
  `AdaptivePool2d` and `Flatten` — a naive `features[-2:]` would unfreeze pooling (no parameters).
  It now filters to parameter-bearing modules only.
- `ImageFolder.samples` holds `(path, class_index)`, NOT class names — the mapped loader must go
  through `inner.classes[index]`.
- `requires_grad` is **not saved** in the checkpoint (by design: `model_kwargs.freeze=True`, evaluation
  always builds frozen for inference). Unfreezing is a training-time decision.

### Quiz recap (Sprint 4)

1. What is the "real-world gap"? → baseline accuracy/F1 on PlantVillage (clean) vs PlantDoc (field).
2. How do you score a 38-class model on PlantDoc? → `--map-to-pv` translates PlantDoc classes to PV
   labels; unmatched classes are dropped.
3. What does Stage 2 train? → last 2 backbone blocks at low LR + the head at higher LR, warm-started
   from the Stage 1 checkpoint.
4. Why low LR on the backbone? → it already learned good features; big updates would destroy them
   (catastrophic forgetting).

### Interview Q&As practiced (Sprint 4)

- **Q: How do you measure whether your model generalizes to the real world?** A: Cross-dataset
  evaluation — same model, same label space, photos from a different, harder source (PlantDoc field
  shots vs PlantVillage lab shots). The drop quantifies the gap.
- **Q: Why fine-tune instead of retraining?** A: Warm-starting preserves what the model learned about
  diseases in the lab; a small-LR, partial unfreeze adapts to field-photo conditions. Caution — our
  Sprint 4 run proved this is NOT forgetting-free: fine-tuning on PlantDoc alone dropped PlantVillage F1
  from 0.95 to 0.31 (catastrophic forgetting). That's exactly why Sprint 5 trains on both datasets
  together.

## Phase 4 - Sprint 5: Mixed-Domain Training (✅ done, 2026-08-13)

**Learned in:** Sprint 5 build + real run — training on PlantVillage + PlantDoc together so the model
keeps both domains (the fix for the Sprint 4 forgetting).

### Sprint 5 result (real run)

| variant | dataset | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| **mixed** | plantdoc_test | 0.4478 | 0.4160 | 0.4284 | 0.4107 |
| **mixed** | plantvillage_test | **0.9687** | 0.9565 | 0.9633 | **0.9592** |
| mixed_aug | plantdoc_test | 0.4174 | 0.3786 | 0.3354 | 0.3264 |
| mixed_aug | plantvillage_test | 0.9460 | 0.9376 | 0.9360 | 0.9355 |

**Finding 3 — mixed training fixes the forgetting, and then some.** Both gates passed: PlantVillage F1
**0.9592** (`mixed`) is even *higher* than the 0.9501 baseline, and PlantDoc F1 **0.4107** beats the
0.1116 field baseline ~3.7×. Training on both datasets every epoch keeps the lab AND learns the field.

**But it's a trade-off, not a free lunch.** Mixed gained less on PlantDoc than pure fine-tuning did,
because PlantDoc is only ~5% of each mixed epoch:

| model (all real runs) | PlantVillage F1 | PlantDoc F1 |
|---|---|---|
| baseline (PV only) | 0.9501 | 0.1116 |
| pv_plus_plantdoc (PD fine-tune) | 0.3116 | 0.5090 |
| both (PD fine-tune + aug) | 0.3168 | 0.5578 |
| **mixed (PV+PD together)** | **0.9592** | 0.4107 |
| mixed_aug | 0.9355 | 0.3264 |

No single model wins both columns yet: the fine-tunes own field photos (`both` 0.5578) but forget the
lab; `mixed` owns the lab and is the only one usable on both domains. Next lever: oversample PlantDoc
so it gets a bigger share of each mixed epoch — chase `both`'s PlantDoc number without losing
PlantVillage.

### The problem it solves

Sprint 4 fine-tuned on PlantDoc only and forgot PlantVillage (F1 0.95 → 0.31). Sprint 5 trains on
**both datasets in the same epoch** (`torch.utils.data.ConcatDataset`): the 38-class label space is
already aligned via `class_map.json`, so each batch just mixes lab and field photos.

### Variants

- **v5 `mixed`**: `--init-from best_plantvillage_stage1.pt --mix-with plantdoc --tag mixed` (no aug)
  → `best_plantvillage_mixed.pt`.
- **v6 `mixed_aug`**: same + `--augment` → `best_plantvillage_mixed_aug.pt`.

### How it's implemented

- `build_loaders(..., mix_with="plantdoc")` (`ml/data_loading.py`): builds the PlantVillage train/val
  datasets *and* the mapped PlantDoc train/val datasets, then concatenates them into one train loader
  and one val loader. `class_names` = PlantVillage's 38 classes (shared head).
- `ml/train.py --mix-with plantdoc`: forwards the flag; the epoch loop, optimizer and best-val
  checkpoint logic are unchanged. Checkpoint files keep the `best_plantvillage_<tag>.pt` scheme.
- Evaluation unchanged: `ml/evaluate.py` scores any checkpoint on either test set.
- Caveat: PlantDoc is ~5% of each epoch (2,107 vs 43,429 images). The run confirmed the concern —
  `mixed` got PlantDoc F1 0.4107 vs the pure fine-tune's 0.5578. Fix if we chase the field number:
  oversample PlantDoc inside the mixed loader.

### The notebook steps (Sprint 5 numbering)

Same four jobs as Sprint 4, different step numbers:

| step | what it does | changes model? | produces |
|---|---|---|---|
| **train** (Step 4/6) | epoch loop on PlantVillage+PlantDoc (`--mix-with`), saves `best_plantvillage_mixed*.pt` | **YES** | `.pt` files |
| **evaluate** (Step 5/7) | loads a checkpoint, scores it on ONE test set (plantdoc then plantvillage) | no | metrics + `cm_<variant>.png` + one CSV row each |
| **verdict** (Step 8) | reads the CSV, checks the TWO gates (PlantDoc F1 up, no forgetting) | no | printed table + "DONE"/"NOT met" |
| **predict** (Step 9) | classifies 2-3 field + 2-3 lab photos, prints top-3 labels | no | human sanity check |

The evaluate steps produce the CSV rows; the verdict reads them back — it adds no numbers of its own.

### Quiz recap (Sprint 5)

1. Why did Sprint 4 forget PlantVillage? → it trained on PlantDoc only (2,107 images, 10 epochs) and
   overwrote the lab-domain head.
2. How does mixed training fix it? → every epoch sees both datasets, so the head keeps lab knowledge
   while learning field photos.
3. How do two datasets share one 38-class head? → `--map-to-pv` translates PlantDoc folders to
   PlantVillage labels; `ConcatDataset` just concatenates the two datasets.
4. What are the two success gates? → PlantDoc F1 beats the 0.1116 baseline AND PlantVillage F1 stays
   near 0.9501 (no forgetting).

## Phase 5 - Sprint 6: Oversampling PlantDoc in Mixed Training (built & smoke-tested, awaiting the real run)

**Learned in:** Sprint 6 build — repeating the PlantDoc train set inside the mixed loader so the
shared head spends more of each epoch on field photos.

### The problem it solves

Sprint 5's `mixed` kept both domains (PV 0.9592 / PD 0.4107 F1) but PlantDoc lagged the pure
fine-tune (`both` 0.5578), because PlantDoc is only ~5% of each mixed epoch (2,107 vs 43,429 images).
Sprint 6 repeats the PlantDoc train set N times (`--plantdoc-repeat 8` → ~28% of every epoch) to chase
the field number without giving up the lab.

### Variant

- **v7 `mixed_upsampled`**: same as `mixed` + `--plantdoc-repeat 8` → `best_plantvillage_mixed_upsampled.pt`.
- No augmentation variant: aug has hurt field scores twice (`mixed_aug` < `mixed`, Sprint 3 aug < baseline).

### How it's implemented

- `build_loaders(..., plantdoc_repeat=N)` (`ml/data_loading.py`): in the `mix_with` branch, the mapped
  PlantDoc train dataset is repeated N times in the `ConcatDataset`
  (`[pv] + [pd_train] * plantdoc_repeat`); val/test unchanged.
- `ml/train.py --plantdoc-repeat N`: forwards the flag (default 1); header shows
  `plantvillage+plantdoc x8 (60285 train images, 38 classes)`.

### Quiz recap (Sprint 6)

1. Why was `mixed`'s PlantDoc F1 (0.4107) below `both`'s (0.5578)? → PlantDoc is ~5% of each mixed
   epoch; `both` spent 100% of its time on PlantDoc.
2. What does `--plantdoc-repeat 8` change? → PlantDoc appears 8× per epoch (~28% of batches), so the
   head sees field photos far more without dropping the lab set.
3. Why not just train on PlantDoc only? → that's `both`: great field (0.5578) but catastrophic
   forgetting (PlantVillage 0.3168).

## Phase 6 - Sprint 7: stronger backbones for the field ceiling (real run in, 2026-08-16)

**Learned in:** Sprint 7 build + real run. MobileNetV2 looked like the bottleneck (its best field
score is `both`'s 0.5578), so we repeated the field-strong and fine-tune-then-mix recipes on
ResNet-50 and EfficientNet-B0. The headline: **backbone capacity does NOT change the lab-vs-field
trade-off** — every backbone shows the same two outcomes, and no single model clears both gates yet.

### Where Sprint 6 left us (real run, 2026-08-14)

| model | PlantVillage F1 | PlantDoc F1 |
|---|---|---|
| baseline (PV only) | 0.9501 | 0.1116 |
| both (PD fine-tune) | 0.3168 | 0.5578 |
| mixed | 0.9592 | 0.4107 |
| **mixed_upsampled** (repeat 8) | **0.9362** | **0.4122** |

Oversampling PlantDoc 8x barely moved the field score (+0.0015, noise) and cost lab (-0.023). Lesson:
the field bottleneck is NOT epoch share — `both` proved the head can hit 0.5578 when it spends all
its time on field photos. The bottleneck is MobileNetV2 itself plus the one-shared-head balancing
problem. The PlantDoc paper reported ~0.70 accuracy with ResNet-50 on its native classes/split.

### Sprint 7 result (real run)

| model (real runs) | PlantVillage F1 | PlantDoc F1 |
|---|---|---|
| baseline (MobileNetV2, PV only) | 0.9501 | 0.1116 |
| both (MobileNetV2, PD fine-tune) | 0.3168 | 0.5578 |
| both_resnet50 (v11) | 0.2857 | **0.6554** |
| both_efficientnet (v12) | 0.4541 | 0.5719 |
| mixed (MobileNetV2) | **0.9592** | 0.4107 |
| v13 mixed_from_field_resnet50 | 0.9589 | 0.4238 |

**The 3-backbone finding — capacity does not buy both domains.** Every backbone splits into the same
two outcomes:

- **Field-strong (v11/v12):** high field, catastrophic lab forgetting. `both_resnet50` set a new
  field ceiling (**0.6554**, clearing the 0.60 gate) but its lab F1 (0.2857) is the worst yet — the
  bigger the backbone, the deeper it digs into field photos and the worse it forgets the lab.
  `both_efficientnet` is the most balanced (0.5719 / 0.4541) but fails both gates.
- **Mixed (v13):** lab recovered (0.9589, gate PASS), field collapses back to ~0.42 (gate FAIL). The
  field-strong warm start barely helped (0.4238 vs `mixed`'s 0.4107 — ~noise on the 230-image test
  set). **Verdict: field 0.4238 < 0.60 -> FAIL | lab 0.9589 >= 0.85 -> PASS.**

So the lab-vs-field tension is **structural, not a MobileNetV2 limitation**: mixed training re-learns
the shared head on lab-dominated batches (PlantDoc is ~5% of every epoch) and wipes the field again,
on every backbone. The consistent Pareto frontier — no model above 0.60 field **and** 0.85 lab at the
same time — is itself the publishable empirical finding.

**Notebook bug found along the way (fixed, `0a70a8f`):** Stage-2 checkpoint paths were built from the
backbone variable name (`efficientnet_b0`) while training tags used short names (`stage2_efficientnet`),
so v12's evaluate/init-from/audit looked for a non-existent `..._efficientnet_b0.pt`
(`FileNotFoundError`). `resnet50` worked only by coincidence. Fix: explicit `STAGE2_CKPT` map in Step 1.
Moral: don't derive filenames from variable names — the CSV dedup (`variant + checkpoint`) then treats
re-runs with the same tag as the same run.

### The honest ceiling (why 0.60 is plausible, 0.70 is aspirational)

- The PlantDoc test set is only ~230 mapped images: F1 swings of +/-0.03-0.05 are noise (mixed 0.4107
  vs mixed_upsampled 0.4122 was exactly that). A 0.60+ number is a real jump, not luck.
- ResNet-50 (25.6M params) and EfficientNet-B0 (5.3M) both beat MobileNetV2 (3.5M) on ImageNet; on
  2,107 field images that capacity buys better robustness to the background clutter/lighting PlantDoc
  photos have.
- Caps that stay regardless of backbone: PlantDoc's crowd-sourced label noise (some photos are simply
  mislabeled), the small test set, and the mapping onto PlantVillage classes. Label cleaning and more
  real photos are the levers beyond the backbone.

### Why one self-contained model (not two heads)

Every single-model run so far started from the **lab-strong** side: `baseline` (lab 0.95, field 0.11)
-> mixed training kept the lab (0.9592) but only learned field 0.4107. The fix is to start from the
**field-strong** side: warm-start from the field-strong Stage-2 checkpoint (via `--init-from`) and
continue with mixed PlantVillage+PlantDoc training (`--mix-with plantdoc`) to re-learn the lab while
keeping the field. One shared head, no domain toggle, no router — the app ships a single model that
handles real photos on its own.

**Why not the two-head model?** two-head needs to know a photo's domain to pick a head. A "field photo"
toggle is a poor product, and a lab-vs-field style classifier is its own hard problem (the plan's own
caveat). Confidence fusion also fails: the lab head is *confidently wrong* on field photos (that is the
0.11 field F1). A single mixed model avoids routing entirely.

### Variants

- **v11 `both_resnet50`**: the Sprint 4 `both` recipe on ResNet-50 (Stage 1 PlantVillage head-only
  with `--backbone resnet50`, then Stage 2 PlantDoc fine-tune + `--augment`, last 2 blocks unfrozen).
  Artifacts: `best_plantvillage_stage1_resnet50.pt` -> `best_plantdoc_stage2_resnet50.pt`.
- **v12 `both_efficientnet`**: the same recipe on EfficientNet-B0
  (`best_plantvillage_stage1_efficientnet.pt` -> `best_plantdoc_stage2_efficientnet.pt`).
- **v13 `mixed_from_field_<winner>`**: the fine-tune-then-mix warm start — `--init-from` the field
  winner (the higher PlantDoc F1 of v11/v12), then `--mix-with plantdoc` head-only mixed training,
  10 epochs. The single self-contained candidate. Result: field 0.4238 / lab 0.9589 (field gate FAIL).
- **v13_x8 `mixed_from_field_<winner>_x8` (next test)**: v13 + `--plantdoc-repeat 8` (~28% field
  share). Needs a DISTINCT tag so it saves its own checkpoint and logs its own CSV row (dedup keys on
  `variant + checkpoint`; same tag would be skipped as a duplicate and overwrite the old checkpoint).

### How it's implemented (new code)

- `ml/model.py`: `MODEL_FEATURE_DIMS` now includes `resnet50` (2048). `build_model` sets the head on
  `model.fc` for ResNets, `model.classifier` otherwise. `head_module(model)` returns whichever the
  backbone uses (so `train.py` does not hard-code `.classifier`). `_feature_modules(model)` lists the
  backbone's feature modules (MobileNet/EfficientNet: `features`; ResNet: `conv1..layer4`), and
  `unfreeze_last_blocks` filters to parameter-bearing modules then unfreezes the last N — generic
  across backbones.
- `ml/train.py --backbone`: choice from `MODEL_FEATURE_DIMS`, saved in `model_kwargs` so
  evaluate/predict rebuild the same architecture.
- `ml/evaluate.py --tta`: test-time augmentation (average softmax over the image + its horizontal
  flip). Worth ~1-3 points on the small field set; keep OFF for apples-to-apples ablations unless the
  whole table is re-run with it.
- `scripts/audit_plantdoc_labels.py`: scores every PlantDoc image with a field-strong checkpoint and
  writes image path, mapped class, predicted class, confidence (sorted least-confident first) to CSV
  for manual mislabel review.

### The notebook steps (Sprint 7 numbering)

| step | what it does | changes model? | produces |
|---|---|---|---|
| **train Stage 1** (Step 4) | head-only PlantVillage per new backbone | **YES** | `best_plantvillage_stage1_<backbone>.pt` |
| **train v11/v12** (Steps 5/6) | Sprint 4 `both` recipe on each new backbone | **YES** | `best_plantdoc_stage2_<backbone>.pt` |
| **evaluate v11/v12** (Step 7) | field ceiling per backbone, picks winner | no | metrics + CSV rows |
| **train v13** (Step 8) | fine-tune-then-mix on the winner | **YES** | `best_plantvillage_mixed_from_field_<winner>.pt` |
| **evaluate v13** (Step 9) | both gates on the single model | no | metrics + CSV rows |
| **label audit** (Step 10) | flags PlantDoc mislabels with the field-strong winner | no | `plantdoc_label_audit_<winner>.csv` |
| **verdict** (Step 11) | landscape + both gates | no | printed verdict |
| **predict** (Step 12) | 2 field + 2 lab photos | no | human sanity check |

### The two gates (must both pass on ONE checkpoint, no routing)

1. **Field target:** PlantDoc F1 >= **0.60** (stretch 0.70) — the level MobileNetV2 never reached.
2. **Lab recovered:** PlantVillage F1 >= 0.85.

**Decision rule (as applied):** v13 did NOT pass — field 0.4238 < 0.60 (lab 0.9589 >= 0.85 OK). The
field-strong backbone cleared 0.60 (0.6554) but no single model keeps the lab. Per the plan the
fallback is the two-head domain model with an automatic lab/field router — invoked only if the
cheaper single-model levers below fail.

### Future thoughts (decided 2026-08-16)

1. **Cheapest decisive test — give the field a bigger mixed share.** v13 trains head-only with PlantDoc
   at its natural ~5% of every epoch; that is why the head re-collapses to lab. The trainer already
   supports `--plantdoc-repeat N` (Sprint 6 used 8 -> ~28% field share and lab stayed 0.9362, >= 0.85).
   Re-run v13 as `mixed_from_field_resnet50_x8` (distinct tag -> distinct checkpoint -> distinct CSV
   row). If field climbs with lab still >= 0.85, single-model is alive; if it plateaus below 0.60,
   accept the frontier and pivot.
2. **Do NOT switch datasets.** A cleaner field dataset is not the fix: (a) the v13 collapse is a recipe
   problem (epoch share), not labels; (b) every other field dataset (PlantVillage-Taiwan, Cassava Leaf,
   AI Challenger, PlantCLEF) has its own taxonomy, so remapping into our 38 classes would re-introduce
   the very label ambiguity we would be escaping; (c) it would break the paper's PlantVillage+PlantDoc
   continuity.
3. **If x8 plateaus: label audit (Step 10).** `audit_plantdoc_labels.py` quantifies PlantDoc's
   crowd-sourced mislabels — that caps the field-strong *ceiling* (0.6554), not the v13 collapse, and
   is the lever for the 0.70 stretch.
4. **Real phone photos** stay the product-aligned lever (the app's success criteria already require
   them) and the paper's differentiator: labels we control, already in the 38-class schema.
5. **Router fallback** (only if single-model is ruled out): lab-strong `mixed` + field-strong
   `both_resnet50` behind an automatic router. Product concern: a domain toggle is poor UX and a
   lab-vs-field style classifier is its own hard problem.
6. **Paper angle:** whatever single-model run concludes, the 3-backbone x {field-strong, mixed}
   landscape is the empirical robustness study. A third dataset (e.g., Cassava Leaf or
   PlantVillage-Taiwan) used as an *external validation* set — not merged into training — is a good
   paper addition.

### Quiz recap (Sprint 7)

1. Why repeat the recipe on a new backbone? → MobileNetV2's field ceiling is ~0.56; the PlantDoc paper
   reached ~0.70 accuracy with ResNet-50, and its extra capacity should survive real-photo clutter
   better on the same recipe.
2. What does "fine-tune, then mix" (v13) change? → it warm-starts from the field-strong Stage-2
   checkpoint, so the head re-learns the lab from a point already good on field photos.
3. Why not two heads? → two-head needs a domain decision to pick a head (toggle = poor product, router
   = its own hard problem, confidence fusion fails on confidently-wrong lab predictions).
4. Why does `ml/model.py` need `head_module()`? → torchvision ResNets call the head `fc`, MobileNets/
   EfficientNets call it `classifier`; the optimizer must grab the right module per backbone.
5. Why is 0.60 the real gate rather than 0.70? → our evaluation maps PlantDoc onto the 38-class
   PlantVillage space (harder than PlantDoc's native classes), and the ~230-image test set makes
   small deltas noisy.
6. What is the cheapest lever past a plateau? → label cleaning: `audit_plantdoc_labels.py` scores each
   PlantDoc image with the field-strong model and lists the least-confident ones for manual review.
7. Why did v13 (warm-started mix) still collapse on field (0.4238)? → mixed training re-trains the
   shared head on lab-dominated batches (PlantDoc ~5% of every epoch), wiping the field again
   regardless of the backbone's capacity. The warm start only bought ~noise over `mixed` (0.4107).
8. Why is a 0.42 vs 0.41 difference noise? → the PlantDoc test set is ~230 mapped images; +/-0.03-0.05
   F1 swings are noise, so comparisons must beat that band.
9. Why not fix the collapse with a cleaner dataset? → the collapse is an epoch-share/recipe problem,
   not a data problem, and other field datasets have different taxonomies (remapping re-introduces
   label noise). New data is a *ceiling* lever (audit, phone photos), not a *collapse* fix.
