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

## Phase 4 - Sprint 5: Mixed-Domain Training (built & smoke-tested, awaiting the real run)

**Learned in:** Sprint 5 build — training on PlantVillage + PlantDoc together so the model keeps both
domains (the fix for the Sprint 4 forgetting).

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
- Caveat: PlantDoc is ~5% of each epoch (2,107 vs 43,429 images). If its signal is too weak we can
  oversample PlantDoc later.

### Quiz recap (Sprint 5)

1. Why did Sprint 4 forget PlantVillage? → it trained on PlantDoc only (2,107 images, 10 epochs) and
   overwrote the lab-domain head.
2. How does mixed training fix it? → every epoch sees both datasets, so the head keeps lab knowledge
   while learning field photos.
3. How do two datasets share one 38-class head? → `--map-to-pv` translates PlantDoc folders to
   PlantVillage labels; `ConcatDataset` just concatenates the two datasets.
4. What are the two success gates? → PlantDoc F1 beats the 0.1116 baseline AND PlantVillage F1 stays
   near 0.9501 (no forgetting).
