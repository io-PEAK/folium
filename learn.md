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

### Results so far (from actual runs only)

**Sprint 1 baseline** — `baseline_pv_only_no_aug`, PlantVillage **test** split (5,431 images),
MobileNetV2 frozen head-only, 5 epochs, batch 32, lr 1e-3, no augmentation (2026-08-11):

| variant | dataset | accuracy | precision | recall | f1 |
|---|---|---|---|---|---|
| baseline_pv_only_no_aug | plantvillage_test | 0.9613 | 0.9482 | 0.9561 | 0.9501 |

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
