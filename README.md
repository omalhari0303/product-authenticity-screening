# AI-Powered Product Authenticity Screening (Computer Vision)

An educational prototype that screens product images for visual patterns consistent with genuine or
potentially counterfeit goods. It covers the full ML workflow: preprocessing, model comparison,
evaluation, explainability (Grad-CAM) and a Streamlit app.

> **Scope:** this is a learning project. It does **not** authenticate luxury goods. Reliable
> authentication needs suitable labelled data, real-world validation and calibration. Model scores are
> uncalibrated softmax outputs, not probabilities.

## Architecture

```
upload image -> preprocess (RGB, pad+resize 224) -> CNN / MobileNetV2 -> score -> Grad-CAM -> Streamlit
```

| Component | File |
|---|---|
| 1. Image input & preprocessing | `src/preprocessing.py` |
| 2. Models (LogReg, CNN, MobileNetV2) | `src/models.py` |
| 3. Evaluation (acc, P/R/F1, confusion matrix) | `src/evaluate.py` |
| 4. Explainability (Grad-CAM) | `src/gradcam.py` |
| 5. Web app | `app/streamlit_app.py` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # sanity check
```
On **Google Colab**: upload/clone the repo, `!pip install -r requirements.txt`, choose a GPU runtime, and run the commands below with `!` prefixes.

## Workflow

**Phase 1: foundations (Fashion-MNIST).** Learn the pipeline; this is *not* an authenticity model.
```bash
python -m src.train_fashion_mnist --epochs 5
```
Outputs a LogReg-vs-CNN comparison table and confusion matrices in `reports/fashion_mnist/`.

**Phase 2: authenticity pipeline.** Put real labelled data in `data/authenticity/` (see `data/README.md`),
or smoke-test with synthetic data:
```bash
python scripts/make_demo_dataset.py                    # synthetic, for testing code only
python -m src.train_authenticity --data-dir data/demo_synthetic --model logreg
python -m src.train_authenticity --data-dir data/demo_synthetic --model cnn --epochs 10
python -m src.train_authenticity --data-dir data/authenticity --model mobilenet   # transfer learning + fine-tune
```
Each run saves `models/authenticity_model.keras`, `models/class_names.json`, and test metrics +
confusion matrix under `reports/authenticity_<model>/`.

**Phase 3: app.**
```bash
streamlit run app/streamlit_app.py
```

## Design decisions worth knowing (good interview talking points)

- **Rescaling lives inside the model**, so training and inference can't drift apart.
- **No horizontal-flip augmentation:** flipping a logo creates a "fake-looking" image for the wrong reason.
- **Padding instead of stretching** keeps stitch geometry and logo proportions intact.
- **Class weights** handle genuine/counterfeit imbalance; judge models by **recall and precision on the
  counterfeit class**, not accuracy alone (the cost of a false "genuine" differs from a false flag).
- **MobileNetV2 is built inline** (not nested) so Grad-CAM can reach `out_relu` directly.
- **Fine-tuning** unfreezes only the last 30 backbone layers, BatchNorm kept frozen, at lr=1e-5.

## Known limitations / next steps
### Dataset and Evaluation Limitations

- The project includes the Goldnet dataset for product authenticity
  screening and model evaluation.
- A synthetic demo dataset is also provided for application testing.
- Model performance depends on the quality, diversity, and
  representativeness of the training data.
- High evaluation scores on a limited dataset do not guarantee
  reliable counterfeit detection in real-world scenarios.
- Further validation across different brands, product categories,
  and image capture conditions is required.
## Structure

```
app/streamlit_app.py     scripts/make_demo_dataset.py     tests/test_preprocessing.py
src/{config,preprocessing,data,models,evaluate,gradcam,train_fashion_mnist,train_authenticity}.py
data/  models/  reports/  requirements.txt
```
