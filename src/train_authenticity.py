"""Phase 2 - train an authenticity screening model on folder-labelled images.

    python -m src.train_authenticity --data-dir data/authenticity --model mobilenet
    python -m src.train_authenticity --data-dir data/authenticity --model cnn --epochs 25
    python -m src.train_authenticity --data-dir data/authenticity --model logreg

Outputs go to models/ (model + class_names.json) and reports/authenticity_<model>/.
"""
import argparse
import json

from .config import CLASS_NAMES_PATH, DEFAULT_MODEL_PATH, IMG_SIZE, MODELS_DIR, POSITIVE_CLASS, REPORTS_DIR, SEED
from .data import balanced_class_weights, class_counts, dataset_to_numpy, load_authenticity_splits, load_split
from .evaluate import compute_metrics, plot_confusion_matrix, predict_keras, save_metrics


def train_logreg(args, data_dir, out):
    import joblib

    from .models import build_logistic_regression

    size = (64, 64)  # raw-pixel logistic regression needs small inputs
    train, names = load_split(data_dir, "train", size, 64, shuffle=False)
    test, _ = load_split(data_dir, "test", size, 64)
    xtr, ytr = dataset_to_numpy(train)
    xte, yte = dataset_to_numpy(test)
    clf = build_logistic_regression().fit(xtr.reshape(len(xtr), -1) / 255.0, ytr)
    pred = clf.predict(xte.reshape(len(xte), -1) / 255.0)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "logreg_baseline.joblib")
    return yte, pred, names


def train_keras(args, data_dir, out):
    import tensorflow as tf

    from .models import build_cnn, build_mobilenet_v2, unfreeze_backbone

    tf.keras.utils.set_random_seed(SEED)
    train, val, test, names = load_authenticity_splits(data_dir, IMG_SIZE, args.batch_size)
    weights = balanced_class_weights(class_counts(data_dir, "train", names))
    print("class order:", names, "| class weights:", weights)

    shape = (*IMG_SIZE, 3)
    model = build_cnn(shape, len(names), augment=True) if args.model == "cnn" else build_mobilenet_v2(shape, len(names))
    MODELS_DIR.mkdir(exist_ok=True)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(DEFAULT_MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]
    model.fit(train, validation_data=val, epochs=args.epochs, class_weight=weights, callbacks=cbs, verbose=2)

    if args.model == "mobilenet" and args.fine_tune_epochs > 0:
        print("\n--- fine-tuning last backbone layers ---")
        unfreeze_backbone(model, last_n=30, lr=1e-5)
        model.fit(train, validation_data=val, epochs=args.fine_tune_epochs, class_weight=weights, callbacks=cbs, verbose=2)

    model.save(DEFAULT_MODEL_PATH)
    y_true, y_pred, _ = predict_keras(model, test)
    return y_true, y_pred, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", choices=["logreg", "cnn", "mobilenet"], default="mobilenet")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--fine-tune-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out = REPORTS_DIR / f"authenticity_{args.model}"
    runner = train_logreg if args.model == "logreg" else train_keras
    y_true, y_pred, names = runner(args, args.data_dir, out)

    MODELS_DIR.mkdir(exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(names))
    m = compute_metrics(y_true, y_pred, names, POSITIVE_CLASS)
    save_metrics(m, out / "test_metrics.json")
    plot_confusion_matrix(y_true, y_pred, names, out / "confusion_matrix.png", f"{args.model} (test)")
    print(f"\nTEST  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")
    if f"{POSITIVE_CLASS}_recall" in m:
        print(f"{POSITIVE_CLASS}: precision={m[f'{POSITIVE_CLASS}_precision']:.3f} recall={m[f'{POSITIVE_CLASS}_recall']:.3f}")
    print("Reports saved to", out)


if __name__ == "__main__":
    main()
