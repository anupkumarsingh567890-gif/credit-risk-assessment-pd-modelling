"""Page 6: Model Evaluation - compare models, move the threshold, review validation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    MODEL_COLUMNS, champion, figure, kpi_row, load_table, load_test_predictions, setup_page,
)

import streamlit as st  # noqa: E402

from src.metrics import all_metrics, confusion  # noqa: E402

setup_page("Model Evaluation & Validation",
           "Out-of-time test: 12,458 loans issued July - December 2011, never used for fitting or tuning")
champ = champion()
preds = load_test_predictions()

st.subheader("Model comparison (out-of-time test)")
metrics = load_table("eval_metrics")
test = metrics[metrics["sample"] == "test"].drop(columns="sample").round(4)
test.insert(1, "champion", test["model"].eq(champ["model"]).map({True: "yes", False: ""}))
st.dataframe(test, hide_index=True, use_container_width=True)
st.caption(f"**Champion: {champ['model']}.** {champ['reason']} Ranking metrics (AUC, Gini, KS) and the "
           "Brier score matter most for a PD model; accuracy is misleading with ~14% defaults.")

st.subheader("Decision threshold")
c1, c2 = st.columns([1, 2])
model = c1.selectbox("Model", list(MODEL_COLUMNS), index=list(MODEL_COLUMNS).index(champ["model"]))
threshold = c1.slider("Flag as predicted default when PD >=", 0.02, 0.60, round(champ["threshold"], 3), 0.005,
                      help="Default: the training default rate")
m = all_metrics(preds["default"], preds[MODEL_COLUMNS[model]], threshold)
with c2:
    kpi_row([("Accuracy", f"{m['accuracy']:.3f}"), ("Precision", f"{m['precision']:.3f}"),
             ("Recall", f"{m['recall']:.3f}"), ("F1", f"{m['f1']:.3f}")])
    st.dataframe(confusion(preds["default"], preds[MODEL_COLUMNS[model]], threshold), use_container_width=True)
    st.caption(f"Threshold-free: AUC {m['roc_auc']:.3f} | Gini {m['gini']:.3f} | KS {m['ks']:.3f} | "
               f"Brier {m['brier']:.4f} | mean PD {m['mean_pd']:.2%} vs observed {m['observed_default_rate']:.2%}")

st.subheader("Curves")
tabs = st.tabs(["ROC", "Precision-Recall", "Calibration", "Confusion matrices"])
with tabs[0]:
    figure("evaluation", "01_roc_curves")
with tabs[1]:
    figure("evaluation", "02_precision_recall_curves")
with tabs[2]:
    figure("evaluation", "03_calibration_curves")
with tabs[3]:
    figure("evaluation", "04_confusion_matrices")

st.subheader("Validation summary (champion)")
validation = load_table("validation_summary")
st.dataframe(validation, hide_index=True, use_container_width=True)
flagged = validation[validation["status"] != "Pass"] if not validation.empty else validation
if len(flagged):
    st.warning("Flagged: " + "; ".join(flagged["check"]) + ". The test period (late 2011) had a higher "
               "default rate than the training period, so predicted PDs are too low on average. In "
               "production the PD level would be recalibrated to recent experience.")

v1, v2 = st.columns(2)
with v1:
    st.markdown("**Train vs cross-validation vs test AUC**")
    st.dataframe(load_table("validation_oot_overfitting").round(4), hide_index=True, use_container_width=True)
    figure("validation", "01_score_psi")
    figure("validation", "03_calibration_deciles")
with v2:
    st.markdown("**Characteristic stability (CSI)**")
    st.dataframe(load_table("validation_csi_features").round(4), hide_index=True, use_container_width=True)
    figure("validation", "04_threshold_sensitivity")
    figure("validation", "05_auc_by_quarter")
figure("validation", "02_feature_importance")
