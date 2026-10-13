"""Leakage-aware fraud risk modeling. Run: python fraud_risk_pipeline.py --data creditcard.csv"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--split", choices=["chronological", "stratified"], default="chronological")
    parser.add_argument("--fp-cost", type=float, default=10)
    parser.add_argument("--fn-cost", type=float, default=500)
    args = parser.parse_args()
    if args.fp_cost < 0 or args.fn_cost <= 0:
        parser.error("Costs must be nonnegative, and fn-cost positive")
    df = pd.read_csv(args.data).dropna(subset=["Class", "Time"])
    expected = {"Class", "Time", "Amount", *(f"V{i}" for i in range(1, 29))}
    if expected - set(df):
        raise ValueError(f"Missing columns: {sorted(expected - set(df))}")
    if not df.Class.isin([0, 1]).all():
        raise ValueError("Class must be binary (0/1)")
    if args.split == "chronological":
        df = df.sort_values("Time", kind="stable")
        a, b = int(len(df)*.6), int(len(df)*.8)
        train, val, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    else:
        train, rest = train_test_split(df, test_size=.4, stratify=df.Class, random_state=42)
        val, test = train_test_split(rest, test_size=.5, stratify=rest.Class, random_state=42)
    if any(s.Class.nunique() < 2 for s in (train, val, test)):
        raise ValueError("Every split must contain both classes; use a larger dataset or stratified split")
    cols = [c for c in df if c != "Class"]
    X, y = train[cols], train.Class.astype(int)
    Xv, yv = val[cols], val.Class.astype(int).to_numpy()
    Xt, yt = test[cols], test.Class.astype(int).to_numpy()
    ratio = (len(y)-y.sum())/y.sum()
    models = {
        "logistic_regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        "xgboost": make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=250, max_depth=4, learning_rate=.05, subsample=.85, colsample_bytree=.85, scale_pos_weight=ratio, eval_metric="logloss", n_jobs=2, random_state=42))
    }
    comparisons = {}
    for name, model in models.items():
        model.fit(X, y)
        prob = model.predict_proba(Xv)[:,1]
        thresholds = np.linspace(.01,.99,99)
        costs = [args.fp_cost*np.sum((prob>=t)&(yv==0)) + args.fn_cost*np.sum((prob<t)&(yv==1)) for t in thresholds]
        idx = int(np.argmin(costs))
        comparisons[name] = {"average_precision": float(average_precision_score(yv,prob)), "threshold": float(thresholds[idx]), "validation_cost": float(costs[idx])}
    selected = max(comparisons, key=lambda k: comparisons[k]["average_precision"])
    threshold = comparisons[selected]["threshold"]
    prob = models[selected].predict_proba(Xt)[:,1]
    pred = (prob>=threshold).astype(int)
    tn,fp,fn,tp = map(int,confusion_matrix(yt,pred,labels=[0,1]).ravel())
    result = {"split":args.split, "rows":len(df), "split_sizes":{"train":len(train),"validation":len(val),"test":len(test)}, "fraud_rates":{"train":float(y.mean()),"validation":float(yv.mean()),"test":float(yt.mean())}, "validation":comparisons, "selected_model":selected, "threshold":threshold, "test":{"precision":float(precision_score(yt,pred,zero_division=0)),"recall":float(recall_score(yt,pred,zero_division=0)),"f1":float(f1_score(yt,pred,zero_division=0)),"average_precision":float(average_precision_score(yt,prob)),"roc_auc":float(roc_auc_score(yt,prob)),"tn":tn,"fp":fp,"fn":fn,"tp":tp,"illustrative_cost":args.fp_cost*fp+args.fn_cost*fn}}
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(result,indent=2))
    pd.DataFrame({"actual":yt,"fraud_score_uncalibrated":prob,"flagged":pred}).to_csv(output/"predictions.csv",index=False)
    if selected=="xgboost":
        try:
            import shap
            sample=Xt.sample(n=min(200,len(Xt)),random_state=42)
            model=models[selected]
            values=shap.TreeExplainer(model[-1]).shap_values(model[0].transform(sample))
            pd.DataFrame({"feature":cols,"mean_abs_shap":np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap",ascending=False).to_csv(output/"shap_importance.csv",index=False)
        except (ImportError,ValueError,TypeError) as e:
            print("Optional SHAP unavailable:",e)
    print(json.dumps(result,indent=2))
    print("Scores are NOT calibrated fraud probabilities. Costs are illustrative.")
if __name__=="__main__":
    main()
