import os
import json
import time
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List

class RecommenderEngine:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        
        # 1. Load pipeline metadata
        metadata_path = os.path.join(artifacts_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        self.threshold: float = self.metadata["optimal_threshold"]
        self.feature_cols: List[str] = self.metadata["feature_columns"]
        self.cat_cols: List[str] = self.metadata["categorical_features"]
        self.model_version: str = self.metadata.get("model_version", "v1.0.0")

        # 2. Load LightGBM booster
        model_path = os.path.join(artifacts_dir, "lgb_model.txt")
        self.booster = lgb.Booster(model_file=model_path)

        # 3. Load lookup stores into memory
        self.user_features = pd.read_parquet(os.path.join(artifacts_dir, "user_features.parquet"))
        self.product_features = pd.read_parquet(os.path.join(artifacts_dir, "product_features.parquet"))
        self.user_product_features = pd.read_parquet(os.path.join(artifacts_dir, "user_product_features.parquet"))
        self.catalog = pd.read_parquet(os.path.join(artifacts_dir, "catalog.parquet"))

        # Index primary keys for fast row selection
        self.user_features.set_index("user_id", inplace=True)
        self.catalog.set_index("product_id", inplace=True)

    def predict(
        self, 
        user_id: int, 
        order_dow: int = 1, 
        order_hour: int = 10, 
        days_since_prior: float = 7.0, 
        top_n: int = 10
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Check user presence
        if user_id not in self.user_features.index:
            return {
                "user_id": user_id,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "threshold_used": self.threshold,
                "total_recommendations": 0,
                "recommendations": []
            }

        # Filter candidate items user has previously purchased
        user_pairs = self.user_product_features[self.user_product_features["user_id"] == user_id].copy()
        if user_pairs.empty:
            return {
                "user_id": user_id,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "threshold_used": self.threshold,
                "total_recommendations": 0,
                "recommendations": []
            }

        # Construct inference feature matrix
        user_row = self.user_features.loc[[user_id]].reset_index()
        infer_df = user_pairs.merge(user_row, on="user_id", how="left")
        infer_df = infer_df.merge(self.product_features, on="product_id", how="left")

        # Inject situational features
        infer_df["current_order_dow"] = order_dow
        infer_df["current_order_hour"] = order_hour
        infer_df["current_days_since_prior"] = days_since_prior

        for col in self.cat_cols:
            if col in infer_df.columns:
                infer_df[col] = infer_df[col].astype("category")

        # Generate model predictions
        X_infer = infer_df[self.feature_cols]
        infer_df["predicted_probability"] = self.booster.predict(X_infer)

        # Filter by optimal threshold and rank
        candidates = infer_df[infer_df["predicted_probability"] >= self.threshold].copy()
        candidates.sort_values(by="predicted_probability", ascending=False, inplace=True)
        top_candidates = candidates.head(top_n)

        # Attach catalog names and categories
        top_products = top_candidates[["product_id", "predicted_probability"]].merge(
            self.catalog, 
            on="product_id", 
            how="left"
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Cast probabilities to float for JSON serialization
        records = []
        for row in top_products.to_dict(orient="records"):
            row["predicted_probability"] = round(float(row["predicted_probability"]), 4)
            records.append(row)

        return {
            "user_id": user_id,
            "latency_ms": round(latency_ms, 2),
            "threshold_used": self.threshold,
            "total_recommendations": len(records),
            "recommendations": records
        }