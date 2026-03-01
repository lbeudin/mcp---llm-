"""
Multi-Agent MPC System
Agents:
1. Request Interpreter & Feature Engineer
4. Interpretation & Validation Agent
5. Behavioral Profiling Agent

Production-oriented modular architecture.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import entropy


# ==========================================================
# Dataclass
# ==========================================================
@dataclass
class FeatureEngineeringOutput:
    selected_features: List[str]
    feature_logic: Dict[str, str]
    preprocessing_pipeline: Pipeline

@dataclass
class ValidationReport:
    cluster_narratives: Dict[int, str]
    quality_assessment: Dict[str, Any]
    improvement_recommendations: List[str]

@dataclass
class ClusterProfile:
    probability_tables: Dict[int, pd.DataFrame]
    descriptive_statistics: Dict[int, pd.DataFrame]
    personas: Dict[int, str]


# ==========================================================
# AGENT 1 — Request Interpreter & Feature Engineer
# ==========================================================


class RequestInterpreterFeatureEngineer:

    def __init__(self):
        # create the map between keywords and features - will help the agent to select relevant features based on the request
        self.intent_map = {
            "volatility": [
                "returns",
                "trade_volume",
                "portfolio_value",
                "vix_level"
            ],
            "risk": [
                "beta",
                "drawdown",
                "exposure_equity",
                "exposure_fixed_income"
            ],
            "behavior": [
                "trades",
                "allocation_entropy",
                "cash_ratio"
            ]

        }

    def interpret_request(self, request: str) -> List[str]:
        request = request.lower()
        selected = []
        for keyword, features in self.intent_map.items():
            if keyword in request:
                selected.extend(features)
        return list(set(selected))



    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "returns" in df.columns:
            df["rolling_volatility_30"] = df["returns"].rolling(30, min_periods=5).std()

        if {"trades", "portfolio_value"}.issubset(df.columns):
            df["turnover_rate"] = df["trades"] / (df["portfolio_value"] + 1e-6)

        if {"trade_volume", "vix_level"}.issubset(df.columns):
            df["vix_sensitivity"] = (
                df["trade_volume"] * df["vix_level"]
            ).rolling(14, min_periods=5).mean()

        if "exposure_equity" in df.columns:
            proportions = df[
                [col for col in df.columns if col.startswith("exposure_")]
            ]
            df["allocation_entropy"] = proportions.apply(
                lambda row: entropy(row + 1e-6), axis=1
            )

        return df

    def build_preprocessing_pipeline(self, df: pd.DataFrame) -> Pipeline:

        numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_filter", VarianceThreshold(0.01)),
            ("scaler", RobustScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        return preprocessor

    def process(self, request: str, df: pd.DataFrame) -> FeatureEngineeringOutput:

        selected_features = self.interpret_request(request)
        df_engineered = self.engineer_features(df)

        preprocessing_pipeline = self.build_preprocessing_pipeline(df_engineered)

        feature_logic = {
            "rolling_volatility_30": "30-day rolling standard deviation of returns",
            "turnover_rate": "Trades divided by portfolio value",
            "vix_sensitivity": "Rolling interaction between trade volume and VIX",
            "allocation_entropy": "Portfolio diversification entropy"
        }

        return FeatureEngineeringOutput(
            selected_features=selected_features,
            feature_logic=feature_logic,
            preprocessing_pipeline=preprocessing_pipeline
        )


# ==========================================================
# AGENT 4 — Interpretation & Validation Agent
# ==========================================================


class InterpretationValidationAgent:

    def evaluate_quality(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        quality = {}

        if len(np.unique(labels)) > 1:
            quality["silhouette_score"] = silhouette_score(X, labels)
        else:
            quality["silhouette_score"] = -1

        return quality

    def generate_cluster_narratives(
        self,
        df: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[int, str]:

        narratives = {}
        df_temp = df.copy()
        df_temp["cluster"] = labels

        for cluster_id in sorted(df_temp["cluster"].unique()):
            cluster_df = df_temp[df_temp["cluster"] == cluster_id]

            avg_vol = cluster_df.get("rolling_volatility_30", pd.Series()).mean()
            avg_turnover = cluster_df.get("turnover_rate", pd.Series()).mean()

            description = f"Cluster {cluster_id}: "

            if avg_vol > df_temp["rolling_volatility_30"].mean():
                description += "High volatility exposure; "
            else:
                description += "Low volatility exposure; "

            if avg_turnover > df_temp["turnover_rate"].mean():
                description += "Active trading behavior."
            else:
                description += "Stable trading behavior."

            narratives[cluster_id] = description

        return narratives

    def recommend_improvements(
        self,
        quality_metrics: Dict[str, float]
    ) -> List[str]:

        recommendations = []

        if quality_metrics.get("silhouette_score", 0) < 0.3:
            recommendations.append(
                "Consider dimensionality reduction (PCA/UMAP)."
            )
            recommendations.append(
                "Test density-based clustering (HDBSCAN)."
            )

        if quality_metrics.get("silhouette_score", 0) < 0.2:
            recommendations.append(
                "Revisit feature engineering and reduce noise variables."
            )

        return recommendations

    def process(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        labels: np.ndarray
    ) -> ValidationReport:

        quality = self.evaluate_quality(X, labels)
        narratives = self.generate_cluster_narratives(df, labels)
        recommendations = self.recommend_improvements(quality)

        return ValidationReport(
            cluster_narratives=narratives,
            quality_assessment=quality,
            improvement_recommendations=recommendations
        )


# ==========================================================
# AGENT 5 — Behavioral Profiling Agent
# ==========================================================

class BehavioralProfilingAgent:

    def compute_probability_tables(
        self,
        df: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[int, pd.DataFrame]:

        df_temp = df.copy()
        df_temp["cluster"] = labels

        categorical_cols = df_temp.select_dtypes(exclude=np.number).columns
        probability_tables = {}

        for cluster_id in sorted(df_temp["cluster"].unique()):
            cluster_df = df_temp[df_temp["cluster"] == cluster_id]
            prob_table = {}

            for col in categorical_cols:
                distribution = cluster_df[col].value_counts(normalize=True)
                prob_table[col] = distribution

            probability_tables[cluster_id] = pd.DataFrame(prob_table)

        return probability_tables

    def compute_descriptive_statistics(
        self,
        df: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict[int, pd.DataFrame]:

        df_temp = df.copy()
        df_temp["cluster"] = labels

        numeric_cols = df_temp.select_dtypes(include=np.number).columns
        stats = {}

        for cluster_id in sorted(df_temp["cluster"].unique()):
            cluster_df = df_temp[df_temp["cluster"] == cluster_id]
            stats[cluster_id] = cluster_df[numeric_cols].describe()

        return stats

    def generate_personas(
        self,
        descriptive_stats: Dict[int, pd.DataFrame]
    ) -> Dict[int, str]:

        personas = {}

        for cluster_id, stats in descriptive_stats.items():
            mean_vol = stats.loc["mean"].get("rolling_volatility_30", 0)
            mean_turnover = stats.loc["mean"].get("turnover_rate", 0)

            persona = f"Cluster {cluster_id}: "

            if mean_vol > 0.02:
                persona += "Market-reactive investor; "
            else:
                persona += "Defensive investor; "

            if mean_turnover > 0.1:
                persona += "High engagement."
            else:
                persona += "Low engagement."

            personas[cluster_id] = persona

        return personas

    def process(
        self,
        df: pd.DataFrame,
        labels: np.ndarray
    ) -> ClusterProfile:

        probability_tables = self.compute_probability_tables(df, labels)
        descriptive_stats = self.compute_descriptive_statistics(df, labels)
        personas = self.generate_personas(descriptive_stats)

        return ClusterProfile(
            probability_tables=probability_tables,
            descriptive_statistics=descriptive_stats,
            personas=personas
        )
