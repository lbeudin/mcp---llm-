import json

import pandas as pd
from scipy.stats import entropy


# ==========================================================
# AGENT 1 — Request Interpreter & Feature Engineer
# ==========================================================

class ConditionalFeatureEngineer:

    def __init__(self):
        self.features = [
            "returns",
            "trade_volume",
            "portfolio_value",
            "vix_level",
            "rolling_volatility_30",
            "turnover_rate",
            "vix_sensitivity",
            "allocation_entropy"
        ]

        self.feature_logic = {
            "emea": self._rolling_volatility_30,
            "momentum": self._turnover_rate,
        }

        self.feature_preprocess= {
             "missing_value_handling": self._rolling_volatility_30}
             "scale": self._rolling_volatility_30}
            #"outlier_treatment": self._turnover_rate,
            # "encoding": self._vix_sensitivity,
            #"dimensionality_reduction": self._allocation_entropy}


    def process(self, df: pd.DataFrame) :
        # lea make copy of df
        with open("features.json", "r") as f:
            config = json.load(f)

        selected_features = config.get("features", [])
        feature_engineering = config.get("feature_engineering", [])
        preprocessing = config.get("preprocessing", [])

        for engineering in feature_engineering :
            func = self.feature_logic.get(engineering.items())
            for features in engineering.values():
                if func and features in self.feature_logic and features in selected_features:
                    df[features] = func(df[features])
                    # todo lea see here if it makes sense to replace or create new column
                else:
                    print(f"Warning: No feature found or no implemtation check Json for {features} in logic {engineering.items}")

        for preprocess in preprocessing :
            func = self.feature_preprocess.get(preprocess.items())
            for features in preprocessing.values():
                if func and features in self.feature_logic and features in selected_features:
                    df[features] = func(df[features])
                else:
                    print(f"Warning: No feature found or no implemtation check Json for {features} in logic {preprocessing.items}")

        self.df = df
        # launch clustering here


    def emea(self,df: pd.series):
        return df.ewm(span=20, adjust=False).mean()


    def momentum(self, df: pd.series,  pct_change = 12):
        return df.pct_change(pct_change)

    def scale(self, df: pd.series):
        return df.pct_change(pct_change)

#
# extends there

