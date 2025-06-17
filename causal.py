import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

# ---------- 1. 数据预处理 ---------- #
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    keep_levels = ['01.超低', '02.低', '03.中', '04.微高', '05.高']
    df = df[df['risk_level'].isin(keep_levels)].copy()

    # 按风险档剔除极端消费额
    def trim(group):
        p95 = group['ins_amt_d30'].quantile(0.95)
        return group[group['ins_amt_d30'] <= p95]
    df = df.groupby('risk_level', group_keys=False).apply(trim)
    return df.reset_index(drop=True)

# ---------- 2. 倾向得分 + 结果模型 ---------- #
def fit_causal_models(df: pd.DataFrame,
                      features: list,
                      n_estimators: int = 200,
                      learning_rate: float = 0.05,
                      max_depth: int = -1):

    X = df[features]
    T = (df['delta_quota'] > 0).astype(int)
    Y = df['ins_amt_d30']

    # 2.1 Propensity model
    ps_model = LGBMClassifier(n_estimators=n_estimators,
                              learning_rate=learning_rate,
                              max_depth=max_depth,
                              objective='binary')
    ps_model.fit(X, T)
    e_hat = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
    w = T / e_hat + (1 - T) / (1 - e_hat)

    # 2.2 Outcome models
    treat_model = LGBMRegressor(n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                objective='regression')
    ctrl_model = LGBMRegressor(n_estimators=n_estimators,
                               learning_rate=learning_rate,
                               max_depth=max_depth,
                               objective='regression')

    treat_model.fit(X[T == 1], Y[T == 1], sample_weight=w[T == 1])
    ctrl_model.fit(X[T == 0], Y[T == 0], sample_weight=w[T == 0])

    mu1 = treat_model.predict(X)
    mu0 = ctrl_model.predict(X)
    tau_hat = mu1 - mu0

    return tau_hat, ps_model, treat_model, ctrl_model

# ---------- 3. 敏感用户筛选 ---------- #
def flag_sensitive(df: pd.DataFrame,
                   tau_hat: np.ndarray,
                   mode: str = 'global',
                   quantile: float = 0.90,
                   abs_threshold: float = None) -> pd.Series:

    df = df.copy()
    df['tau_hat'] = tau_hat

    if abs_threshold is not None:
        threshold_func = lambda g: abs_threshold
    elif mode == 'global':
        global_q = df['tau_hat'].quantile(quantile)
        threshold_func = lambda g: global_q
    elif mode == 'by_risk':
        quantiles = df.groupby('risk_level')['tau_hat'].quantile(quantile)
        threshold_func = lambda g: quantiles.loc[g.name]
    else:
        raise ValueError("mode 应为 'global' 或 'by_risk'")

    sensitive_flag = df.groupby('risk_level')['tau_hat'].apply(
        lambda g: g > threshold_func(g)
    ).reset_index(level=0, drop=True)
    return sensitive_flag.astype(int)

# ---------- 4. 主流程 ---------- #
def identify_quota_sensitive(df: pd.DataFrame,
                             covariate_cols: list,
                             mode: str = 'global',
                             quantile: float = 0.90,
                             abs_threshold: float = None) -> pd.DataFrame:

    df_clean = preprocess(df)
    tau_hat, *_ = fit_causal_models(df_clean, covariate_cols)
    df_clean['is_sensitive'] = flag_sensitive(
        df_clean, tau_hat, mode, quantile, abs_threshold
    )
    return df_clean

# ▼▼▼ 使用示例 ▼▼▼
# covs = [c for c in df.columns if c not in ('delta_quota', 'ins_amt_d30', 'risk_level')]
# result = identify_quota_sensitive(df, covs, mode='by_risk', quantile=0.85)
# 敏感用户列表：
# sensitive_users = result[result['is_sensitive'] == 1]
