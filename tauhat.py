import pandas as pd
import numpy as np
from scipy import stats

def bin_and_evaluate(df: pd.DataFrame,
                     nbins: int = 10,
                     tau_col: str = 'tau_hat',
                     treat_col: str = 'delta_quota',
                     outcome_col: str = 'ins_amt_d30'):
    # 处理指示
    df = df.copy()
    df['T'] = (df[treat_col] > 0).astype(int)
    
    # 等频分箱
    df['tau_bin'] = pd.qcut(-df[tau_col], q=nbins, labels=False)  # -号确保降序
    summary = []

    for b in range(nbins):
        sub = df[df['tau_bin'] == b]
        n_total = len(sub)
        n_treat = sub['T'].sum()
        prop_treat = n_treat / n_total if n_total else np.nan

        mean_outcome = sub[outcome_col].mean()
        mean_treat = sub.loc[sub['T'] == 1, outcome_col].mean()
        mean_ctrl  = sub.loc[sub['T'] == 0, outcome_col].mean()
        uplift     = mean_treat - mean_ctrl

        # 可选：检验 treated 与 control 均值差异显著性
        t_stat, p_val = stats.ttest_ind(sub.loc[sub['T'] == 1, outcome_col],
                                        sub.loc[sub['T'] == 0, outcome_col],
                                        equal_var=False, nan_policy='omit')

        summary.append({'bin': b + 1,
                        'sample_size': n_total,
                        'treated_cnt': n_treat,
                        'treated_prop': prop_treat,
                        'avg_outcome': mean_outcome,
                        'avg_outcome_treat': mean_treat,
                        'avg_outcome_ctrl': mean_ctrl,
                        'uplift': uplift,
                        'p_value': p_val})
    return pd.DataFrame(summary)

# 调用示例
eval_table = bin_and_evaluate(result, nbins=10)
print(eval_table)

import numpy as np
import pandas as pd
from scipy import stats

# ---------- 1. Uplift 曲线与 AUUC ---------- #
def uplift_curve(df: pd.DataFrame,
                 tau_col: str = 'tau_hat',
                 treat_col: str = 'delta_quota',
                 outcome_col: str = 'ins_amt_d30',
                 n_points: int = 100):
    """
    返回两个一维数组：
    x -- 覆盖率比例 (0~1)，按 tau_hat 降序截取前 k% 的观测
    y -- 对应区间的累计真实 uplift
    """
    df = df.copy()
    df['T'] = (df[treat_col] > 0).astype(int)
    df = df.sort_values(tau_col, ascending=False).reset_index(drop=True)

    # 全局基准：总体 treated / control 数量比率
    n_treat_total = df['T'].sum()
    n_ctrl_total  = len(df) - n_treat_total
    ratio_ctrl = n_ctrl_total / max(n_treat_total, 1)

    xs, ys = [0.0], [0.0]        # 起点 (0,0)
    step = len(df) // n_points

    cum_treat_sum = cum_ctrl_sum = cum_treat_cnt = cum_ctrl_cnt = 0.0
    for i, row in df.iterrows():
        if row['T'] == 1:
            cum_treat_sum += row[outcome_col]
            cum_treat_cnt += 1
        else:
            cum_ctrl_sum += row[outcome_col]
            cum_ctrl_cnt += 1

        # 每隔 step 行记录一次曲线点
        if (i + 1) % step == 0 or i == len(df) - 1:
            frac = (i + 1) / len(df)
            # 随机基线：若前 i 条样本随机抽取，则期望 uplift 为 0
            # 因此此处 uplift = adjusted_treat_sum - adjusted_ctrl_sum
            adj_ctrl_sum = cum_ctrl_sum * (cum_treat_cnt / max(cum_ctrl_cnt, 1))
            uplift_val = cum_treat_sum - adj_ctrl_sum
            xs.append(frac)
            ys.append(uplift_val)

    # AUUC：梯形积分
    auuc_val = np.trapz(ys, xs)
    return np.array(xs), np.array(ys), auuc_val


# ---------- 2. Qini 系数 ---------- #
def qini_coefficient(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Qini = AUUC - 随机曲线面积
    随机曲线在 (0,0)-(1,ys[-1]) 之间作一条直线
    """
    auuc = np.trapz(ys, xs)
    random_area = 0.5 * ys[-1]          # 直线三角形面积
    return auuc - random_area


# ---------- 3. 分箱 uplift 与秩相关检验 ---------- #
def bin_rank_correlation(eval_table: pd.DataFrame,
                         uplift_col: str = 'uplift',
                         method: str = 'spearman'):
    """
    在 bin 级别度量 τ̂ 排序是否与真实 uplift 单调一致。
    """
    observed = eval_table[uplift_col]
    ranks    = np.arange(1, len(eval_table) + 1)   # τ̂ 降序 bin rank
    if method == 'spearman':
        corr, p = stats.spearmanr(ranks, observed, nan_policy='omit')
    elif method == 'kendall':
        corr, p = stats.kendalltau(ranks, observed, nan_policy='omit')
    else:
        raise ValueError("method 仅支持 'spearman' 或 'kendall'")
    return corr, p


# ---------- 4. 一站式评估入口 ---------- #
def full_uplift_evaluation(df: pd.DataFrame,
                           nbins: int = 10):
    """
    - 生成十分位分箱统计
    - 计算 uplift 曲线 / AUUC / Qini
    - 计算 Spearman 秩相关
    """
    # 调用上一轮的分箱函数
    eval_tbl = bin_and_evaluate(df, nbins=nbins)

    # Uplift 曲线与 AUUC、Qini
    xs, ys, auuc_val = uplift_curve(df)
    qini_val = qini_coefficient(xs, ys)

    # 秩相关检验
    rho, p_val = bin_rank_correlation(eval_tbl)

    print("AUUC: {:.4f}".format(auuc_val))
    print("Qini: {:.4f}".format(qini_val))
    print("Spearman ρ: {:.3f}  (p={:.4f})".format(rho, p_val))
    return eval_tbl, (xs, ys), {'AUUC': auuc_val, 'Qini': qini_val,
                                'Spearman_rho': rho, 'p_value': p_val}


# ----------- ▼ 使用示例 ▼ ----------- #
# eval_table, (curve_x, curve_y), metrics = full_uplift_evaluation(result, nbins=10)
# eval_table.head()

