# zimdn_optimized.py  ——  优化版 Zero‑Inflated MDN
# 主要改进：处理极度不平衡的零膨胀数据（90%零值）

import math, random, os, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. 公共工具
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------
# 2. 改进的预处理（添加诊断）
# ------------------------------------------------------------
def preprocess_dL(dL: np.ndarray, eps: float = 1e‑6):
    """预处理并诊断数据分布"""
    # 统计零比例
    zero_ratio = (dL == 0).mean()
    print(f"零样本比例: {zero_ratio:.3f}")
    
    pos = dL > 0
    dL_pos = np.log1p(dL[pos])
    
    # 使用更稳健的分位数方法
    q25, q75 = np.percentile(dL_pos, [25, 75])
    iqr = q75 - q25
    med = np.median(dL_pos)
    mad = 1.4826 * np.median(np.abs(dL_pos - med))  # 标准化MAD
    
    print(f"正样本统计: median={np.exp(med)-1:.2f}, MAD={mad:.3f}")
    print(f"正样本范围: [{dL[pos].min():.2f}, {dL[pos].max():.2f}]")
    
    # 使用更稳定的标准化
    dL_proc = np.zeros_like(dL, dtype=np.float32)
    dL_proc[pos] = (dL_pos - med) / max(mad, 0.1)  # 防止除零
    
    # 裁剪极端值
    dL_proc = np.clip(dL_proc, -5, 5)
    
    return dL_proc.astype(np.float32), med, mad, zero_ratio

# ------------------------------------------------------------
# 3. 改进的Dataset（支持平衡采样）
# ------------------------------------------------------------
class CreditDataset(Dataset):
    def __init__(self, X, Z, dL_proc, compute_weights=False):
        feat = np.concatenate([X, Z], axis=-1).astype(np.float32)
        self.x  = torch.from_numpy(feat)
        self.dL = torch.from_numpy(dL_proc)
        
        if compute_weights:
            # 计算样本权重用于平衡采样
            zero_mask = (dL_proc == 0)
            n_zero = zero_mask.sum()
            n_pos = len(dL_proc) - n_zero
            
            # 给零样本和正样本相等的总权重
            self.weights = torch.zeros(len(dL_proc))
            self.weights[zero_mask] = 0.5 / n_zero
            self.weights[~zero_mask] = 0.5 / n_pos
        else:
            self.weights = None

    def __len__(self):  
        return len(self.dL)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dL[idx]

# ------------------------------------------------------------
# 4. 改进的 Zero‑Inflated MDN
# ------------------------------------------------------------
class ZeroInflatedMDN(nn.Module):
    def __init__(self, input_dim: int,
                 hidden: int = 256,  # 增加容量
                 K: int = 5,         # 增加混合分量
                 sigma_min: float = 0.1, 
                 sigma_max: float = 2.0,
                 zero_prior: float = 0.9):  # 添加先验
        super().__init__()
        self.K = K
        self.zero_prior = zero_prior
        
        # 更深的backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),  # 使用BatchNorm
            nn.GELU(),  # 更平滑的激活
            nn.Dropout(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden, hidden // 2),
            nn.GELU()
        )
        
        # 独立的零概率预测头（增加容量）
        self.pi0_head = nn.Sequential(
            nn.Linear(hidden // 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.alpha_head = nn.Linear(hidden // 2, K)
        self.mu_head    = nn.Linear(hidden // 2, K)
        self.sigma_head = nn.Linear(hidden // 2, K)
        
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        
        # 初始化偏置以匹配先验
        with torch.no_grad():
            # 初始化pi0接近真实比例
            self.pi0_head[-1].bias.fill_(math.log(zero_prior / (1 - zero_prior)))

    def _parametrize_sigma(self, raw):
        # 使用softplus确保正值和平滑性
        s = F.softplus(raw)
        return self.sigma_min + s.clamp(max=self.sigma_max - self.sigma_min)

    def forward(self, x):
        h = self.backbone(x)
        
        # 零概率（加入先验正则化的想法）
        pi0_logit = self.pi0_head(h).squeeze(-1)
        pi0 = torch.sigmoid(pi0_logit)
        
        # 混合分量参数
        alpha = F.softmax(self.alpha_head(h), -1)
        mu    = self.mu_head(h)
        sigma = self._parametrize_sigma(self.sigma_head(h))
        
        return pi0, alpha, mu, sigma, pi0_logit

    def log_prob(self, x, l, eps: float = 1e‑8):
        pi0, alpha, mu, sigma, _ = self.forward(x)
        
        # 对零样本
        is_zero = (l == 0)
        log_p_zero = torch.log(pi0.clamp(eps, 1-eps))
        
        # 对正样本
        l_exp = l.unsqueeze(-1).expand_as(mu)
        log_pdf = torch.distributions.Normal(mu, sigma).log_prob(l_exp)
        log_mix = torch.log(alpha + eps) + log_pdf
        log_p_pos = torch.log(1 - pi0.clamp(eps, 1-eps)) + torch.logsumexp(log_mix, -1)
        
        return torch.where(is_zero, log_p_zero, log_p_pos)

# ------------------------------------------------------------
# 5. 改进的损失函数
# ------------------------------------------------------------
class BalancedNLLLoss(nn.Module):
    """平衡的负对数似然损失"""
    def __init__(self, zero_weight: float = 2.0):
        super().__init__()
        self.zero_weight = zero_weight
    
    def forward(self, model, x, l):
        log_p = model.log_prob(x, l)
        is_zero = (l == 0).float()
        
        # 对零样本加权
        weights = 1.0 + is_zero * (self.zero_weight - 1.0)
        return -(log_p * weights).mean()

def regularized_loss(model, x, l, zero_prior=0.9, reg_weight=0.1):
    """带先验正则化的损失"""
    pi0, alpha, mu, sigma, pi0_logit = model(x)
    
    # 基础NLL
    base_loss = -model.log_prob(x, l).mean()
    
    # KL散度正则化（推动pi0接近先验）
    kl_loss = F.kl_div(
        torch.log(pi0.mean().unsqueeze(0).clamp(1e-8, 1-1e-8)),
        torch.tensor([zero_prior]).to(x.device),
        reduction='batchmean'
    )
    
    # 熵正则化（鼓励alpha的多样性）
    entropy = -(alpha * torch.log(alpha + 1e-8)).sum(-1).mean()
    
    # sigma正则化
    sigma_reg = (sigma**2).mean()
    
    total_loss = base_loss + reg_weight * kl_loss - 0.01 * entropy + 0.001 * sigma_reg
    
    return total_loss, {
        'nll': base_loss.item(),
        'kl': kl_loss.item(),
        'entropy': entropy.item(),
        'sigma_reg': sigma_reg.item()
    }

# ------------------------------------------------------------
# 6. 改进的训练策略
# ------------------------------------------------------------
def train_epoch_balanced(model, loader, opt, device, 
                        stage=3, zero_prior=0.9, grad_clip=0.5):
    """使用平衡策略的训练"""
    model.train()
    tot_loss = 0.0
    loss_components = {'nll': 0, 'kl': 0, 'entropy': 0, 'sigma_reg': 0}
    
    for x, l in loader:
        x, l = x.to(device), l.to(device)
        
        if stage == 1:  # Stage 1: 只训练pi0
            pi0, _, _, _, pi0_logit = model(x)
            # 使用focal loss思想
            pt = torch.where(l == 0, pi0, 1 - pi0)
            focal_weight = (1 - pt) ** 2
            
            loss = F.binary_cross_entropy(
                pi0.clamp(1e-6, 1-1e-6), 
                (l == 0).float(),
                weight=focal_weight
            )
            
        elif stage == 2:  # Stage 2: 固定pi0，训练连续部分
            # 只在正样本上训练
            pos_mask = l > 0
            if pos_mask.sum() == 0:
                continue
                
            x_pos, l_pos = x[pos_mask], l[pos_mask]
            loss = -model.log_prob(x_pos, l_pos).mean()
            
        else:  # Stage 3: 联合训练
            loss, components = regularized_loss(model, x, l, zero_prior)
            for k, v in components.items():
                loss_components[k] += v * x.size(0)
        
        opt.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        opt.step()
        tot_loss += loss.item() * x.size(0)
    
    n_samples = len(loader.dataset)
    metrics = {
        'loss': tot_loss / n_samples,
        'grad_norm': grad_norm
    }
    
    if stage == 3:
        for k in loss_components:
            metrics[k] = loss_components[k] / n_samples
            
    return metrics

# ------------------------------------------------------------
# 7. 诊断工具
# ------------------------------------------------------------
def diagnose_predictions(model, loader, device, save_path="diagnosis.png"):
    """诊断预测分布"""
    model.eval()
    all_pi0 = []
    all_l = []
    
    with torch.no_grad():
        for x, l in loader:
            x, l = x.to(device), l.to(device)
            pi0, _, _, _, _ = model(x)
            all_pi0.append(pi0.cpu())
            all_l.append(l.cpu())
    
    all_pi0 = torch.cat(all_pi0)
    all_l = torch.cat(all_l)
    
    # 绘制诊断图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. pi0分布
    axes[0, 0].hist(all_pi0.numpy(), bins=50, alpha=0.7)
    axes[0, 0].axvline(x=(all_l == 0).float().mean(), color='r', 
                       linestyle='--', label='True zero ratio')
    axes[0, 0].set_title('Distribution of π₀')
    axes[0, 0].legend()
    
    # 2. pi0 vs 真实标签
    zero_mask = (all_l == 0)
    axes[0, 1].hist(all_pi0[zero_mask].numpy(), bins=30, alpha=0.5, 
                    label='True zeros', density=True)
    axes[0, 1].hist(all_pi0[~zero_mask].numpy(), bins=30, alpha=0.5, 
                    label='True positives', density=True)
    axes[0, 1].set_title('π₀ by True Label')
    axes[0, 1].legend()
    
    # 3. 校准曲线
    n_bins = 10
    calibration_data = []
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        mask = (all_pi0 >= low) & (all_pi0 < high)
        if mask.sum() > 0:
            pred_prob = all_pi0[mask].mean().item()
            true_prob = (all_l[mask] == 0).float().mean().item()
            calibration_data.append((pred_prob, true_prob))
    
    if calibration_data:
        pred_probs, true_probs = zip(*calibration_data)
        axes[1, 0].plot(pred_probs, true_probs, 'o-')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('Mean Predicted π₀')
        axes[1, 0].set_ylabel('True Zero Fraction')
        axes[1, 0].set_title('Calibration Plot')
    
    # 4. 统计信息
    stats_text = f"""
    Dataset Statistics:
    - Total samples: {len(all_l)}
    - True zero ratio: {(all_l == 0).float().mean():.3f}
    - Mean predicted π₀: {all_pi0.mean():.3f}
    - Std predicted π₀: {all_pi0.std():.3f}
    
    Model Performance:
    - π₀ for true zeros: {all_pi0[zero_mask].mean():.3f} ± {all_pi0[zero_mask].std():.3f}
    - π₀ for true positives: {all_pi0[~zero_mask].mean():.3f} ± {all_pi0[~zero_mask].std():.3f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return {
        'mean_pi0': all_pi0.mean().item(),
        'true_zero_ratio': (all_l == 0).float().mean().item(),
        'pi0_gap': abs(all_pi0.mean().item() - (all_l == 0).float().mean().item())
    }

# ------------------------------------------------------------
# 8. 主训练函数（改进版）
# ------------------------------------------------------------
def run_train_optimized(X_raw, Z_raw, dL_raw,
                       K_fold=3, batch=1024,  # 减小batch size
                       device="cuda" if torch.cuda.is_available() else "cpu",
                       alpha=.9):
    
    # 预处理并获取零比例
    dL_proc, med, mad, zero_ratio = preprocess_dL(dL_raw)
    print(f"\n数据集零比例: {zero_ratio:.3f}")
    
    kf = KFold(n_splits=K_fold, shuffle=True, random_state=42)
    
    for fold, (tr, va) in enumerate(kf.split(X_raw), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{K_fold}")
        print(f"{'='*50}")
        
        # 创建数据集（训练集使用平衡采样）
        ds_tr = CreditDataset(X_raw[tr], Z_raw[tr], dL_proc[tr], compute_weights=True)
        ds_va = CreditDataset(X_raw[va], Z_raw[va], dL_proc[va], compute_weights=False)
        
        # 使用加权采样器
        sampler = WeightedRandomSampler(ds_tr.weights, len(ds_tr), replacement=True)
        dl_tr = DataLoader(ds_tr, batch, sampler=sampler, collate_fn=collate)
        dl_va = DataLoader(ds_va, batch*2, False, collate_fn=collate)
        
        # 创建模型，传入真实零比例作为先验
        model = ZeroInflatedMDN(
            input_dim=ds_tr[0][0].numel(),
            hidden=256,
            K=5,
            zero_prior=zero_ratio
        ).to(device)
        
        print(f"\n--- Stage 1: 训练π₀ (使用focal loss) ---")
        # 冻结其他参数
        for p in model.parameters(): 
            p.requires_grad = False
        for p in model.pi0_head.parameters(): 
            p.requires_grad = True
            
        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-3,  # 较高学习率
            weight_decay=1e-4
        )
        
        # Stage 1: 更多epochs专注于pi0
        for e in range(20):
            metrics = train_epoch_balanced(model, dl_tr, opt1, device, stage=1, zero_prior=zero_ratio)
            if e % 5 == 0:
                diag = diagnose_predictions(model, dl_va, device, f"stage1_fold{fold}_epoch{e}.png")
                print(f"Epoch {e}: loss={metrics['loss']:.4f}, "
                      f"mean_pi0={diag['mean_pi0']:.3f}, "
                      f"pi0_gap={diag['pi0_gap']:.3f}")
        
        print(f"\n--- Stage 2: 训练连续部分 ---")
        # 解冻连续部分参数
        for p in model.parameters(): 
            p.requires_grad = True
        for p in model.pi0_head.parameters(): 
            p.requires_grad = False
            
        opt2 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        for e in range(10):
            metrics = train_epoch_balanced(model, dl_tr, opt2, device, stage=2)
            if e % 5 == 0:
                print(f"Epoch {e}: loss={metrics['loss']:.4f}")
        
        print(f"\n--- Stage 3: 联合微调 ---")
        # 解冻所有参数
        for p in model.parameters(): 
            p.requires_grad = True
            
        # 使用较小的学习率
        opt3 = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(opt3, factor=0.5, patience=10, min_lr=1e-5)
        
        best_score = float('inf')
        patience_counter = 0
        max_patience = 30
        
        for epoch in range(1, 151):
            # 训练
            train_metrics = train_epoch_balanced(
                model, dl_tr, opt3, device, 
                stage=3, zero_prior=zero_ratio
            )
            
            # 评估
            eval_metrics = eval_epoch(model, dl_va, device, alpha)
            
            # 诊断（每10个epoch）
            if epoch % 10 == 0:
                diag = diagnose_predictions(
                    model, dl_va, device, 
                    f"diagnosis_fold{fold}_epoch{epoch}.png"
                )
            
            # 计算综合得分（考虑pi0准确性）
            pi0_penalty = abs(eval_metrics['pi0m'] - zero_ratio) * 10
            score = eval_metrics['nll'] + pi0_penalty + 2.0 * eval_metrics['ks']
            
            scheduler.step(score)
            
            # 打印详细信息
            if epoch % 5 == 0:
                print(f"\n[Fold {fold} Epoch {epoch:03d}]")
                print(f"  Train: loss={train_metrics['loss']:.4f}, "
                      f"nll={train_metrics.get('nll', 0):.4f}, "
                      f"kl={train_metrics.get('kl', 0):.4f}")
                print(f"  Valid: nll={eval_metrics['nll']:.4f}, "
                      f"cov@{alpha}={eval_metrics['cov']:.3f}, "
                      f"ks={eval_metrics['ks']:.3f}")
                print(f"  Zero: pi0m={eval_metrics['pi0m']:.3f} "
                      f"(target={zero_ratio:.3f}, gap={abs(eval_metrics['pi0m']-zero_ratio):.3f})")
                print(f"  Score: {score:.4f} (best={best_score:.4f})")
            
            # Early stopping
            if score < best_score:
                best_score = score
                patience_counter = 0
                torch.save({
                    'model_state': model.state_dict(),
                    'metrics': eval_metrics,
                    'epoch': epoch
                }, f"best_model_fold{fold}.pt")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        print(f"\nFold {fold} 完成: best_score={best_score:.4f}")
        
        # 最终诊断
        checkpoint = torch.load(f"best_model_fold{fold}.pt")
        model.load_state_dict(checkpoint['model_state'])
        final_diag = diagnose_predictions(model, dl_va, device, f"final_diagnosis_fold{fold}.png")
        print(f"最终pi0准确性: mean={final_diag['mean_pi0']:.3f}, "
              f"target={zero_ratio:.3f}, gap={final_diag['pi0_gap']:.3f}")

# ------------------------------------------------------------
# 9. 评估函数（保持原有，略作改进）
# ------------------------------------------------------------
@torch.no_grad()
def eval_epoch(model, loader, device, alpha=.9):
    model.eval()
    nll_sum = cov_sum = 0.0
    pit_pos, pit_zero, ws = [], [], []
    all_pi0 = []

    for x, l in loader:
        x, l = x.to(device), l.to(device)
        
        # NLL
        log_probs = model.log_prob(x, l)
        nll_sum += -log_probs.sum().item()

        # 获取预测
        pi0, a, mu, sigma, _ = model(x)
        all_pi0.append(pi0.cpu())
        
        # Coverage
        pit = mixture_cdf(pi0, a, mu, sigma, l)
        lo, hi = (1 - alpha) / 2, 1 - (1 - alpha) / 2
        cov = ((pit >= lo) & (pit <= hi)).float().mean().item()
        cov_sum += cov * x.size(0)

        # 分离PIT
        pit_zero.append(pit[l == 0].cpu())
        pit_pos.append(pit[l > 0].cpu())

        # IPW
        ws.append(ipw_weight(model, x, l).cpu())

    # 合并结果
    all_pi0 = torch.cat(all_pi0)
    pit_zero = torch.cat(pit_zero, 0) if pit_zero else torch.empty(0)
    pit_pos = torch.cat(pit_pos, 0) if pit_pos else torch.empty(0)
    ws = torch.cat(ws, 0)

    # KS统计（仅对正样本）
    ks = 0.0
    if len(pit_pos) > 10:  # 确保有足够样本
        sorted_pit = torch.sort(pit_pos)[0]
        uniform_cdf = torch.linspace(0, 1, len(pit_pos))
        ks = torch.abs(sorted_pit - uniform_cdf).max().item()

    # ESS
    ess, max_w = ess_and_maxw(ws)

    return {
        'nll': nll_sum / len(loader.dataset),
        'cov': cov_sum / len(loader.dataset),
        'ks': ks,
        'pi0m': all_pi0.mean().item(),  # 使用所有预测的pi0均值
        'ess': ess,
        'max_w': max_w
    }

# 辅助函数（从原代码复制）
def mixture_cdf(pi0, alpha, mu, sigma, l, eps=1e-8):
    l_ = l.unsqueeze(-1)
    z = (l_ - mu) / (sigma + eps)
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    F_pos = (alpha * Phi).sum(-1)
    F = torch.where(l == 0, pi0, pi0 + (1 - pi0) * F_pos)
    return F.clamp(eps, 1 - eps)

def ess_and_maxw(w):
    w_sum = w.sum()
    ess = (w_sum**2 / torch.clamp((w**2).sum(), 1e-9)).item()
    return ess, w.max().item()

@torch.no_grad()
def ipw_weight(model, x, l, log_clip=(-20, 20), w_clip=50.):
    logp = model.log_prob(x, l).clamp(*log_clip)
    return torch.exp(-logp).clamp(max=w_clip)

# ------------------------------------------------------------
# 10. 使用示例
# ------------------------------------------------------------
if __name__ == "__main__":
    set_seed(2025)
    
    # 模拟你的数据分布（90%零值）
    n_samples = 200_000
    X_raw = np.random.randn(n_samples, 20).astype(np.float32)
    Z_raw = np.random.randn(n_samples, 8).astype(np.float32)
    
    # 生成90%零值的数据
    dL_raw = np.zeros(n_samples, dtype=np.float32)
    pos_mask = np.random.rand(n_samples) > 0.90  # 10%正样本
    dL_raw[pos_mask] = np.random.exponential(200, pos_mask.sum())
    
    print(f"生成数据: {n_samples}样本, 零比例={(dL_raw==0).mean():.3f}")
    
    # 运行优化后的训练
    run_train_optimized(X_raw, Z_raw, dL_raw, K_fold=3, batch=1024)