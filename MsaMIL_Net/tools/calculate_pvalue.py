'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-03-25 09:09:58
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-03-25 09:27:20
FilePath: /MsaMIL/MsaMIL_Net/tools/calculate_pvalue.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

print("==================================================")
print("  DgMsa-MIL vs CLAM 统计显著性检验 (MIA/TMI 标准)  ")
print("==================================================\n")

# 1. 读取远程 Linux 服务器上的两个 CSV 文件
path_base = "/private/ljh-data/shared/MsaMIL/MsaMIL_Net/data/JiXian.csv"
path_ours = "/private/ljh-data/shared/MsaMIL/MsaMIL_Net/data/pvalue_input_train_val_test.csv"

try:
    df_base = pd.read_csv(path_base)
    df_ours = pd.read_csv(path_ours)
    print("成功读取预测数据！")
except Exception as e:
    print(f"读取数据失败，请检查路径: {e}")
    exit()

# 2. 核心严谨性：通过 slide_id 进行患者级别的严格配对合并
# 防止因为数据顺序打乱导致配对检验失效
df_merged = pd.merge(
    df_base[['slide_id', 'true_label', 'pred_prob']], 
    df_ours[['slide_id', 'true_label', 'prob_Adenocarcinoma']], 
    on=['slide_id', 'true_label'], 
    suffixes=('_base', '_ours')
)
print(f"成功严格配对 {len(df_merged)} 例患者数据。\n")

# 提取标签与预测概率
y_true = df_merged['true_label'].values
y_pred_base = df_merged['pred_prob'].values
y_pred_ours = df_merged['prob_Adenocarcinoma'].values

# =========================================================
# 检验 A：McNemar 检验 (评估二分类决策 0.5 阈值下的实际临床收益)
# =========================================================
label_base = (y_pred_base >= 0.5).astype(int)
label_ours = (y_pred_ours >= 0.5).astype(int)

# b: 你的模型算对了，但基线算错了
b = np.sum((label_ours == y_true) & (label_base != y_true))
# c: 你的模型算错了，但基线算对了
c = np.sum((label_ours != y_true) & (label_base == y_true))

# 执行带连续性校正的 McNemar 检验
table = [[0, b], [c, 0]]
mcnemar_result = mcnemar(table, exact=False, correction=True)

print("--- [1] 临床分类错误率比较 (McNemar's Test) ---")
print(f"DgMsa-MIL 纠正了基线的错误: {b} 例")
print(f"基线纠正了 DgMsa-MIL 的错误: {c} 例")
print(f">> McNemar P-value: {mcnemar_result.pvalue:.4e}")
if mcnemar_result.pvalue < 0.05:
    print("   [结论] DgMsa-MIL 在误诊/漏诊率上的降低具有统计学显著性！\n")


# =========================================================
# 检验 B：配全 Bootstrap 检验 (MIA 认可的 DeLong AUC 比较替代方案)
# =========================================================
def paired_auc_bootstrap(y_true, y_pred1, y_pred2, n_bootstraps=2000):
    np.random.seed(42) # 固定随机种子，保证每次运行 P 值绝对一致
    diffs = []
    indices = np.arange(len(y_true))
    
    for i in range(n_bootstraps):
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        # 确保重采样的子集里既有正样本也有负样本，否则无法算 AUC
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        auc1 = roc_auc_score(y_true[boot_idx], y_pred1[boot_idx])
        auc2 = roc_auc_score(y_true[boot_idx], y_pred2[boot_idx])
        diffs.append(auc2 - auc1)
    
    # 计算 p 值：基线 AUC >= DgMsa-MIL AUC 的极端情况概率
    p_value = np.sum(np.array(diffs) <= 0) / len(diffs)
    return p_value

print("--- [2] 全局特征流形比较 (Paired Bootstrap for AUC) ---")
print("正在执行 2000 次强配对重采样 (预计耗时十余秒)...")
p_boot = paired_auc_bootstrap(y_true, y_pred_base, y_pred_ours)

print(f">> Paired AUC Bootstrap P-value: {p_boot:.4e}")
if p_boot < 0.05:
    print("   [结论] DgMsa-MIL 在全局 AUC 上的提升具有统计学显著性！\n")