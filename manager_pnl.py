"""客户平仓盈亏按管理人、通用合约品种、结算币种净额汇总。"""
import re
import numpy as np
import pandas as pd

PNL_COLUMN = "浮动收益（结算 未扣除费用）"

# 明确列出法律实体和简称，避免按“永利”等模糊词误合并不同公司。
LEGAL_MANAGER_NAMES = {
    "浙江旌安私募基金有限公司": "旌安",
    "杭州遂玖私募基金管理有限公司": "遂玖",
    "杭州钧富投资管理合伙企业（有限合伙）": "钧富",
    "浙江钱唐永利资产管理有限公司": "钱唐永利",
    "上海春雷私募基金管理有限公司": "春雷",
    "北京恒德时代私募基金管理有限公司": "恒德时代",
    "北京朝晖知行私募基金管理有限公司": "朝晖知行",
    "北京源润资产投资有限公司": "源润",
    "厦门宁水私募基金管理有限公司": "宁水",
    "山东融升私募基金管理有限公司": "融升",
    "杭州君圭私募基金管理有限公司": "君圭",
    "杭州昊恩私募基金管理有限公司": "昊恩",
    "杭州润洲私募基金管理有限公司": "润洲",
}
ACCOUNT_PREFIX_RULES = [
    (r"^(?:北京)?雪球", "雪球"),
    (r"^旌安", "旌安"),
    (r"^遂[玖久]", "遂玖"),
    (r"^钧富", "钧富"),
    (r"^明汯", "明汯"),
    (r"^永利(?:商品|双盈|进取|\d+号)", "钱唐永利"),
]


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def resolve_manager(row, overrides=None):
    customer_id = re.sub(r"\.0$", "", _text(row["客户编号"]))
    if overrides and customer_id in overrides:
        return overrides[customer_id], "客户编号指定"
    name, short = _text(row["客户名称"]), _text(row["客户简称"])
    if "作为管理人" in name:
        legal_name = name.split("作为管理人", 1)[0].strip()
        return LEGAL_MANAGER_NAMES.get(legal_name, legal_name), "客户全称提取管理人"
    for pattern, manager in ACCOUNT_PREFIX_RULES:
        if re.search(pattern, short) or re.search(pattern, name):
            return manager, "明确账户名称规则"
    # 法人客户保留完整名称；未识别的基金不能猜测归属，显式标注。
    label = name or short
    if not label:
        raise ValueError(f"客户 {customer_id} 缺少名称")
    return label, "未识别管理人，暂单列" if "基金" in label else "独立法人客户"


def rank_manager_product_pnl(closing, start_date, end_date, product_resolver,
                             manager_overrides=None, top_n=5):
    """按平仓日期筛选，先净额汇总再区分盈利/亏损，不跨币种相加。"""
    if top_n < 1:
        raise ValueError("top_n 必须为正整数")
    detail = closing.copy()
    detail.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in detail.columns]
    required = ["平仓日期", "簿记账户", "客户编号", "客户名称", "客户简称",
                "标的", "结算货币", PNL_COLUMN]
    missing = [c for c in required if c not in detail]
    if missing:
        raise ValueError(f"平仓文件缺少字段：{missing}")
    detail = detail[detail["簿记账户"].map(_text).eq("南向对客")].copy()
    detail["平仓日期"] = pd.to_datetime(detail["平仓日期"], errors="coerce")
    if detail["平仓日期"].isna().any():
        raise ValueError("平仓日期无效，不能确定统计期间")
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("统计起止日期无效")
    detail = detail[
        detail["平仓日期"].ge(start) & detail["平仓日期"].lt(end + pd.Timedelta(days=1))
    ].copy()
    if detail.duplicated().any():
        raise ValueError("期间内存在完全重复的平仓记录，请核对后再汇总")
    detail["客户平仓盈亏"] = pd.to_numeric(detail[PNL_COLUMN], errors="coerce")
    if not np.isfinite(detail["客户平仓盈亏"]).all():
        raise ValueError("客户平仓盈亏存在缺失或无效值，不能当作零")
    for col in ["客户编号", "标的", "结算货币"]:
        if detail[col].map(_text).eq("").any():
            raise ValueError(f"{col} 存在缺失值")
    detail["客户编号"] = detail["客户编号"].map(lambda v: re.sub(r"\.0$", "", _text(v)))
    detail["结算货币"] = detail["结算货币"].map(_text)
    managers = [resolve_manager(row, manager_overrides) for _, row in detail.iterrows()]
    detail["管理人"] = [m[0] for m in managers]
    detail["归并依据"] = [m[1] for m in managers]
    detail["品种"] = detail["标的"].map(product_resolver)
    if detail["品种"].map(_text).eq("").any():
        raise ValueError("有标的无法解析为品种")
    conflicts = detail.groupby("客户编号")["管理人"].nunique()
    if conflicts.gt(1).any():
        raise ValueError(f"同一客户编号的管理人冲突：{conflicts[conflicts.gt(1)].index.tolist()}")
    summary = detail.groupby(["管理人", "品种", "结算货币"], as_index=False).agg(
        客户平仓盈亏=("客户平仓盈亏", "sum"),
        账户数=("客户编号", "nunique"),
        平仓记录数=("客户编号", "size"),
        包含账户=("客户简称", lambda s: "、".join(sorted(set(s.map(_text))))),
        具体标的=("标的", lambda s: "、".join(sorted(set(s.map(_text))))),
    )
    summary = summary.sort_values(
        ["结算货币", "客户平仓盈亏", "管理人", "品种"],
        ascending=[True, False, True, True]
    ).reset_index(drop=True)
    def top(sign):
        # 排名与显示均按分，不把浮点噪声误列为盈亏。
        rounded = summary["客户平仓盈亏"].round(2)
        ranked = summary[rounded.gt(0) if sign > 0 else rounded.lt(0)].sort_values(
            ["结算货币", "客户平仓盈亏", "管理人", "品种"],
            ascending=[True, sign < 0, True, True]
        ).groupby("结算货币", group_keys=False).head(top_n).copy()
        ranked.insert(0, "排名", ranked.groupby("结算货币").cumcount() + 1)
        return ranked.reset_index(drop=True)
    mapping = detail[["客户编号", "客户简称", "客户名称", "管理人", "归并依据"]].drop_duplicates()
    return summary, top(1), top(-1), detail, mapping
