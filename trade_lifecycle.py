"""按流水编号关联开仓、平仓记录；不通过缺失记录推断开平仓。"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

PNL_FIELD = "浮动收益（结算 未扣除费用）"
RESULT_COLUMNS = [
    "客户交易方向", "交易手数", "开平仓", "匹配状态", "已匹配开仓手数",
    "已匹配平仓手数", "待确认手数", "客户平仓盈亏_结算未扣费用",
    "盈亏结算币种", "盈亏覆盖范围", "开仓来源行", "平仓来源行",
    "平仓合约编号", "平仓文件日期",
]


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _customer_id(value):
    return re.sub(r"\.0$", "", _text(value))


def _number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _direction(value):
    return {"B": "B", "S": "S", "买": "B", "卖": "S",
            "买入": "B", "卖出": "S", "客户买入": "B",
            "客户卖出": "S"}.get(_text(value).upper())


def _prepare(frame, required, label):
    frame = frame.copy()
    frame.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in frame.columns]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{label}缺少字段：{missing}")
    frame["来源Excel行"] = np.arange(2, len(frame) + 2)
    frame["流水编号"] = frame["流水编号"].map(_text)
    frame = frame[frame["簿记账户"].map(_text).eq("南向对客")].copy()
    return frame


def read_position_file(path):
    """XLSB 优先使用 calamine；没有安装时使用 pandas/pyxlsb。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到开平仓文件：{path}")
    if path.suffix.lower() == ".xlsb":
        try:
            from python_calamine import CalamineWorkbook
        except ImportError:
            try:
                return pd.read_excel(path, engine="pyxlsb")
            except ImportError as exc:
                raise ImportError(
                    "读取 .xlsb 需要 python-calamine 或 pyxlsb；"
                    "在 Notebook 执行 %pip install python-calamine"
                ) from exc
        with CalamineWorkbook.from_path(str(path)) as workbook:
            rows = workbook.get_sheet_by_index(0).to_python()
        if not rows:
            raise ValueError(f"文件没有数据：{path}")
        return pd.DataFrame(rows[1:], columns=rows[0])
    return pd.read_excel(path)


def classify_trades(trades, opening, closing):
    """返回一行对应一笔流水的明细和匹配统计，绝不放大原交易行数。

    开仓取初始开仓手数；平仓手数=了结数量/平仓表合约乘数。
    平仓表交易方向是原持仓方向，不覆盖流水中的本次客户交易方向。
    部分匹配时只显示已匹配平仓部分的盈亏，未匹配部分不填零。
    """
    needed = ["流水编号", "簿记账户", "客户编号", "客户简称", "标的全称",
              "交易日期", "买卖方向", "手数", "价格", "Notional"]
    missing = [c for c in needed if c not in trades]
    if missing:
        raise ValueError(f"交易流水缺少字段：{missing}。请使用原始流水构建的 df。")
    base = trades.copy().reset_index(drop=True)
    base["流水编号"] = base["流水编号"].map(_text)
    if base["流水编号"].eq("").any() or base["流水编号"].duplicated().any():
        raise ValueError("待匹配交易流水编号缺失或重复，不能安全分配平仓盈亏。")
    if not base["簿记账户"].map(_text).eq("南向对客").all():
        raise ValueError("客户开平仓匹配仅支持南向对客流水。")
    opening = _prepare(opening, [
        "流水编号", "合约编号", "簿记账户", "客户编号", "标的", "交易日期",
        "交易方向", "初始开仓手数", "期初价格",
    ], "开仓表")
    closing = _prepare(closing, [
        "流水编号", "合约编号", "簿记账户", "客户编号", "标的",
        "平仓日期", "了结数量", "合约乘数", "了结价格", "结算货币", PNL_FIELD,
    ], "平仓表")
    open_groups = {k: g for k, g in opening.groupby("流水编号") if k}
    close_groups = {k: g for k, g in closing.groupby("流水编号") if k}
    results = []
    for _, trade in base.iterrows():
        key = trade["流水编号"]
        og, cg = open_groups.get(key), close_groups.get(key)
        lots, price = abs(_number(trade["手数"])), _number(trade["价格"])
        side = _direction(trade["买卖方向"])
        if not np.isfinite(lots) or not np.isfinite(price) or side is None:
            raise ValueError(f"流水 {key} 的手数、价格或方向无效。")
        errors = []
        evidence = {
            "客户交易方向": {"B": "买入", "S": "卖出"}[side],
            "交易手数": lots,
            "开平仓": "未匹配",
            "匹配状态": "开仓/平仓文件均未找到该流水编号",
            "已匹配开仓手数": np.nan,
            "已匹配平仓手数": np.nan,
            "待确认手数": lots,
            "客户平仓盈亏_结算未扣费用": np.nan,
            "盈亏结算币种": "",
            "盈亏覆盖范围": "无已匹配平仓记录",
            "开仓来源行": "",
            "平仓来源行": "",
            "平仓合约编号": "",
            "平仓文件日期": "",
        }
        open_lots, close_lots = 0.0, 0.0
        for label, group, price_col in [
            ("开仓", og, "期初价格"), ("平仓", cg, "了结价格")
        ]:
            if group is None:
                continue
            evidence[label + "来源行"] = ", ".join(group["来源Excel行"].astype(str))
            if (group["客户编号"].map(_customer_id) != _customer_id(trade["客户编号"])).any():
                errors.append(label + "客户编号不一致")
            if (group["标的"].map(lambda v: _text(v).upper()) != _text(trade["标的全称"]).upper()).any():
                errors.append(label + "标的不一致")
            prices = pd.to_numeric(group[price_col], errors="coerce")
            if not np.isclose(prices, price, rtol=1e-8, atol=1e-6).all():
                errors.append(label + "价格不一致")
        if og is not None:
            if len(og) != 1:
                errors.append("同一流水匹配多条开仓记录")
            if og["交易方向"].map(_direction).ne(side).any():
                errors.append("开仓方向与流水不一致")
            odates = pd.to_datetime(og["交易日期"], errors="coerce")
            # pyxlsb 原生读取可能返回 Excel 序列日期。
            numeric_dates = pd.to_numeric(og["交易日期"], errors="coerce")
            if numeric_dates.notna().all() and numeric_dates.between(20000, 100000).all():
                odates = pd.to_datetime(numeric_dates, unit="D", origin="1899-12-30")
            tdate = pd.to_datetime(trade["交易日期"], errors="coerce")
            if pd.isna(tdate) or odates.isna().any() or odates.dt.normalize().ne(tdate.normalize()).any():
                errors.append("开仓日期与流水不一致")
            qty = pd.to_numeric(og["初始开仓手数"], errors="coerce")
            if not (np.isfinite(qty) & qty.ge(0)).all():
                errors.append("开仓手数无效")
            else:
                open_lots = qty.sum()
        pnl, currency = np.nan, ""
        if cg is not None:
            # 同一流水可以了结多个原合约，但不能重复计入同一了结行。
            duplicate_cols = ["合约编号", "平仓日期", "了结数量", "了结价格", PNL_FIELD]
            if cg.duplicated(duplicate_cols).any():
                errors.append("存在重复的平仓了结记录")
            qty = pd.to_numeric(cg["了结数量"], errors="coerce")
            multiplier = pd.to_numeric(cg["合约乘数"], errors="coerce")
            if not (np.isfinite(qty) & qty.ge(0) & np.isfinite(multiplier) & multiplier.gt(0)).all():
                errors.append("平仓数量或合约乘数无效")
            else:
                close_lots = (qty / multiplier).sum()
            currencies = cg["结算货币"].map(_text)
            currency = "/".join(sorted(set(currencies) - {""}))
            profits = pd.to_numeric(cg[PNL_FIELD], errors="coerce")
            if currencies.eq("").any() or currencies.nunique() != 1:
                errors.append("同一流水的盈亏币种缺失或不一致")
            if not np.isfinite(profits).all():
                errors.append("客户平仓盈亏缺失或无效")
            else:
                pnl = profits.sum()
            evidence["平仓合约编号"] = "; ".join(cg["合约编号"].map(_text))
            evidence["平仓文件日期"] = "; ".join(sorted(set(cg["平仓日期"].map(_text))))
        if errors:
            evidence.update({"开平仓": "待核对", "匹配状态": "；".join(errors),
                             "盈亏覆盖范围": "字段冲突，未输出盈亏"})
        elif og is not None or cg is not None:
            total = open_lots + close_lots
            delta = lots - total
            kind = "开仓及平仓" if open_lots > 0 and close_lots > 0 else ("平仓" if close_lots > 0 else "开仓")
            evidence.update({
                "已匹配开仓手数": open_lots, "已匹配平仓手数": close_lots,
                "待确认手数": delta, "盈亏结算币种": currency,
            })
            if total <= 0 or delta < -1e-8:
                evidence.update({"开平仓": "待核对", "匹配状态": "匹配手数为零或超过流水手数",
                                 "盈亏覆盖范围": "数量冲突，未输出盈亏"})
            else:
                full = np.isclose(delta, 0, atol=1e-8, rtol=0)
                evidence.update({
                    "开平仓": kind if full else "部分匹配（" + kind + "）",
                    "匹配状态": "流水编号及字段、手数核对通过" if full else "已匹配部分，剩余手数需补充文件核对",
                    "客户平仓盈亏_结算未扣费用": pnl if cg is not None else np.nan,
                    "盈亏覆盖范围": (
                        f"仅覆盖已匹配平仓 {close_lots:g} 手" if cg is not None else "开仓不适用"
                    ),
                })
        results.append(evidence)
    annotated = pd.concat([base, pd.DataFrame(results, index=base.index, columns=RESULT_COLUMNS)], axis=1)
    audit = annotated.groupby(["开平仓", "匹配状态"], dropna=False).size().reset_index(name="交易笔数")
    return annotated, audit


def summarize_position_activity(detail):
    """仅同一结算币种内汇总盈亏；明细中的缺失盈亏不变成零。"""
    keys = ["客户简称", "bbg_ticker", "客户交易方向", "开平仓", "盈亏结算币种"]
    work = detail.copy()
    work["Notional_USD"] = pd.to_numeric(work["Notional"], errors="raise").abs()
    return work.groupby(keys, dropna=False, as_index=False).agg(
        Notional_USD=("Notional_USD", "sum"),
        交易笔数=("流水编号", "size"),
        交易手数=("交易手数", "sum"),
        已匹配开仓手数=("已匹配开仓手数", lambda s: s.sum(min_count=1)),
        已匹配平仓手数=("已匹配平仓手数", lambda s: s.sum(min_count=1)),
        待确认手数=("待确认手数", "sum"),
        客户平仓盈亏_结算未扣费用=("客户平仓盈亏_结算未扣费用", lambda s: s.sum(min_count=1)),
    ).sort_values("Notional_USD", ascending=False).reset_index(drop=True)
