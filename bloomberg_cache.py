"""在彭博电脑上保存查询结果；缓存不包含客户或交易流水。"""
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np
import pandas as pd


def generate_bloomberg_cache(tickers, fetcher, cache_root="bloomberg_cache"):
    """fetcher 传入 Notebook 中的 get_bloomberg_data。

    每次创建独立目录。只有资料完整时才写入缓存；
    queried_at 是本次查询时间，不是彭博行情自身的时间戳。
    """
    tickers = list(dict.fromkeys(tickers))
    if not tickers or any(not isinstance(t, str) or not t.strip() for t in tickers):
        raise ValueError("标的列表为空或包含无效标的，请先清理。")
    started = datetime.now(timezone.utc)
    data = fetcher(tickers).copy()
    completed = datetime.now(timezone.utc)
    required = ["Generic_Ticker", "Contract_Size", "Category", "Currency", "FX_Rate_to_USD"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"查询结果缺少字段：{missing}")
    if not data.index.is_unique:
        raise ValueError("查询结果包含重复标的。")
    data = data.reindex(tickers)
    invalid = pd.Series(False, index=data.index)
    for col in ["Contract_Size", "FX_Rate_to_USD"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        invalid |= ~(np.isfinite(data[col]) & data[col].gt(0))
    for col in ["Generic_Ticker", "Category"]:
        invalid |= data[col].isna() | data[col].astype(str).str.strip().eq("")
    invalid |= ~data["Currency"].astype("string").str.fullmatch(r"[A-Z]{3}").fillna(False)
    if invalid.any():
        raise ValueError(f"缓存未保存：以下标的资料不完整或无效：{data.index[invalid].tolist()}")
    if data.groupby("Currency")["FX_Rate_to_USD"].nunique().gt(1).any():
        raise ValueError("同一币种存在不同汇率，缓存未保存。")
    if not data.loc[data["Currency"].eq("USD"), "FX_Rate_to_USD"].eq(1).all():
        raise ValueError("USD 汇率必须为 1，缓存未保存。")

    cache_dir = Path(cache_root) / completed.strftime("%Y%m%dT%H%M%S_%fZ")
    cache_dir.mkdir(parents=True, exist_ok=False)
    data.index.name = "bbg_ticker"
    # 完整查询结果可以直接按 bbg_ticker 合并回流水。
    data.to_csv(cache_dir / "bbg_data.csv", encoding="utf-8-sig")
    data.drop(columns="FX_Rate_to_USD").to_csv(
        cache_dir / "contract_metadata.csv", encoding="utf-8-sig")
    fx = data[["Currency", "FX_Rate_to_USD"]].drop_duplicates().copy()
    fx["queried_at_utc"] = completed.isoformat()
    fx.to_csv(cache_dir / "fx_rates.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "schema_version": 1,
        "query_started_at_utc": started.isoformat(),
        "query_completed_at_utc": completed.isoformat(),
        "ticker_count": len(data),
        "unclassified_tickers": data.index[data["Category"].eq("Unclassified")].tolist(),
        "fx_basis": "PX_LAST retrieved during this run; not historical trade-date FX",
        "note": "查询时间不是行情时间。复制整个目录以在其他电脑复用；缺失资料不会默认填 1 或 USD。",
    }
    # 最后写 manifest，只有该文件存在才表示导出完成。
    (cache_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"缓存已保存：{cache_dir.resolve()}（{len(data)} 个标的）")
    return data, cache_dir
