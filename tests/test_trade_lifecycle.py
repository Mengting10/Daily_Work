import unittest
import pandas as pd
from trade_lifecycle import classify_trades, summarize_position_activity, PNL_FIELD


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.trades = pd.DataFrame([{
            "流水编号": "T1", "簿记账户": "南向对客", "客户编号": 123.0,
            "客户简称": "客户A", "标的全称": "CLZ6 Comdty.NYMEX",
            "bbg_ticker": "CLZ6 Comdty", "交易日期": "2026-09-10",
            "买卖方向": "B", "手数": 10, "价格": 100, "Notional": 10000,
        }])
        self.opening = pd.DataFrame([{
            "流水编号": "T1", "合约编号": "T1", "簿记账户": "南向对客",
            "客户编号": "123", "标的": "CLZ6 Comdty.NYMEX",
            "交易日期": "2026-09-10", "交易方向": "客户买入",
            "初始开仓手数": 10, "期初价格": 100,
        }])
        self.closing = pd.DataFrame([{
            "流水编号": "T1", "合约编号": "OLD1", "簿记账户": "南向对客",
            "客户编号": 123, "标的": "CLZ6 Comdty.NYMEX",
            "平仓日期": "2026-09-16", "了结数量": 6000, "合约乘数": 1000,
            "了结价格": 100, "结算货币": "CNY", PNL_FIELD: 500,
        }, {
            "流水编号": "T1", "合约编号": "OLD2", "簿记账户": "南向对客",
            "客户编号": 123, "标的": "CLZ6 Comdty.NYMEX",
            "平仓日期": "2026-09-16", "了结数量": 4000, "合约乘数": 1000,
            "了结价格": 100, "结算货币": "CNY", PNL_FIELD: -100,
        }])

    def run_match(self, opening=None, closing=None):
        return classify_trades(
            self.trades,
            self.opening.iloc[:0] if opening is None else opening,
            self.closing.iloc[:0] if closing is None else closing,
        )[0].iloc[0]

    def test_opening_no_pnl(self):
        row = self.run_match(opening=self.opening)
        self.assertEqual(row["开平仓"], "开仓")
        self.assertTrue(pd.isna(row["客户平仓盈亏_结算未扣费用"]))

    def test_split_closes_sum_once_and_keep_customer_direction(self):
        result, _ = classify_trades(self.trades, self.opening.iloc[:0], self.closing)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["开平仓"], "平仓")
        self.assertEqual(result.iloc[0]["已匹配平仓手数"], 10)
        self.assertEqual(result.iloc[0]["客户平仓盈亏_结算未扣费用"], 400)
        self.assertEqual(result.iloc[0]["客户交易方向"], "买入")
        self.assertEqual(result.Notional.sum(), self.trades.Notional.sum())

    def test_mixed_and_partial(self):
        opening = self.opening.copy()
        opening["初始开仓手数"] = 4
        row = self.run_match(opening, self.closing.iloc[:1])
        self.assertEqual(row["开平仓"], "开仓及平仓")
        row = self.run_match(closing=self.closing.iloc[:1])
        self.assertEqual(row["开平仓"], "部分匹配（平仓）")
        self.assertEqual(row["待确认手数"], 4)
        self.assertEqual(row["客户平仓盈亏_结算未扣费用"], 500)

    def test_no_match_not_inferred_as_open(self):
        self.assertEqual(self.run_match()["开平仓"], "未匹配")

    def test_field_conflicts_suppress_pnl(self):
        for field, value in [("客户编号", 999), ("标的", "OTHER"),
                             ("了结价格", 200), ("合约乘数", 0),
                             (PNL_FIELD, None)]:
            with self.subTest(field=field):
                close = self.closing.copy()
                close.loc[0, field] = value
                row = self.run_match(closing=close)
                self.assertEqual(row["开平仓"], "待核对")
                self.assertTrue(pd.isna(row["客户平仓盈亏_结算未扣费用"]))

    def test_duplicate_close_and_excess_quantity(self):
        row = self.run_match(closing=pd.concat([self.closing, self.closing.iloc[:1]]))
        self.assertEqual(row["开平仓"], "待核对")
        row = self.run_match(self.opening, self.closing)
        self.assertEqual(row["开平仓"], "待核对")

    def test_mixed_currency_and_zero_profit(self):
        close = self.closing.copy()
        close.loc[0, "结算货币"] = "USD"
        self.assertEqual(self.run_match(closing=close)["开平仓"], "待核对")
        close[PNL_FIELD] = 0
        close["结算货币"] = "CNY"
        self.assertEqual(self.run_match(closing=close)["客户平仓盈亏_结算未扣费用"], 0)

    def test_duplicate_trade_and_empty(self):
        with self.assertRaises(ValueError):
            classify_trades(pd.concat([self.trades, self.trades]), self.opening, self.closing)
        result, audit = classify_trades(self.trades.iloc[:0], self.opening, self.closing)
        self.assertTrue(result.empty)
        self.assertTrue(audit.empty)

    def test_summary_keeps_missing_profit(self):
        result, _ = classify_trades(self.trades, self.opening, self.closing.iloc[:0])
        self.assertTrue(pd.isna(summarize_position_activity(result).iloc[0]["客户平仓盈亏_结算未扣费用"]))


if __name__ == "__main__":
    unittest.main()
