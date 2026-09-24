import unittest
import pandas as pd
from manager_pnl import rank_manager_product_pnl, resolve_manager, PNL_COLUMN


class ManagerPnlTests(unittest.TestCase):
    def row(self, name, account, pnl, ticker="CLZ6", currency="CNY", date="2026-09-10"):
        return {"客户编号": account, "客户名称": name, "客户简称": name,
                "平仓日期": date, "簿记账户": "南向对客", "标的": ticker,
                "结算货币": currency, PNL_COLUMN: pnl}

    def test_net_accounts_and_contracts_before_ranking(self):
        data = pd.DataFrame([
            self.row("北京雪球1号基金", "1", 100, "CLZ6"),
            self.row("北京雪球2号基金", "2", -80, "CLV6"),
            self.row("遂玖1号基金", "3", -40),
            self.row("遂久2号基金", "4", 10),
            self.row("独立客户", "5", 50),
        ])
        summary, gain, loss, _, _ = rank_manager_product_pnl(
            data, "2026-09-09", "2026-09-15", lambda t: t[:2])
        self.assertEqual(gain["管理人"].tolist(), ["独立客户", "雪球"])
        self.assertEqual(gain["客户平仓盈亏"].tolist(), [50, 20])
        self.assertEqual(loss["客户平仓盈亏"].tolist(), [-30])
        self.assertEqual(summary["客户平仓盈亏"].sum(), data[PNL_COLUMN].sum())

    def test_dates_currency_and_zero(self):
        data = pd.DataFrame([
            self.row("雪球1号基金", "1", 10),
            self.row("雪球1号基金", "1", 20, currency="USD"),
            self.row("雪球1号基金", "1", 1000, date="2026-09-16"),
            self.row("独立客户", "2", 0),
        ])
        summary, gain, loss, detail, _ = rank_manager_product_pnl(
            data, "2026-09-09", "2026-09-15", lambda t: "CL1")
        self.assertEqual(len(detail), 3)
        self.assertEqual(len(gain), 2)
        self.assertTrue(loss.empty)
        self.assertEqual(set(gain["结算货币"]), {"CNY", "USD"})

    def test_no_ambiguous_yongli_merge_and_overrides(self):
        corporate = self.row("浙江物产永利实业", "1", 0)
        fund = self.row("永利商品精选2号私募证券投资基金-LJL", "2", 0)
        legal = self.row("浙江钱唐永利资产管理有限公司作为管理人代表永利9号基金", "3", 0)
        self.assertEqual(resolve_manager(corporate)[0], "浙江物产永利实业")
        self.assertEqual(resolve_manager(fund)[0], resolve_manager(legal)[0])
        self.assertEqual(resolve_manager(fund, {"2": "指定管理人"})[0], "指定管理人")

    def test_bad_profit_and_duplicate_rejected(self):
        bad = pd.DataFrame([self.row("客户", "1", None)])
        with self.assertRaises(ValueError):
            rank_manager_product_pnl(bad, "2026-09-09", "2026-09-15", str)
        dup = pd.DataFrame([self.row("客户", "1", 10)] * 2)
        with self.assertRaises(ValueError):
            rank_manager_product_pnl(dup, "2026-09-09", "2026-09-15", str)


if __name__ == "__main__":
    unittest.main()
