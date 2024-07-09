# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from enum import Enum
from typing import Final, Dict, Tuple, List
import numpy as np
import pandas as pd

"""
本文件主要用来存放资产配置所需要的参数
# TODO 后续调整应放入配置文件或者数据库
"""


class TimeSeriesType(Enum):
    """
    定义资产历史数据类型的枚举值
    """
    YIELD = 'yield'
    PRICE = 'price'
    PRICE_CHANGE = 'price_change'


class DataOriginalSrc(Enum):
    """
    定义投资数据的初始来源，例如：wind,bloomberg,other
    """
    WIND: str = 'wind'
    BLOOMBERG: str = 'bloomberg'
    OTHER: str = 'other'


class AnnualizedMultiplier(Enum):
    """
    定义投资数据的年化乘数
    """
    MONTHS: int = 12
    YEAR: int = 1
    TRADE_DAYS: int = 250


@dataclass
class ExcelFileSetting:
    """
    定义excel源数据文件的配置参数
    """
    # 源数据存放地址
    SOURCE_PATH = "%s/data/SAAData.xlsx" % (os.getcwd())

    # 结果存放地址
    OUTPUT_PATH = "%s/result/efficient_frontier.xlsx" % (os.getcwd())

    # 测试数据存放地址
    TEST_PATH = "%s/result/test.xlsx" % (os.getcwd())

    """
    存放数据使用的标签名称，用作表名或列名
    """
    LABEL_ASSETS_NAME = "assets_name"
    LABEL_EXPECTED_RETURN = "expected_return"
    LABEL_EXPECTED_VOLATILITY = "expected_volatility"
    LABEL_SKEWNESS = "skewness"
    LABEL_EXCESS_KURTOSIS = "excess_kurtosis"
    LABEL_BENCHMARK_NAME = "benchmark_name"
    LABEL_ASSETS_PARAS = "assets_paras"
    LABEL_CORRELATIONS = "correlations"
    LABEL_EFFICIENT_FRONTIER = "efficient_frontier"
    LABEL_PORTFOLIO_NAME = "portfolio_name"
    LABEL_SHARP_RATIO = "sharp_ratio"
    LABEL_IMAGE = "image"

    # 各类数据存放的sheet名
    DATA_LIST = ["HistoryYield", "HistoryIndex"]

    # 各类数据的日期均放在首列
    INDEX_COLUMN: Final[int] = 0

    # TODO 市场基准的历史数据的初始来源、证券ID和初始文件中列名应对应起来，待后续再完善
    # 各类资产的历史数据列名与证券代码
    # 国债采用中证10年期中国国债收益率
    GOV_BOND_COLUMN = "China_gov_bond_yield"
    GOV_BOND_SECURITY_ID = ''
    GOV_BOND_DATA_MULTIPLIER = 0.01
    # 货币基金采用wind货币基金指数
    MONEY_FOND_COLUMN = "money_fund"
    MONEY_FOND_SECURITY_ID = 'H11025.CSI'
    # 境内权益采用沪深300全收益指数
    CHINA_EQUITY_COLUMN = "China_equity"
    CHINA_EQUITY_SECURITY_ID = 'H00300.CSI'
    # 境内信用债采用中债高信用等级债券财富（总值）指数
    CHINA_CREDIT_COLUMN = "China_credit"
    CHINA_CREDIT_SECURITY_ID = 'CBA01901.CS'
    # 香港权益采用恒生指数R
    HK_EQUITY_COLUMN = "HK_equity"
    HK_EQUITY_SECURITY_ID = 'HSIRH.HI'

    # 各类资产的数据频率
    DATA_FREQ = {}

    # 年化乘数，此处因为使用月数据因此为12
    MONTHS: Final[int] = 12

    # 设定源数据的日期标签
    DATE_INDEX: Final[str] = "Date"


@dataclass
class HistoryDate:
    # TODO 定义计算市场基准所采用历史数据的时间区间，后续应对各类市场基准设置单独的时间区间，此处先简化处理
    START_DATE = '2014-06-30'
    END_DATE = '2024-05-31'
    TIME_INTERVAL = ('2014-06-30', '2024-05-31')


class MarketDataPara:
    """
    定义计算市场指数所需要的参数
    """
    def __init__(self, name: str, security_id: str, data_src: DataOriginalSrc = DataOriginalSrc.WIND,
                 weight: float = 1, start_date: str = HistoryDate.START_DATE, end_date: str = HistoryDate.END_DATE):
        self.name = name
        self.security_id = security_id
        self.data_src = data_src
        self.weight = weight
        self.start_date = start_date
        self.end_date = end_date


class BenchmarkPara:
    """
    定义计算市场基准所需要的参数，考虑混合基准，需检查合计权重是否为100%
    """
    def __init__(self, name: str, market_data: List[MarketDataPara],
                 data_type: TimeSeriesType = TimeSeriesType.PRICE_CHANGE):
        self.name = name
        # 检查市场指数的权重是否加总为100%
        if sum(x.weight for x in market_data) != 1:
            raise ValueError("传递了无效的参数值:market_data, 构建市场基准的指数权重相加应该等于100%")
        self.market_data = market_data
        self.market_data_type = data_type


@dataclass
class BenchmarkSetting:
    """
    初始化市场基准的配置参数, 为便于与从excel文件中读取的数据相关联，所有市场指数的名称均保持和excel中对应列的名称一致
    # TODO 后续应将所有配置参数改由配置文件存放
    """
    # 货币基金的市场基准，采用单一基准，权重为100%
    money_fund_index = MarketDataPara(ExcelFileSetting.MONEY_FOND_COLUMN,
                                      ExcelFileSetting.MONEY_FOND_SECURITY_ID, DataOriginalSrc.WIND, 1,
                                      HistoryDate.START_DATE, HistoryDate.END_DATE)
    money_fund_bench_para = BenchmarkPara('money_fund', [money_fund_index], TimeSeriesType.PRICE_CHANGE)

    # 30年期国债的市场基准
    gov_bond_index = MarketDataPara(ExcelFileSetting.GOV_BOND_COLUMN,
                                    ExcelFileSetting.GOV_BOND_SECURITY_ID, DataOriginalSrc.WIND, 1,
                                    HistoryDate.START_DATE, HistoryDate.END_DATE)
    gov_bond_bench_para = BenchmarkPara('30_year_gov_bond', [gov_bond_index], TimeSeriesType.YIELD)

    # 信用债券的市场基准
    credit_bond_index = MarketDataPara(ExcelFileSetting.CHINA_CREDIT_COLUMN,
                                       ExcelFileSetting.CHINA_CREDIT_SECURITY_ID, DataOriginalSrc.WIND, 1,
                                       HistoryDate.START_DATE, HistoryDate.END_DATE)
    credit_bond_bench_para = BenchmarkPara('High_level_credit', [credit_bond_index], TimeSeriesType.PRICE_CHANGE)

    # 境内权益的市场基准
    China_equity_index = MarketDataPara(ExcelFileSetting.CHINA_EQUITY_COLUMN,
                                        ExcelFileSetting.CHINA_EQUITY_SECURITY_ID, DataOriginalSrc.WIND, 1,
                                        HistoryDate.START_DATE, HistoryDate.END_DATE)
    China_equity_bench_para = BenchmarkPara('CSI300', [China_equity_index], TimeSeriesType.PRICE_CHANGE)

    # 香港权益的市场基准
    HK_equity_index = MarketDataPara(ExcelFileSetting.HK_EQUITY_COLUMN,
                                     ExcelFileSetting.HK_EQUITY_SECURITY_ID, DataOriginalSrc.WIND, 1,
                                     HistoryDate.START_DATE, HistoryDate. END_DATE)
    HK_equity_bench_para = BenchmarkPara('HSIRH', [HK_equity_index], TimeSeriesType.PRICE_CHANGE)

    # 汇总市场基准设定
    bench_paras = {money_fund_bench_para.name: money_fund_bench_para,
                   gov_bond_bench_para.name: gov_bond_bench_para,
                   credit_bond_bench_para.name: credit_bond_bench_para,
                   China_equity_bench_para.name: China_equity_bench_para,
                   HK_equity_bench_para.name: HK_equity_bench_para
                   }

    # 市场无风险利率设定为2.3%
    RISK_FREE_RATE = 0.023


@dataclass
class AssetSetting:
    """
    初始化大类资产的配置参数，主要是需要将资产名称和市场基准名称关联起来
    # TODO 后续存入配置文件或数据库
    """
    # 先用4类资产做测试
    ASSETS_NAME = ['cash', 'gov_bond', 'credit_bond', 'China_equity', 'HK_equity']

    # 存储资产名称与基准名称的对应关系
    ASSET_BENCHMARKS = {'cash': 'money_fund', 'gov_bond': '30_year_gov_bond', 'credit_bond': 'High_level_credit',
                        'China_equity': 'CSI300', 'HK_equity': 'HSIRH'}


@dataclass
class PortfolioSetting:
    """
    初始化投资组合的配置参数
    """
    PORTFOLIO_NAME = '4_assets_portfolio'
    # 建立4类资产的投资组合
    ASSETS_NAME = ['cash', 'gov_bond', 'credit_bond', 'China_equity']

    # 假设权益资产配置不超过30%
    # TODO 后续各类资产的权重限制条件可存入配置文件或数据库

    # 用于存储各类资产权重的上下限，未设置的默认为（0，1）
    WEIGHTS_INTERVAL = {'China_equity': (0, 0.3)}

    # 将不等约束条件
    """
        以下是官网的示例，摘录备查，其中'jac'代表降梯度函数，即导数
        ineq_cons = {'type': 'ineq',
                 'fun' : lambda x: np.array([1 - x[0] - 2*x[1],
                                             1 - x[0]**2 - x[1],
                                             1 - x[0]**2 + x[1]])
        eq_cons = {'type': 'eq',
               'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
               'jac' : lambda x: np.array([2.0, 1.0])}
    """
    INEQ_CONS = [{'type': 'ineq', 'fun': lambda weights: 0.3 - weights[3]}]


@dataclass
class PlotSetting:
    """
    记录生成图像的参数设定
    """
    # 生成图像的存放地点，只有有效前沿的图
    EF_IMAGE_PATH = "%s/result/efficient_frontier.png" % (os.getcwd())

    # 生成图像的存放地点，含资本市场线的图
    CML_IMAGE_PATH = "%s/result/capital_market_line.png" % (os.getcwd())

    # 画布大小设定
    FIGURE_SIZE = (10, 6)

    # 散点图个数
    SCATTER_COUNTS = 1000

    # 存放标签名称
    LABEL_CML = "capital_market_line"


@dataclass
class MonteCarloSetting:
    """
    记录蒙特卡洛模拟使用的参数
    """
    # 蒙特卡洛的次数
    MONTE_CARLO_CONUTS = 10000


if __name__ == '__main__':

    # weights = [1/5 for x in range(5)]
    # print(weights)

    # series1 = pd.Series(data={"one": 1, "two": 2}, index=["one", "two"])
    # series2 = pd.Series(data={"one": 3, "two": 4}, index=["one", "two"])
    """
    a1 = np.array([0.1, 0.2, 0.3, 0.4])
    a2 = np.array([0.5, 0.6, 0.7, 0.8])
    a3 = np.outer(a1, a2)
    df1 = pd.DataFrame([a1, a2])

    series1 = df1[0]
    print(series1)

    print(series1[1])

    print(df1)
    """

    # initial_guess = np.random.dirichlet(np.ones(5))
    # print(initial_guess)

    # is_special_zone = list(map(lambda name: name in AssetSetting.weights_zone, AssetSetting.ASSETS_NAME))
    # print(is_special_zone)
    # test1 = np.random.random()
    # print(test1)

    a = np.array([1, -1])

    result = np.all(a > 0)

    print(np.linspace(0.0, 0.25))



    """
    data = {
        'returns_a': [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03, 0.04, -0.01, 0.02],
        'returns_b': [0.02, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03, 0.04, -0.01, 0.02]
    }
    df = pd.DataFrame(data)

    df_rolling = (1 + df).rolling(window=5).apply(lambda x: x.prod()) - 1

    test = df['returns_a'].tolist()

    sumprod = np.prod([x + 1 for x in test[0:5]]) - 1

    print([x + 1 for x in test[0:5]])

    print(df_rolling)

    print(sumprod)

    cml_x = np.linspace(0.0, 0.25)

    print("cml_x = np.linspace(0.0, 0.25): %s" % cml_x)
    """
