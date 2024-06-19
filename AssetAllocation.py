# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats
from . import DataProcess as dp
from . import SAAconfig as config
from enum import Enum
from typing import Final, Dict, Tuple, List
from dataclasses import dataclass


class MarketDataPara:
    """
    定义计算市场指数所需要的参数
    """
    def __init__(self, name: str, security_id: str, data_src: config.DataOriginalSrc = config.DataOriginalSrc.WIND,
                 weight: str = 1, start_date: str = config.HistoryDate.START_DATE, end_date: str = config.HistoryDate.END_DATE):
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
    def __init__(self, name: str, market_data: List[MarketDataPara]):
        self.name = name
        # 检查市场指数的权重是否加总为100%
        if sum(x.weight for x in market_data) != 1:
            raise ValueError("传递了无效的参数值:market_data, 构建市场基准的指数权重相加应该等于100%")
        self.market_data = market_data


@dataclass
class BenchmarkSetting:
    """
    初始化市场基准的配置参数, 为便于与从excel文件中读取的数据相关联，所有市场指数的名称均保持和excel中对应列的名称一致
    # TODO 后续应将所有配置参数改由配置文件存放
    """
    # 货币基金的市场基准，采用单一基准，权重为100%
    money_fund_index = MarketDataPara(config.ExcelFileSetting.MONEY_FOND_COLUMN,
                                      config.ExcelFileSetting.MONEY_FOND_SECURITY_ID, config.DataOriginalSrc.WIND, 1,
                                      config.HistoryDate.START_DATE, config.HistoryDate.END_DATE)
    money_fund_bench_para = BenchmarkPara('money_fund', [money_fund_index])

    # 10年期国债的市场基准
    gov_bond_index = MarketDataPara(config.ExcelFileSetting.GOV_BOND_COLUMN,
                                    config.ExcelFileSetting.GOV_BOND_SECURITY_ID, config.DataOriginalSrc.WIND, 1,
                                    config.HistoryDate.START_DATE, config.HistoryDate.END_DATE)
    gov_bond_bench_para = BenchmarkPara('10_year_gov_bond', [gov_bond_index])

    # 信用债券的市场基准
    credit_bond_index = MarketDataPara(config.ExcelFileSetting.CHINA_CREDIT_COLUMN,
                                       config.ExcelFileSetting.CHINA_CREDIT_SECURITY_ID, config.DataOriginalSrc.WIND, 1,
                                       config.HistoryDate.START_DATE, config.HistoryDate.END_DATE)
    credit_bond_bench_para = BenchmarkPara('High_level_credit', [credit_bond_index])

    # 境内权益的市场基准
    China_equity_index = MarketDataPara(config.ExcelFileSetting.CHINA_EQUITY_COLUMN,
                                        config.ExcelFileSetting.CHINA_EQUITY_SECURITY_ID, config.DataOriginalSrc.WIND, 1,
                                        config.HistoryDate.START_DATE, config.HistoryDate.END_DATE)
    China_equity_bench = BenchmarkPara('CSI300', [China_equity_index])

    # 香港权益的市场基准
    HK_equity_index = MarketDataPara(config.ExcelFileSetting.HK_EQUITY_COLUMN,
                                     config.ExcelFileSetting.HK_EQUITY_SECURITY_ID, config.DataOriginalSrc.WIND, 1,
                                     config.HistoryDate.START_DATE, config.HistoryDate. END_DATE)
    HK_equity_bench = BenchmarkPara('HSIRH', [HK_equity_index])


@dataclass
class AssetSetting:
    """
    初始化大类资产的配置参数，主要是需要将资产名称和市场基准名称关联起来
    """
    ASSET_BENCHMARKS = {'cash': 'money_fund', 'gov_Bond': '10_year_gov_bond', 'credit_bond': 'High_level_credit',
                        'China_equity': 'CSI300', 'HK_equity': 'HSIRH'}


class TimeSeriesType(Enum):
    """
    定义资产历史数据类型的枚举值
    """
    YIELD = 'yield'
    PRICE = 'price'
    PRICE_CHANGE = 'price_change'


class Benchmark:
    """
       用来定义市场基准，可以直接用年化收益率、波动率等参数构造，也可以根据传入的历史时间序列计算出所需参数
    """
    def __init__(self, name: str, annul_return: float, annul_volatility: float,
                 annul_skewness: float, annul_excess_kurtosis: float, start_date: str, end_date: str):
        self.name = name
        self.annul_return = annul_return
        self.annul_volatility = annul_volatility
        self.annul_skewness = annul_skewness
        self.annul_excess_kurtosis = annul_excess_kurtosis
        self.start_date = start_date
        self.end_date = end_date

    @classmethod
    def benchmark_from_timeseires(cls, name: str, timeseries: pd.Series, start_date: str, end_date: str,
                                  type: TimeSeriesType = TimeSeriesType.PRICE_CHANGE):
        """
        根据传入的
        :param name:
        :param timeseries:
        :param start_date:
        :param end_date:
        :param type:
        :return:Benchmark instance
        """
        # 通过传入的日期序列参数计算收益率、波动率、偏度和超值峰度（-3）
        annual_return, volatility, skewness, excess_kurtosis = Benchmark.stats_benchmark(timeseries, type)
        benchmark = cls(name, annual_return, volatility, skewness, excess_kurtosis, start_date, end_date)
        return benchmark

    # 计算时间序列的统计指标
    # 基本假设：利率曲线类指标均为年化收益率，价格类指标为每月收盘价较上月的涨跌幅
    @staticmethod
    def stats_benchmark(timeseries: pd.Series, type: TimeSeriesType = TimeSeriesType.PRICE_CHANGE):
        """
        计算时间序列的统计指标
        :param timeseries:
        :param type:价格、利率或者价格变动
        :return:annul_return, annul_volatility, skewness, excess_kurtosis 数据均为float
        """
        # annul_return = 0.0
        # annul_volatility = 0.0
        # skewness = 0.0
        # excess_kurtosis = 0.0
        if type == TimeSeriesType.YIELD:
            # 如果时间序列为利率数据，一般都是直接传入的年化利率
            annul_return = timeseries.mean()
            annul_volatility = timeseries.std()
            skewness = stats.skew(timeseries.astype(float))
            excess_kurtosis = stats.kurtosis(timeseries.astype(float))
        elif type == TimeSeriesType.PRICE_CHANGE:
            # 如果时间序列为价格变动，默认为月度价格变动，收益率与波动率需进行年化
            annul_return = (1 + timeseries.mean()) ^ dp.MONTHS - 1
            annul_volatility = timeseries.std() * np.sqrt(dp.MONTHS)
            skewness = stats.skew(timeseries.astype(float))
            excess_kurtosis = stats.kurtosis(timeseries.astype(float))
        elif type == TimeSeriesType.PRICE:
            # 如果时间序列为价格，则需要先处理为价格涨跌幅数据
            annul_return = (1 + timeseries.pct_change().mean()) ^ dp.MONTHS - 1
            annul_volatility = timeseries.pct_change().std() * np.sqrt(dp.MONTHS)
            skewness = stats.skew(timeseries.pct_change().astype(float))
            excess_kurtosis = stats.kurtosis(timeseries.pct_change().astype(float))
        else:
            raise ValueError("传递了无效的参数值 type %s，可选参数为yield和price" % type)

        return annul_return, annul_volatility, skewness, excess_kurtosis


class Asset:
    """
    定义资产类型，输入参数为资产名，预期收益、预期波动率和市场基准
    """
    def __init__(self, name: str, expected_return: float, expected_volatility: float, benchmark: Benchmark):
        self.name = name
        self.expected_return = expected_return
        self.expected_volatility = expected_volatility
        self.annul_skewness = benchmark.annul_skewness
        self.annul_excess_kurtosis = benchmark.annul_excess_kurtosis
        self.benchmark = benchmark


class Portfolio:
    """
    定义投资组合
    """
    def __init__(self, name: str, assets: list, weights: list, correlations):
        self.name = name
        self.assets = assets
        self.weights = weights
        self.correlations = correlations

    # 返回资产名称列表
    def asset_names(self):
        asset_names = []
        for asset in self.assets:
            asset_names.append(asset.name)
        return asset_names

    # 返回资产预期收益的一维数组
    def returns_array(self):
        expected_returns = []
        for x in self.assets:
            expected_returns.append(x.expected_return)
        return np.array(expected_returns)

    # 返回资产权重的一维数组
    def weights_array(self):
        return np.array(self.weights)

    # 返回投资组合的预期收益率
    def portfolio_return(self):
        expected_return = self.returns_array().dot(self.weights)
        return expected_return

    # 返回投资组合的预期波动率
    def portfolio_volatility(self):
        weights_vols = np.multiply(self.weights_array(), self.returns_array())
        variance = (weights_vols.dot(np.transpose(weights_vols))).dot(self.correlations)
        volatility = variance ** 0.5
        return volatility


if __name__ == '__main__':
    # 从Excel中读取基准历史收益率数据，后续可改为从数据库中读取
    dfs = dp.read_data_from_execl(dp.SOURCE_PATH, dp.DATA_LIST, dp.INDEX_COLUMN)

    # 清除Excel中的无关数据，后续可在此步骤中加入数据清洗的环节
    dfs_prepared = dp.data_prepare(dfs)

    # 定义市场基准取数的历史区间
    start_date = START_DATE
    end_date = END_DATE

    # 获取历史利率曲线数据
    df_yield = dfs_prepared[dp.DATA_LIST[0]]

    # 获取历史国债到期收益率数据，源数据来自Wind，因此除以100计算收益率数据
    gov_bond_yield_series = df_yield[dp.GOV_BOND_COLUMN] / 100
    gov_bond_history_yields = gov_bond_yield_series[start_date:end_date]
    print("gov_bond_history_yields.name : %s " % gov_bond_history_yields.name)

    # 获取历史指数价格数据，转换为价格涨跌幅便于后续使用
    df_return = dfs_prepared[dp.DATA_LIST[1]].pct_change()

    # 获取货币基金历史收益率数据
    money_fund_series = df_return[dp.MONEY_FOND_COLUMN]
    money_fund_history_returns = money_fund_series[start_date:end_date]

    # 获取中国权益历史收益率数据
    china_equity_series = df_return[dp.CHINA_EQUITY_COLUMN]
    china_equity_history_returns = china_equity_series[start_date:end_date]

    # 获取中国信用债历史收益率数据
    china_credit_series = df_return[dp.CHINA_CREDIT_COLUMN]
    china_credit_history_returns = china_credit_series[start_date:end_date]

    # 获取香港权益历史收益率数据
    hk_equity_series = df_return[dp.HK_EQUITY_COLUMN]
    hk_equity_history_returns = hk_equity_series[start_date:end_date]

    # 构建投资组合，为各类资产设定名称与参数，为方便构造，暂时将各类资产的参数都参照历史市场基准设定
    # 后续扩展的时候可以对各类资产的参数做精细化调整，例如结合宏观经济数据设定各类资产的预期收益率，
    # 通过自回归模型（GARCH）去设定各类资产的预期波动率，其他参数和相关系数矩阵可沿用市场历史基准参数

    # 定义国债资产
    # gov_bond_annul_return,gov_bond_annul_volatility,gov_bond_skewness




