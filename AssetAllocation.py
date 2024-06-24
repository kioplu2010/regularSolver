# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats
from regularSolver import DataProcess as dp
from regularSolver import SAAconfig as config
from enum import Enum
from typing import Final, Dict, Tuple, List
from dataclasses import dataclass


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
                                  type: config.TimeSeriesType = config.TimeSeriesType.PRICE_CHANGE):
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
    def stats_benchmark(timeseries: pd.Series, type: config.TimeSeriesType = config.TimeSeriesType.PRICE_CHANGE):
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
        if type == config.TimeSeriesType.YIELD:
            # 如果时间序列为利率数据，一般都是直接传入的年化利率
            annul_return = timeseries.mean()
            annul_volatility = timeseries.std()
            skewness = stats.skew(timeseries.astype(float))
            excess_kurtosis = stats.kurtosis(timeseries.astype(float))
        elif type == config.TimeSeriesType.PRICE_CHANGE:
            # 如果时间序列为价格变动，默认为月度价格变动，收益率与波动率需进行年化
            annul_return = (1 + timeseries.mean()) ^ config.AnnualizedMultiplier.MONTHS - 1
            annul_volatility = timeseries.std() * np.sqrt(config.AnnualizedMultiplier.MONTHS)
            skewness = stats.skew(timeseries.astype(float))
            excess_kurtosis = stats.kurtosis(timeseries.astype(float))
        elif type == config.TimeSeriesType.PRICE:
            # 如果时间序列为价格，则需要先处理为价格涨跌幅数据
            annul_return = (1 + timeseries.pct_change().mean()) ^ config.AnnualizedMultiplier.MONTHS - 1
            annul_volatility = timeseries.pct_change().std() * np.sqrt(config.AnnualizedMultiplier.MONTHS)
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
        weights_vols = np.dot(self.weights_array(), self.returns_array())
        variance = sum(weights_vols * np.transpose(weights_vols) * self.correlations)
        volatility = variance ** 0.5
        return volatility


if __name__ == '__main__':
    """
    以下模块的作用是读取源数据并整理成需要的数据备用
    """
    # 从Excel中读取基准历史收益率数据，后续可改为从数据库中读取
    dfs_src = dp.read_data_from_execl(config.ExcelFileSetting.SOURCE_PATH, config.ExcelFileSetting.DATA_LIST,
                                      config.ExcelFileSetting.INDEX_COLUMN)

    # 清除Excel中的无关数据，后续可在此步骤中加入数据清洗的环节
    dfs_prepared = dp.data_prepare(dfs_src)

    # 定义市场基准取数的历史区间
    start_date = config.HistoryDate.START_DATE
    end_date = config.HistoryDate.END_DATE

    # 获取历史利率曲线数据
    df_yield = dfs_prepared[config.ExcelFileSetting.DATA_LIST[0]]

    # 获取历史国债到期收益率数据，源数据来自Wind，因此除以100计算收益率数据
    gov_bond_yield_series = df_yield[config.ExcelFileSetting.GOV_BOND_COLUMN] * config.ExcelFileSetting.GOV_BOND_DATA_MULTIPLIER
    gov_bond_history_yields = gov_bond_yield_series[start_date:end_date]

    # 获取历史指数价格数据，转换为价格涨跌幅便于后续使用
    df_return = dfs_prepared[config.ExcelFileSetting.DATA_LIST[1]].pct_change()

    # 获取货币基金历史收益率数据
    money_fund_series = df_return[config.ExcelFileSetting.MONEY_FOND_COLUMN]
    money_fund_history_returns = money_fund_series[start_date:end_date]

    # 获取中国权益历史收益率数据
    china_equity_series = df_return[config.ExcelFileSetting.CHINA_EQUITY_COLUMN]
    china_equity_history_returns = china_equity_series[start_date:end_date]

    # 获取中国信用债历史收益率数据
    china_credit_series = df_return[config.ExcelFileSetting.CHINA_CREDIT_COLUMN]
    china_credit_history_returns = china_credit_series[start_date:end_date]

    # 获取香港权益历史收益率数据
    hk_equity_series = df_return[config.ExcelFileSetting.HK_EQUITY_COLUMN]
    hk_equity_history_returns = hk_equity_series[start_date:end_date]

    # 合并源数据，包括国债利率、货币基金、中国权益、中国信用债与香港权益
    data_merger = pd.DataFrame({gov_bond_history_yields.name: gov_bond_history_yields,
                                money_fund_history_returns.name: money_fund_history_returns,
                                china_equity_history_returns.name: china_equity_history_returns,
                                china_credit_history_returns.name: china_credit_history_returns,
                                hk_equity_history_returns.name: hk_equity_history_returns})

    """
    以下模块的作用是计算市场基准的参数，并以此为基础建立各大类资产
    """
    # 建立空的市场基准数据集以备逐一加入市场基准，用于后续的相关系数矩阵计算
    bench_df = pd.DataFrame()

    # 建立空的资产大类列表
    asset_list = []

    # 构建投资组合，为各类资产设定名称与参数，为方便构造，暂时将各类资产的参数都参照历史市场基准设定
    # TODO 后续扩展的时候可以对各类资产的参数做精细化调整，例如结合宏观经济数据设定各类资产的预期收益率，
    # TODO 通过自回归模型（GARCH）去设定各类资产的预期波动率，其他参数和相关系数矩阵可沿用市场历史基准参数
    for asset_name in config.AssetSetting.ASSETS_NAME:
        # 通过各类资产对应的市场基准获取基准名称
        benchmark_name = config.AssetSetting.ASSET_BENCHMARKS[asset_name]
        # 通过基准名称获取对应的市场基准数据
        benchmark_para = config.BenchmarkSetting.bench_paras[benchmark_name]

        # 考虑混合基准的情形，将基准数据乘以权重
        bench_series_list = list(map(lambda market_data_para:
                                     data_merger[market_data_para.name] * market_data_para.weight,
                                     benchmark_para.market_data))

        # 将乘以权重后的市场基准加总获得混合基准
        weighted_bench_series = sum(bench_series_list)

        # 将混合基准加入市场基准数据集,汇总的前提是所有市场基准有相同的index，即相同的日期index
        # TODO 当前将所有市场基准的时间区间设置为一致，且用日期作为所有时间序列的index，因此可以直接合并
        # TODO 如果计算各类市场基准的年化收益率、波动率的时间区间不一致，则应提前将时间序列的index调整为一致
        bench_df[benchmark_name] = weighted_bench_series

        # 建立市场基准
        benchmark = Benchmark.benchmark_from_timeseires(benchmark_name, weighted_bench_series, start_date, end_date,
                                                        benchmark_para.market_data_type)

        # TODO 建立资产大类，暂时先用市场基准的年化收益率、波动率作为预期收益率和波动率
        # TODO 后续优化时再对各类资产的预期收益率与预期波动率做调整，可考虑针对每一种大类资产类型单独建模
        asset = Asset(asset_name, benchmark.annul_return, benchmark.annul_volatility, benchmark)

        # 将大类资产加入列表
        asset_list.append(asset)

    """
    以下模块的作用是初始化投资组合
    """
    # 计算相关系数矩阵,默认为计算皮尔逊相关系数（'pearson'）
    corr_matrix = bench_df.corr()

    # 计算资产种类
    asset_types = len(config.AssetSetting.ASSETS_NAME)

    # 初始化资产权重，等权简化处理
    asset_weights = [1 / asset_types for x in range(asset_types)]

    # 建立初始投资组合
    portfolio = Portfolio(config.PortfolioSetting.PORTFOLIO_NAME, asset_list, asset_weights, corr_matrix)








