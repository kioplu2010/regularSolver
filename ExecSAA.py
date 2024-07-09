# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco
import scipy.interpolate as sci
from scipy import stats
from regularSolver import DataProcess as dp
from regularSolver import SAAconfig as config
from regularSolver import SaveResult as save
from regularSolver.AssetAllocation import Benchmark, Asset, Portfolio, Markowitz
from regularSolver.MonteCarlo import MonteCarlo

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

    # 获取历史利率曲线数据,源数据来自Wind，因此除以100计算收益率数据
    df_yield = (dfs_prepared[config.ExcelFileSetting.DATA_LIST[0]] *
                             config.ExcelFileSetting.GOV_BOND_DATA_MULTIPLIER)

    # 获取历史利率曲线的滚动年化数据，由于数据本身是年化收益率，滚动收益率直接12个月平均值
    df_yield_rolling = df_yield.rolling(window=config.ExcelFileSetting.MONTHS).apply(lambda x: x.mean())

    # 获取历史指数价格数据，转换为价格涨跌幅便于后续使用
    df_return = dfs_prepared[config.ExcelFileSetting.DATA_LIST[1]].pct_change()

    # 获取历史指数价格涨跌幅的滚动年化数据
    df_return_rolling = (1 + df_return).rolling(window=config.ExcelFileSetting.MONTHS).apply(lambda x: x.prod()) - 1

    # 合并原始数据,并获取需要的时间区间数据
    df_data_merge_src = pd.merge(df_yield ,df_return, left_index=True, right_index=True)
    df_data_merge = df_data_merge_src[start_date:end_date]

    # 合并滚动年化数据，便于计算相关系统矩阵
    df_data_rolling_src = pd.merge(df_yield_rolling, df_return_rolling, left_index=True, right_index=True)
    df_data_merge_rolling = df_data_rolling_src[start_date:end_date]

    """
    需要对每类市场基准的历史收益率数据作特殊处理时再启动此段
    # 获取历史国债到期收益率数据，源数据来自Wind，因此除以100计算收益率数据
    gov_bond_yield_series = (df_yield[config.ExcelFileSetting.GOV_BOND_COLUMN] *
                             config.ExcelFileSetting.GOV_BOND_DATA_MULTIPLIER.value)
    gov_bond_history_yields = gov_bond_yield_series[start_date:end_date]

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

    """
    以下模块的作用是计算市场基准的参数，并以此为基础建立各大类资产的模型
    """
    # 建立空的市场基准数据集以备逐一加入市场基准，用于后续的相关系数矩阵计算
    df_bench_rolling = pd.DataFrame()

    # 建立空的资产大类列表
    asset_list = []

    # 构建投资组合，为各类资产设定名称与参数，为方便构造，暂时将各类资产的参数都参照历史市场基准设定
    # TODO 后续扩展的时候可以对各类资产的参数做精细化调整，例如结合宏观经济数据设定各类资产的预期收益率，
    # TODO 通过自回归模型（GARCH）去设定各类资产的预期波动率，其他参数和相关系数矩阵可沿用市场历史基准参数
    for asset_name in config.PortfolioSetting.ASSETS_NAME:
        # 通过各类资产对应的市场基准获取基准名称
        benchmark_name = config.AssetSetting.ASSET_BENCHMARKS[asset_name]
        # 通过基准名称获取对应的市场基准数据
        benchmark_para = config.BenchmarkSetting.bench_paras[benchmark_name]

        # 考虑混合基准的情形，将基准数据乘以权重
        bench_series_list = list(map(lambda market_data_para:
                                     df_data_merge[market_data_para.name] * market_data_para.weight,
                                     benchmark_para.market_data))

        # 将乘以权重后的市场基准加总获得混合基准
        weighted_bench_series = sum(bench_series_list)

        # 获取滚动年化基准
        bench_series_list_rolling = list(map(lambda market_data_para:
                                     df_data_merge_rolling[market_data_para.name] * market_data_para.weight,
                                     benchmark_para.market_data))

        weighted_bench_rolling = sum(bench_series_list_rolling)

        # 将混合基准加入市场基准数据集,汇总的前提是所有市场基准有相同的index，即相同的日期index
        # TODO 当前将所有市场基准的时间区间设置为一致，且用日期作为所有时间序列的index，因此可以直接合并
        # TODO 如果计算各类市场基准的年化收益率、波动率的时间区间不一致，则应提前将时间序列的index调整为一致
        # df_bench[benchmark_name] = weighted_bench_series_rolling
        df_bench_rolling[benchmark_name] = weighted_bench_rolling


        # 建立市场基准
        benchmark = Benchmark.benchmark_from_timeseires(benchmark_name, weighted_bench_series, start_date, end_date,
                                                        benchmark_para.market_data_type)

        # TODO 建立资产大类，暂时先用市场基准的年化收益率、波动率作为预期收益率和波动率
        # TODO 后续优化时再对各类资产的预期收益率与预期波动率做调整，可考虑针对每一种大类资产类型单独建模
        # TODO 后续可考虑在此处引入Black-Littleman模型调整资产的预期收益率和波动率
        asset = Asset(asset_name, benchmark.annul_return, benchmark.annul_volatility, benchmark)

        # 将大类资产加入列表
        asset_list.append(asset)

    """
    以下模块的作用是初始化投资组合
    """
    # 计算相关系数矩阵,默认为计算皮尔逊相关系数（'pearson'）
    corr_matrix = df_bench_rolling.corr()

    # 计算资产种类
    asset_types = len(config.PortfolioSetting.ASSETS_NAME)

    # 初始化资产权重，等权简化处理
    origin_weights = [1 / asset_types for x in range(asset_types)]

    # 建立初始投资组合
    origin_portfolio = Portfolio(config.PortfolioSetting.PORTFOLIO_NAME, asset_list, origin_weights, corr_matrix)

    """
    以下模块的作用是计算马科维兹有效前沿
    """
    portfolios = Markowitz.get_efficient_frontier(Markowitz.min_volatility, origin_portfolio,
                                                  config.PortfolioSetting.INEQ_CONS, 200)

    """
    以下模块的作用是保存计算结果，将数据保存到excel，并单独保存图像
    """
    save.SaveFrontier.save_data_to_local(portfolios)

    """
    以下模块的作用是用选定的投资组合做模特卡洛模拟
    """
    selected_portfolio = portfolios[0]

    selected_portfolio.weights = config.MonteCarloSetting.SELECTED_WEIGHTS

    monte_carlo = MonteCarlo(selected_portfolio, config.MonteCarloSetting.START_AMOUNT,
                             config.MonteCarloSetting.NET_CASH_FLOW, config.MonteCarloSetting.TERMS,
                             config.MonteCarloSetting.MONTE_CARLO_CONUTS)

    # 获取收益率序列与投资金额序列
    returns, amounts = monte_carlo.get_simulations_returns()

    # 保存蒙特卡洛模拟结果
    save.SaveMonteCarlo.save_data_to_local(returns, amounts)

    print("Well Done!")