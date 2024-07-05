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
from typing import Final, Dict, Tuple, List


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
    def stats_benchmark(timeseries: pd.Series,
                        type: config.TimeSeriesType = config.TimeSeriesType.PRICE_CHANGE) -> (float, float, float, float):
        """
        计算时间序列的统计指标
        :param timeseries:
        :param type:价格、利率或者价格变动
        :return:annul_return, annul_volatility, skewness, excess_kurtosis 数据均为float
        """
        annul_return = 0.0
        annul_volatility = 0.0
        skewness = 0.0
        excess_kurtosis = 0.0
        if type == config.TimeSeriesType.YIELD:
            # 如果时间序列为利率数据，一般都是直接传入的年化利率
            annul_return = timeseries.mean()
            annul_volatility = timeseries.std()
            skewness = stats.skew(timeseries.astype(float))
            excess_kurtosis = stats.kurtosis(timeseries.astype(float))
        elif type == config.TimeSeriesType.PRICE_CHANGE:
            # 如果时间序列为价格变动，默认为月度价格变动，收益率与波动率需进行年化
            annul_return = (1 + timeseries.mean()) ** float(config.AnnualizedMultiplier.MONTHS.value) - 1
            annul_volatility = timeseries.std() * np.sqrt(float(config.AnnualizedMultiplier.MONTHS.value))
            skewness = stats.skew(timeseries.astype(float))
            excess_kurtosis = stats.kurtosis(timeseries.astype(float))
        elif type == config.TimeSeriesType.PRICE:
            # 如果时间序列为价格，则需要先处理为价格涨跌幅数据
            annul_return = (1 + timeseries.pct_change().mean()) ** float(config.AnnualizedMultiplier.MONTHS.value) - 1
            annul_volatility = timeseries.pct_change().std() * np.sqrt(float(config.AnnualizedMultiplier.MONTHS.value))
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

    # 返回资产种类
    def get_assets_types(self):
        return len(self.assets)

    # 返回资产名称列表
    def get_asset_names(self):
        asset_names = []
        for asset in self.assets:
            asset_names.append(asset.name)
        return asset_names

    # 返回资产预期收益的一维数组
    def get_returns_array(self):
        expected_returns = []
        for x in self.assets:
            expected_returns.append(x.expected_return)
        return np.array(expected_returns)

    # 返回资产预期波动率的一维数组
    def get_vols_array(self):
        expected_vols = []
        for x in self.assets:
            expected_vols.append(x.expected_volatility)
        return np.array(expected_vols)

    # 返回资产权重的一维数组
    def get_weights_array(self):
        return np.array(self.weights)

    # 返回投资组合的预期收益率
    def get_portfolio_return(self):
        expected_return = self.get_returns_array().dot(self.weights)
        return expected_return

    # 返回投资组合的预期波动率
    def get_portfolio_volatility(self):
        # 考虑资产权重和资产波动率都是一维数组，此处为获得矩阵应求外积
        weights_vols = np.outer(self.get_weights_array(), self.get_vols_array())
        variance = np.sum((weights_vols * np.transpose(weights_vols) * self.correlations).values)
        volatility = variance ** 0.5
        return volatility

    # 返回投资组合的夏普比率
    def get_portfolio_sharpe(self):
        sharp_ratio = (self.get_portfolio_return() - config.BenchmarkSetting.RISK_FREE_RATE) / self.get_portfolio_volatility()
        return sharp_ratio


class Markowitz:
    """
    计算马科维兹有效前沿，通过scipy的Optimization模块计算，后续增加用cvxpy计算以获取更加精确的解
    """
    # 优化求解目标函数，给定目标收益率的情况下求最小波动率
    @staticmethod
    def min_volatility(weights: List[float], portfolio: Portfolio) -> float:
        portfolio.weights = weights
        return portfolio.get_portfolio_volatility()

    # 各类资产权重的边界，都大于0小于1
    @staticmethod
    def get_bounds(assets_types: int) -> list:
        return [(0, 1)] * assets_types

    @staticmethod
    def get_efficient_frontier(optimize_func: callable, origin_portfolio: Portfolio,
                               constraints: List, nportfolios: int = 100) -> List[Portfolio]:
        """
        计算有效前沿
        :param origin_portfolio: 初始化的投资组合，资产权重为等权，其他参数已计算完成
        :param constraints:各类资产权重的约束条件
        :param nportfolios:默认计算100个点
        :param optimize_func:默认优化目标为求最小波动率，
        :return:
        """
        # 各类资产中预期回报最小的资产
        min_return = np.min(origin_portfolio.get_returns_array())

        # 各类资产中预期回报最大的资产
        max_return = np.max(origin_portfolio.get_returns_array())

        # 获取资产种类
        ntypes = origin_portfolio.get_assets_types()

        # 获取各类资产的收益率
        expected_returns = origin_portfolio.get_returns_array()

        # 初始化资产权重，等权简化处理
        initial_weights = np.array([1 / ntypes for x in range(ntypes)])

        # 生成组合目标收益率序列
        target_returns = np.linspace(min_return, max_return, nportfolios)

        # 初始化投资组合列表
        target_ports = []

        # 默认约束为各类资产权重相加应等于1
        default_cons = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]

        # 获取各类资产权重的上下限
        opt_bounds = Markowitz.get_bounds(ntypes)

        # 保存优化求解失败次数
        failures = 0

        # 计算有效前沿
        for target_return in target_returns:

            # 增加约束组合收益率等于目标收益率
            target_return_cons = [{'type': 'eq', 'fun': lambda weights: np.sum(expected_returns * weights)
                                   - target_return}]

            # 汇总各类资产权重的设置
            opt_cons = default_cons + target_return_cons + constraints

            # 调用优化函数计算结果
            result = sco.minimize(fun=optimize_func, x0=initial_weights, args=(origin_portfolio,),
                                  method='SLSQP', bounds=opt_bounds, constraints=opt_cons)

            # 如果求解成功，将结果加入投资组合列表
            if result.success:
                # 各投资组合的命名方式为：初始投资组合名称 + 预期收益率
                result_name = origin_portfolio.name + str(" {:.2%}").format(target_return)
                result_weights = result.x.tolist()
                target_ports.append(Portfolio(result_name, origin_portfolio.assets, result_weights,
                                              origin_portfolio.correlations))
            else:
                failures += 1

        print("optimize failures: %s" % failures)

        return target_ports


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

    print("Well Done!")





