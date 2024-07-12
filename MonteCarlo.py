# -*- coding: utf-8 -*-
from regularSolver.AssetAllocation import Portfolio
from regularSolver import SAAconfig as config
import numpy as np
from scipy import stats
from scipy.stats import norm, skewnorm
from typing import List


class MonteCarlo:
    """
    用来对已确定各类资产权重的投资组合做蒙特考虑模拟
    """
    def __init__(self, portfolio: Portfolio, start_amount: float, net_cash_flow: List[float], years: int,
                 counts=config.MonteCarloSetting.MONTE_CARLO_CONUTS):
        """
        初始化
        :param portfolio:选中的投资组合
        :param start_amount:期初投资组合净值
        :param net_cash_flow:每年资金净流入与流出金额
        :param years: 默认模拟3年视角下的资产配置结果
        :param counts:默认模拟10000次随机结果
        """
        self.portfolio = portfolio
        self.start_amount = start_amount
        self.net_cash_flow = net_cash_flow
        self.years = years
        self.counts = counts

    def get_simulations_returns(self):

        # 累计资金占用
        accumulated_average_fund = np.ones(self.counts) * self.start_amount

        # 累计投资金额
        accumulated_invest_amount = np.ones(self.counts) * self.start_amount

        # 初始化收益率矩阵列表
        accumulated_returns = np.ones(self.counts)

        # 资产种类
        asset_types = self.portfolio.get_assets_types()

        for x in range(self.years):

            # 初始化收益率矩阵
            returns = np.zeros((self.counts, asset_types))

            # 生成标准正态分布的随机数
            standard_normals = norm.rvs(size=(self.counts, asset_types))

            # 使用Cholesky分解引入相关性，L为正定矩阵
            L = np.linalg.cholesky(self.portfolio.correlations)

            # 为正态分布数据引入相关性
            correlated_normals = np.dot(standard_normals, L.T)

            cash_flow = self.net_cash_flow[x]

            # 生成随机收益率
            for i in range(asset_types):
                # 获取当前资产类型
                asset = self.portfolio.assets[i]

                returns[:, i] = asset.expected_return + correlated_normals[:, i] * asset.expected_volatility

                """
                skewnorm.rvs不适合用来生成随机收益率数据，主要原因是若数据正偏，则平均值应大于期望值，继续使用平均值作期望值，那生成
                的随机数据的平均值会明显大于原来的平均值（新的期望值）
                
                # 如果为正态分布使用标准化数据获取随机收益率
                if asset.annul_skewness == 0:
                    returns[:, i] = asset.expected_return + correlated_normals[:, i] * asset.expected_volatility
                else:
                    # 若偏度不为零，使用skewnorm分布生成具有偏度的随机数
                    skew_returns = skewnorm.rvs(a=asset.annul_skewness, loc=asset.expected_return,
                                                 scale=asset.expected_volatility, size=self.counts)
                """

                """
                print("asset mean: %s std: %s skew: %s kurtosis: %s" % (asset.expected_return,
                                                                        asset.expected_volatility,
                                                                        asset.annul_skewness,
                                                                        asset.annul_excess_kurtosis))

                adj1_mean = skew_returns.mean()
                adj1_std = skew_returns.std()
                adj1_skewness = stats.skew(skew_returns)
                adj1_excess_kurtosis = stats.kurtosis(skew_returns)

                print("adj1 mean: %s std: %s skew: %s kurtosis: %s" % (adj1_mean,
                                                                        adj1_std,
                                                                        adj1_skewness,
                                                                        adj1_excess_kurtosis))

                # 为避免生成的随机数据偏离，对生成的随机数据作调整
                skew_std = skew_returns.std()
                skew_returns = skew_returns * asset.expected_volatility / skew_std

                skew_mean = skew_returns.mean()

                skew_returns = skew_returns + (asset.expected_return - skew_mean)

                returns[:, i] = skew_returns

                
                adj2_mean = skew_returns.mean()
                adj2_std = skew_returns.std()
                adj2_skewness = stats.skew(skew_returns)
                adj2_excess_kurtosis = stats.kurtosis(skew_returns)
                """

                """
                # 若超值峰度不为0，调整随机收益率峰度
                if asset.annul_excess_kurtosis != 0:
                    std_norm = norm.rvs(size=self.counts)
                    adjusted_returns = (std_norm * np.sqrt(1 + asset.annul_excess_kurtosis / 3) * np.std(returns[:, i])
                                        + np.mean(returns[:, i]))
                    returns[:, i] = adjusted_returns
    
                    adjusted_mean = adjusted_returns.mean()
                    adjusted_std = adjusted_returns.std()
                    adjusted_skewness = stats.skew(adjusted_returns)
                    adjusted_excess_kurtosis = stats.kurtosis(adjusted_returns)
    
                    print("adj3 mean: %s std: %s skew: %s kurtosis: %s" % (adjusted_mean,
                                                                           adjusted_std,
                                                                           adjusted_skewness,
                                                                           adjusted_excess_kurtosis))
    
                """

            # 计算投资组合收益率
            portfolio_returns = np.dot(returns, np.array(self.portfolio.weights))

            # 计算投资组合累计收益率
            accumulated_returns = accumulated_returns * (portfolio_returns + 1)

            # 计算资金占用
            accumulated_average_fund = accumulated_invest_amount + 0.5 * cash_flow

            # 计算期末金额
            accumulated_invest_amount = (1 + portfolio_returns) * accumulated_average_fund

        # 计算平均年化收益率
        # annual_returns = np.power(accumulated_returns, 1 / self.years) - 1
        accumulated_returns = accumulated_returns - 1

        # 返回结果
        return accumulated_returns, accumulated_invest_amount


if __name__ == '__main__':

    """
    skew_returns = skewnorm.rvs(a=0.1, loc=0, scale=1, size=10000)

    # skew_returns = norm.rvs(size=100000)

    mean = skew_returns.mean()
    std = skew_returns.std()
    skewness = stats.skew(skew_returns)
    excess_kurtosis = stats.kurtosis(skew_returns)

    print("mean: %s" % mean)
    print("std: %s" % std)
    print("skewness: %s" % skewness)
    print("excess_kurtosis: %s" % excess_kurtosis)
    """

    aa = [0.027, 0.001, 0.008]

    bb = np.power(aa, 1/3)

    print(bb)
















