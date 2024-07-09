# -*- coding: utf-8 -*-
from regularSolver.AssetAllocation import Portfolio
from regularSolver import SAAconfig as config
import numpy as np
from scipy.stats import norm, skewnorm


class MonteCarlo:
    """
    用来对已确定各类资产权重的投资组合做蒙特考虑模拟
    """
    def __init__(self, portfolio: Portfolio, years: int = 3, counts=config.MonteCarloSetting.MONTE_CARLO_CONUTS):
        """
        初始化
        :param portfolio:选中的投资组合
        :param years: 默认模拟3年视角下的资产配置结果
        :param counts:默认模拟10000次随机结果
        """
        self.portfolio = portfolio
        self.years = years
        self.counts = counts

    def get_simulations(self):
        for x in range(self.counts):



