# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from regularSolver.AssetAllocation import Portfolio
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from typing import Final, Dict, Tuple, List
from regularSolver import SAAconfig as config


class SaveFrontier:
    """
    用来保存计算好的有效前沿
    """

    @staticmethod
    def save_all_to_excel(portfolios: List[Portfolio]):
        """
        此函数用于保存所有数据和图像至excel
        数据保存项：
        1.各类资产的名称、预期收益率、预期波动率、偏度、超值峰度、基准名称
        2.相关系数矩阵
        3.有效前沿上的投资组合各类资产权重、组合预期收益率、组合预期波动率、组合夏普比率
        图像保存项：散点图、有效前沿和资本市场线生成图片
        :param portfolios:已计算好的有效前沿上的资产组合列表
        :return:保存到当地文件，无需输出
        """
        # 各类资产参数，由于所有投资组合的参数均为一致，所以取第一个投资组合即可
        assets = portfolios[0].assets

        # 获取资产名称列表
        asset_names = [asset.name for asset in assets]

        # 获取各类资产的参数,包括预期收益率、预期波动率、偏度、超值峰度、基准名称
        asset_data = [[asset.expected_return, asset.expected_volatility, asset.annul_skewness,
                      asset.annul_excess_kurtosis, asset.benchmark.name] for asset in assets]

        # 获取资产属性的标签名称
        asset_para_names = [config.ExcelFileSetting.LABEL_EXPECTED_RETURN,
                            config.ExcelFileSetting.LABEL_EXPECTED_VOLATILITY,
                            config.ExcelFileSetting.LABEL_SKEWNESS,
                            config.ExcelFileSetting.LABEL_EXCESS_KURTOSIS,
                            config.ExcelFileSetting.LABEL_BENCHMARK_NAME]

        # 建立DataFrame存储存放各类资产参数
        df_assets_paras = pd.DataFrame(data=asset_data, index=asset_names, columns=asset_para_names)

        # 存放相关系数矩阵
        df_correlations = portfolios[0].correlations

        # 存放有效前沿，包括组合名称、各类资产权重、组合预期收益率、组合预期波动率、组合夏普比率
        # 获取投资组合名称列表
        portfolio_names = [portfolio.name for portfolio in portfolios]

        # 获取各投资组合的资产权重、组合预期收益率、组合预期波动率、组合夏普比率
        portfolio_data = list(map(lambda portfolio: portfolio.weights + [portfolio.get_portfolio_return(),
                                                                         portfolio.get_portfolio_volatility(),
                                                                         portfolio.get_portfolio_sharpe()], portfolios))

        # 按投资组合数据的顺序建好列名
        portfolio_data_names = asset_names + [config.ExcelFileSetting.LABEL_EXPECTED_RETURN,
                                             config.ExcelFileSetting.LABEL_EXPECTED_VOLATILITY,
                                             config.ExcelFileSetting.LABEL_SHARP_RATIO]

        # 建立DataFrame存储存放投资组合数据
        df_efficient_frontier = pd.DataFrame(data=portfolio_data, index=portfolio_names, columns=portfolio_data_names)

        # 存放到excel
        with pd.ExcelWriter(config.ExcelFileSetting.OUTPUT_PATH) as writer:
            # 将资产参数写入excel
            df_assets_paras.to_excel(writer, sheet_name=config.ExcelFileSetting.LABEL_ASSETS_PARAS,
                                     index_label=config.ExcelFileSetting.LABEL_ASSETS_NAME)

            # 将相关系数矩阵写入excel
            df_correlations.to_excel(writer, sheet_name=config.ExcelFileSetting.LABEL_CORRELATIONS,
                                     index_label=config.ExcelFileSetting.LABEL_CORRELATIONS)

            # 将有效前沿写入excel
            df_efficient_frontier.to_excel(writer, sheet_name=config.ExcelFileSetting.LABEL_EFFICIENT_FRONTIER,
                                           index_label=config.ExcelFileSetting.LABEL_PORTFOLIO_NAME)






