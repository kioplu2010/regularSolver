# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as sci
import scipy.optimize as sco
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
    def merge_interval(interval_list: List[Tuple[float, float]]):
        """
        此函数用来合并数据区间
        :param interval_list: 包含（down, up)的列表
        :return: (down, up)，若down大于up，返回（0,0）
        """
        down = max([x[0] for x in interval_list])
        up = min([y[0] for y in interval_list])
        if down <= up:
            return tuple(down, up)
        else:
            return tuple(0.0, 0.0)

    @staticmethod
    def get_random_weights(weights_interval: Dict[str, Tuple[float, float]], counts: int = 1000,
                           constraints: List = []) -> pd.DataFrame:
        """
        此函数的作用是返回符合约束条件的资产权重的随机数组，当前只考虑了对每个资产分别设定上下限的情况
        # TODO 考虑更加复杂的约束，用constraints传递，每次生成好的数据做检查，直至满足条件的数量达标为止
        # TODO 需要注意的是，单资产的上下限在生成随机数据时已经满足要求，因此此处使用的constraints与计算有效前沿时比要去除重复
        :param weights_interval:各类资产权重的上下限区间，不包含默认（0，1）的上下限区间
        :param counts:需要生成的随机数组的个数，默认1000
        :param constraints:除开资产上下限之外的复杂约束
        :return:返回生成好的资产权重随机数组，返回随机数组的个数不一定等于输入的要求个数
        """
        # 记录合格的返回值个数
        qualified_counts = 0

        # 保存合格的返回数据
        qualified_result = []

        # 资产类型数
        ntype = len(config.PortfolioSetting.ASSETS_NAME)

        # 加总各类资产权重的下限
        down_sum = 0.0

        # 检查各类资产权重的上下限区间是否处于（0,1）之间
        for limit in weights_interval.values():
            if limit[0] < 0 or limit[1] > 1:
                raise ValueError("传递了无效的参数值:weights_interval, 各类资产权重的上下限区间应该处于（0,1）之间")
            elif limit[0] > limit[1]:
                raise ValueError("传递了无效的参数值:weights_interval, 各类资产权重的下限应该小于等于上限")
            else:
                down_sum += limit[0]

        # 检查各类资产的下限加总是否小于1
        if down_sum > 1:
           raise ValueError("传递了无效的参数值:weights_interval, 各类资产权重的下限加总应该小于等于1")

        # 保存有额外上下限区间的资产下标
        is_extra_loc = []

        # 保存没有额外上下限区间的资产下标
        no_extra_loc = []

        # 检查各类资产是否有额外的上下限区间并保存结果
        for iloc in range(ntype):
            # 获取资产名称
            name = config.PortfolioSetting.ASSETS_NAME[iloc]

            # 检查并存放结果
            if name in weights_interval:
                is_extra_loc.append(iloc)
            else:
                no_extra_loc.append(iloc)

        # 建立随机数生成器
        random_gen = np.random.default_rng()

        # 生成资产权重的随机数组，考虑执行效率，最多执行需要的个数乘以10次，避免死循环
        # 执行次数达到上限后，有多少符合结果的返回多少
        for x in range(counts * 10):
            # 先生成资产权重的随机数组，每个元素均初始化为0
            weights = np.zeros(ntype)
            extra_weights = np.zeros(ntype)

            # 为保证随机性，每次将有额外上下限区间的资产下标随机排列一次
            shuffle_loc = is_extra_loc
            random_gen.shuffle(shuffle_loc)

            # 保存生成资产权重随机数的区间
            random_interval = (0, 1)

            # 用来保存已生成的随机数加总
            random_sum = 0.0

            # 用来保存已生成的随机数的下行加总
            random_down_sum = 0.0

            """
            为有额外上下限区间的资产生成权重，先生成（0,1）的随机数，假设为x，然后处理 y=（b-a) * x + a，得到 y∈（a,b)
            """
            for i in shuffle_loc:
                # 获取资产名称
                name = config.PortfolioSetting.ASSETS_NAME[i]

                # 根据资产名称获取上下限
                interval_i = weights_interval[name]

                # 合并区间，把剩余资产权重的上限和此类资产的上下限合并
                interval_merge = SaveFrontier.merge_interval([random_interval, interval_i])

                # 生成符合条件的随机数 y=（b-a) * x + a
                weight_i = np.random.random() * (interval_merge[1] - interval_merge[0]) + interval_merge[0]

                # 保存已生成的资产权重
                extra_weights[i] = weight_i

                # 加总已生成的资产权重
                random_sum += weight_i

                # 加总已生成的资产权重的下限
                random_down_sum += interval_merge[0]

                # 更新生成资产权重随机数的区间
                # 第一个区间保障下一个随机数据与之前已生成的随机数据加总小于等于1
                # 第二个区间保障下一个随机数据生成后不会让后续的随
                random_interval = SaveFrontier.merge_interval([(0, 1 - random_sum),
                                                               (0, 1 - down_sum + random_down_sum)])

            """
            为没有额外上下限区间的资产生成权重
            """
            # 没有额外上下限区间的资产种类
            no_extra_ntype = len(no_extra_loc)

            # 生成资产权重，都处于（0,1）之间
            no_extra_weights = random_gen.random(size=no_extra_ntype)

            # 加总初始权重
            initial_weights_sum = np.sum(no_extra_weights)

            # 转化数据，确保sum(no_extra) + sum(extra) = 1
            no_extra_weights = [weight / initial_weights_sum * (1 - random_sum) for weight in no_extra_weights]

            """
            将有额外上下限区间的资产和没有没有额外上下限区间的资产的权重合并
            """
            for xloc in is_extra_loc:
                weights[xloc] = extra_weight[xloc]

            for yloc in no_extra_loc:
                weights[yloc] = no_extra_weight[yloc]

            # 存储检查结果
            is_qualified = False

            # 检查生成的资产权重随机数组是否满足更加复杂的约束条件
            for constraint in constraints:
                # 获取约束函数
                check_func = constraints["func"]

                # 输入生成的资产权重随机数组，获取执行结果
                check_result = check_func(weights)

                # 检查执行结果是否满足约束条件
                if constraints['type'] == "ineq":
                    is_qualified = np.all(check_result > 0)
                elif constraints['type'] == "eq":
                    is_qualified = np.all(check_result == 0)
                else:
                    is_qualified = True

            # 若检查结果符合约束条件，将生成的资产权重随机数组加入返回值
            if is_qualified:
                qualified_counts += 1
                qualified_result.append(weights.tolist())

            # 若符合要求条件的随机数组已达到要求个数，结束循环
            if qualified_counts == counts:
                break

        # 将结果数据放入dataFrame
        df_result = pd.DataFrame(data=qualified_result, columns=config.PortfolioSetting.ASSETS_NAME)

        return df_result


    @staticmethod
    def tangent_equations(paras: List, tck: tuple,  risk_free_return=config.BenchmarkSetting.RISK_FREE_RATE):
        """
        此函数的作用是作为fsolve优化函数的参数，优化目标是调整paras，使得返回值都等于0，要求返回参数个数与传入参数个数相等
        :param paras:第一个参数为无风险收益率R_f，一般来说无风险收益率为预先设定值，所以其实第一个参数可以省略，同步去掉第一个返回值即可
                     第二个参数为斜率，即是 R_m/σ_m，假设为b
                     第三个参数为波动率，即是σ_m，是函数的参数
                     由传入的三个参数可以得到直线函数Y(x) = bx + R_f
        :param tck: 根据收益率序列与波动率序列得到的有效前沿光滑曲线的参数，假设曲线代表的函数为Z(X)
        :param risk_free_return:无风险收益率
        :return:
        """
        # 约束条件1，无风险利率等于传入值
        eq1 = risk_free_return - paras[0]

        # 约束条件2，Y(x) = Z(x)，约束切点必须在有效前沿上，对于相同的x，有相同的y值，即直线为连接点（x, Z(x))与点(0, R_f)
        # splev函数的第一参数即为x，tck为曲线参数，der表示求导数，例如der=0代表返回原函数的值，der=0代表返回一阶导数的值
        eq2 = risk_free_return + paras[1] * paras[2] - sci.splev(paras[2], tck, der=0)

        # 约束条件3，dY(x) = dZ(x),由于Y(x)是直线，dY(x)等于b
        # dY(x)为直线的斜率，dZ(x)为一阶导数，代表曲线切线的斜率，当两者相等时直线与曲线相切
        eq3 = paras[1] - sci.splev(paras[2], tck, der=1)

        return eq1, eq2, eq3

    @staticmethod
    def save_data_to_local(portfolios: List[Portfolio]):
        """
        此函数用于保存所有数据至excel，单独保存图片
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

        """
        以下模块的作用是生成画布, 在画布上生成图像
        """
        # 生成画布
        plt.figure(config.PlotSetting.FIGURE_SIZE)
        """
        # 资本市场线（Capital Market Line）
        # 相关假设：可以按无风险利率借入或借出现金，现金无波动率，所以借入或借出资金后的组合收益率与组合波动率如下
        # 公式: E(R_p) = Q * R_m + (1-Q) * R_f  σ_p = Q * σ_m 其中Q是新组合中风险资产的占比
        # 公式: E(R_p) = R_f + [E(R_m) - R_f] * [σ_p / σ_m]
        # 生成资本市场线，本质上是求有效前沿上的点（x, y)与点(0, R_f)，此两点相连恰好与曲线相切
        # （此判断不对，待验证）又因为sharp = (R_p - R_f) / σ_m，在无风险收益率确定的前提下，此问题等同于求组合夏普比率最大的点
        # 所以如果不考虑展示图的平滑效果，可以直接用传入参数系列投资组合中夏普比率最大的点和（0，R_f)点相连得到CML
        # 而且从实用角度看，平滑过的有效前沿只有组合预期收益率和组合波动率的数据，并没有各类资产的权重，因此无实际应用的意义
        """
        # 考虑CML对于实际的投资组合选择的作用本身也极为有限，此处为追求展示效果，仍然生成光滑的曲线
        # 先确定资本市场线的范围，基本上波动率（x轴）能够覆盖有效前沿，略微超出一些就行
        # 获取有效前沿上的序列，组合收益率、组合波动率、夏普比率
        series_return = df_efficient_frontier[config.ExcelFileSetting.LABEL_EXPECTED_RETURN]
        series_vol = df_efficient_frontier[config.ExcelFileSetting.LABEL_EXPECTED_VOLATILITY]
        series_sharp = df_efficient_frontier[config.ExcelFileSetting.LABEL_SHARP_RATIO]

        # 一般来说最小波动率应该是第一个组合，考虑部分目标函数使得优化器计算的有效前沿有一段下弯，以下操作可用来排除下弯数据
        # 获取最小组合波动率的index
        index_min = np.argmin(series_vol)
        up_series_return = series_return.iloc[index_min:]
        up_series_vol = series_vol.iloc[index_min:]
        up_series_sharp = series_sharp.iloc[index_min:]

        # 获取收益率与波动率的光滑曲线
        tck = sci.splrep(up_series_vol, up_series_return)

        # 假设（x,y)是曲线上的点，获取从点（0,R_f)出发的曲线切线
        initial_x = (np.min(series_vol) + np.max(series_vol)) / 2
        initial_paras = [config.BenchmarkSetting.RISK_FREE_RATE, 0.5, initial_x]
        tangent_line_x = sco.fsolve(SaveFrontier.tangent_equations, initial_paras,
                                    args=(tck, config.BenchmarkSetting.RISK_FREE_RATE))

        """
        # 画散点图
        """
        # 先生成符合约束条件的资产权重随机数组

        # plt.scatter(portfolio_volatilities, portfolio_returns,
        #            c=(portfolio_returns - 0.02) / portfolio_volatilities, marker='o')

        # 获取最大的组合波动率
        max_vol = series_vol.max()

        # 预设CML的x轴从0至最大组合波动率
        cml_x = np.linspace(0.0, max_vol)














