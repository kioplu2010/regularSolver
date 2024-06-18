# -*- coding: utf-8 -*-
import os
from typing import Final
from dataclasses import dataclass
from enum import Enum

"""
本文件主要用来存放资产配置所需要的参数
# TODO 后续调整应放入配置文件或者数据库
"""

@dataclass
class History:
# TODO 定义计算市场基准所采用历史数据的时间区间，后续应对各类市场基准设置单独的时间区间，此处先简化处理
START_DATE = '2014-06-30'
END_DATE = '2024-05-31'
TIME_INTERVAL = ('2014-06-30', '2024-05-31')

class DataOriginalSrc(Enum):
    """
    定义投资数据的初始来源，例如：wind,bloomberg,other
    """
    WIND = 'wind'
    BLOOMBERG = 'bloomberg'
    OTHER = 'other'

class MarketDataPara:
    """
    定义计算市场指数所需要的参数
    """
    def __init__(self, name: str, security_id: str, data_src: DataOriginalSrc = DataOriginalSrc.WIND,
                 weight: str = 1, start_date: str = START_DATE, end_date: str = END_DATE):
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
class ExcelFileSetting:
    """
    定义excel源数据文件的配置参数
    """
    # 源数据存放地址
    SOURCE_PATH = "%s/data/SAAData.xlsx" % (os.getcwd())

    # 各类数据存放的sheet名
    DATA_LIST = ["HistoryYield", "HistoryIndex"]

    # 各类数据的日期均放在首列
    INDEX_COLUMN: Final[int] = 0

    # TODO 市场基准的历史数据的初始来源、证券ID和初始文件中列名应对应起来，待后续再完善
    # 各类资产的历史数据列名与证券代码
    # 国债采用中证10年期中国国债收益率
    GOV_BOND_COLUMN = "China_gov_bond_yield"
    GOV_BOND_SECURITY_ID = ''
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





if __name__ == '__main__':
    pass