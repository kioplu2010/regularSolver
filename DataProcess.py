# -*- coding: utf-8 -*-
import pandas as pd
from scipy import stats
from . import SAAconfig as config


def read_data_from_execl(file_path: str, sheet_list: list, index_column: int = 0):
    """
    从excel读取数据
    :param file_path:
    :param sheet_list:
    :param index_column:
    :return: dict of DataFrame
    """
    src_dfs = pd.read_excel(file_path, sheet_list, index_col=index_column)
    return src_dfs


def data_clean(df: pd.DataFrame):
    """
    清洗读入的数据清洗，清除日期标签之前的无关数据，为便于以后使用，将数据按日期调整为正序
    # TODO 后续此函数应根据数据类型进行数据清洗，加入对空值或异常数据的处理
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    date_row_num = 0

    row_labels = df.index.array
    # print("row_labels %s " % row_labels)

    for x in range(len(row_labels)):
        # if row_labels[x] == DATE_INDEX:
        if str(row_labels[x]).strip().capitalize() == config.ExcelFileSetting.DATE_INDEX:
            date_row_num = x
            break
            # print("date_row_num : %s " % date_row_num)

    # 以DATE_INDEX为标志去掉前面的无关数据
    header = df.iloc[date_row_num]
    df_cleaned = df[date_row_num + 1:]

    # 将index转换为DatetimeIndex
    date_index = pd.DatetimeIndex(data=df_cleaned.index.array, freq='infer')
    # print("date_index %s freq %s" % (date_index, date_index.freq))
    df_cleaned.index = date_index
    # 更新列名为包含DATE_INDEX的行
    df_cleaned.columns = header

    # 为便于使用，统一调整为按日期正序排列
    df_sorted = df_cleaned.sort_index(ascending=True)

    return df_sorted


def data_prepare(src_dfs: dict):
    """
    准备源数据,主要是对数据作清洗便于后续使用
    :param src_dfs: dict of DataFrames
    :return:
    """
    prepared_dfs = {}
    for sheet_name, df in src_dfs.items():
        prepared_dfs[sheet_name] = data_clean(df)
    return prepared_dfs


if __name__ == '__main__':
    dfs = read_data_from_execl(config.ExcelFileSetting.SOURCE_PATH, config.ExcelFileSetting.DATA_LIST,
                               config.ExcelFileSetting.INDEX_COLUMN)
    dfs_prepared = data_prepare(dfs)

    # 获取历史到期收益率数据
    df_yield = dfs_prepared[config.ExcelFileSetting.DATA_LIST[0]]

    # print("df_yield.index %s " % df_yield.index)
    # print("df_yield.columns %s " % df_yield.columns)

    gov_bond_yield_series = df_yield[config.ExcelFileSetting.GOV_BOND_COLUMN]

    start_date = '2014-06-30'
    end_date = '2024-05-31'

    history_yield = gov_bond_yield_series[start_date:end_date] / 100
    annul_return = history_yield.mean()
    annul_volatility = history_yield.std()
    skewness = stats.skew(history_yield.astype(float))
    excess_kurtosis = stats.kurtosis(history_yield.astype(float))

    # print("history_yield annul_return %s " % annul_return)
    # print("history_yield annul_volatility %s " % annul_volatility)
    # print("history_yield skewness %s " % skewness)
    # print("history_yield excess_kurtosis %s " % excess_kurtosis)

    # print("gov_bond_yield_series.index %s " % gov_bond_yield_series.index)

    # print("gov_bond_yield_series.array %s " % gov_bond_yield_series.array)

    # print("gov_bond_yield_series to period() %s " % gov_bond_yield_series.to_period(freq='M'))

    # 获取市场指数历史价格数据
    df_price = dfs_prepared[config.ExcelFileSetting.DATA_LIST[1]].pct_change()
    print(df_price.head())


"""
    for sheet_name, df in dfs_prepared.items():
        print(f"Sheet Name: {sheet_name}")
        print("df index: %s " % df.index)
        print("df columns: %s " % df.columns)
        print(df.head())
"""
