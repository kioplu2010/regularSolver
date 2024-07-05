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

    pass


