import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
from numpy import random


def get_share_data_to_excel(token,ts_code,start_time,end_time,excel_path):
    """
        获取股票数据，并存在excel中
        参数 :
        token : tushare 接口token
        ts_code : 股票代码
        start_time : 股票开始日期
        end_time : 股票截止日期
        excel_path : excel路径名
    """
    tsapi_get = ts.pro_api(token)
    df = tsapi_get.daily(ts_code=ts_code, start_date=start_time, end_date=end_time)
    df.to_csv(excel_path)  # 保存数据

def get_share_data_to_return(token,ts_code,start_time,end_time):
    """
        获取股票数据，并返回
        参数 :
        token : tushare 接口token
        ts_code : 股票代码
        start_time : 股票开始日期
        end_time : 股票截止日期
    """
    tsapi_get = ts.pro_api(token)
    df = tsapi_get.daily(ts_code=ts_code, start_date=start_time, end_date=end_time)
    return df


def normality_test(array):
    """
        检验显著性的函数
    """
    print('进行股票数据正态分布检验: ')
    print('标准试验p值为%.3f，表示不为正态分布是显著的，就是在误差允许范围内可以认为该收益率数据服从正态分布' % scs.normaltest(array)[1])

def test_normality(excel_path):
    """
        检验正态分布
        参数 :
        excel_path : 股票数据excel路径名
    """
    data = pd.read_csv(excel_path)[::-1] # 要求时间向后递增，原数据是向上递增的
    df0 = data['pct_chg'] # 获得收益率字段数据
    log_data = np.array(np.log(df0 + 11).dropna()) # 注意，这里使用的是对数收益率
    normality_test(log_data) # 检验正态分布
    return data


def share_path(data,date,times,pe):
    """
        蒙特卡罗模拟股票数据
        参数：
        data : 股票数据
        excel_path : 股票数据excel路径名
        date : 模拟期间的交易日期
        times : 模拟次数
        pe : 所获得的最后一天的股价
        返回：
        pt : 全部模拟数据列表(包含日期）
        pt1 : 全部模拟数据列表(不包含日期,只有股票信息）
    """
    sigma = (data['pct_chg'] / 100).std()  # 波动率
    n = 243  # 历史价格时间长度
    dt = 1 / n  # 单位时间
    sigs = sigma * np.sqrt(dt) + 0.005  # 漂移项
    mu = (data['pct_chg'] / 100).mean()  # 期望收益率
    drt = mu * dt  # 扰动项

    pt = []  # 全部模拟数据列表(包含日期）
    pt1 = []  # 全部模拟数据列表(不包含日期）
    # 蒙特卡洛模拟
    for i in range(times):  # 控制次数
        pn = pe  # 初始化股价
        p = []  # 单次模拟情况，包含日期
        p1 = []  # 单次模拟情况，不包含日期
        #p.append(pe) # 计入初始股价
        for days in range(0, 243):  # 控制天数，交易日期是243天
            pn = pn + pn * (random.normal(drt, sigs))  # 产生新股价
            if pn < 0.1:  # 确保股价大于等于一毛钱，如果低于一毛钱就令等于前一天价格
                pn = p[-1]
            pnlist = [date[242 - days]]
            pnlist.append(pn)  # 加上对应的日期
            p1.append(pn)
            p.append(pnlist)

        pt.append(p)
        pt1.append(p1)
    return pt,pt1


def Snowball_compute(getin_rate,getout_rate,ann_rate,out_date_list,pt,pe):
    """
        计算模拟出的每支股票路径的雪球收益
        参数：
        getin_rate : 敲入
        getout_rate : 敲出
        ann_rate : 年化收益
        out_date_list : 每月的观察日
        pt : 带时间的股票数据
        pe : 所获得的最后一天的股价
        返回：
        Returns : 每支股票路径的收益
    """
    Returns = []  # 存收益率的列表
    for share in pt:
        # print(share[-1][1])
        oflag = 0
        iflag = 0
        # 先判断是否敲出
        for date in out_date_list:
            # 遍历每个月15号
            mflag = 0
            while mflag == 0:
                for day in share:
                    if date == day[0]:  # 该月15号是交易日期
                        mflag = 1
                        if day[1] > pe * getout_rate: # 发生敲出
                            # print(day)
                            oflag = 1
                            if int(day[0][4:6])<5 : Returns.append(ann_rate * (int(day[0][4:6]) - 5 + 1 + 12) / 12)  # 收益率
                            else : Returns.append(ann_rate * (int(day[0][4:6]) - 5 + 1) / 12)  # 收益率
                            break
                if mflag == 0: # 该月15号不是交易日期
                    date = str(int(date)+1)  # 日期向后延
                    # print(date)
            if oflag == 1 : break  # 已敲出的就不需要再进行其他计算
        if oflag == 1: continue  # 已敲出的就不需要再进行其他计算,开始计算下一次模拟

        # 判断是否发生敲入
        for day in share:
            if day[1]< pe*getin_rate:  # 发生敲入
                iflag = 1
                break
        if iflag == 1: # 若发生敲入
            if share[-1][1] > pe :  # 发生了敲入，但是到期标的上涨了
                Returns.append(0)
            else:  # 计算到期下跌了多少
                Returns.append((share[-1][1]-pe) / pe)
        else:  # 没有发生敲入，收益就是年化收益
            Returns.append(ann_rate)

    return Returns

def average_returns(Returns,times):
    """
        计算平均收益率
        参数：
        Returns : 每支股票路径的收益
        times : 蒙特卡洛模拟的股票路径数
        返回：
        average : 期权价格
    """
    average = 0
    for i in Returns:
        average += i
    average /= times
    average *= 100
    return average


def Get_Option_price(Returns,times,pe,r,T,getout_rate):
    """
        按看跌期权计算价格
        参数：
        Returns : 每支股票路径的收益
        times : 蒙特卡洛模拟的股票路径数
        pe : 所获得的最后一天的股价
        r : 无风险利率
        T : 行权时间
        getout_rate : 敲出
        返回：
        sums : 期权价格
    """
    sums = 0
    for i in Returns:
        p = max(0,  pe * getout_rate - (i+1)*pe ) * (np.exp(-r * T)) / times
        sums += p
    return sums



def visualization(pt1, pe, getout_rate, getin_rate):
    """
        蒙特卡洛模拟的股票数据可视化
        参数：
        pt1 : 股票数据
        pe : 所获得的最后一天的股价
        getout_rate : 敲出
        getin_rate : 敲入
    """
    pt1 = pd.DataFrame(pt1).T  # 全部模拟数据
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.plot(pt1)
    plt.axhline(y=pe*getout_rate, linestyle='--', label='敲出线', color = "red")
    plt.axhline(y=pe*getin_rate, linestyle='--', label='敲入线',color = "green")
    plt.legend(loc='upper left')
    plt.title('茅台-蒙特卡罗模拟-股价模拟')
    plt.xlabel('时间/天')
    plt.ylabel('价格/元')
    plt.show()


if __name__=="__main__":
    '''
    全局参数设置：
    '''
    times = 10000  # 蒙特卡洛模拟次数
    getin_rate = 0.8  # 敲入
    getout_rate = 1.03  # 敲出
    ann_rate = 0.2  # 年化收益率
    r = 0.038  # 无风险利率
    T = 1  # 行权时间 1年
    print('雪球产品初始参数设置: ')
    print('蒙特卡洛模拟次数: %d '%times)
    print('敲入: %.2f '%getin_rate)
    print('敲出: %.2f '%getout_rate)
    print('年化收益率: %.2f ' %ann_rate)
    print('行权时间: %d ' %T)
    print('敲出观察日: 每月15号')
    # 观察日期列表
    out_date_list = ['20210615', '20210715', '20210815', '20210915', '20211015', '20211115',
                     '20211215', '20220115', '20220215', '20220315', '20220415', '20220515']

    '''
    开始运行：
    '''
    token = "d318ea464366dea908dbea5e23e4e8d17fc31942a44a251ac5d05938"  # tushare接口token
    excel_path ='D:/SUFE_xjx/专业课/金融建模/茅台股票信息.csv'  # 文件保存地址
    #get_share_data_to_excel(token, '600519.SH', '20210601', '20220601', excel_path)  # 获取数据,保存在excel中
    data = test_normality(excel_path)  # 股票数据正态分布检验
    pe = data['close'].iloc[0]  # 最后一天股价
    df_date = get_share_data_to_return(token, '600519.SH', '20210601', '20220601')  # 获取交易数据
    date = df_date['trade_date']  # 获取交易日期
    pt, pt1 = share_path(data, date, times, pe)  # 蒙特卡罗模拟股票数据
    returns = Snowball_compute(getin_rate,getout_rate,ann_rate,out_date_list,pt,pe)  # 获取雪球产品收益率
    sums = Get_Option_price(returns,times,pe,r,T,getout_rate)  # 按看跌期权计算价格
    print("股票历史回测波动率为 %.2f"%(100*(data['pct_chg'] / 100).std())+"%")  # 波动率
    print("蒙特卡洛模拟结果如下：" )
    print("按欧式看跌期权计算的期权价格为 %.2f"%sums)
    average = average_returns(returns,times)  # 计算平均收益率
    print("雪球产品的预期收益率是%.2f"%average+'%')
    visualization(pt1, pe, getout_rate, getin_rate)  # 股票数据可视化

