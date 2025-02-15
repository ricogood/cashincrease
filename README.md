# cash capital increase stategy 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 读取并处理股票日K数据 ===
df_stock = pd.read_csv(r"D:\現金增資\newmerged_data_2016_2024.csv")
df_stock.columns = df_stock.columns.str.strip()
df_stock['CODE'] = df_stock['CODE'].astype(str).str.strip().str.zfill(4)
df_stock['DATE'] = pd.to_datetime(df_stock['DATE'], format='%Y-%m-%d')

# === 读取并处理现金增资数据 ===
df_increase = pd.read_excel(r"D:\現金增資\capital_increase_separated.xlsx")
df_increase['公司代碼'] = df_increase['STOCK_CODE'].astype(str).str.extract(r'^(\d+)').iloc[:, 0].str.zfill(4)
df_increase['股東會日'] = pd.to_datetime(df_increase['股東會日'])
df_increase['事件日'] = pd.to_datetime(df_increase['事件日'])

# 筛选股东会日在 2016-11-08 至 2024-11-09 之间的资料
df_filtered = df_increase[
    (df_increase['股東會日'] >= '2016-11-08') & (df_increase['股東會日'] <= '2024-11-09')
].copy()

# === 读取并处理 EPS 数据 ===
file_path = r"D:\現金增資\TEJ20241123010703\20241123010710.csv"
df_eps = pd.read_csv(file_path, encoding='utf-16', sep='\t')
df_eps = df_eps[['CODE', 'DATE', 'EPS']]
df_eps['CODE'] = df_eps['CODE'].astype(str).str.zfill(4)
df_eps['DATE'] = pd.to_datetime(df_eps['DATE'], format='%Y%m')

# 初始化结果列表
trade_results = []

# 手续费率和交易税率
commission_rate = 0.001425
transaction_tax_rate = 0.003
stop_loss_threshold = 0.4

# 遍历每一条记录，模拟交易
for index, row in df_filtered.iterrows():
    company_code = row['公司代碼']
    announcement_date = row['股東會日']
    event_date = row['事件日']
    planned_exit_date = event_date - pd.Timedelta(days=1)

    # 提取该公司的日K数据
    company_data = df_stock[df_stock['CODE'] == company_code].copy()
    if company_data.empty:
        continue

    # 设置日期索引并排序
    company_data.set_index('DATE', inplace=True)
    company_data.sort_index(inplace=True)

    # 确定买入日期
    tentative_buy_date = announcement_date + pd.Timedelta(days=3)
    trading_days = company_data.index[company_data.index >= tentative_buy_date]

    if not trading_days.empty:
        actual_buy_date = trading_days.min()
        buy_price = company_data.loc[actual_buy_date]['CLOSE']
    else:
        continue

    # === EPS 过滤器 ===
    eps_start_date = actual_buy_date - pd.DateOffset(years=3)
    eps_data = df_eps[
        (df_eps['CODE'] == company_code) &
        (df_eps['DATE'] >= eps_start_date) &
        (df_eps['DATE'] <= actual_buy_date)
    ]

    if eps_data.empty or (eps_data['EPS'] < 0).any():
        continue

    # 获取持有期间的交易日期
    holding_period = company_data.loc[actual_buy_date:planned_exit_date].index
    if holding_period.empty:
        continue

    stop_loss_triggered = False
    for date in holding_period:
        current_low = company_data.loc[date]['LOW']
        drawdown = (buy_price - current_low) / buy_price

        if drawdown >= stop_loss_threshold:
            stop_loss_triggered = True
            actual_sell_date = date
            sell_price = current_low
            break

    if not stop_loss_triggered:
        try:
            sell_price = company_data.loc[planned_exit_date]['CLOSE']
            actual_sell_date = planned_exit_date
        except KeyError:
            previous_trading_day = company_data[company_data.index < planned_exit_date].index.max()
            if pd.isna(previous_trading_day):
                continue
            sell_price = company_data.loc[previous_trading_day]['CLOSE']
            actual_sell_date = previous_trading_day

    # 计算交易费用和净利润
    investment_amount = 100000
    buy_commission = investment_amount * commission_rate
    total_buy_cost = investment_amount + buy_commission
    shares_bought = investment_amount / buy_price
    gross_sell_amount = shares_bought * sell_price
    sell_commission = gross_sell_amount * commission_rate
    transaction_tax = gross_sell_amount * transaction_tax_rate
    total_sell_costs = sell_commission + transaction_tax
    net_sell_amount = gross_sell_amount - total_sell_costs
    profit = net_sell_amount - total_buy_cost
    return_rate = profit / total_buy_cost

    trade_results.append({
        '公司代码': company_code,
        '买入日期': actual_buy_date.date(),
        '卖出日期': actual_sell_date.date(),
        '买入价': buy_price,
        '卖出价': sell_price,
        '持有天数': (actual_sell_date - actual_buy_date).days,
        '收益': profit,
        '收益率': return_rate
    })

# === 计算每日收益和持仓 ===
all_dates = pd.date_range(start=df_stock['DATE'].min(), end=df_stock['DATE'].max())
daily_pnl = pd.DataFrame(index=all_dates)
daily_pnl['daily_profit'] = 0.0
daily_pnl['holding_count'] = 0

for result in trade_results:
    sell_date = pd.Timestamp(result['卖出日期'])
    buy_date = pd.Timestamp(result['买入日期'])
    daily_pnl.loc[sell_date, 'daily_profit'] += result['收益']
    daily_pnl.loc[buy_date:sell_date, 'holding_count'] += 1

daily_pnl['cumulative_profit'] = daily_pnl['daily_profit'].cumsum()

# === 持仓统计 ===
max_holding = daily_pnl['holding_count'].max()
avg_holding = daily_pnl['holding_count'][daily_pnl['holding_count'] > 0].mean()

# === 绘制 Equity Curve ===
plt.figure(figsize=(12, 6))
plt.plot(daily_pnl.index, daily_pnl['cumulative_profit'], label='profit', marker='o')
plt.title('Equity Curve')
plt.xlabel('date')
plt.ylabel('cumulative profit')
plt.grid(True)
plt.legend()
plt.show()

# === 输出最终结果 ===
if trade_results:
    df_results = pd.DataFrame(trade_results)
    total_investment = len(df_results) * 100000

    # 计算总利润和平均利润
    total_profit = df_results['收益'].sum()
    average_profit_per_trade = total_profit / len(df_results)

    # 输出结果
    print(f"总投入本金：{total_investment:.2f} 元")
    print(f"总利润：{total_profit:.2f} 元")
    print(f"每笔交易平均利润：{average_profit_per_trade:.2f} 元")
    print(f"总交易笔数：{len(df_results)}")
    print(f"最大持仓档数：{max_holding}")
    print(f"平均持仓档数：{avg_holding:.2f}")

 # === 绘制收益率分布图（收益率小于0用绿色，大于0用红色）===
plt.figure(figsize=(10, 6))

# 分离收益率大于0和小于0的部分
returns_positive = df_results[df_results['收益率'] > 0]['收益率']
returns_negative = df_results[df_results['收益率'] <= 0]['收益率']

# 绘制分布图
sns.histplot(returns_positive, bins=20, kde=True, color='red', label='> 0')
sns.histplot(returns_negative, bins=20, kde=True, color='green', label='<= 0')

# 标注平均收益率
average_return = df_results['收益率'].mean()
plt.axvline(average_return, color='blue', linestyle='--', label=f'Average: {average_return:.2%}')

# 添加图例和标题
plt.title('Return Distribution')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()
print("没有符合条件的交易。")
