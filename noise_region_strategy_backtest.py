#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import os
import time as time_module

# 策略配置参数
CONFIG = {
    "data_file": None,  # 设置为None会自动查找最新的BTC数据文件
    "source": "okx",
    "lookback_days": 3,  # 用于计算波动率的历史数据天数
    "k1": 1.5,  # 上边界乘数
    "k2": 1.5,  # 下边界乘数
    "trade_start_time": time(16, 0),  # 比特币24小时交易，可以自定义交易时段
    "trade_end_time": time(23, 50),
    "max_trades_per_day": 3,  # 每日最大交易次数
    "commission_rate": 0.0005,  # 手续费率（0.05%）
    "slippage": 0.0003,  # 滑点（0.03%）
    "start_date": "2025-05-10",  # 只处理从这个日期开始的数据
    "initial_balance": 100000,  # 初始资金金额
    "check_interval": 10  # 检查间隔（分钟），同时应用于入场和止损检查
}

class NoiseRegionStrategyFixed:
    def __init__(self, data, lookback_days=2, k1=1.2, k2=1.2, 
                 trade_start_time=time(9, 30), trade_end_time=time(15, 30),
                 max_trades_per_day=3, commission_rate=0.0005, slippage=0.0003,
                 initial_balance=100000, check_interval=10):
        """
        噪声区域交易策略初始化 - 固定时间间隔检查版本
        
        参数:
        data: DataFrame - 包含时间戳、开盘价、最高价、最低价、收盘价的数据
        lookback_days: int - 用于计算波动率的历史数据天数
        k1: float - 上边界乘数，调整上边界sigma的影响力
        k2: float - 下边界乘数，调整下边界sigma的影响力
        trade_start_time: time - 交易开始时间
        trade_end_time: time - 交易结束时间
        max_trades_per_day: int - 每日最大交易次数限制
        commission_rate: float - 交易手续费率（单边）
        slippage: float - 滑点率
        initial_balance: float - 初始资金金额
        check_interval: int - 检查间隔（分钟），同时应用于入场和止损检查
        """
        # 数据预处理
        self.data = data.copy()
        self.data['date'] = self.data['timestamp'].dt.date
        self.data['time'] = self.data['timestamp'].dt.time
        
        # 策略参数
        self.lookback_days = lookback_days
        self.k1 = k1
        self.k2 = k2
        self.trade_start_time = trade_start_time
        self.trade_end_time = trade_end_time
        self.max_trades_per_day = max_trades_per_day
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.check_interval = check_interval
        
        # 资金管理
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.balance_history = []
        
        # 交易记录
        self.trades = []
        
        # 计算结果存储
        self.results = None
        self.performance = None
        self.bounds = None
        
        # 添加标志，标记数据点是否为检查点
        self._mark_check_points()
        
    def _mark_check_points(self):
        """标记所有的检查点（符合时间间隔的点）"""
        self.data['is_check_point'] = False
        
        # 按日期分组处理
        for date, group in self.data.groupby('date'):
            # 计算每个时间点的总分钟数
            minutes_times = [(t.hour * 60 + t.minute) for t in group['time']]
            
            # 找出所有符合检查间隔的时间点
            check_indices = [i for i, m in enumerate(minutes_times) if m % self.check_interval == 0]
            
            # 标记这些检查点
            if check_indices:
                date_indices = group.index[check_indices]
                self.data.loc[date_indices, 'is_check_point'] = True
        
        print(f"已标记 {self.data['is_check_point'].sum()} 个检查点，总数据点数 {len(self.data)}")
        
    def calculate_bounds(self):
        """计算每个时间点的上下边界"""
        # 为每一行数据添加上下边界
        bounds_data = []
        
        # 获取所有唯一日期
        unique_dates = sorted(self.data['date'].unique())
        total_dates = len(unique_dates) - self.lookback_days
        
        print(f"开始计算边界，总计 {total_dates} 个交易日...")
        
        # 对每个交易日计算边界
        for i, current_date in enumerate(unique_dates):
            if i < self.lookback_days:
                continue
                
            if (i - self.lookback_days) % 10 == 0 or (i - self.lookback_days) == total_dates - 1:
                progress = (i - self.lookback_days + 1) / total_dates * 100
                print(f"处理日期: {current_date} ({progress:.1f}%)")
                
            # 获取当日数据
            current_day_data = self.data[self.data['date'] == current_date].copy()
            
            # 获取当日开盘价
            day_open = current_day_data.iloc[0]['open']
            
            # 获取前一日收盘价
            prev_date = unique_dates[i-1]
            prev_day_data = self.data[self.data['date'] == prev_date]
            prev_close = prev_day_data.iloc[-1]['close']
            
            # 确定上下边界参考价
            upper_ref = max(day_open, prev_close)
            lower_ref = min(day_open, prev_close)
            
            # 预先计算该日不同时间点的sigma，避免重复计算
            unique_times = current_day_data['time'].unique()
            time_count = len(unique_times)
            print(f"当日 {current_date} 有 {time_count} 个不同时间点需要计算sigma")
            
            start_overall = time_module.time()
            batch_size = 100  # 批量处理大小
            time_sigma = {}
            
            # 批量处理时间点，每批次打印一次进度
            for batch_idx in range(0, time_count, batch_size):
                batch_end = min(batch_idx + batch_size, time_count)
                batch_times = unique_times[batch_idx:batch_end]
                
                # 批量计算这些时间点的sigma
                for time_val in batch_times:
                    time_sigma[time_val] = self._calculate_sigma(unique_dates[:i], time_val, day_open)
                
                # 打印进度
                elapsed = time_module.time() - start_overall
                print(f"已处理 {batch_end}/{time_count} 个时间点 (耗时: {elapsed:.2f}秒)")
                
                # 每批次强制执行垃圾回收，减少内存占用
                import gc
                gc.collect()
            
            # 为当日每个时间点计算边界
            for _, row in current_day_data.iterrows():
                current_time = row['time']
                
                # 从预计算的值获取sigma
                sigma = time_sigma[current_time]
                
                # 计算上下边界
                upper_bound = upper_ref * (1 + self.k1 * sigma)
                lower_bound = lower_ref * (1 - self.k2 * sigma)
                
                # 添加到结果中
                bounds_row = row.copy()
                bounds_row['upper_bound'] = upper_bound
                bounds_row['lower_bound'] = lower_bound
                bounds_row['sigma'] = sigma
                bounds_row['day_open'] = day_open
                bounds_row['prev_close'] = prev_close
                bounds_row['upper_ref'] = upper_ref
                bounds_row['lower_ref'] = lower_ref
                
                bounds_data.append(bounds_row)
        
        # 转换为DataFrame
        print("边界计算完成，正在处理数据...")
        self.bounds = pd.DataFrame(bounds_data)
        return self.bounds
    
    def _calculate_sigma(self, historical_dates, current_time, current_day_open):
        """计算特定时间点的历史波动率sigma"""
        # 使用缓存来提高性能
        cache_key = f"{hash(tuple(historical_dates))}-{current_time}"
        if not hasattr(self, '_sigma_cache'):
            self._sigma_cache = {}
        if cache_key in self._sigma_cache:
            return self._sigma_cache[cache_key]
            
        start_time = time_module.time()
        
        historical_changes = []
        
        # 使用最近的lookback_days天数据
        recent_dates = historical_dates[-self.lookback_days:]
        
        # 使用向量化操作优化
        filter_mask = (self.data['date'].isin(recent_dates)) & (self.data['time'] == current_time)
        filtered_data = self.data[filter_mask]
        
        # 打印调试信息（仅打印一次）
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"调试: 历史日期数: {len(recent_dates)}, 过滤后数据量: {len(filtered_data)}")
            print(f"调试: 历史日期: {recent_dates}")
            print(f"调试: 当前时间点: {current_time}")
            
        # 创建开盘价缓存
        if not hasattr(self, '_open_price_cache'):
            self._open_price_cache = {}
            
        # 按日期分组处理
        for hist_date, group in filtered_data.groupby('date'):
            # 从开盘价缓存获取开盘价，避免每次重新筛选数据
            if hist_date in self._open_price_cache:
                hist_day_open = self._open_price_cache[hist_date]
            else:
                # 获取当日开盘数据
                day_data = self.data[self.data['date'] == hist_date]
                
                if len(day_data) == 0:
                    continue
                    
                hist_day_open = day_data.iloc[0]['open']
                self._open_price_cache[hist_date] = hist_day_open
            
            if len(group) > 0:
                # 计算相对于开盘价的绝对变动率
                close_price = group.iloc[0]['close']
                change_rate = abs(close_price / hist_day_open - 1)
                historical_changes.append(change_rate)
        
        # 计算平均变动率作为sigma
        result = np.mean(historical_changes) if historical_changes else 0.0
        
        # 保存到缓存
        self._sigma_cache[cache_key] = result
        
        # 计算耗时
        end_time = time_module.time()
        elapsed_time = end_time - start_time
        
        # 如果单次计算时间超过0.1秒，则可能是性能瓶颈点
        if elapsed_time > 0.1 and not hasattr(self, '_time_warning_printed'):
            self._time_warning_printed = True
            print(f"警告: sigma计算耗时 {elapsed_time:.2f} 秒")
            
        return result
        
    def run_backtest(self):
        """运行回测 - 只在固定时间间隔检查点进行决策"""
        if self.bounds is None:
            self.calculate_bounds()
        
        print("开始回测过程...")
        
        # 初始化回测结果
        backtest_results = self.bounds.copy()
        backtest_results['position'] = 0  # 0: 空仓, 1: 多头, -1: 空头
        backtest_results['signal'] = 0    # 0: 无信号, 1: 做多, -1: 做空, 2: 平多, -2: 平空
        backtest_results['stop_loss'] = np.nan
        backtest_results['trades_today'] = 0
        backtest_results['balance'] = self.initial_balance  # 添加资金余额列
        
        position = 0  # 当前持仓状态
        entry_price = 0  # 入场价格
        stop_loss = np.nan  # 止损价格
        trades_today = 0  # 当日交易次数
        current_date = None
        pnl = 0  # 当前交易盈亏
        trade_count = 0  # 总交易次数计数
        position_size = 0  # 持仓数量
        last_check_time = None  # 上次检查时间
        
        # 添加is_check_point列到回测结果
        backtest_results['is_check_point'] = False
        for index, row in backtest_results.iterrows():
            # 计算当前分钟数
            current_time = row['time']
            current_minutes = current_time.hour * 60 + current_time.minute
            
            # 检查是否是检查点
            if current_minutes % self.check_interval == 0:
                backtest_results.at[index, 'is_check_point'] = True
        
        # 打印检查点数量
        check_points_count = backtest_results['is_check_point'].sum()
        print(f"回测中共有 {check_points_count} 个检查点，总数据点 {len(backtest_results)}")
        
        # 逐行处理数据
        for i in range(len(backtest_results)):
            row = backtest_results.iloc[i]
            
            # 检查是否是新的交易日
            if current_date != row['date']:
                current_date = row['date']
                trades_today = 0
                last_check_time = None  # 新交易日重置检查时间
                
            backtest_results.at[i, 'trades_today'] = trades_today
            
            # 检查是否在交易时段内
            current_time = row['time']
            in_trading_hours = (self.trade_start_time <= current_time <= self.trade_end_time)
            
            # 更新止损价格 (这里可以每分钟都更新，因为只是计算不是执行)
            if not np.isnan(stop_loss):
                if position == 1:  # 多头追踪止损
                    stop_loss = max(stop_loss, row['upper_bound'])
                elif position == -1:  # 空头追踪止损
                    stop_loss = min(stop_loss, row['lower_bound'])
                backtest_results.at[i, 'stop_loss'] = stop_loss
            
            # 创建当前时间的datetime对象
            current_datetime = datetime.combine(row['date'], current_time)
            
            # 确定是否是检查点
            is_check_point = row['is_check_point']
            
            # 平仓条件检查（只在检查点进行）
            if position != 0 and is_check_point and in_trading_hours:
                # 止损触发检查
                if (position == 1 and row['close'] < stop_loss) or (position == -1 and row['close'] > stop_loss):
                    # 计算盈亏（考虑滑点和手续费）
                    exit_price = row['close'] * (1 - position * self.slippage)
                    
                    # 计算交易总值和手续费
                    trade_value = position_size * exit_price
                    exit_fee = trade_value * self.commission_rate
                    
                    # 更新资金余额
                    if position == 1:  # 多头平仓
                        self.current_balance = trade_value - exit_fee
                    else:  # 空头平仓
                        profit_ratio = (entry_price - exit_price) / entry_price  # 空头收益率
                        self.current_balance = self.current_balance * (1 + profit_ratio) - exit_fee
                        
                    pnl = self.current_balance - self.balance_history[-1] if self.balance_history else self.current_balance - self.initial_balance
                    self.balance_history.append(self.current_balance)
                    
                    # 记录交易
                    self.trades.append({
                        'entry_date': entry_date,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_date': row['date'],
                        'exit_time': row['time'],
                        'exit_price': exit_price,
                        'position': position,
                        'position_size': position_size,
                        'pnl': pnl,
                        'balance': self.current_balance,
                        'exit_reason': 'stop_loss'
                    })
                    
                    # 更新信号和持仓
                    signal = 2 if position == 1 else -2  # 2: 平多, -2: 平空
                    backtest_results.at[i, 'signal'] = signal
                    backtest_results.at[i, 'balance'] = self.current_balance
                    
                    print(f"止损平仓: 价格={exit_price:.2f}, 数量={position_size:.6f}, 盈亏={pnl:.2f}, 余额={self.current_balance:.2f}")
                    
                    position = 0
                    position_size = 0
                    stop_loss = np.nan
                    trades_today += 1
                    trade_count += 1
                    backtest_results.at[i, 'trades_today'] = trades_today
                    
                    if trade_count % 50 == 0:
                        print(f"已处理 {trade_count} 笔交易...")
            
            # 交易时段结束强制平仓
            if current_time >= self.trade_end_time and position != 0:
                # 计算盈亏
                exit_price = row['close'] * (1 - position * self.slippage)
                
                # 计算交易总值和手续费
                trade_value = position_size * exit_price
                exit_fee = trade_value * self.commission_rate
                
                # 更新资金余额
                if position == 1:  # 多头平仓
                    self.current_balance = trade_value - exit_fee
                else:  # 空头平仓
                    profit_ratio = (entry_price - exit_price) / entry_price  # 空头收益率
                    self.current_balance = self.current_balance * (1 + profit_ratio) - exit_fee
                    
                pnl = self.current_balance - self.balance_history[-1] if self.balance_history else self.current_balance - self.initial_balance
                self.balance_history.append(self.current_balance)
                
                # 记录交易
                self.trades.append({
                    'entry_date': entry_date,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_date': row['date'],
                    'exit_time': row['time'],
                    'exit_price': exit_price,
                    'position': position,
                    'position_size': position_size,
                    'pnl': pnl,
                    'balance': self.current_balance,
                    'exit_reason': 'session_end'
                })
                
                # 更新信号和持仓
                signal = 2 if position == 1 else -2
                backtest_results.at[i, 'signal'] = signal
                backtest_results.at[i, 'balance'] = self.current_balance
                
                print(f"收盘平仓: 价格={exit_price:.2f}, 数量={position_size:.6f}, 盈亏={pnl:.2f}, 余额={self.current_balance:.2f}")
                
                position = 0
                position_size = 0
                stop_loss = np.nan
                trades_today += 1
                trade_count += 1
                backtest_results.at[i, 'trades_today'] = trades_today
                
                if trade_count % 50 == 0:
                    print(f"已处理 {trade_count} 笔交易...")
            
            # 开仓条件检查（仅当空仓且在交易时段内且当日交易次数未达上限且是检查点）
            if position == 0 and in_trading_hours and trades_today < self.max_trades_per_day and is_check_point:
                # 多头入场: 价格 > 上边界
                if row['close'] > row['upper_bound']:
                    position = 1
                    entry_price = row['close'] * (1 + self.slippage)  # 考虑滑点
                    # 记录交易前的资金余额
                    self.balance_history.append(self.current_balance)
                    # 全仓计算持仓数量，减去手续费
                    position_size = self.current_balance * (1 - self.commission_rate) / entry_price
                    stop_loss = row['upper_bound']
                    entry_date = row['date']
                    entry_time = row['time']
                    backtest_results.at[i, 'signal'] = 1
                    backtest_results.at[i, 'stop_loss'] = stop_loss
                    trades_today += 1
                    trade_count += 1
                    backtest_results.at[i, 'trades_today'] = trades_today
                    
                    # 记录交易开始信息
                    print(f"多头入场: 价格={entry_price:.2f}, 数量={position_size:.6f}, 余额={self.current_balance:.2f}")
                    
                    if trade_count % 50 == 0:
                        print(f"已处理 {trade_count} 笔交易...")
                
                # 空头入场: 价格 < 下边界
                elif row['close'] < row['lower_bound']:
                    position = -1
                    entry_price = row['close'] * (1 - self.slippage)  # 考虑滑点
                    # 记录交易前的资金余额
                    self.balance_history.append(self.current_balance)
                    # 空头全仓，计算持仓数量，减去手续费
                    position_size = self.current_balance * (1 - self.commission_rate) / entry_price
                    stop_loss = row['lower_bound']
                    entry_date = row['date']
                    entry_time = row['time']
                    backtest_results.at[i, 'signal'] = -1
                    backtest_results.at[i, 'stop_loss'] = stop_loss
                    trades_today += 1
                    trade_count += 1
                    backtest_results.at[i, 'trades_today'] = trades_today
                    
                    # 记录交易开始信息
                    print(f"空头入场: 价格={entry_price:.2f}, 数量={position_size:.6f}, 余额={self.current_balance:.2f}")
                    
                    if trade_count % 50 == 0:
                        print(f"已处理 {trade_count} 笔交易...")
            
            # 更新当前持仓
            backtest_results.at[i, 'position'] = position
        
        # 保存回测结果
        self.results = backtest_results
        
        print(f"回测完成，共处理 {trade_count} 笔交易")
        
        # 计算回测性能指标
        print("计算性能指标...")
        self._calculate_performance()
        
        return self.results, self.performance
    
    def _calculate_performance(self):
        """计算回测性能指标"""
        if not self.trades:
            self.performance = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl_per_trade': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'final_balance': self.initial_balance,
                'return_pct': 0
            }
            return
        
        # 转换交易记录为DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        # 计算基本指标
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
        final_balance = self.current_balance
        return_pct = (final_balance / self.initial_balance - 1) * 100
        
        # 计算最大回撤
        if 'balance' in trades_df.columns:
            # 使用资金曲线计算回撤
            balance_series = pd.Series([self.initial_balance] + trades_df['balance'].tolist())
            running_max = balance_series.cummax()
            drawdown_pct = (running_max - balance_series) / running_max * 100
            max_drawdown = drawdown_pct.max() if len(drawdown_pct) > 0 else 0
        else:
            # 传统方式计算回撤
            cumulative_pnl = trades_df['pnl'].cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = running_max - cumulative_pnl
            max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        # 计算夏普比率（假设无风险利率为0，使用每日收益率）
        if len(trades_df) > 1:
            # 按日期分组计算每日收益
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            daily_returns = trades_df.groupby(trades_df['entry_date'].dt.date)['pnl'].sum()
            
            # 计算夏普比率
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 保存性能指标
        self.performance = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'return_pct': return_pct
        }
        
        return self.performance
    
    def plot_results(self, output_dir='./output', filename='noise_region_fixed_backtest.png'):
        """绘制回测结果图表"""
        if self.results is None:
            print("请先运行回测")
            return
        
        print("开始绘制结果图表...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 选择一个交易日的数据进行可视化
        dates = self.results['date'].unique()
        if len(dates) > 2:  # 选择第三个交易日（前两天是用于计算历史波动率的）
            sample_date = dates[2]
        else:
            sample_date = dates[-1]
        
        day_data = self.results[self.results['date'] == sample_date].copy()
        
        # 创建画布
        fig, axs = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 绘制价格和边界
        axs[0].plot(day_data['timestamp'], day_data['close'], label='Close Price', color='black')
        axs[0].plot(day_data['timestamp'], day_data['upper_bound'], label='Upper Bound', color='red', linestyle='--')
        axs[0].plot(day_data['timestamp'], day_data['lower_bound'], label='Lower Bound', color='green', linestyle='--')
        
        if 'stop_loss' in day_data.columns:
            axs[0].plot(day_data['timestamp'], day_data['stop_loss'], label='Stop Loss', color='purple', linestyle=':')
        
        # 标记检查点
        check_points = day_data[day_data['is_check_point']]
        axs[0].scatter(check_points['timestamp'], check_points['close'], color='yellow', marker='o', s=30, label='Check Point')
        
        # 标记交易信号
        buy_signals = day_data[day_data['signal'] == 1]
        sell_signals = day_data[day_data['signal'] == -1]
        close_long = day_data[day_data['signal'] == 2]
        close_short = day_data[day_data['signal'] == -2]
        
        axs[0].scatter(buy_signals['timestamp'], buy_signals['close'], color='green', marker='^', s=100, label='Buy')
        axs[0].scatter(sell_signals['timestamp'], sell_signals['close'], color='red', marker='v', s=100, label='Sell')
        axs[0].scatter(close_long['timestamp'], close_long['close'], color='blue', marker='x', s=100, label='Close Long')
        axs[0].scatter(close_short['timestamp'], close_short['close'], color='orange', marker='x', s=100, label='Close Short')
        
        axs[0].set_title(f'Noise Region Strategy - Fixed Interval - {sample_date}')
        axs[0].set_ylabel('Price')
        axs[0].legend()
        axs[0].grid(True)
        
        # 绘制持仓状态
        axs[1].plot(day_data['timestamp'], day_data['position'], label='Position', color='blue', drawstyle='steps-post')
        axs[1].set_ylabel('Position')
        axs[1].set_yticks([-1, 0, 1])
        axs[1].set_yticklabels(['Short', 'Flat', 'Long'])
        axs[1].grid(True)
        
        # 绘制检查点标记
        axs[2].step(day_data['timestamp'], day_data['is_check_point'].astype(int), where='post', color='orange', label='Check Points')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Check Point')
        axs[2].set_yticks([0, 1])
        axs[2].set_yticklabels(['No', 'Yes'])
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
        # 绘制整体盈亏曲线
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
            trades_df = trades_df.sort_values('entry_date')
            
            # 绘制累计盈亏曲线
            cumulative_pnl = trades_df['pnl'].cumsum()
            
            plt.figure(figsize=(15, 6))
            plt.plot(cumulative_pnl.index, cumulative_pnl.values, label='Cumulative PnL')
            plt.title('Cumulative PnL')
            plt.xlabel('Trade #')
            plt.ylabel('PnL')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'fixed_cumulative_pnl.png'))
            plt.close()
            
            # 绘制资金曲线
            if 'balance' in trades_df.columns:
                balance_history = [self.initial_balance] + trades_df['balance'].tolist()
                trade_indices = range(len(balance_history))
                
                plt.figure(figsize=(15, 6))
                plt.plot(trade_indices, balance_history, label='Account Balance', color='green')
                plt.title('Account Balance History')
                plt.xlabel('Trade #')
                plt.ylabel('Balance')
                plt.grid(True)
                
                # 标记最大回撤
                if len(balance_history) > 1:
                    balance_series = pd.Series(balance_history)
                    running_max = balance_series.cummax()
                    drawdown = (running_max - balance_series) / running_max
                    max_dd_idx = drawdown.idxmax()
                    peak_idx = running_max.iloc[:max_dd_idx+1].idxmax()
                    
                    plt.plot([peak_idx, max_dd_idx], [running_max.iloc[peak_idx], balance_series.iloc[max_dd_idx]], 
                             'r--', linewidth=2, label=f'Max Drawdown: {drawdown.iloc[max_dd_idx]*100:.2f}%')
                
                plt.legend()
                plt.savefig(os.path.join(output_dir, 'fixed_balance_history.png'))
                plt.close()
    
    def save_results(self, output_dir='./output'):
        """保存回测结果到CSV文件"""
        if self.results is None:
            print("请先运行回测")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存回测结果
        print("保存回测结果...")
        self.results.to_csv(os.path.join(output_dir, 'fixed_backtest_results.csv'), index=False)
        
        # 保存交易记录
        if self.trades:
            print("保存交易记录...")
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(os.path.join(output_dir, 'fixed_trades.csv'), index=False)
        
        # 保存性能指标
        if self.performance:
            print("保存性能指标...")
            pd.DataFrame([self.performance]).to_csv(os.path.join(output_dir, 'fixed_performance.csv'), index=False)
            
        print(f"回测结果已保存到 {output_dir} 目录")

def load_data(filepath, source='okx'):
    """
    加载K线数据，支持多种数据源格式
    
    参数:
    filepath: str - 数据文件路径
    source: str - 数据源类型，目前支持'okx'
    
    返回:
    DataFrame - 处理后的数据
    """
    data = pd.read_csv(filepath)
    
    if source.lower() == 'okx':
        # OKX数据处理
        if 'timestamp' in data.columns:
            # 如果timestamp是数值类型的毫秒时间戳
            if pd.api.types.is_numeric_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'].astype(int), unit='ms')
            else:
                # 尝试直接转换
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                
        # 如果存在date列但没有timestamp列，使用date列
        elif 'date' in data.columns and 'timestamp' not in data.columns:
            data['timestamp'] = pd.to_datetime(data['date'])
    else:
        # 默认处理：确保timestamp列为datetime类型
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # 确保必要的列存在
    required_columns = ['timestamp', 'open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"数据文件缺少必要的列: {', '.join(missing_columns)}")
    
    return data

def main():
    # 从配置中获取参数
    data_file = CONFIG["data_file"]
    source = CONFIG["source"]
    
    # 如果未指定数据文件，则查找最新的数据文件
    if data_file is None:
        data_files = [f for f in os.listdir('./') if f.endswith('.csv') and (
            f.startswith('BTC_USDT_') or f.startswith('BTC-USDT_'))]
        if not data_files:
            print("未找到BTC-USDT数据文件，请先运行okx_candle_data.py获取数据或者指定数据文件路径")
            return
        
        data_file = sorted(data_files)[-1]  # 获取最新的文件
    
    print(f"使用数据文件: {data_file}")
    
    # 加载数据
    try:
        data = load_data(data_file, source=source)
        print(f"成功加载数据，共 {len(data)} 条记录")
        
        # 根据配置过滤数据
        if "start_date" in CONFIG and CONFIG["start_date"]:
            start_date = pd.to_datetime(CONFIG["start_date"]).date()
            data = data[data['timestamp'].dt.date >= start_date]
            print(f"按日期过滤后数据量: {len(data)} 条记录 (从 {start_date} 开始)")

    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 初始化策略
    strategy = NoiseRegionStrategyFixed(
        data=data,
        lookback_days=CONFIG["lookback_days"],
        k1=CONFIG["k1"],
        k2=CONFIG["k2"],
        trade_start_time=CONFIG["trade_start_time"],
        trade_end_time=CONFIG["trade_end_time"],
        max_trades_per_day=CONFIG["max_trades_per_day"],
        commission_rate=CONFIG["commission_rate"],
        slippage=CONFIG["slippage"],
        initial_balance=CONFIG["initial_balance"],
        check_interval=CONFIG["check_interval"]
    )
    
    # 运行回测
    results, performance = strategy.run_backtest()
    
    # 输出回测结果
    print("\n==== 回测性能 ====")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    # 绘制回测结果
    strategy.plot_results()
    
    # 保存结果
    strategy.save_results()

if __name__ == "__main__":
    main()   