#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hmac
import base64
import urllib.parse
import hashlib
import datetime
import time
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import pytz

# OKX API配置
API_KEY = "a16ccaed-3adc-4a37-8484-4d9dd0cb940c"
SECRET_KEY = "8EE34D5B82E1ED5FA441021E20FEC08C"
PASSPHRASE = "Hello2022"
BASE_URL = "https://www.okx.com"

# 用户配置参数
SYMBOL = "BTC-USDT"  # 交易对
BAR = "1m"           # K线周期：1m, 5m, 15m, 1H, 4H, 1D, 1W, 1M
START_DATE = "2025-01-01"  # 起始日期 (包含)
END_DATE = "2025-05-20"    # 结束日期 (包含)
LIMIT = 100  # 每次请求的数据点数量，最大300
OUTPUT_FILE = f"{SYMBOL}_{BAR}_{START_DATE}_{END_DATE}.csv"  # 输出文件名

# 是否使用本地时区
USE_LOCAL_TIMEZONE = True  # 设置为False以使用UTC时区
LOCAL_TIMEZONE = 'Asia/Shanghai'  # 根据需要修改您的本地时区

# 签名
def sign(message, secret_key):
    mac = hmac.new(bytes(secret_key, encoding='utf8'), bytes(message, encoding='utf-8'), digestmod='sha256')
    d = mac.digest()
    return base64.b64encode(d).decode()

# 添加签名等验证信息
def pre_hash(timestamp, method, request_path, body):
    return str(timestamp) + method + request_path + body

# 获取iso格式时间戳
def get_timestamp():
    now = datetime.utcnow()
    t = now.isoformat("T", "milliseconds")
    return t + "Z"

# 日期字符串转为时间戳(毫秒)，考虑时区
def date_to_timestamp(date_str):
    try:
        # 尝试解析包含时间的日期字符串
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # 如果失败，尝试解析仅包含日期的字符串，并设置时间为00:00:00
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    if USE_LOCAL_TIMEZONE:
        # 如果使用本地时区，将本地时间转换为UTC时间
        local_tz = pytz.timezone(LOCAL_TIMEZONE)
        dt = local_tz.localize(dt)
        utc_dt = dt.astimezone(pytz.UTC)
        timestamp = int(utc_dt.timestamp() * 1000)
    else:
        # 否则直接假设输入的日期字符串就是UTC时间
        timestamp = int(dt.timestamp() * 1000)
    
    return timestamp

# 时间戳转为可读日期，考虑时区
def timestamp_to_date(timestamp, return_utc=False):
    dt_utc = datetime.fromtimestamp(int(timestamp) / 1000, tz=pytz.UTC)
    
    if not return_utc and USE_LOCAL_TIMEZONE:
        # 如果需要本地时间，将UTC时间转换为本地时间
        local_tz = pytz.timezone(LOCAL_TIMEZONE)
        dt_local = dt_utc.astimezone(local_tz)
        return dt_local.strftime("%Y-%m-%d %H:%M:%S")
    else:
        # 否则返回UTC时间
        return dt_utc.strftime("%Y-%m-%d %H:%M:%S")

# 发送请求
def api_request(method, request_path, params=None):
    url = BASE_URL + request_path
    timestamp = get_timestamp()
    
    if params:
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        request_path += '?' + query_string
        url += '?' + query_string
    
    body = ''
    
    # 签名
    message = pre_hash(timestamp, method, request_path, body)
    signed_message = sign(message, SECRET_KEY)
    
    headers = {
        "OK-ACCESS-KEY": API_KEY,
        "OK-ACCESS-SIGN": signed_message,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": PASSPHRASE,
        "Content-Type": "application/json"
    }
    
    response = requests.request(method, url, headers=headers)
    return response.json()

# 获取K线数据
def get_candle_data(symbol, bar, after=None, limit=100):
    """
    获取K线数据
    
    参数:
    - symbol: 交易对
    - bar: K线周期
    - after: 获取此时间戳之后(更早)的数据
    - limit: 返回结果的数量限制(最大为300)
    """
    request_path = "/api/v5/market/history-candles"
    
    params = {
        'instId': symbol,
        'bar': bar,
        'limit': limit
    }
    
    if after:
        params['after'] = after
    
    result = api_request('GET', request_path, params)
    
    if result.get('code') == '0':
        return result.get('data', [])
    else:
        print(f"获取数据失败: {result.get('msg', '未知错误')}")
        return []

# 处理数据并保存为CSV
def process_and_save_data(data, symbol, bar, output_file):
    if not data:
        print("没有数据可处理")
        return None
    
    # 定义列名
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # 转换价格为float
    for col in ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote']:
        df[col] = df[col].astype(float)
    
    # 添加两个时间列：UTC时间和本地时间
    df['date_utc'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
    
    if USE_LOCAL_TIMEZONE:
        local_tz = pytz.timezone(LOCAL_TIMEZONE)
        df['date'] = df['date_utc'].dt.tz_convert(local_tz)
    else:
        df['date'] = df['date_utc']
    
    # 按时间排序（从旧到新）
    df = df.sort_values('date')
    
    # 保存为CSV
    df.to_csv(output_file, index=False)
    print(f"数据已保存至 {output_file}")
    
    return df

# 获取指定日期范围内的所有K线数据
def get_all_candles_in_range(symbol, bar, start_date, end_date, limit=100):
    """
    获取指定日期范围内的所有K线数据
    
    参数:
    - symbol: 交易对
    - bar: K线周期
    - start_date: 开始日期 (格式: YYYY-MM-DD)
    - end_date: 结束日期 (格式: YYYY-MM-DD)
    - limit: 每次请求的数据量限制
    
    返回:
    - 按时间排序的K线数据列表
    """
    time_desc = "本地时间" if USE_LOCAL_TIMEZONE else "UTC时间"
    print(f"获取 {symbol} 的 {bar} K线数据，从 {start_date} 到 {end_date}（{time_desc}）...")
    
    # 转换日期为时间戳
    # 注意：我们需要将end_date加上一天，以确保包含end_date这一天的所有数据
    if ' ' not in end_date:  # 如果end_date只包含日期，没有时间部分
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_obj = end_date_obj + timedelta(days=1)  # 加一天
        end_date = end_date_obj.strftime("%Y-%m-%d")
    
    end_timestamp = str(date_to_timestamp(end_date))
    start_timestamp = date_to_timestamp(start_date)
    
    print(f"转换后的时间戳范围：{start_timestamp} - {end_timestamp}")
    print(f"对应的UTC时间：{timestamp_to_date(str(start_timestamp), True)} - {timestamp_to_date(str(end_timestamp), True)}")
    
    all_data = []
    earliest_timestamp = None
    
    # 第一次请求使用end_date作为after参数
    current_after = end_timestamp
    
    # 持续获取数据，直到达到start_date或没有更多数据
    while True:
        print(f"请求数据：after={current_after} ({timestamp_to_date(current_after)})...")
        batch_data = get_candle_data(symbol, bar, after=current_after, limit=limit)
        
        if not batch_data:
            print("没有更多数据")
            break
        
        # 过滤掉早于start_date的数据
        valid_data = []
        for item in batch_data:
            item_timestamp = int(item[0])
            if item_timestamp >= start_timestamp:
                valid_data.append(item)
            else:
                # 找到了比start_date更早的数据，可以停止
                print(f"已达到开始日期 {start_date}")
                break
        
        all_data.extend(valid_data)
        
        # 如果批次数据数量小于limit或已找到早于start_date的数据，结束循环
        if len(batch_data) < limit or len(valid_data) < len(batch_data):
            break
        
        # 更新after参数为当前批次中最早的时间戳
        current_after = batch_data[-1][0]
        print(f"已获取 {len(all_data)} 条数据，继续获取... (最早时间: {timestamp_to_date(current_after)})")
    
    # 根据时间戳从旧到新排序
    sorted_data = sorted(all_data, key=lambda x: int(x[0]))
    
    print(f"总共获取了 {len(sorted_data)} 条数据")
    
    return sorted_data

def main():
    # 获取指定日期范围内的所有K线数据
    data = get_all_candles_in_range(SYMBOL, BAR, START_DATE, END_DATE, LIMIT)
    
    # 处理并保存数据
    if data:
        df = process_and_save_data(data, SYMBOL, BAR, OUTPUT_FILE)
        
        # 打印数据范围
        if not df.empty:
            first_date = df['date'].min().strftime('%Y-%m-%d %H:%M:%S')
            last_date = df['date'].max().strftime('%Y-%m-%d %H:%M:%S')
            print(f"数据日期范围: {first_date} 至 {last_date}")
            
            # 打印每天第一条记录的时间，检查是否从00:00开始
            if 'D' in BAR or 'W' in BAR or 'M' in BAR:
                # 对于日线及以上周期，检查每日开盘时间
                pass
            else:
                # 对于分钟线，检查每日第一条记录的时间
                df['date_day'] = df['date'].dt.date
                daily_first = df.groupby('date_day').first()
                print("\n每日第一条记录时间:")
                for idx, row in daily_first.iterrows():
                    print(f"{idx}: {row['date'].strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()