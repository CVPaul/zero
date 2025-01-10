#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import numba as nb
import numpy as np
import pandas as pd

from typing import List
from types import SimpleNamespace
from quark.db.ailab import Client
from quark.db.ailab import SRC_EVENT_MAP
from zero.trans.strategy.common import StructBase


class Platform(StructBase):

    def __init__(self, tick: int = 0, n_threads: int = 0, source = "PK"):
        super().__init__()
        assert source in SRC_EVENT_MAP or source == "MD"
        self.tick = tick
        self.done = False
        self.n_threads = n_threads
        self.mds = np.array([])
        self.mask = np.array([])
        self.positions = np.array([])
        self.ask_orders_ = np.array([])
        self.bid_orders_ = np.array([])
        self.ask_order_loc = np.array([])
        self.bid_order_loc = np.array([])
        self.filled_orders = np.array([])
        self.filled_orders_cnt = 0
        self.max_hold_orders = 20
        self.g_order_id = 0
        self.timestamp = 0
        self.source_ = source
        self.last_ticks = []

    def list(self):
        cli = Client()
        db = cli.db['tick_overview']
        print('>>> Supported symbols:')
        for doc in db.find():
            print(f"    - {doc['symbol']}({doc['start']}~{doc['end']})")

    def reset(self, date: str, tickers: List[str], end_date: str = '', repeat=1):
        self.done = False
        self.repeat = repeat
        self.tickers = tickers
        self.n_tickers = len(tickers)
        self.ticker2ii = {
            self.tickers[i]:i
            for i in range(self.n_tickers)}
        self.last_ticks = [None] * self.n_tickers
        if not end_date:
            end_date = date
        # init the data buffers
        n_tickers = self.n_tickers * self.repeat
        self.mds = np.zeros((n_tickers, self.md_size), dtype=np.float32)
        self.mask = np.zeros(n_tickers, dtype=np.bool_)
        self.positions = np.zeros((n_tickers, self.position_size), dtype=np.float32)
        self.ask_orders_ = np.zeros((n_tickers * self.max_hold_orders, self.order_size), dtype=np.float32)
        self.bid_orders_ = np.zeros((n_tickers * self.max_hold_orders, self.order_size), dtype=np.float32)
        self.ask_order_loc = np.zeros(n_tickers, np.int32)
        self.bid_order_loc = np.zeros(n_tickers, np.int32)
        self.filled_orders = np.zeros((n_tickers * self.max_hold_orders, self.order_size), dtype=np.float32)
        # load history tick data
        cli = Client()
        if self.source_ and self.source_ in SRC_EVENT_MAP:
            self.data = cli.read2(tickers, self.source_, date, end_date, return_df=False)
        else:
            self.data = cli.read(tickers, 'tick', date, end_date, return_df=False)

    def step(self):
        try:
            tick = next(self.data)
            if self.source_ == 'PK':
                self.timestamp = tick['E']
                tick = self._pankou_to_tick(tick)
            elif self.source_ == 'DP':
                self.timestamp = tick['E']
                tick = self._depth_to_tick(tick)
            elif self.source_ == 'MD':
                tick = SimpleNamespace(**tick)
                self.timestamp = int(tick.datetime.timestamp() * 1000)
            else:
                raise RuntimeError(f"unsupported source type:{self.source_}!")
            self.filled_orders_cnt = 0
            self.match(tick)
            self.on_tick(tick)
            return tick
        except StopIteration:
            self.done = True
            return None

    def _pankou_to_tick(self, pk):
        tick = SimpleNamespace()
        tick.symbol = pk['s']
        tick.ask_price_1 = float(pk['a'])
        tick.bid_price_1 = float(pk['b'])
        tick.ask_volume_1 = float(pk['A'])
        tick.bid_volume_1 = float(pk['B'])
        return tick

    def _depth_to_tick(self, dp):
        tick = SimpleNamespace()
        tick.symbol = dp['s']
        for i in range(1, 11):
            setattr(tick, f'ask_price_{i}', float(dp['a'][i-1][0]))
            setattr(tick, f'ask_volume_{i}', float(dp['a'][i-1][1]))
            setattr(tick, f'bid_price_{i}', float(dp['b'][i-1][0]))
            setattr(tick, f'bid_volume_{i}', float(dp['b'][i-1][1]))
        return tick
    
    @property
    def bid_orders(self):
        return self.bid_orders_[self.bid_orders_[:, self.order_id] > 0]

    @property
    def ask_orders(self):
        return self.ask_orders_[self.ask_orders_[:, self.order_id] > 0]

    def take(self, actions: np.array):
        for action in actions:
            if action[self.action] == 0:
                self.add(action)
            else:
                self.cancel(action)

    def add(self, action: np.array):
        ii = int(action[self.instrument_id] + 0.5)
        st = ii * self.max_hold_orders
        self.g_order_id += 1
        action[self.insert_time] = self.timestamp % self.time_scale_
        action[self.insert_base] = self.timestamp // self.time_scale_
        action[self.order_id] = self.g_order_id
        volume = action[self.total_volume]
        if action[self.direction] ==  0: # buy
            pi = self.bid_order_loc[ii]
            self.bid_orders_[st + pi] = action
            if action[self.side] == 0: # long-buy --> long-open
                self.positions[ii][self.long_unfilled_buy] += volume
            else: # short-buy --> short-close
                self.positions[ii][self.short_unfilled_buy] += volume
            self.bid_order_loc[ii] += 1
            assert self.bid_order_loc[ii] <= self.max_hold_orders
        else: # sell
            pi = self.ask_order_loc[ii]
            self.ask_orders_[st + pi] = action
            # print("pos21", self.positions)
            if action[self.side] == 0: # long-sell --> long-close
                self.positions[ii][self.long_unfilled_sell] += volume
            else: # short-sell --> short-open
                self.positions[ii][self.short_unfilled_sell] += volume
            self.ask_order_loc[ii] += 1
            assert self.ask_order_loc[ii] <= self.max_hold_orders
        prob = action[self.trade_prob]
        if prob > 1e-8 and np.random.uniform(0, 1) <= prob:
            if self.last_ticks[ii]:
                self.match(self.last_ticks[ii])

    def cancel(self, action: np.array):
        ii = int(action[self.instrument_id] + 0.5)
        st = ii * self.max_hold_orders
        oi = action[self.order_id]
        rs = []
        if action[self.direction] ==  0: # buy
            pi = self.bid_order_loc[ii]
            for i in range(st, st + pi):
                if oi == self.bid_orders_[i][self.order_id]:
                    order = self.bid_orders_[i]
                    volume = order[self.total_volume] - order[self.trade_volume]
                    if action[self.side] == 0: # long-buy --> long-open
                        self.positions[ii][self.long_unfilled_buy] -= volume
                    else: # short-buy --> short-close
                        self.positions[ii][self.short_unfilled_buy] -= volume
                    self.bid_orders_[i, self.order_id] = 0
                else:
                    rs.append(i)
            self.bid_order_loc[ii] = len(rs)
            if pi != len(rs):
                if self.bid_order_loc[ii] > 0:
                    self.bid_orders_[st:st + len(rs)] = self.bid_orders_[rs]
        else: # sell
            pi = self.ask_order_loc[ii]
            for i in range(st, st + pi):
                if oi == self.ask_orders_[i][self.order_id]:
                    order = self.ask_orders_[i]
                    volume = order[self.total_volume] - order[self.trade_volume]
                    if action[self.side] == 0: # long-sell --> long-close
                        self.positions[ii][self.long_unfilled_sell] -= volume
                    else: # short-sell --> short-open
                        self.positions[ii][self.short_unfilled_sell] -= volume
                    self.ask_orders_[i, self.order_id] = 0
                else:
                    rs.append(i)
            self.ask_order_loc[ii] = len(rs)
            if pi != len(rs):
                if self.ask_order_loc[ii] > 0:
                    self.ask_orders_[st:st + len(rs)] = self.ask_orders_[rs]

    def match(self, tick):
        # print("before match:", self.positions)
        ii = self.ticker2ii[tick.symbol]
        st = ii * self.max_hold_orders
        self.last_ticks[ii] = tick
        # buy side
        rs = []
        bi = self.bid_order_loc[ii]
        price = tick.ask_price_1
        for i in range(st, st + bi):
            if abs(price / self.bid_orders_[i][self.price] - 1.0) < 1e-7:
                order = self.bid_orders_[i]
                self.filled_orders[self.filled_orders_cnt] = order
                self.filled_orders_cnt += 1
                volume = order[self.total_volume] - order[self.trade_volume]
                if order[self.side] == 0: # long-buy --> long-open
                    self.positions[ii][self.long_unfilled_buy] -= volume
                    self.positions[ii][self.long_buy] += volume
                else: # short-buy --> short-close
                    self.positions[ii][self.short_unfilled_buy] -= volume
                    self.positions[ii][self.short_buy] += volume
                self.bid_orders_[i, self.order_id] = 0
            else:
                rs.append(i)
        self.bid_order_loc[ii] = len(rs)
        if bi != len(rs):
            if self.bid_order_loc[ii] > 0:
                self.bid_orders_[st:st + len(rs)] = self.bid_orders_[rs]
        # sell side
        rs = []
        ai = self.ask_order_loc[ii]
        price = tick.bid_price_1
        for i in range(st, st + ai):
            if abs(price / self.ask_orders_[i][self.price] - 1.0) < 1e-7:
                order = self.ask_orders_[i]
                self.filled_orders[self.filled_orders_cnt] = order
                self.filled_orders_cnt += 1
                volume = order[self.total_volume] - order[self.trade_volume]
                if order[self.side] == 0: # long-sell --> long-close
                    self.positions[ii][self.long_unfilled_sell] -= volume
                    self.positions[ii][self.long_sell] += volume
                else: # short-sell --> short-open
                    self.positions[ii][self.short_unfilled_sell] -= volume
                    self.positions[ii][self.short_sell] += volume
                self.ask_orders_[i, self.order_id] = 0
            else:
                rs.append(i)
        self.ask_order_loc[ii] = len(rs)
        if ai != len(rs):
            if self.ask_order_loc[ii] > 0:
                self.ask_orders_[st:st + len(rs)] = self.ask_orders_[rs]

    def on_tick(self, tick):
        ii = self.ticker2ii[tick.symbol]
        md = self.mds[ii]
        if self.source_ == "PK": # 盘口
            md[7] = float(tick.bid_price_1)
            md[17] = float(tick.ask_price_1)
            md[27] = float(tick.bid_volume_1)
            md[37] = float(tick.ask_volume_1)
        else:
            if self.source_ == "MD": # not depth --> md snapshot
                md[0] = tick.pre_close
                md[1] = tick.open_price
                md[2] = tick.high_price
                md[3] = tick.low_price
                md[4] = tick.last_price
                md[5] = tick.last_price
                md[6] = tick.volume
            md[7] = tick.bid_price_1
            md[8] = tick.bid_price_2
            md[9] = tick.bid_price_3
            md[10] = tick.bid_price_4
            md[11] = tick.bid_price_5
            md[12] = tick.bid_price_6
            md[13] = tick.bid_price_7
            md[14] = tick.bid_price_8
            md[15] = tick.bid_price_9
            md[16] = tick.bid_price_10
    
            md[17] = tick.ask_price_1
            md[18] = tick.ask_price_2
            md[19] = tick.ask_price_3
            md[20] = tick.ask_price_4
            md[21] = tick.ask_price_5
            md[22] = tick.ask_price_6
            md[23] = tick.ask_price_7
            md[24] = tick.ask_price_8
            md[25] = tick.ask_price_9
            md[26] = tick.ask_price_10
    
            md[27] = tick.bid_volume_1
            md[28] = tick.bid_volume_2
            md[29] = tick.bid_volume_3
            md[30] = tick.bid_volume_4
            md[31] = tick.bid_volume_5
            md[32] = tick.bid_volume_6
            md[33] = tick.bid_volume_7
            md[34] = tick.bid_volume_8
            md[35] = tick.bid_volume_9
            md[36] = tick.bid_volume_10
    
            md[37] = tick.ask_volume_1
            md[38] = tick.ask_volume_2
            md[39] = tick.ask_volume_3
            md[40] = tick.ask_volume_4
            md[41] = tick.ask_volume_5
            md[42] = tick.ask_volume_6
            md[43] = tick.ask_volume_7
            md[44] = tick.ask_volume_8
            md[45] = tick.ask_volume_9
            md[46] = tick.ask_volume_10
        md[47] = 1 # lot_size





