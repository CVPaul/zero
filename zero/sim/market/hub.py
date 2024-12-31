#!/usr/bin/env python
#-*- coding:utf-8 -*-


import os
import numpy as np
import pandas as pd

from typing import List
from types import SimpleNamespace
from quark.db.ailab import Client
from zero.trans.strategy.common import StructBase


class Platform(StructBase):

    def __init__(self, tick: int = 0, n_threads: int = 0):
        super().__init__()
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

    def list(self):
        cli = Client()
        db = cli.db['tick_overview']
        print('>>> Supported symbols:')
        for doc in db.find():
            print(f"    - {doc['symbol']}({doc['start']}~{doc['end']})")

    def reset(self, date: str, tickers: List[str], end_date: str = ''):
        self.done = False
        self.tickers = tickers
        self.n_tickers = len(tickers)
        self.ticker2ii = {
            self.tickers[i]:i
            for i in range(self.n_tickers)}
        if not end_date:
            end_date = date
        # init the data buffers
        self.mds = np.zeros((self.n_tickers, self.md_size), dtype=np.float32)
        self.mask = np.zeros(self.n_tickers, dtype=np.bool_)
        self.positions = np.zeros((self.n_tickers, self.position_size), dtype=np.float32)
        self.ask_orders_ = np.zeros((self.n_tickers * self.max_hold_orders, self.order_size), dtype=np.float32)
        self.bid_orders_ = np.zeros((self.n_tickers * self.max_hold_orders, self.order_size), dtype=np.float32)
        self.ask_order_loc = np.zeros(self.n_tickers, np.int32)
        self.bid_order_loc = np.zeros(self.n_tickers, np.int32)
        self.filled_orders = np.zeros((self.n_tickers * self.max_hold_orders, self.order_size), dtype=np.float32)
        # load history tick data
        cli = Client()
        self.data = cli.read(tickers, 'tick', date, end_date, return_df=False)

    def step(self):
        try:
            tick = next(self.data)
            tick = SimpleNamespace(**tick)
            self.filled_orders_cnt = 0
            self.timestamp = int(tick.datetime.timestamp() * 1000)
            self.match(tick)
            self.on_tick(tick)
            return tick
        except StopIteration:
            self.done = True
            return None

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

    def cancel(self, action: np.array):
        ii = int(action[self.instrument_id] + 0.5)
        st = ii * self.max_hold_orders
        oi = action[self.order_id]
        j = st
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
                else:
                    j += 1
                if i != j:
                    self.bid_orders_[j] = self.bid_orders_[i]
                    self.bid_orders_[i][self.order_id] = 0 # reset
            self.bid_order_loc[ii] = j - st
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
                else:
                    j += 1
                if i != j:
                    self.ask_orders_[j] = self.ask_orders_[i]
                    self.ask_orders_[i][self.order_id] = 0 # reset
            self.ask_order_loc[ii] = j - st
        return pi != j

    def match(self, tick):
        # print("before match:", self.positions)
        price = tick.last_price
        ii = self.ticker2ii[tick.symbol]
        st = ii * self.max_hold_orders
        # buy side
        rs = []
        bi = self.bid_order_loc[ii]
        for i in range(st, st + bi):
            if price <= self.bid_orders_[i][self.price]:
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
            else:
                rs.append(i)
        self.bid_order_loc[ii] = len(rs)
        if bi != len(rs):
            if self.bid_order_loc[ii] > 0:
                self.bid_orders_[st:st + len(rs)] = self.bid_orders_[rs]
            self.bid_orders_[st+len(rs):, self.order_id] = 0 # reset
        # sell side
        rs = []
        ai = self.ask_order_loc[ii]
        for i in range(st, st + ai):
            if price >= self.ask_orders_[i][self.price]:
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
            else:
                rs.append(i)
        self.ask_order_loc[ii] = len(rs)
        if ai != len(rs):
            if self.ask_order_loc[ii] > 0:
                self.ask_orders_[st:st + len(rs)] = self.ask_orders_[rs]
            self.ask_orders_[st + len(rs):, self.order_id] = 0 # reset

    def on_tick(self, tick):
        ii = self.ticker2ii[tick.symbol]
        md = self.mds[ii]
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





