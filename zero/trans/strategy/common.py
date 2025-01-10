#!/usr/bin/env python
#-*- coding:utf-8 -*-


import torch.nn as nn


class StructBase(nn.Module):

    def __init__(self):
        super().__init__()
        # this value must aligin with online(zero_trader.cc)'s version
        self.api_version_ = 2
        self.time_scale_ = 1000000
        # md
        self.prevclose = 0
        self.open = 1
        self.high = 2
        self.low = 3
        self.close = 4
        self.last = 5
        self.volume = 6
        self.bidprice = 7
        self.askprice = 17
        self.bidvolume = 27
        self.askvolume = 37
        self.lot_size = 47
        self.md_size = 48
        # position
        self.long_init_pos = 0
        self.long_buy = 1
        self.long_sell = 2
        self.long_unfilled_buy = 3
        self.long_unfilled_sell = 4
        self.short_init_pos = 5
        self.short_buy = 6
        self.short_sell = 7
        self.short_unfilled_buy = 8
        self.short_unfilled_sell = 9
        self.long_trade_volume = 10
        self.short_trade_volume = 11
        self.position_size = 12
        # order
        self.instrument_id = 0
        self.order_id = 1
        self.action = 2
        self.price = 3
        self.side = 4
        self.direction = 5
        self.total_volume = 6
        self.trade_volume = 7
        self.insert_time = 8
        self.order_type = 9
        self.insert_base = 10
        self.trade_prob = 11
        self.order_size = 12