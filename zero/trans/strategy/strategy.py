#!/usr/bin/env python
#-*- coding:utf-8 -*-


import torch

from typing import List
from zero.trans.strategy.common import StructBase


class Strategy(StructBase):

    def __init__(self):
        super().__init__()

    @torch.jit.export
    def calc_pos(self, positions: torch.FloatTensor):
        long_pos = positions[self.long_init_pos] + \
            positions[self.long_buy] + positions[self.long_unfilled_buy] - \
            positions[self.long_sell] - positions[self.long_unfilled_sell]
        short_pos = positions[self.short_init_pos] - \
            positions[self.short_buy] - positions[self.short_unfilled_buy] + \
            positions[self.short_sell] + positions[self.short_unfilled_sell]
        return long_pos, short_pos

    @torch.jit.export
    def get_insert_time(self, orders: torch.Tensor):
        return orders[self.insert_base].to(torch.float64) * self.time_scale + orders[self.insert_time]

    @torch.jit.export
    def cancel_timeout_orders(self, timestamp: float, bid_orders: torch.Tensor, ask_orders: torch.Tensor, timeout: float) -> torch.FloatTensor:
        cancel_orders: List[torch.Tensor] = []
        for orders in [bid_orders, ask_orders]:
            if orders.size(0):
                ts = self.get_insert_time(orders)
                slc_orders = ts < (timestamp - timeout)
                if slc_orders.any():
                    cancel_orders.append(orders[:, slc_orders])
        if len(cancel_orders) > 1:
            result = torch.cat(cancel_orders, dim=1)
        elif len(cancel_orders) == 1:
            result = cancel_orders[0]
        else:
            result = torch.zeros(self. order_size, 0, dtype=torch.float32, device=bid_orders.device)
        result[self.action] = 1 # cancel
        return result

    @torch.jit.export
    def insert_orders(self, instrument_ids: torch.Tensor, prices: torch.Tensor, volumes: torch.Tensor, directions: torch.Tensor, sides: torch.Tensor) -> torch.Tensor:
        n_orders = instrument_ids.numel()
        orders = torch.zeros(self.order_size, n_orders, dtype=torch.float32, device=instrument_ids.device)
        if n_orders:
            orders[self.instrument_id] = instrument_ids
            orders[self.price] = prices
            orders[self.total_volume] = volumes
            orders[self.side] = sides
            orders[self.direction] = directions
        return orders

    @torch.jit.export
    def insert_order(self, instrument_id: int, price: float, volume: int, direction: int, side: int) -> torch.Tensor:
        orders = torch.zeros(self.order_size, 1, dtype=torch.float32)
        orders[self.instrument_id] = instrument_id
        orders[self.price] = price
        orders[self.total_volume] = volume
        orders[self.side] = side
        orders[self.direction] = direction
        return orders