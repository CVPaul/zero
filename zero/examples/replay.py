#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import torch
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, "/home/ubuntu/dlli/zero")
sys.path.insert(0, "/home/ubuntu/quark")
from zero.sim.market.hub import Platform



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', type=str, default='20250110')
    parser.add_argument('--end-date', type=str, default='20250111')
    args = parser.parse_args()

    p = Platform(source="PK")
    p.reset(args.start_date, ['ETHUSDT'], args.end_date)

    tick = 0
    end = pd.to_datetime(args.end_date).timestamp() * 1e3

    while p.done != True:
        p.step()
        # simulate trader add orders
        if tick % 1000000 == 0:
            # print(p.mds[0])
            print(p.timestamp)
            action = np.array([[0, tick//1000000, 0, p.mds[0][17], 0, 0, p.mds[0][37], 0, 0, 0, 0, 0]])
                            # [1, tick//1000000, 0, p.mds[1][17], 0, 0, 1, 0, 0, 0, 0, 0]])
            p.take(action)
        tick += 1
    
    os.makedirs('/home/ubuntu/dlli/outputs', exist_ok=True)
    p.dump('/home/ubuntu/dlli/outputs')
    

if __name__ == '__main__':
    main()

