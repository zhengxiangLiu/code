
from datetime import datetime
import copy
import json
import numpy as np
import operator
import pandas as pd
import os
import argparse


class SectorPreprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.market_name = market_name
    def generate_sector_relation(self, industry_ticker_file,
                                     selected_tickers_fname):
        selected_tickers = np.genfromtxt(
            os.path.join(self.data_path,  selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print(selected_tickers[0:5])
        print('#tickers selected:', len(selected_tickers))
        ticker_index = {}
        for index, ticker in enumerate(selected_tickers):
            ticker_index[ticker] = index
        with open(industry_ticker_file,'r') as fin:
            industry_tickers=json.load(fin)
        print('#industries: ', len(industry_tickers))
        valid_industry_count=0
        valid_industry_index={}
        for industry in industry_tickers.keys():
            if len(industry_tickers[industry])>1:
                valid_industry_index[industry]=valid_industry_count
                valid_industry_count+=1
        ticker_relation_embedding=np.zeros([len(selected_tickers),valid_industry_count + 1], dtype=int)
        one_hot_industry_embedding = np.identity(valid_industry_count + 1,dtype=int)
        print("#ticker_relation_embedding:",ticker_relation_embedding.shape)
        print("#one_hot_industry_embedding:",one_hot_industry_embedding.shape)
        for industry in valid_industry_index.keys():
            cur_ind_tickers=industry_tickers[industry]
            if len(cur_ind_tickers)<=1:
                print('shit industry:',industry)
                continue
            ind_ind = valid_industry_index[industry]
            for i in range(len(cur_ind_tickers)):
                left_tic_ind=ticker_index[cur_ind_tickers[i]]
                ticker_relation_embedding[left_tic_ind][ind_ind]=1

        print('ok')
        # np.save(self.market_name + '_industry_relation',
        #         ticker_relation_embedding)
        
if __name__ == '__main__':
    desc = "pre-process sector data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = '../data/'
    if args.market is None:
        #args.market = 'NASDAQ'
        args.market = 'NYSE'

    processor = SectorPreprocessor(args.path, args.market)

    processor.generate_sector_relation(
        os.path.join('data/relation/sector_industry/',
                     processor.market_name + '_industry_ticker.json'),
        processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    )