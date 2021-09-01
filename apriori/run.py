

import colorsys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import threading
# we need to install mlxtend on anaconda prompt by typing 'pip install mlxtend'
from mlxtend.frequent_patterns import apriori # مكتبة الخوارزمية
from mlxtend.frequent_patterns import association_rules
store_data = pd.read_csv('BreadBasket_DMS.csv') # تححميل الداتا سيت

class test(threading.Thread): # كلاسس تحليل الدقة
    def __init__(self):
        threading.Thread.__init__(self)
        Items = {}
        for item in store_data['Item']:
            if item in Items:
                Items[item] = Items[item] + 1
            else:
                Items[item] = 1

        keys = []
        vals = []
        for i, k in Items.items():
            if k > 30:
                keys.append(i)
                vals.append(k)

        store_data['Quantity']= 1
        store_data.head(7)
        self.basket = store_data.groupby(['Transaction', 'Item'])['Quantity'].sum().unstack().fillna(0)
        print(self.basket.head())
        basket_sets = self.basket.applymap(self.encode_units)
        # الدقه 0.03
        frequent_itemsets = pd.DataFrame.from_records(apriori(basket_sets, min_support=0.03, use_colnames=True))
        file_object = open('result.txt', 'a') # استخراج الدقه الي ملف
        file_object.write(str(frequent_itemsets))
        file_object.close()

        print(frequent_itemsets)
        print(f"MAX SUPPORT :{frequent_itemsets['support'].max()} | ITEM NAME ")

        print(f"MIN SUPPORT :{frequent_itemsets['support'].min()} | ITEM NAME ")

        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        print(rules.head())
        plt.bar(keys, vals, label="Items sold")
        plt.rcParams["figure.figsize"] = [20,10]
        plt.ylabel ('Number of Transactions in Percentage')
        plt.xlabel ('Items Sold')
        plt.xticks(list(keys), rotation=90)
        plt.legend (bbox_to_anchor=(1, 1), loc="best", borderaxespad=0.)
        plt.show()
    def encode_units(self, x):
            if x <= 0:
                return 0
            if x >= 1:
                return 1
if __name__ == '__main__':
     n = threading.Thread(target=test)
     n.start()