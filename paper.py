import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import math
import scipy.optimize as opt


'''
    if u need to use these function, first u need to change the direction in every function
'''



'''
    process data 1, choosing which stock num that you want and then, catching the data 
'''

def mthProcess1(year, date, stockNum):
    result = []

    f1 = open('/Users/liuzhaolun/PycharmProjects/TSMC/data/give/data/' + year + '/smth' + year + '/mth' + year + date, 'r')

    # read file1 and store to array
    flag1 = 1
    for line in f1:
        if flag1:
            flag1 = 0
            continue
        else:
            temp = []
            if line[8:12] == stockNum:
                for word in line.split(','):
                    temp.append(word.replace('\"', '').replace('\n', ''))
                result.append(temp)

    # save to a file
    f = open('/Users/liuzhaolun/PycharmProjects/TSMC/data/' + year + '/buffer.txt', 'w')
    for haha in result:
        temp = ''
        for word in haha:
            temp += word + ' '
        temp += '\n'
        f.write(temp)

    df = pd.read_csv('/Users/liuzhaolun/PycharmProjects/TSMC/data/give/data/' + year + '/buffer.txt')
    df.to_csv('/Users/liuzhaolun/PycharmProjects/TSMC/data/give/data/' + year + '/v1(' + stockNum + ')/mth' + year + date + '.csv', index=0)

    f = open('/Users/liuzhaolun/PycharmProjects/TSMC/data/give/data/' + year + '/buffer.txt', 'w')
    f.close()

'''
    process data 2, catch the helpful data column
'''

def mthProcess2(year, date, stockNum):
    df = pd.read_csv('/home/alpha/Desktop/give/data/' + year + '/v1(' + stockNum + ')/mth' + year + date + '.csv', sep="  ",
                     header=None)
    df.columns = ["split_a", "split_b"]
    global strB, strYM
    for i in range(len(df)):
        strYM = str(df.loc[i, 'split_a'])
        df.loc[i, 'MTH-MTHDATE'] = strYM[0:8]
        #df.loc[i, 'MTH-STKNO'] = strYM[8:12]

    print('phase1')
    for i in range(len(df)):
        strB = df.loc[i, 'split_b']
        if len(strB) != 49:
            df = df.drop(i)
            print(i)
        else:
            df.loc[i, 'MTH-MTHTIME'] = strB[2:10]  # 8
            df.loc[i, 'MTH-BUYSELL'] = strB[0]     # 1
            df.loc[i, 'MTH-EXCD'] = strB[1]        # 1
            #df.loc[i, 'MTH-RECNO'] = strB[10:18]   # 8
            #df.loc[i, 'MTH-ODRNO'] = strB[18:23]   # 5
            df.loc[i, 'MTH-MTHPR'] = strB[23:30]   # 7
            df.loc[i, 'MTH-MTHSHR'] = strB[30:39]  # 9
            #df.loc[i, 'MTH-MTHPRT'] = strB[39:43]  # 4
            #df.loc[i, 'MTH-ODRTPE'] = strB[43]     # 1
            df.loc[i, 'MTH-MARK'] = strB[44]       # 1
            #df.loc[i, 'MTH-BRKID'] = strB[45:49]   # 4


    print('phase2')
    df = df.drop(['split_a', 'split_b'], axis=1)
    #df = df.sort_values(["MTH-MTHDATE", "MTH-MTHSHR"])
    #df = df.sort_values(["MTH-MTHDATE", "MTH-MTHTIME", "MTH-MTHSHR"])
    df = df.dropna()
    df.to_csv('/home/alpha/Desktop/give/data/' + year + '/v2(' + stockNum + ')/mth' + year + date + '_v2.csv', index=0)

'''
    process data 3, delete non-useful and error data
'''

def mthProcess3(year, date, stockNum):
    df = pd.read_csv('/home/alpha/Desktop/give/data/' + year + '/v2(' + stockNum + ')/mth' + year + date + '_v2.csv', encoding='utf-8')
    #df = df.sort_values(["MTH-MTHTIME", "MTH-MTHSHR"])
    print(df.head())

    i = 0
    beg = 0
    list = []
    flg = 0
    while(i < len(df)):
        if i == len(df) - 1:
            break
        if df.loc[i, 'MTH-MTHTIME'] > 13300000:
            beg = i
            flg = 1
            break
        if df.loc[i, 'MTH-MTHSHR'] == df.loc[i+1, 'MTH-MTHSHR']:
            i += 2
        else:
            #print("-------"+str(i))
            list.append(i)
            i += 1

    if flg == 1:
        for j in range(beg, len(df)):
            df = df.drop(j)

    for j in list:
        df = df.drop(j)

    #df = df.sort_values(["MTH-MTHDATE", "MTH-MTHSHR"])
    df.to_csv('/home/alpha/Desktop/give/data/' + year + '/v3(' + stockNum + ')/mth' + year + date + '_v3.csv', index=0)


'''
    merge the data, for example: 120 days: 2018/1/1-2018-6-30
'''

def txtMerger(year, date, frame, stockNum):
    result = []

    f1 = open('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S'+frame+'.txt', 'r')
    f2 = open('/home/alpha/Desktop/give/data/' + year + '/v3(' + stockNum + ')/mth' + year + date + '_v3.csv', 'r')
    # f3 = open('/Users/liuzhaolun/PycharmProjects/TSMC/data/2015/01/20150107/mth20150107_v1.csv', 'r')

    # read file1 and store to array
    flag1 = 1
    for line in f1:
        if flag1:
            flag1 = 0
            continue
        else:
            temp = []
            for word in line.split(','):
                temp.append(word.replace('\"', '').replace('\n', ''))
            result.append(temp)

    # read file2 and store to array
    flag2 = 1
    count = 0
    for line in f2:
        if flag2:
            flag2 = 0
            continue
        else:
            temp = []
            for word in line.split(','):
                temp.append(word.replace('\"', '').replace('\n', ''))
            result.append(temp)

    # save to a file
    f = open('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S'+frame+'.txt', 'w')
    for haha in result:
        temp = ''
        for word in haha:
            temp += word + ' '
        temp += '\n'
        f.write(temp)

'''
    process order imbalance, change the buy and Sell
'''

def OI(frame, stockNum):
    df = pd.read_csv('/Users/liuzhaolun/PycharmProjects/TSMC/data/new_Chao-Lun/data/Clear(' + stockNum + ')/S' + frame + '.txt',
                     header=None, delim_whitespace=True, names=["MTH-MTHDATE", "MTH-MTHTIME", "MTH-BUYSELL",
                                                                "MTH-EXCD", "MTH-MTHPR", "MTH-MTHSHR",
                                                                "MTH-MARK"], index_col=False)

    for i in range(len(df)):
        if i == 0:
            df.loc[i, 'MTH-BUYSELL'] = 'B'
            continue
        if df.loc[i, 'MTH-MTHPR'] - df.loc[i - 1, 'MTH-MTHPR'] > 0:
            df.loc[i, 'MTH-BUYSELL'] = 'B'
        elif df.loc[i, 'MTH-MTHPR'] - df.loc[i - 1, 'MTH-MTHPR'] == 0:
            df.loc[i, 'MTH-BUYSELL'] = df.loc[i - 1, 'MTH-BUYSELL']
        elif df.loc[i, 'MTH-MTHPR'] - df.loc[i - 1, 'MTH-MTHPR'] < 0:
            df.loc[i, 'MTH-BUYSELL'] = 'S'


    df.to_csv('/Users/liuzhaolun/PycharmProjects/TSMC/data/new_Chao-Lun/data/Clear(' + stockNum + ')/S' + frame + '.txt',
              encoding='utf-8', index=0)

'''
    process every bucket order imbalance and other info.
'''

def VPIN_Change(days, frame, stockNum):
    df = pd.read_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S' + frame + '.txt', encoding='utf-8')

    print(df.head())

    # buffer: store remainder
    df1 = pd.read_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/buffer.csv', encoding='utf-8')

    sum_F = 0
    sum_I = 0
    sum_J = 0
    sum_M = 0

    sum_shares = 0
    bucket_num = 1
    df1_rowCount = 0

    max = [0] * 100000
    min = [1000] * 100000
    day = 0
    countDAY = 0
    firstTime = 1
    lstBucket = [0] * 100000  # bucket 的總量  ex: #1 : 4000 shares
    lstBucketRemainder = [0] * 100000

    count_date = 1

    for i in range(len(df)):
        if i == 0:
            continue
        if df.loc[i, 'MTH-MTHDATE'] != df.loc[i-1, 'MTH-MTHDATE']:
            count_date += 1
        if count_date > 10:
            sum_shares += df.loc[i, 'MTH-MTHSHR']

    bucket_volume = int(sum_shares / (days * 50))

    print(bucket_volume)
    print('phase1')

    buy_count = 0
    sell_count = 0
    aggregated_Buy_Volume = 0
    aggregated_Sell_Volume = 0
    for i in range(len(df)):

        lstBucket[bucket_num] += df.loc[i, 'MTH-MTHSHR']

        ''' buffer 裡有值 '''
        if df1_rowCount > 0 and firstTime == 1:
            lstBucket[bucket_num] += df1.loc[df1_rowCount-1, 'MTH-MTHSHR']

        if lstBucket[bucket_num] <= bucket_volume:
            if firstTime == 1:
                # initialize
                aggregated_Buy_Volume = 0
                aggregated_Sell_Volume = 0

                if df1_rowCount > 0:
                    if df1.loc[df1_rowCount-1, 'MTH-BUYSELL'] == 'B':
                        aggregated_Buy_Volume += df1.loc[df1_rowCount-1, 'MTH-MTHSHR']
                        buy_count += 1

                    elif df1.loc[df1_rowCount-1, 'MTH-BUYSELL'] == 'S':
                        aggregated_Sell_Volume += df1.loc[df1_rowCount-1, 'MTH-MTHSHR']
                        sell_count += 1

                firstTime = 0

            '''
                create 4 columns
            '''

            df.loc[i, 'Aggregated Volume Bucket'] = lstBucket[bucket_num]

            if df.loc[i, 'MTH-BUYSELL'] == 'B':
                buy_count += 1
                aggregated_Buy_Volume += df.loc[i, 'MTH-MTHSHR']

            elif df.loc[i, 'MTH-BUYSELL'] == 'S':
                sell_count += 1
                aggregated_Sell_Volume += df.loc[i, 'MTH-MTHSHR']


        # 計算bucket量，超出平均
        if lstBucket[bucket_num] > bucket_volume:
            lstBucketRemainder[bucket_num] = lstBucket[bucket_num] - bucket_volume  # remainder
            lstBucket[bucket_num] = bucket_volume                                   # ex: #1 :3000 shares stable
            #lstBucket[bucket_num] += lstBucketRemainder[bucket_num]                 # ex: #2 :add 1000 shares (remainder)
            #df.loc[i, 'MTH-MTHSHR'] -= lstBucketRemainder[bucket_num]                # 將最後一個超出量的row volume 調整
            '''
                create 4 columns
            '''

            if df.loc[i, 'MTH-BUYSELL'] == 'B':
                buy_count += 1
                aggregated_Buy_Volume += df.loc[i, 'MTH-MTHSHR']
            if df.loc[i, 'MTH-BUYSELL'] == 'S':
                sell_count += 1
                aggregated_Sell_Volume += df.loc[i, 'MTH-MTHSHR']

            df.loc[i, 'Aggregated Volume Bucket'] = bucket_volume
            df.loc[i, 'Aggregated Buy Volume'] = aggregated_Buy_Volume
            df.loc[i, 'Aggregated Sell Volume'] = aggregated_Sell_Volume
            df.loc[i, 'Order Imbalance'] = abs(aggregated_Buy_Volume - aggregated_Sell_Volume)
            df.loc[i, 'Buy Volume'] = buy_count
            df.loc[i, 'Sell Volume'] = sell_count

            '''
                新增多出來的row到df1當作buffer
            '''
            df1.loc[df1_rowCount, 'MTH-MTHDATE'] = df.loc[i, 'MTH-MTHDATE']
            df1.loc[df1_rowCount, 'MTH-MTHTIME'] = df.loc[i, 'MTH-MTHTIME']
            df1.loc[df1_rowCount, 'MTH-BUYSELL'] = df.loc[i, 'MTH-BUYSELL']
            df1.loc[df1_rowCount, 'MTH-MTHPR'] = df.loc[i, 'MTH-MTHPR']
            df1.loc[df1_rowCount, 'MTH-MTHSHR'] = lstBucketRemainder[bucket_num]
            df1.loc[df1_rowCount, 'MTH-MARK'] = df.loc[i, 'MTH-MARK']
            df1.loc[df1_rowCount, 'BUCKET'] = bucket_num


            df.loc[i, 'BUCKET'] = bucket_num
            df.loc[i, 'SAVE'] = bucket_num


            lstBucket[bucket_num] = lstBucketRemainder[bucket_num]

            bucket_num += 1
            df1_rowCount += 1
            firstTime = 1
            buy_count = 0
            sell_count = 0
            aggregated_Buy_Volume = 0
            aggregated_Sell_Volume = 0

    print('phase2')
    for i in range(len(df)):
        strMark = df.loc[i, 'MTH-MARK']
        if df.loc[i, 'Aggregated Volume Bucket'] == bucket_volume:
            if strMark == 'F':
                sum_F += 1
            if strMark == 'I':
                sum_I += 1
            if strMark == 'J':
                sum_J += 1
            if strMark == 'M':
                sum_M += 1
            if df.loc[i, 'SAVE'] >= 0:
                df.loc[i, 'MARK_F'] = sum_F
                df.loc[i, 'MARK_I'] = sum_I
                df.loc[i, 'MARK_J'] = sum_J
                df.loc[i, 'MARK_M'] = sum_M
            sum_F = 0
            sum_I = 0
            sum_J = 0
            sum_M = 0
        else:
            if strMark == 'F':
                sum_F += 1
            if strMark == 'I':
                sum_I += 1
            if strMark == 'J':
                sum_J += 1
            if strMark == 'M':
                sum_M += 1
            if df.loc[i, 'SAVE'] >= 0:
                df.loc[i, 'MARK_F'] = sum_F
                df.loc[i, 'MARK_I'] = sum_I
                df.loc[i, 'MARK_J'] = sum_J
                df.loc[i, 'MARK_M'] = sum_M

    print('phase3')
    for i in range(len(df)):
        if i == 0:
            if df.loc[i, 'MTH-MTHPR'] > max[day]:
                max[day] = df.loc[i, 'MTH-MTHPR']
            if df.loc[i, 'MTH-MTHPR'] < min[day]:
                min[day] = df.loc[i, 'MTH-MTHPR']
        if i > 0:
            if df.loc[i, 'MTH-MTHDATE'] != df.loc[i - 1, 'MTH-MTHDATE']:
                for j in range(countDAY, i):
                    if j == i - 1 or df.loc[j, 'SAVE'] >= 0:
                        df.loc[j, 'HIGH'] = max[day]
                        df.loc[j, 'LOW'] = min[day]
                        df.loc[j, 'OPEN'] = df.loc[countDAY, 'MTH-MTHPR']
                        df.loc[j, 'CLOSE'] = df.loc[i - 1, 'MTH-MTHPR']
                    # print('QQ' + str(j))
                countDAY = i
                day += 1
            if df.loc[i, 'MTH-MTHPR'] > max[day]:
                max[day] = df.loc[i, 'MTH-MTHPR']
            if df.loc[i, 'MTH-MTHPR'] < min[day]:
                min[day] = df.loc[i, 'MTH-MTHPR']
            if i + 1 == len(df):
                for j in range(countDAY, i + 1):
                    if j == i or df.loc[j, 'SAVE'] >= 0:
                        df.loc[j, 'HIGH'] = max[day]
                        df.loc[j, 'LOW'] = min[day]
                        df.loc[j, 'OPEN'] = df.loc[countDAY, 'MTH-MTHPR']
                        df.loc[j, 'CLOSE'] = df.loc[i, 'MTH-MTHPR']

                countDAY = i + 1
                day += 1

    df = df.dropna()
    df = df.drop(['MTH-EXCD'], axis=1)

    df.to_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S' + frame + '_v1.csv',
              encoding='utf-8', index=0)

    df1.to_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/buffer.csv', encoding='utf-8', index=0)


'''
    calculate VPIN
'''

def VPIN2(frame, stockNum):
    df = pd.read_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S' + frame + '_v1.csv',
                     encoding='utf-8')
    for i in range(len(df)):
        df.loc[i, 'Order Imbalance(buy-sell)'] = df.loc[i, 'Aggregated Buy Volume'] - df.loc[i, 'Aggregated Sell Volume']
    bucket = 0
    for i in range(len(df)):
        sum = 0
        if (bucket+50) > len(df):
            break
        for j in range(bucket,bucket+50):
            sum += df.loc[j, 'Order Imbalance']
        bucket += 1
        df.loc[i, 'VPIN'] = sum / (50 * df.loc[i, 'Aggregated Volume Bucket'])

    for i in range(len(df)):
        if i == 0:
            df.loc[i, 'Y_TREND'] = 'A'
        elif i == len(df) - 1:
            break
        else:
            # 漲
            if (df.loc[i + 1, 'MTH-MTHPR'] - df.loc[i, 'MTH-MTHPR']) >= 0:
                df.loc[i, 'Y_TREND'] = 'A'
            # 持平
            elif (df.loc[i + 1, 'MTH-MTHPR'] - df.loc[i, 'MTH-MTHPR']) < 0:
                df.loc[i, 'Y_TREND'] = 'B'

    for i in range(len(df)):
        if i == 0:
            df.loc[i, 'Price-Momentum(abs)'] = 0
            df.loc[i, 'Price-Momentum(not abs)'] = 0
        elif i == len(df) - 1:
            break
        else:
            df.loc[i, 'Price-Momentum(abs)'] = abs(df.loc[i + 1, 'MTH-MTHPR'] - df.loc[i, 'MTH-MTHPR'])
            df.loc[i, 'Price-Momentum(not abs)'] = df.loc[i + 1, 'MTH-MTHPR'] - df.loc[i, 'MTH-MTHPR']

    df = df.dropna(subset=["VPIN", "Y_TREND", "Price-Momentum(abs)", "Price-Momentum(not abs)"])

    df.to_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S' + frame + '_v2.csv',
              encoding='utf-8', index=0)


'''
    calculate different technical index, we can set the n paramaters to adjust index
'''

def MVM(n, frame, stockNum):
    df = pd.read_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S' + frame + '_v2.csv', encoding='utf-8')

    change = 1
    lstMVM = [0] * 10000    # MVM
    lstMVA = [0] * 10000    # MVA
    count = [0] * 10000
    min = [1000] * 1000        # last n days minimum
    max = [0] * 1000        # last n days maximum
    up = [0] * 1000  # RSI up movement
    down = [0] * 1000  # RSI down movement
    M = [0] * 1000       # CCI Mt
    SM = [0] * 1000      # CCI SMt
    D = [0] * 1000       # CCI Dt
    lastNdaysClosePrice = [0] * 1000
    for i in range(len(df)):
        if i+1 == len(df):
            lstMVM[change] += df.loc[i, 'Aggregated Volume Bucket']
            lstMVA[change] = df.loc[i, 'CLOSE']
            #lstKline[change] = df.loc[i, 'CLOSE']
            M[change] = df.loc[i, 'HIGH'] + df.loc[i, 'LOW'] + df.loc[i, 'CLOSE']
            count[change] += 1
            break
        if df.loc[i, 'MTH-MTHDATE'] != df.loc[i + 1, 'MTH-MTHDATE']:
            lstMVM[change] += df.loc[i, 'Aggregated Volume Bucket']
            lstMVA[change] = df.loc[i, 'CLOSE']
            #lstKline[change] = df.loc[i, 'CLOSE']
            M[change] = df.loc[i, 'HIGH'] + df.loc[i, 'LOW'] + df.loc[i, 'CLOSE']
            count[change] += 1
            change += 1
        else:
            lstMVM[change] += df.loc[i, 'Aggregated Volume Bucket']
            lstMVA[change] = df.loc[i, 'CLOSE']
            #lstKline[change] = df.loc[i, 'CLOSE']
            M[change] = df.loc[i, 'HIGH'] + df.loc[i, 'LOW'] + df.loc[i, 'CLOSE']
            count[change] += 1

    print(change)
    sum_MVM = [0] * 1000  # volume
    sum_MVA = [0] * 1000  # close price
    cell = [0] * 1000
    b = n+1
    a = 1
    '''
        MVA  MVM
    '''
    for i in range(1,(change-n)+1):
        for j in range(a,b):
            cell[i] += count[j]
            sum_MVM[i] += lstMVM[j]
            sum_MVA[i] += lstMVA[j]
        a += 1
        b += 1
        #print(count[i])
        #print(cell[i])

    '''
        K%  LW_R%  A/D
    '''
    for i in range(1, (change - n) + 1):
        if i == 1:
            x = count[i - 1]
            y = cell[i]
        for j in range(x, y):
            if df.loc[j, 'HIGH'] > max[i]:
                max[i] = df.loc[j, 'HIGH']
            if df.loc[j, 'LOW'] < min[i]:
                min[i] = df.loc[j, 'LOW']

        lastNdaysClosePrice[i] = df.loc[x, 'CLOSE']
        print(lastNdaysClosePrice[i])
        x += count[i]
        y += count[i + n]
        if i == change-n:
            for j in range(change-n+1, change+1):
                lastNdaysClosePrice[j] = df.loc[x, 'CLOSE']
                x += count[j]
                #print(lastNdaysClosePrice[j])
    '''
        RSI  CCI
    '''
    for i in range(1, (change - n) + 1):
        if i == 1:
            aa = 1
            bb = n
        for j in range(aa,bb):
            if lastNdaysClosePrice[j + 1] - lastNdaysClosePrice[j] >= 0:
                up[i] += abs(round((lastNdaysClosePrice[j + 1] - lastNdaysClosePrice[j]), 2))
            else:
                down[i] += abs(round((lastNdaysClosePrice[j + 1] - lastNdaysClosePrice[j]), 2))
            SM[i] += (M[j]/n)
        for j in range(aa,bb):
            D[i] += ((M[j]-SM[j])/n)
        print(up[i])
        print(down[i])
        print('---------')
        aa += 1
        bb += 1

    '''
        all technical indicator
    '''
    for i in range(1,(change-n)+1):
        for j in range(cell[i], cell[i]+count[i+n]):
            df.loc[j, 'MAX_' + str(n)] = max[i]
            df.loc[j, 'MIN_' + str(n)] = min[i]
            df.loc[j, 'MVM_' + str(n)] = round(sum_MVM[i] / n, 2)
            df.loc[j, 'MVA_' + str(n)] = round(sum_MVA[i] / n, 2)
            df.loc[j, 'K%_' + str(n)] = round(((lastNdaysClosePrice[i] - min[i]) / (max[i] - min[i])) * 100, 2)
            df.loc[j, 'LW_R%_' + str(n)] = round(((df.loc[j, 'HIGH'] - lastNdaysClosePrice[i]) /
                                                 (df.loc[j, 'HIGH'] - df.loc[j, 'LOW'])) * 100, 2)
            df.loc[j, 'A/D_' + str(n)] = round((df.loc[j, 'HIGH'] - lastNdaysClosePrice[i + 1]) /
                                               (df.loc[j, 'HIGH'] - df.loc[j, 'LOW']), 2)
            df.loc[j, 'RSI_' + str(n)] = round(100 - (100 / (1 + ((up[i] / n) / (down[i] / n)))), 2)
            df.loc[j, 'CCI_' + str(n)] = round((M[i] - SM[i]) / (0.015 * D[i]), 2)
            #print(j)
        cell[i+1] = cell[i] + count[i+n]


    #df = df.dropna()

    df.to_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S' + frame + '_v2.csv',
              encoding='utf-8', index=0)

'''
    calculate close and VPIN mometum
'''

def momentum(frame, stockNum):
    df = pd.read_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S' + frame + '_v2.csv',
                     encoding='utf-8')
    for i in range(len(df)):
        if i == 0:
            continue
        df.loc[i, 'CLOSE_Momentum'] = df.loc[i, 'CLOSE'] - df.loc[i-1, 'CLOSE']
        df.loc[i, 'VPIN_Momentum'] = df.loc[i, 'VPIN'] - df.loc[i - 1, 'VPIN']
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna()

    df.to_csv('/home/alpha/Desktop/give/data/Clear(' + stockNum + ')/S' + frame + '_v3.csv',
              encoding='utf-8', index=0)


'''
    calculate PIN indicators
'''

def PIN(frame, stockNum):
    df = pd.read_csv('/Users/liuzhaolun/PycharmProjects/TSMC/data/new_Chao-Lun/data/Clear(' + stockNum + ')/NEW_PIN/RS' + frame + '_v3.csv',
                     encoding='utf-8')

    for i in range(len(df)):
        df.loc[i, 'PIN'] = df.loc[i, 'Order Imbalance']/(df.loc[i, 'Aggregated Buy Volume']+df.loc[i, 'Aggregated Sell Volume'])

    df = df.dropna(subset=["PIN"])

    df.to_csv('/Users/liuzhaolun/PycharmProjects/TSMC/data/new_Chao-Lun/data/Clear(' + stockNum + ')/NEW_PIN/RS' + frame + '_v3.csv',
              encoding='utf-8', index=0)

'''
    machine learning models, show the Precision Recall F1-measure
'''
def ensemble():
    df = pd.read_csv('/Users/liuzhaolun/PycharmProjects/TSMC/data/Clear(2338)/S14_v3.csv', encoding='utf-8')
    X = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'MTH-MTHPR', 'MVM_3', 'MVM_5', 'MVM_10', 'MVA_3', 'MVA_5', 'MVA_10',
            'K%_3', 'K%_5', 'K%_10', 'LW_R%_3', 'LW_R%_5', 'LW_R%_10', 'A/D_3', 'A/D_5', 'A/D_10',
            'RSI_3', 'RSI_5', 'RSI_10', 'CCI_3', 'CCI_5', 'CCI_10', 'Order Imbalance(buy-sell)', 'VPIN']]
    y = df['Y_TREND']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train = X[0:3698]
    X_test = X[3698:4300]
    y_train = y[0:3698]
    y_test = y[3698:4300]

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)


    tree = DecisionTreeClassifier(criterion='entropy', max_depth=200)
    tree.fit(X_train, y_train)
    print('DecisionTree(Entropy)')
    print(metrics.classification_report(y_test, tree.predict(X_test)))
    print('---------------------')

    tree = DecisionTreeClassifier(criterion='gini', max_depth=200)
    tree.fit(X_train, y_train)
    print('DecisionTree(Gini)')
    print(metrics.classification_report(y_test, tree.predict(X_test)))
    print('---------------------')


    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    print('RandomForest')
    print(metrics.classification_report(y_test, rfc.predict(X_test)))
    print('---------------------')


'''
    main function
'''
if __name__ == '__main__':
    '''
        have the process data, there are five list
    '''

    YY07 = ['1218', '1219', '1220', '1221', '1224', '1225', '1226', '1227', '1228', '1231']

    YY08 = ['0102', '0103', '0104', '0107', '0108', '0109', '0110', '0111', '0114', '0115',
            '0116', '0117', '0118', '0121', '0122', '0123', '0124', '0125', '0128', '0129',
            '0130', '0131', '0201', '0212', '0213', '0214', '0215', '0218', '0219', '0220',
            '0221', '0222', '0225', '0226', '0227', '0229', '0303', '0304', '0305', '0306',
            '0307', '0310', '0311', '0312', '0313', '0314', '0317', '0318', '0319', '0320',
            '0321', '0324', '0325', '0326', '0327', '0328', '0331', '0401', '0402', '0403',
            '0407', '0408', '0409', '0410', '0411', '0414', '0415', '0416', '0417', '0418',
            '0421', '0422', '0423', '0424', '0425', '0428', '0429', '0430', '0502', '0505',
            '0506', '0507', '0508', '0509', '0512', '0513', '0514', '0515', '0516', '0519',
            '0520', '0521', '0522', '0523', '0526', '0527', '0528', '0529', '0530', '0602',
            '0603', '0604', '0605', '0606', '0609', '0610', '0611', '0612', '0613', '0616',
            '0617', '0618', '0619', '0620', '0623', '0624', '0625', '0626', '0627', '0630',
            '0701', '0702', '0703', '0704', '0707', '0708', '0709', '0710', '0711', '0714',
            '0715', '0716', '0717', '0718', '0721', '0722', '0723', '0724', '0725', '0729',
            '0730', '0731', '0801', '0804', '0805', '0806', '0807', '0808', '0811', '0812',
            '0813', '0814', '0815', '0818', '0819', '0820', '0821', '0822', '0825', '0826',
            '0827', '0828', '0829', '0901', '0902', '0903', '0904', '0905', '0908', '0909',
            '0910', '0911', '0912', '0915', '0916', '0917', '0918', '0919', '0922', '0923',
            '0924', '0925', '0926', '0930', '1001', '1002', '1003', '1006', '1007', '1008',
            '1009', '1013', '1014', '1015', '1016', '1017', '1020', '1021', '1022', '1023',
            '1024', '1027', '1028', '1029', '1030', '1031', '1103', '1104', '1105', '1106',
            '1107', '1110', '1111', '1112', '1113', '1114', '1117', '1118', '1119', '1120',
            '1121', '1124', '1125', '1126', '1127', '1128', '1201', '1202', '1203', '1204',
            '1205', '1208', '1209', '1210', '1211', '1212', '1215', '1216', '1217', '1218',
            '1219', '1222', '1223', '1224', '1225', '1226', '1229', '1230', '1231']

    YY09 = ['0105', '0106', '0107', '0108', '0109', '0110', '0112', '0113', '0114', '0115',
            '0116', '0117', '0119', '0120', '0121', '0202', '0203', '0204', '0205', '0206',
            '0209', '0210', '0211', '0212', '0213', '0216', '0217', '0218', '0219', '0220',
            '0223', '0224', '0225', '0226', '0227', '0302', '0303', '0304', '0305', '0306',
            '0309', '0310', '0311', '0312', '0313', '0316', '0317', '0318', '0319', '0320',
            '0323', '0324', '0325', '0326', '0327', '0330', '0331', '0401', '0402', '0403',
            '0406', '0407', '0408', '0409', '0410', '0413', '0414', '0415', '0416', '0417',
            '0420', '0421', '0422', '0423', '0424', '0427', '0428', '0429', '0430', '0504',
            '0505', '0506', '0507', '0508', '0511', '0512', '0513', '0514', '0515', '0518',
            '0519', '0520', '0521', '0522', '0525', '0526', '0527', '0601', '0602', '0603',
            '0604', '0605', '0606', '0608', '0609', '0610', '0611', '0612', '0615', '0616',
            '0617', '0618', '0619', '0622', '0623', '0624', '0625', '0626', '0629', '0630',
            '0701', '0702', '0703', '0706', '0707', '0708', '0709', '0710', '0713', '0714',
            '0715', '0716', '0717', '0720', '0721', '0722', '0723', '0724', '0727', '0728',
            '0729', '0730', '0731', '0803', '0804', '0805', '0806', '0810', '0811', '0812',
            '0813', '0814', '0817', '0818', '0819', '0820', '0821', '0824', '0825', '0826',
            '0827', '0828', '0831', '0901', '0902', '0903', '0904', '0907', '0908', '0909',
            '0910', '0911', '0914', '0915', '0916', '0917', '0918', '0921', '0922', '0923',
            '0924', '0925', '0928', '0928', '0928', '0929', '0930', '1001', '1002', '1005',
            '1006', '1007', '1008', '1009', '1012', '1013', '1014', '1015', '1016', '1019',
            '1020', '1021', '1022', '1023', '1026', '1027', '1028', '1029', '1030', '1102',
            '1103', '1104', '1105', '1106', '1109', '1110', '1111', '1112', '1113', '1116',
            '1117', '1118', '1119', '1120', '1123', '1124', '1125', '1126', '1127', '1130',
            '1201', '1202', '1203', '1204', '1207', '1208', '1209', '1210', '1211', '1214',
            '1215', '1216', '1217', '1218', '1221', '1222', '1223', '1224', '1225', '1228',
            '1229', '1230', '1231']

    YY10 = ['0104', '0105', '0106', '0107', '0108', '0111', '0112', '0113', '0114', '0115',
            '0118', '0119', '0120', '0121', '0122', '0125', '0126', '0127', '0128', '0129',
            '0201', '0202', '0203', '0204', '0205', '0206', '0208', '0209', '0210', '0222',
            '0223', '0224', '0225', '0226', '0301', '0302', '0303', '0304', '0305', '0308',
            '0309', '0310', '0311', '0312', '0315', '0316', '0317', '0318', '0319', '0322',
            '0323', '0324', '0325', '0326', '0329', '0330', '0331', '0401', '0402', '0406',
            '0407', '0408', '0409', '0412', '0413', '0414', '0415', '0416', '0419', '0420',
            '0421', '0422', '0423', '0426', '0427', '0428', '0429', '0430', '0503', '0504',
            '0505', '0506', '0507', '0510', '0511', '0512', '0513', '0514', '0517', '0518',
            '0519', '0520', '0521', '0524', '0525', '0526', '0527', '0528', '0531', '0601',
            '0602', '0603', '0604', '0607', '0608', '0609', '0610', '0611', '0614', '0615',
            '0617', '0618', '0621', '0622', '0623', '0624', '0625', '0628', '0629', '0630',
            '0701', '0702', '0705', '0706', '0707', '0708', '0709', '0712', '0713', '0714',
            '0715', '0716', '0719', '0720', '0721', '0722', '0723', '0726', '0727', '0728',
            '0729', '0730', '0802', '0803', '0804', '0805', '0806', '0809', '0810', '0811',
            '0812', '0813', '0816', '0817', '0818', '0819', '0820', '0823', '0824', '0825',
            '0826', '0827', '0830', '0831', '0901', '0902', '0903', '0906', '0907', '0908',
            '0909', '0910', '0913', '0914', '0915', '0916', '0917', '0920', '0921', '0923',
            '0924', '0927', '0928', '0929', '0930', '1001', '1004', '1005', '1006', '1007',
            '1008', '1011', '1012', '1013', '1014', '1015', '1018', '1019', '1020', '1021',
            '1022', '1025', '1026', '1027', '1028', '1029', '1101', '1102', '1103', '1104',
            '1105', '1108', '1109', '1110', '1111', '1112', '1115', '1116', '1117', '1118',
            '1119', '1122', '1123', '1124', '1125', '1126', '1129', '1130', '1201', '1202',
            '1203', '1206', '1207', '1208', '1209', '1210', '1213', '1214', '1215', '1216',
            '1217', '1220', '1221', '1222', '1223', '1224', '1227', '1228', '1229', '1230',
            '1231']

    YY11 = ['0103', '0104', '0105', '0106', '0107', '0110', '0111', '0112', '0113', '0114',
            '0117', '0118', '0119', '0120', '0121', '0124', '0125', '0126', '0127', '0128',
            '0208', '0209', '0210', '0211', '0214', '0215', '0216', '0217', '0218', '0221',
            '0222', '0223', '0224', '0225', '0301', '0302', '0303', '0304', '0307', '0308',
            '0309', '0310', '0311', '0314', '0315', '0316', '0317', '0318', '0321', '0322',
            '0323', '0324', '0325', '0328', '0329', '0330', '0331', '0401', '0406', '0407',
            '0408', '0411', '0412', '0413', '0414', '0415', '0418', '0419', '0420', '0421',
            '0422', '0425', '0426', '0427', '0428', '0429', '0503', '0504', '0505', '0506',
            '0509', '0510', '0511', '0512', '0513', '0516', '0517', '0518', '0519', '0520',
            '0523', '0524', '0525', '0526', '0527', '0530', '0531', '0601', '0602', '0603',
            '0607', '0608', '0609', '0610', '0610', '0613', '0614', '0615', '0616', '0617',
            '0620', '0621', '0622', '0623', '0624', '0627', '0628', '0629', '0630', '0701',
            '0704', '0705', '0706', '0707', '0708', '0711', '0712', '0713', '0714', '0715',
            '0718', '0719', '0720', '0721', '0722', '0725', '0726', '0727', '0728', '0729',
            '0801', '0802', '0803', '0804', '0805', '0808', '0809', '0810', '0811', '0812',
            '0815', '0816', '0817', '0818', '0819', '0822', '0823', '0824', '0825', '0826',
            '0829', '0830', '0831', '0901', '0902', '0905', '0906', '0907', '0908', '0909',
            '0913', '0914', '0915', '0916', '0919', '0920', '0921', '0922', '0923', '0926',
            '0927', '0928', '0929', '0930', '1003', '1004', '1005', '1006', '1007', '1011',
            '1012', '1013', '1014', '1017', '1018', '1019', '1020', '1021', '1024', '1025',
            '1026', '1027', '1028', '1031', '1101', '1102', '1103', '1104', '1107', '1108',
            '1109', '1110', '1111', '1114', '1115', '1116', '1117', '1118', '1121', '1122',
            '1123', '1124', '1125', '1128', '1129', '1130', '1201', '1202', '1205', '1206',
            '1207', '1208', '1209', '1212', '1213', '1214', '1215', '1216', '1219', '1220',
            '1221', '1222', '1223', '1226', '1227', '1228', '1229', '1230']

    YY12 = ['0102', '0103', '0104', '0105', '0106', '0109', '0110', '0111', '0112', '0113',
            '0116', '0117', '0118', '0130', '0131', '0201', '0202', '0203', '0204', '0206',
            '0207', '0208', '0209', '0210', '0213', '0214', '0215', '0216', '0217', '0220',
            '0221', '0222', '0223', '0224', '0229', '0301', '0302', '0303', '0305', '0306',
            '0307', '0308', '0309', '0312', '0313', '0314', '0315', '0316', '0319', '0320',
            '0321', '0322', '0323', '0326', '0327', '0328', '0329', '0330', '0402', '0403',
            '0405', '0406', '0409', '0410', '0411', '0412', '0413', '0416', '0417', '0418',
            '0419', '0420', '0423', '0424', '0425', '0426', '0427', '0430', '0502', '0503',
            '0504', '0507', '0508', '0509', '0510', '0511', '0514', '0515', '0516', '0517',
            '0518', '0521', '0522', '0523', '0524', '0525', '0528', '0529', '0530', '0531',
            '0601', '0604', '0605', '0606', '0607', '0608', '0611', '0612', '0613', '0614',
            '0615', '0618', '0619', '0620', '0621', '0622', '0625', '0626', '0627', '0628',
            '0629', '0702', '0703', '0704', '0705', '0706', '0709', '0710', '0711', '0712',
            '0713', '0716', '0717', '0718', '0719', '0720', '0723', '0724', '0725', '0726',
            '0727', '0730', '0731', '0801', '0803', '0806', '0807', '0808', '0809', '0810',
            '0813', '0814', '0815', '0816', '0817', '0820', '0821', '0822', '0823', '0824',
            '0827', '0828', '0829', '0830', '0831', '0903', '0904', '0905', '0906', '0907',
            '0910', '0911', '0912', '0913', '0914', '0917', '0918', '0919', '0920', '0921',
            '0924', '0925', '0926', '0927', '0928', '1001', '1002', '1003', '1004', '1005',
            '1008', '1009', '1011', '1012', '1015', '1016', '1017', '1018', '1019', '1022',
            '1023', '1024', '1025', '1026', '1029', '1030', '1031', '1101', '1102', '1105',
            '1106', '1107', '1108', '1109', '1112', '1113', '1114', '1115', '1116', '1119',
            '1120', '1121', '1122', '1123', '1126', '1127', '1128', '1129', '1130', '1203',
            '1204', '1205', '1206', '1207', '1210', '1211', '1212', '1213', '1214', '1217',
            '1218', '1219', '1220', '1221', '1222', '1224', '1225', '1226', '1227', '1228']

    YY15 = ['0105', '0106', '0107', '0108', '0109', '0112', '0113', '0114', '0115', '0116',    # 1
            '0119', '0120', '0121', '0122', '0123', '0126', '0127', '0128', '0129', '0130',    # 2
            '0202', '0203', '0204', '0205', '0206', '0209', '0210', '0211', '0212', '0213',    # 3
            '0224', '0225', '0226', '0302', '0303', '0304', '0305', '0306', '0309', '0310',    # 4
            '0311', '0312', '0313', '0316', '0317', '0318', '0319', '0320', '0323', '0324',    # 5
            '0325', '0326', '0327', '0330', '0331', '0401', '0402', '0407', '0408', '0409',    # 6
            '0410', '0413', '0414', '0415', '0416', '0417', '0420', '0421', '0422', '0423',    # 7
            '0424', '0427', '0428', '0429', '0430', '0504', '0505', '0506', '0507', '0508',    # 8
            '0511', '0512', '0513', '0514', '0515', '0518', '0519', '0520', '0521', '0522',    # 9
            '0525', '0526', '0527', '0528', '0529', '0601', '0602', '0603', '0604', '0605',    # 10
            '0608', '0609', '0610', '0611', '0612', '0615', '0616', '0617', '0618', '0622',    # 11
            '0623', '0624', '0625', '0626', '0629', '0630', '0701', '0702', '0703', '0706',    # 12
            '0707', '0708', '0709', '0713', '0714', '0715', '0716', '0717', '0720', '0721',    # 13
            '0722', '0723', '0724', '0727', '0728', '0729', '0730', '0731', '0803', '0804',    # 14
            '0805', '0806', '0807', '0810', '0811', '0812', '0813', '0814', '0817', '0818',    # 15
            '0819', '0820', '0821', '0824', '0825', '0826', '0827', '0828', '0831', '0901',    # 16
            '0902', '0903', '0904', '0907', '0908', '0909', '0910', '0911', '0914', '0915',    # 17
            '0916', '0917', '0918', '0921', '0922', '0923', '0924', '0925', '0930', '1001',    # 18
            '1002', '1005', '1006', '1007', '1008', '1012', '1013', '1014', '1015', '1016',    # 19
            '1019', '1020', '1021', '1022', '1023', '1026', '1027', '1028', '1029', '1030',    # 20
            '1102', '1103', '1104', '1105', '1106', '1109', '1110', '1111', '1112', '1113',    # 21
            '1116', '1117', '1118', '1119', '1120', '1123', '1124', '1125', '1126', '1127',    # 22
            '1130', '1201', '1202', '1203', '1204', '1207', '1208', '1209', '1210', '1211',    # 23
            '1214', '1215', '1216', '1217', '1218', '1221', '1222', '1223', '1224', '1225',    # 24
            '1228', '1229', '1230', '1231']

    YYALL = ['1218', '1219', '1220', '1221', '1224', '1225', '1226', '1227', '1228', '1231',

             '0102', '0103', '0104', '0107', '0108', '0109', '0110', '0111', '0114', '0115',
             '0116', '0117', '0118', '0121', '0122', '0123', '0124', '0125', '0128', '0129',
             '0130', '0131', '0201', '0212', '0213', '0214', '0215', '0218', '0219', '0220',
             '0221', '0222', '0225', '0226', '0227', '0229', '0303', '0304', '0305', '0306',
             '0307', '0310', '0311', '0312', '0313', '0314', '0317', '0318', '0319', '0320',
             '0321', '0324', '0325', '0326', '0327', '0328', '0331', '0401', '0402', '0403',
             '0407', '0408', '0409', '0410', '0411', '0414', '0415', '0416', '0417', '0418',
             '0421', '0422', '0423', '0424', '0425', '0428', '0429', '0430', '0502', '0505',
             '0506', '0507', '0508', '0509', '0512', '0513', '0514', '0515', '0516', '0519',
             '0520', '0521', '0522', '0523', '0526', '0527', '0528', '0529', '0530', '0602',
             '0603', '0604', '0605', '0606', '0609', '0610', '0611', '0612', '0613', '0616',
             '0617', '0618', '0619', '0620', '0623', '0624', '0625', '0626', '0627', '0630',  # 1
             '0701', '0702', '0703', '0704', '0707', '0708', '0709', '0710', '0711', '0714',
             '0715', '0716', '0717', '0718', '0721', '0722', '0723', '0724', '0725', '0729',
             '0730', '0731', '0801', '0804', '0805', '0806', '0807', '0808', '0811', '0812',
             '0813', '0814', '0815', '0818', '0819', '0820', '0821', '0822', '0825', '0826',
             '0827', '0828', '0829', '0901', '0902', '0903', '0904', '0905', '0908', '0909',
             '0910', '0911', '0912', '0915', '0916', '0917', '0918', '0919', '0922', '0923',  # 2
             '0924', '0925', '0926', '0930', '1001', '1002', '1003', '1006', '1007', '1008',
             '1009', '1013', '1014', '1015', '1016', '1017', '1020', '1021', '1022', '1023',
             '1024', '1027', '1028', '1029', '1030', '1031', '1103', '1104', '1105', '1106',
             '1107', '1110', '1111', '1112', '1113', '1114', '1117', '1118', '1119', '1120',
             '1121', '1124', '1125', '1126', '1127', '1128', '1201', '1202', '1203', '1204',
             '1205', '1208', '1209', '1210', '1211', '1212', '1215', '1216', '1217', '1218',  # 3
             '1219', '1222', '1223', '1224', '1225', '1226', '1229', '1230', '1231',

             '0105', '0106', '0107', '0108', '0109', '0110', '0112', '0113', '0114', '0115',
             '0116', '0117', '0119', '0120', '0121', '0202', '0203', '0204', '0205', '0206',
             '0209', '0210', '0211', '0212', '0213', '0216', '0217', '0218', '0219', '0220',
             '0223', '0224', '0225', '0226', '0227', '0302', '0303', '0304', '0305', '0306',
             '0309', '0310', '0311', '0312', '0313', '0316', '0317', '0318', '0319', '0320',
             '0323', '0324', '0325', '0326', '0327', '0330', '0331', '0401', '0402', '0403',
             '0406', '0407', '0408', '0409', '0410', '0413', '0414', '0415', '0416', '0417',
             '0420', '0421', '0422', '0423', '0424', '0427', '0428', '0429', '0430', '0504',
             '0505', '0506', '0507', '0508', '0511', '0512', '0513', '0514', '0515', '0518',
             '0519', '0520', '0521', '0522', '0525', '0526', '0527', '0601', '0602', '0603',
             '0604', '0605', '0606', '0608', '0609', '0610', '0611', '0612', '0615', '0616',
             '0617', '0618', '0619', '0622', '0623', '0624', '0625', '0626', '0629', '0630',
             '0701', '0702', '0703', '0706', '0707', '0708', '0709', '0710', '0713', '0714',
             '0715', '0716', '0717', '0720', '0721', '0722', '0723', '0724', '0727', '0728',
             '0729', '0730', '0731', '0803', '0804', '0805', '0806', '0810', '0811', '0812',
             '0813', '0814', '0817', '0818', '0819', '0820', '0821', '0824', '0825', '0826',
             '0827', '0828', '0831', '0901', '0902', '0903', '0904', '0907', '0908', '0909',
             '0910', '0911', '0914', '0915', '0916', '0917', '0918', '0921', '0922', '0923',
             '0924', '0925', '0928', '0928', '0928', '0929', '0930', '1001', '1002', '1005',
             '1006', '1007', '1008', '1009', '1012', '1013', '1014', '1015', '1016', '1019',
             '1020', '1021', '1022', '1023', '1026', '1027', '1028', '1029', '1030', '1102',
             '1103', '1104', '1105', '1106', '1109', '1110', '1111', '1112', '1113', '1116',
             '1117', '1118', '1119', '1120', '1123', '1124', '1125', '1126', '1127', '1130',
             '1201', '1202', '1203', '1204', '1207', '1208', '1209', '1210', '1211', '1214',
             '1215', '1216', '1217', '1218', '1221', '1222', '1223', '1224', '1225', '1228',
             '1229', '1230', '1231',

             '0104', '0105', '0106', '0107', '0108', '0111', '0112', '0113', '0114', '0115',
             '0118', '0119', '0120', '0121', '0122', '0125', '0126', '0127', '0128', '0129',
             '0201', '0202', '0203', '0204', '0205', '0206', '0208', '0209', '0210', '0222',
             '0223', '0224', '0225', '0226', '0301', '0302', '0303', '0304', '0305', '0308',
             '0309', '0310', '0311', '0312', '0315', '0316', '0317', '0318', '0319', '0322',
             '0323', '0324', '0325', '0326', '0329', '0330', '0331', '0401', '0402', '0406',
             '0407', '0408', '0409', '0412', '0413', '0414', '0415', '0416', '0419', '0420',
             '0421', '0422', '0423', '0426', '0427', '0428', '0429', '0430', '0503', '0504',
             '0505', '0506', '0507', '0510', '0511', '0512', '0513', '0514', '0517', '0518',
             '0519', '0520', '0521', '0524', '0525', '0526', '0527', '0528', '0531', '0601',
             '0602', '0603', '0604', '0607', '0608', '0609', '0610', '0611', '0614', '0615',
             '0617', '0618', '0621', '0622', '0623', '0624', '0625', '0628', '0629', '0630',
             '0701', '0702', '0705', '0706', '0707', '0708', '0709', '0712', '0713', '0714',
             '0715', '0716', '0719', '0720', '0721', '0722', '0723', '0726', '0727', '0728',
             '0729', '0730', '0802', '0803', '0804', '0805', '0806', '0809', '0810', '0811',
             '0812', '0813', '0816', '0817', '0818', '0819', '0820', '0823', '0824', '0825',
             '0826', '0827', '0830', '0831', '0901', '0902', '0903', '0906', '0907', '0908',
             '0909', '0910', '0913', '0914', '0915', '0916', '0917', '0920', '0921', '0923',
             '0924', '0927', '0928', '0929', '0930', '1001', '1004', '1005', '1006', '1007',
             '1008', '1011', '1012', '1013', '1014', '1015', '1018', '1019', '1020', '1021',
             '1022', '1025', '1026', '1027', '1028', '1029', '1101', '1102', '1103', '1104',
             '1105', '1108', '1109', '1110', '1111', '1112', '1115', '1116', '1117', '1118',
             '1119', '1122', '1123', '1124', '1125', '1126', '1129', '1130', '1201', '1202',
             '1203', '1206', '1207', '1208', '1209', '1210', '1213', '1214', '1215', '1216',
             '1217', '1220', '1221', '1222', '1223', '1224', '1227', '1228', '1229', '1230',
             '1231',

             '0103', '0104', '0105', '0106', '0107', '0110', '0111', '0112', '0113', '0114',
             '0117', '0118', '0119', '0120', '0121', '0124', '0125', '0126', '0127', '0128',
             '0208', '0209', '0210', '0211', '0214', '0215', '0216', '0217', '0218', '0221',
             '0222', '0223', '0224', '0225', '0301', '0302', '0303', '0304', '0307', '0308',
             '0309', '0310', '0311', '0314', '0315', '0316', '0317', '0318', '0321', '0322',
             '0323', '0324', '0325', '0328', '0329', '0330', '0331', '0401', '0406', '0407',
             '0408', '0411', '0412', '0413', '0414', '0415', '0418', '0419', '0420', '0421',
             '0422', '0425', '0426', '0427', '0428', '0429', '0503', '0504', '0505', '0506',
             '0509', '0510', '0511', '0512', '0513', '0516', '0517', '0518', '0519', '0520',
             '0523', '0524', '0525', '0526', '0527', '0530', '0531', '0601', '0602', '0603',
             '0607', '0608', '0609', '0610', '0610', '0613', '0614', '0615', '0616', '0617',
             '0620', '0621', '0622', '0623', '0624', '0627', '0628', '0629', '0630', '0701',
             '0704', '0705', '0706', '0707', '0708', '0711', '0712', '0713', '0714', '0715',
             '0718', '0719', '0720', '0721', '0722', '0725', '0726', '0727', '0728', '0729',
             '0801', '0802', '0803', '0804', '0805', '0808', '0809', '0810', '0811', '0812',
             '0815', '0816', '0817', '0818', '0819', '0822', '0823', '0824', '0825', '0826',
             '0829', '0830', '0831', '0901', '0902', '0905', '0906', '0907', '0908', '0909',
             '0913', '0914', '0915', '0916', '0919', '0920', '0921', '0922', '0923', '0926',
             '0927', '0928', '0929', '0930', '1003', '1004', '1005', '1006', '1007', '1011',
             '1012', '1013', '1014', '1017', '1018', '1019', '1020', '1021', '1024', '1025',
             '1026', '1027', '1028', '1031', '1101', '1102', '1103', '1104', '1107', '1108',
             '1109', '1110', '1111', '1114', '1115', '1116', '1117', '1118', '1121', '1122',
             '1123', '1124', '1125', '1128', '1129', '1130', '1201', '1202', '1205', '1206',
             '1207', '1208', '1209', '1212', '1213', '1214', '1215', '1216', '1219', '1220',
             '1221', '1222', '1223', '1226', '1227', '1228', '1229', '1230',

             '0102', '0103', '0104', '0105', '0106', '0109', '0110', '0111', '0112', '0113',
             '0116', '0117', '0118', '0130', '0131', '0201', '0202', '0203', '0204', '0206',
             '0207', '0208', '0209', '0210', '0213', '0214', '0215', '0216', '0217', '0220',
             '0221', '0222', '0223', '0224', '0229', '0301', '0302', '0303', '0305', '0306',
             '0307', '0308', '0309', '0312', '0313', '0314', '0315', '0316', '0319', '0320',
             '0321', '0322', '0323', '0326', '0327', '0328', '0329', '0330', '0402', '0403',
             '0405', '0406', '0409', '0410', '0411', '0412', '0413', '0416', '0417', '0418',
             '0419', '0420', '0423', '0424', '0425', '0426', '0427', '0430', '0502', '0503',
             '0504', '0507', '0508', '0509', '0510', '0511', '0514', '0515', '0516', '0517',
             '0518', '0521', '0522', '0523', '0524', '0525', '0528', '0529', '0530', '0531',
             '0601', '0604', '0605', '0606', '0607', '0608', '0611', '0612', '0613', '0614',
             '0615', '0618', '0619', '0620', '0621', '0622', '0625', '0626', '0627', '0628',
             '0629', '0702', '0703', '0704', '0705', '0706', '0709', '0710', '0711', '0712',
             '0713', '0716', '0717', '0718', '0719', '0720', '0723', '0724', '0725', '0726',
             '0727', '0730', '0731', '0801', '0803', '0806', '0807', '0808', '0809', '0810',
             '0813', '0814', '0815', '0816', '0817', '0820', '0821', '0822', '0823', '0824',
             '0827', '0828', '0829', '0830', '0831', '0903', '0904', '0905', '0906', '0907',
             '0910', '0911', '0912', '0913', '0914', '0917', '0918', '0919', '0920', '0921',
             '0924', '0925', '0926', '0927', '0928', '1001', '1002', '1003', '1004', '1005',  # 19
             '1008', '1009', '1011', '1012', '1015', '1016', '1017', '1018', '1019', '1022',
             '1023', '1024', '1025', '1026', '1029', '1030', '1031', '1101', '1102', '1105',
             '1106', '1107', '1108', '1109', '1112', '1113', '1114', '1115', '1116', '1119',
             '1120', '1121', '1122', '1123', '1126', '1127', '1128', '1129', '1130', '1203',
             '1204', '1205', '1206', '1207', '1210', '1211', '1212', '1213', '1214', '1217',
             '1218', '1219', '1220', '1221', '1222', '1224', '1225', '1226', '1227', '1228']  # 20


    '''
        start the program, e.q 2330
    '''

    for i in range(len(YYALL)):
        if i < 1261:
            year = '2012'
        if i < 1011:
            year = '2011'
        if i < 763:
            year = '2010'
        if i < 512:
            year = '2009'
        if i < 259:
            year = '2008'
        if i < 10:
            year = '2007'
        mthProcess1(year, YYALL[i], '2330')
        mthProcess2(year, YYALL[i], '2330')
        mthProcess3(year, YYALL[i], '2330')
        print(i)

    '''
        merge the data to conduct the S1.txt, we need to adjust the loop para until conduct the S20.txt
    '''
    for i in range(0, 130):
        if i < 1261:
            year = '2012'
        if i < 1011:
            year = '2011'
        if i < 763:
            year = '2010'
        if i < 512:
            year = '2009'
        if i < 259:
            year = '2008'
        if i < 10:
            year = '2007'

        txtMerger(year, YYALL[i], '1', '2308')
        txtMerger(year, YYALL[i], '1', '2330')
        print(i)


    for i in range(1,21):
        VPIN_Change(120, str(i), '2338')
        VPIN2(str(i), '2330')
        MVM(3, str(i), '2330')
        MVM(5, str(i), '2330')
        MVM(10, str(i), '2330')
        momentum(str(i), '2330')
        print(str(i) + '------- 2330')

    '''
        running the ml models
    '''
    ensemble()




    # under is process the 2338 stock num, because there are many days do not have buy or sell data
    '''
    n = 0
    for i in range(1140, len(YYALL)):
        if i < 10:
            n = 0
        elif i < 259:
            n = len(YY07)
            # 2008
            if i == 39 + n or i == 45 + n or i == 47 + n or i == 222 + n or i == 223 + n or i == 232 + n:
                continue
            elif (i >= 233 + n and i <= 240 + n) or (i >= 243 + n and i <= 245 + n):
                continue
        elif i < 512:
            n = len(YY07) + len(YY08)
            # 2009
            if (i >= 4 + n and i <= 29 + n) or i == 160 + n:
                continue
        elif i < 763:
            n = len(YY07) + len(YY08) + len(YY09)
            # 2010
            if (i >= 172 + n and i <= 182 + n) or i == 218 + n or i == 219 + n:
                continue
        elif i < 1011:
            n = len(YY07) + len(YY08) + len(YY09) + len(YY10)
            # 2011
            if i == 62 + n or i == 95 + n or i == 106 + n or i == 113 + n or i == 129 + n or i == 141 + n or i == 153 + n:
                continue
            elif i == 157 + n or i == 158 + n or i == 166 + n or i == 179 + n or i == 181 + n or i == 189 + n or i == 191 + n:
                continue
            elif i == 194 + n or i == 196 + n or i == 199 + n or i == 205 + n or i == 206 + n or i == 208 + n or i == 210 + n or i == 214 + n:
                continue
            elif i == 217 + n or i == 228 + n or i == 229 + n or i == 232 + n or i == 237 + n or i == 241 + n:
                continue

        elif i < 1261:
            n = len(YY07) + len(YY08) + len(YY09) + len(YY10) + len(YY11)
            # 2012
            if i == 0 + n or i == 5 + n or i == 6 + n or i == 7 + n or i == 10 + n or i == 30 + n or i == 35 + n or i == 37 + n or i == 41 + n or i == 49 + n:
                continue
            elif i == 53 + n or i == 55 + n or i == 55 + n or (i >= 73 + n and i <= 77 + n) or i == 80 + n or i == 81 + n or (i >= 84 + n and i <= 86 + n):
                continue
            elif i == 91 + n or i == 92 + n or i == 94 + n or i == 95 + n or i == 98 + n or i == 100 + n or (i >= 102 + n and i <= 118 + n) or i == 120 + n:
                continue
            elif (i >= 124 + n and i <= 132 + n) or i == 135 + n or i == 136 + n or i == 141 + n or i == 143 + n or i == 185 + n or i == 187 + n or i == 188 + n or i == 190 + n:
                continue
            elif (i >= 193 + n and i <= 196 + n) or i == 199 + n or i == 200 + n or i == 204 + n or (i >= 206 + n and i <= 219 + n) or i == 227 + n:
                continue
            elif i == 233 + n or i == 234 + n or i == 235 + n or i == 238 + n:
                continue


        if i < 1261:
            year = '2012'
        if i < 1011:
            year = '2011'
        if i < 763:
            year = '2010'
        if i < 512:
            year = '2009'
        if i < 259:
            year = '2008'
        if i < 10:
            year = '2007'

        txtMerger(year, YYALL[i], '20', '2338')
        #txtMerger(year, YYALL[i], '20', '2324')
        #txtMerger(year, YYALL[i], '20', '2330')
        #txtMerger(year, YYALL[i], '20', '2377')
        #txtMerger(year, YYALL[i], '20', '3481')
        print(i)
    '''
    '''
    for i in range(6, 21):
        VPIN(120, str(i), '2330')
        print(i)

    '''

    '''
    print(len(YY08))
    print(len(YY09))
    print(len(YY10))
    print(len(YY11))
    print(len(YY12))
    print(len(YY07)+len(YY08)+len(YY09)+len(YY10)+len(YY11)+len(YY12))
    print(len(YYALL))
    '''

    ''' 2338 mthProcess 1-3'''
    '''
    n = -1
    for i in range(len(YY07)):
        n += 1
        #mthProcess1('2007', YY07[i], '2338')
        mthProcess2('2007', YY07[i], '2338')
        mthProcess3('2007', YY07[i], '2338')
        print(i)

    # 39 45 47 222 223 232 233-240 243-245no data bug

    for i in range(len(YY08)):
        n += 1
        if i == 39 or i == 45 or i == 47 or i == 222 or i == 223 or i == 232:
            continue
        if (i >= 233 and i <= 240) or (i >= 243 and i <= 245):
            continue
        #mthProcess1('2008', YY08[i], '2338')
        mthProcess2('2008', YY08[i], '2338')
        mthProcess3('2008', YY08[i], '2338')
        print(i)
    
    # 4-29  160data bug
    for i in range(len(YY09)):
        #n += 1
        if (i >=4 and i <= 29) or i == 160:
            continue
        #mthProcess1('2009', YY09[i], '2338')
        mthProcess2('2009', YY09[i], '2338')
        mthProcess3('2009', YY09[i], '2338')
        print(i)

    #print(n)
    
    # 172 - 182  218-219 no data bug

    for i in range(len(YY10)):

        if (i >= 172 and i <= 182) or i == 218 or i == 219:
            continue
        #mthProcess1('2010', YY10[i], '2338')
        mthProcess2('2010', YY10[i], '2338')
        mthProcess3('2010', YY10[i], '2338')
        print(i)
    

    # 62 95 106 113 129 141 153 157-158 166 179 181 189 191 194 196 199 205-206 208 210 214  217 228-229 232 237 241  no data bug
    for i in range(len(YY11)):
        if i == 62 or i == 95 or i == 106 or i == 113 or i == 129 or i == 141 or i == 153:
            continue
        elif i == 157 or i == 158 or i == 166 or i == 179 or i == 181 or i == 189 or i == 191:
            continue
        elif i == 194 or i == 196 or i == 199 or i == 205 or i == 206 or i == 208 or i == 210 or i == 214:
            continue
        elif i == 217 or i == 228 or i == 229 or i == 232 or i == 237 or i == 241:
            continue
        #mthProcess1('2011', YY11[i], '2338')
        mthProcess2('2011', YY11[i], '2338')
        mthProcess3('2011', YY11[i], '2338')
        print(i)
   
    # 0 5-7 10 30 35 37 41 49 53 55 73-77 80-81 84-86 91-92 94-95 98 100 102-118 120 124-132 135-136 141no data bug
    # 143 185 187 188 190 193-196 199-200 204 206-219 227 233-235 238

    for i in range(len(YY12)):
        if i == 0 or i == 5 or i== 6 or i == 7 or i == 10 or i == 30 or i == 35 or i == 37 or i == 41 or i == 49:
            continue
        elif i == 53 or i == 55 or i == 55 or (i >= 73 and i <= 77) or i == 80 or i == 81 or (i >= 84 and i <= 86):
            continue
        elif i == 91 or i == 92 or i == 94 or i == 95 or i == 98 or i == 100 or (i >= 102 and i <= 118) or i == 120:
            continue
        elif (i >= 124 and i <= 132) or i == 135 or i == 136 or i == 141 or i == 143 or i == 185 or i == 187 or i == 188 or i == 190:
            continue
        elif (i >= 193 and i <= 196) or i == 199 or i == 200 or i == 204 or (i >= 206 and i <= 219) or i == 227:
            continue
        elif i == 233 or i == 234 or i == 235 or i == 238:
            continue
        #mthProcess1('2012', YY12[i], '2338')
        mthProcess2('2012', YY12[i], '2338')
        mthProcess3('2012', YY12[i], '2338')
        print(i)
    '''

    '''
    for i in range(1,21):
        VPIN(120, str(i), '2338')
        print(str(i) + '------- 2338')
    '''


    print(1)
