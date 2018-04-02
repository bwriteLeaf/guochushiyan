import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

class weekObj:
    def __init__(self,weekDate,weekValue,monthValue,zValue):
        self.weekDate=weekDate
        self.weekValue=weekValue
        self.monthValue = monthValue
        self.zValue = zValue
        self.res = weekValue*monthValue*zValue


tmp = np.loadtxt("co-all.csv", dtype=np.str, delimiter=",")
weekDate = tmp[0:, 0].astype(np.str)
weekValue = tmp[0:, 1].astype(np.float)

tmp = np.loadtxt("mon-all.csv", dtype=np.str, delimiter=",")
monthDate = tmp[0:, 0].astype(np.str)
monthValue = tmp[0:, 1].astype(np.float)

reslist=[]

for d in range(0,len(monthDate)):
    mon_date = monthDate[d]
    index = mon_date.rfind('/')
    mon_str = str(mon_date[0:index])
    weekNumlist = []
    weekall = 0
    weekcnt = 0
    for i in range(0,len(weekDate)):
        week_date = str(weekDate[i])
        index2 = week_date.rfind('/')
        week_str = str(week_date[0:index2])
        if week_str==mon_str:
            weekNumlist.append(i)
            weekall = weekall + weekValue[i]
            weekcnt = weekcnt +1
    if weekcnt >1:
        weekavg = weekall/weekcnt
        for j in weekNumlist:
            week_str = weekDate[j]
            w = weekObj(weekDate[j],weekValue[j],monthValue[d],1/weekavg)
            reslist.append(w)

for w in reslist:
    print(str(w.weekDate)+","+str(w.res))
