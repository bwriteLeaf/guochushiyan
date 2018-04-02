import datetime
import numpy as np
from main import trainTest
import sklearn
import matplotlib
import matplotlib.pyplot as plt

class Documentt:
    def __init__(self, date, posCnt,negCnt):
        self.date = date
        self.posCnt = posCnt
        self.negCnt = negCnt
        self.sub = self.posCnt-self.negCnt
        self.add = self.posCnt+self.negCnt

class DaySenti:
    def __init__(self, date,senti):
        self.date = date
        self.senti = senti

def getsenticnt(wordlist,content):
    cnt = 0
    for senword in wordlist:
        if content.find(senword) != -1:
            cnt = cnt+1
    return cnt

def getUpDown(wti_value):
    wti_y = []
    wti_y.append(0)
    for i in range(1,len(wti_value)): #注意差分序列的第一个值不能取
        if wti_value[i]<wti_value[i-1] :
            wti_y.append(0)
        else:
            wti_y.append(1)
    return wti_y

def getFileName(docCnt,option,param_m,param_h,param_l):
    dataHead = "data_"
    resHead = "res_"
    dataFile = dataHead+str(option)+"_"+str(docCnt)+"_"+str(param_m)+str(param_h)+str(param_l)+".csv"
    resFile = resHead+str(option)+"_"+str(docCnt)+"_"+str(param_m)+str(param_h)+str(param_l)+".csv"
    attCnt = param_m+param_l
    return dataFile,attCnt,resFile

def alignDate(wti_date,wti_value,senDate):
    ret_date = []
    ret_value = []
    for i in range(0,len(wti_date)):
        if wti_date[i] in senDate:
            ret_date.append(wti_date[i])
            ret_value.append(wti_value[i])
    return  ret_date,ret_value



def dataPre(docCnt,option,param_m,param_h,param_l):

    inPathHead = "C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/reuters/reuters_"
    outPath = getFileName(docCnt,option,param_m,param_h,param_l)[0]
    wtiPath = "wti-daliy-use.csv"

    posList = ["positive","positives","success","successes","successful","succeed","succeeds","succeeding",
               "succeeded","accomplish","accomplishes","accomplishing","accomplished","accomplishment",
               "accomplishments","strong","strength","strengths","certain","certainly","definite","solid","excellent",
               "good","leading","achieve","achieves","achieved","achieving","achievement","achievements",
               "progress","progressing","deliver","delivers","delivered","delivering","leader","leading","pleased",
               "reward","rewards","rewarding","rewarded","opportunity","opportunities","enjoy","enjoys",
               "enjoying","enjoyed","encouraged","encouraging","up","increase","increases","increasing",
               "increased","rise","rises","rising","rose","risen","improve","improves","improving","improved",
               "improvements","strengthen","strengthens","strengthening","strengthened","stronger","strongest",
               "better","best","more","most","above","record","high","higher","highest","greater","greatest","larger",
               "largest","grow","grows","growing","grew","grown","growth","expand","expands","expanding",
               "expanded","expansion","exceed","exceeded","exceeding","beat","beats","beating"]

    negList = ["negative","negatives","fail","fails","failing","failure","weak","weakness","weaknesses","difficult",
               "difficulty","hurdle","hurdles","obstacle","obstacles","slump","slumps","slumping","slumped",
               "uncertain","uncertainty","unsettled","unfavorable","downturn","depressed",
               "disappoint","disappoints","disappointing","disappointed","disappointment",
               "risk","risks","risky","threat","threats","penalty","penalties","down","decrease","decreases",
               "decreasing","decreased","decline","declines","declining","declined","fall","falls","falling",
               "fell","fallen","drop","drops","dropping","dropped","deteriorate","deteriorates",
               "deteriorating","deteriorated","worsen","worsens","worsening","weaken","weakens",
               "weakening","weakened","worse","worst","low","lower","lowest","less","least","smaller","smallest","shrink",
               "shrinks","shrinking","shrunk","below","under","challenge","challenges","challenging","challenged"]
    posCnt = 0
    negCnt = 0
    doclist = []
    dateset = set()
    sentilist = []
    for txtNum in range(0,docCnt):
        txtPath = inPathHead+str(txtNum)+".txt"
        #read txt method three
        f = open(txtPath,"r", encoding='UTF-8')
        lines = f.readlines()
        title = lines[0]
        date_str = lines[1]
        content = lines[2]
        f.close()

        timeobj = datetime.datetime.strptime(date_str, "%B %d, %Y\n")
        time = timeobj.strftime("%Y/%m/%d")

        if not(timeobj.weekday()==5 or timeobj.weekday()==6):
            posCnt = getsenticnt(posList,content)
            negCnt = getsenticnt(negList, content)
            doc = Documentt(time,posCnt,negCnt)
            doclist.append(doc)
            dateset.add(time)

    for date in dateset:
        suball = 0
        addall = 0
        for doc in doclist:
            if doc.date == date:
                suball = suball+doc.sub
                addall = addall+doc.add
        if addall == 0:
            print(date)
        else:
            senti = suball/addall
            robj = DaySenti(date,senti)
            sentilist.append(robj)

    sentilist = sorted(sentilist, key=lambda robj: robj.date)
    senValue = [robj.senti for robj in sentilist]
    senDate = [robj.date for robj in sentilist]
    senValue = sklearn.preprocessing.scale(senValue)

## 处理wti序列
    tmp = np.loadtxt(wtiPath, dtype=np.str, delimiter=",")
    wti_date = tmp[1:, 0].astype(np.str)  # 加载日期部分
    wti_value = tmp[1:, 1].astype(np.float)  # 加载数据部分

    wti_time = []
    for i in range(0,len(wti_date)):
        timeobj = datetime.datetime.strptime(wti_date[i], "%Y/%m/%d")
        wti_time.append(timeobj.strftime("%Y/%m/%d"))
    wti_date = wti_time

    wti_date,wti_value = alignDate(wti_date,wti_value,senDate)
    wti_y = getUpDown(wti_value)
    wti_value = sklearn.preprocessing.scale(wti_value)

    senDate,senValue=alignDate(senDate,senValue,wti_date)

    id = lineChartPlot([wti_value,senValue],["wti","sentiment"],wti_date)
    f = plt.figure(id)
    f.savefig('pic.png')

    f = open(outPath, 'w')
    if option == "type1":
        for j in range(0, param_m):
            print("x"+str(j)+',', file=f,end='')
        print("y", file=f)

        for i in range(param_m-1,len(wti_value)-param_h):
            ##变量值
            print(str(wti_value[i]), file=f,end='')
            for j in range(1,param_m):
                print(','+str(wti_value[i-j]), file=f,end='')

            ##标签
            print(','+str(wti_y[i+param_h]), file=f)

    elif option == "type2":
        for j in range(0, param_m+param_l):
            print("x" + str(j) + ',', file=f, end='')
        print("y", file=f)

        for i in range(max(param_m - 1,param_l-1), len(wti_value) - param_h):
            ##变量值
            print(str(wti_value[i]), file=f, end='') #xt
            for j in range(1, param_m):
                print(',' + str(wti_value[i - j]), file=f, end='')

            for k in range(0, param_l):#sl
                print(',' + str(senValue[i - k]), file=f, end='')

            ##标签
            print(',' + str(wti_y[i + param_h]), file=f)
    f.close()

def preAndTest(docCnt,option,param_m,param_h,param_l,test_classifiers):
    dataPre(docCnt, option, param_m, param_h, param_l)
    # dataFile, attCnt, outFile = getFileName(docCnt, option, param_m, param_h, param_l)
    # trainTest(dataFile, attCnt, outFile,test_classifiers)

# 绘制分组线图
    # dataList: 待绘制的数据集，为小数
    # dataLabelList: 与dataList相对应的标签，被用作图例
    # xAxisLabelList: x轴数据标签
    # xLable x轴标签
    # yLable y轴标签
def lineChartPlot(dataList, dataLabelList, xAxisLabelList, xLable=None, yLable=None):
    legends = []
    if len(dataList) == 0:
        return None

    case_cnt = len(dataList)

    for i in range(0, case_cnt):
        xs = xAxisLabelList
        ys = dataList[i]
        l = plt.plot(xs, ys, label=dataLabelList[i])
        legends.append(l)

    if case_cnt >= 2:
        plt.legend()

    if yLable is not None:
        plt.ylabel(yLable)
    if xLable is not None:
        plt.xlabel(xLable)

    return 1

if __name__ == '__main__':
    test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'GBDT']
    preAndTest(7785, "type2", 4, 2, 3,test_classifiers)
