import datetime
import numpy as np
from main import trainTest
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import denoise
from weekdeal import weekInYear
import os
import drawing

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
    attCnt = param_m + param_l
    if option.startswith("type1"):
        attCnt = param_m
    return dataFile,attCnt,resFile

def alignDate(wti_date,wti_value,senDate):
    ret_date = []
    ret_value = []
    for i in range(0,len(wti_date)):
        if wti_date[i] in senDate:
            ret_date.append(wti_date[i])
            ret_value.append(wti_value[i])
    return  ret_date,ret_value

def denoiseData(datalist):
    mat = []
    for i in range(0,len(datalist)):
        data = datalist[i]
        nlist = []
        nlist.append(data)
        mat.append(nlist)
    rawret = denoise.DenoisMat(mat)
    ret = [x[0] for x in rawret]
    return ret



def dataPre(docCnt,option,figureHelper):
    inPathHead = "C:/Users/user/Documents/YJS/国储局Ⅱ期/实验/reuters/reuters_"
    wtiPath = "wti-daliy-use.csv"
    picPath = "pic.png"
    outBasicPath = "wti-senti.csv"
    picPath_wti = "pic-wti.png"


    isWeek = False
    if option.endswith("week"):
        isWeek = True
        wtiPath = "wti-week-use.csv"
        picPath = "pic-week.png"
        picPath_wti = "pic-week-wti.png"
        outBasicPath = "wti-senti-week.csv"

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
##遍历所有文档，计算每个文档的情绪词个数，创建文档对象将该文档日期加入日期集合；
        if not(timeobj.weekday()==5 or timeobj.weekday()==6):
            posCnt = getsenticnt(posList,content)
            negCnt = getsenticnt(negList, content)
            doc = Documentt(time,posCnt,negCnt)
            doclist.append(doc)
            if isWeek:
                if timeobj.weekday()==4:
                    dateset.add(time)
            else:
                dateset.add(time)

#  遍历日期集合，每一个日期寻找所有对应的文档，将个数相加，计算出每个日期的情绪值，保存在最终结果中，并按日期排序
#  按周思路：日期集合只包含所有的周中的一天，如周五，寻找时寻找该周五对应的所有周一~周四
    for date in dateset:
        suball = 0
        addall = 0
        for doc in doclist:
            if isWeek:
                if weekInYear(doc.date) == weekInYear(date):
                    suball = suball+doc.sub
                    addall = addall+doc.add
            else:
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


## 处理wti序列
    tmp = np.loadtxt(wtiPath, dtype=np.str, delimiter=",")
    wti_date = tmp[1:, 0].astype(np.str)  # 加载日期部分
    wti_value = tmp[1:, 1].astype(np.float)  # 加载数据部分

    wti_time = []
    for i in range(0,len(wti_date)):
        timeobj = datetime.datetime.strptime(wti_date[i], "%Y/%m/%d")
        wti_time.append(timeobj.strftime("%Y/%m/%d"))
    wti_date = wti_time

## 日期对齐
    wti_date,wti_value = alignDate(wti_date,wti_value,senDate)
    senDate, senValue = alignDate(senDate, senValue, wti_date)

    wti_timeobj = []
    for i in range(0, len(wti_date)):
        timeobj = datetime.datetime.strptime(wti_date[i], "%Y/%m/%d")
        wti_timeobj.append(timeobj)


    figureHelper.linePlot([wti_value], ["wti"], wti_timeobj, picPath_wti,xLable="date",yLable="wti oil price")

## 两序列标准化
    wti_value = sklearn.preprocessing.scale(wti_value)
    senValue = sklearn.preprocessing.scale(senValue)

    # wti_value = wti_value.reshape(1, -1)
    # senValue = senValue.reshape(1, -1)


    ## 作图 同时打印输出两序列
    if option.startswith("draw"):
        figureHelper.linePlot([wti_value,senValue],["wti","sentiment"],wti_timeobj,picPath)
        fdata = open(outBasicPath, 'w')
        print("date,wti,sentiment",file=fdata)
        for i in range(0,len(wti_date)):
            print(wti_date[i]+","+str(wti_value[i])+","+str(senValue[i]), file=fdata)
        fdata.close()

def loadHpData(option,figureHelper):
    hpPath = "wti-senti-hp.csv"
    picPath = "pic-hp.png"
    if option.endswith("week"):
        hpPath = "wti-senti-week-hp.csv"
        picPath = "pic-hp-week.png"
    tmp = np.loadtxt(hpPath, dtype=np.str, delimiter=",")
    wti_value = tmp[1:, 1].astype(np.float)  # 加载数据部分
    senValue = tmp[1:, 2].astype(np.float)  # 加载类别标签部分
    wti_date = tmp[1:, 0].astype(np.str)

    wti_timeobj = []
    for i in range(0, len(wti_date)):
        timeobj = datetime.datetime.strptime(wti_date[i], "%Y/%m/%d")
        wti_timeobj.append(timeobj)

    figureHelper.linePlot([wti_value, senValue], ["wti", "sentiment"], wti_timeobj, picPath)

    ## 得到wti标签序列
    wti_y = getUpDown(wti_value)

    return wti_value,wti_y,senValue

def dataOutput(docCnt,option,param_m,param_h,param_l,wti_value,wti_y,senValue):
    outPath = getFileName(docCnt,option,param_m,param_h,param_l)[0]
    f = open(outPath, 'w')
    if option.startswith("type1"):
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

    elif option.startswith("type2"):
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

def preAndTest(docCnt,option,param_m,param_h,param_l,test_classifiers,figureHelper):
    wti_value, wti_y, senValue = loadHpData(option,figureHelper)
    dataOutput(docCnt, option, param_m, param_h, param_l,wti_value, wti_y, senValue)
    dataFile, attCnt, outFile = getFileName(docCnt, option, param_m, param_h, param_l)
    trainTest(dataFile, attCnt, outFile,test_classifiers)


if __name__ == '__main__':
    figureHelper = drawing.FigureHelper()
    # test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'GBDT','SVM']
    # types = ["type1","type2","type1week","type2week"]
    # for type in types:
    #     preAndTest(7785, type, 4, 1, 3,test_classifiers,figureHelper)
    #     pass


    types2 = ["draw","drawweek"]
    for type2 in types2:
        dataPre(7785, type2,figureHelper)
        # loadHpData(type2,figureHelper)

