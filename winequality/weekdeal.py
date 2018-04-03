# -*- coding=utf-8 -*-
import datetime


def week_get(vdate):
    dayscount = datetime.timedelta(days=vdate.isoweekday())
    dayfrom = vdate - dayscount + datetime.timedelta(days=1)
    dayto = vdate - dayscount + datetime.timedelta(days=7)
    print(' ~~ '.join([str(dayfrom), str(dayto)]))
    week7 = []
    i = 0
    while (i <= 6):
        week7.append('周' + str(i + 1) + ': ' + str(dayfrom + datetime.timedelta(days=i)))
        i += 1
    return week7


def weekInYear(vdate_str):
    date = vdate_str
    yearWeek = datetime.date(int(date[0:4]), int(date[5:7]), int(date[8:10])).isocalendar()[0:2]
    return str(yearWeek[0]) + '#' + str(yearWeek[1])


if __name__ == '__main__':
    vdate_str = '2018-12-30'
    vdate = datetime.datetime.strptime(vdate_str, '%Y-%m-%d')
    print(weekInYear(vdate_str))

    # for week in week_get(vdate):
    #     for weekYear in (weekInYear(vdate_str).split()):
    #         print(weekYear, week)