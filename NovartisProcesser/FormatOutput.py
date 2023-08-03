# -*- coding: utf-8 -*-

import prettytable as pt

dic1 = {'预期寿命' : '>9个月@是由',
'ANC' : '≥1.5 x 109/L@，血小板≥100',
'血小板' : '≥100 x 109/L@，血红蛋白≥9',
'肝总胆红素' : '≤2 xULN@.对于患有',
'综合征' : '≤3 xULN@允许的丙氨酸氨基转移酶',
'肾eGFR' : '≥50 mL/min@/1.73 m2',
'白蛋白' : '≥2.5 g/dL@.人类免疫缺陷病毒'}

def formatdic(dic):
    tb = pt.PrettyTable()
    for key in dic1:
        tb.add_row([key, dic1[key]])
    print(tb)

def formatlist1(lis):
    tb = pt.PrettyTable()
    names = ['English name', 'Chinese name', 'Criteria', 'Information']
    tb.field_names = names
    tb._max_width = {'Information' : 60}
    for item in lis:
        store = list()
        for (i, key) in enumerate(item):
            store.append(item[key])
        if(i!=3):
            store.append('')
        tb.add_row(store)
    print(tb)

def formatlist2(lis):
    tb = pt.PrettyTable()
    names = ['SQL', 'Information']
    tb.field_names = names
    for item in lis:
        store = list()
        for (i, key) in enumerate(item):
            store.append(item[key])
        if(i!=1):
            store.append('')
        tb.add_row(store)
    print(tb)
