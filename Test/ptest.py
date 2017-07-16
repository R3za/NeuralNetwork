def testList(dictele):
    for k in dictele.keys():
        dictele[k][0]+=dictele[k][1]


dictt={1:[1,2],2:[2,4]}
testList(dictt)
print dictt
