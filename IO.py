import numpy as np
import re

def readFile(f, nnType):
    if nnType == 'dnn':
        inputData = {}
    elif nnType == 'rnn':
        inputData = []
    keyOrder = []
    my_file = open(f,"r+")
    for line in open(f):
        line = my_file.readline()
        s = re.split(" |\n",line)
        s.pop()
        s[1:] = [float(x) for x in s[1:]]
        if nnType == 'dnn':
            inputData[s[0]] = np.asarray(s[1:])
        elif nnType == 'rnn':
            inputData.append(np.asarray(s[1:]))
        keyOrder.append(s[0])
    my_file.close()
    return inputData, keyOrder

def readLabel(f, featureSize, nnType):
    if nnType == 'dnn':
        label = {}
    elif nnType == 'rnn':
        label = []
    my_file = open(f,'r+')
    for line in open(f):
        line = my_file.readline()
        s = re.split(',|\n',line)
        s.pop()
        number = str2int(s[1])
        labelElement = []
        for i in range(featureSize):
            if i == number:
                labelElement.append(1)
            else:
                labelElement.append(0)
        if nnType == 'dnn':
            label[s[0]] = np.asarray(labelElement)
        elif nnType == 'rnn':
            label.append(np.asarray(labelElement))
    my_file.close()
    return label


def writeFile(f1, f2, possibilityVectors, outputData, keyOrder):

    file = open(f1,"w")
    file.write('Id,Prediction' + '\n')
    for index in range(len(keyOrder)):
        if index != 0 and index != len(keyOrder) - 1  :
            # print index, keyOrder[index]
            if outputData[keyOrder[index-1]] == outputData[keyOrder[index+1]] :
                outputData[keyOrder[index]] = outputData[keyOrder[index-1]]
        outputData[keyOrder[index]] = mrg48to39(outputData[keyOrder[index]])
        file.write(keyOrder[index] + ',' + outputData[keyOrder[index]] + '\n')
    file.close()
    file = open(f2,"w")
    for line in possibilityVectors:
        for index in range(len(line)):
            if index != len(line) - 1:
                file.write(str(line[index]) + " ")
            else:
                file.write(str(line[index]))
        file.write('\n')
    file.close()



def mrg48to39(string):
    if string == "ao":
        string = "aa"
    elif string == "ax":
        string = "ah"
    elif string == "epi" or string == "cl" or string == "vcl":
        string = "sil"
    elif string == "el":
        string = "l"
    elif string == "en":
        string = "n"
    elif string == "ix":
        string = "ih"
    return string


def str2int(string):
    value = -1
    if string == "aa":
        value = 0
    elif string == "ae":
        value = 1
    elif string == "ah":
        value = 2
    elif string == "aw":
        value = 3
    elif string == "ay":
        value = 4
    elif string == "b":
        value = 5
    elif string == "ch":
        value = 6
    elif string == "sil":
        value = 7
    elif string == "d":
        value = 8
    elif string == "dh":
        value = 9
    elif string == "dx":
        value = 10
    elif string == "eh":
        value = 11
    elif string == "l":
        value = 12
    elif string == "n":
        value = 13
    elif string == "er":
        value = 14
    elif string == "ey":
        value = 15
    elif string == "f":
        value = 16
    elif string == "g":
        value = 17
    elif string == "hh":
        value = 18
    elif string == "ih":
        value = 19
    elif string == "iy":
        value = 20
    elif string == "jh":
        value = 21
    elif string == "k":
        value = 22
    elif string == "m":
        value = 23
    elif string == "ng":
        value = 24
    elif string == "ow":
        value = 25
    elif string == "oy":
        value = 26
    elif string == "p":
        value = 27
    elif string == "r":
        value = 28
    elif string == "sh":
        value = 29
    elif string == "s":
        value = 30
    elif string == "th":
        value = 31
    elif string == "t":
        value = 32
    elif string == "uh":
        value = 33
    elif string == "uw":
        value = 34
    elif string == "v":
        value = 35
    elif string == "w":
        value = 36
    elif string == "y":
        value = 37
    elif string == "z":
        value = 38
    elif string == "ao":
        value = 39
    elif string == "ax":
        value = 40
    elif string == "epi":
        value = 41
    elif string == "cl":
        value = 42
    elif string == "vcl":
        value = 43
    elif string == "el":
        value = 44
    elif string == "en":
        value = 45
    elif string == "ix":
        value = 46
    elif string == "zh":
        value = 47
    if value != -1:
        return value
    else:
        print ("input string is not fit!\n")


def int2str(num):
    string = ""
    if num == 0:
        string = "aa"
    elif num == 1:
        string = "ae"
    elif num == 2:
        string = "ah"
    elif num == 3:
        string = "aw"
    elif num == 4:
        string = "ay"
    elif num == 5:
        string = "b"
    elif num == 6:
        string = "ch"
    elif num == 7:
        string = "sil"
    elif num == 8:
        string = "d"
    elif num == 9:
        string = "dh"
    elif num == 10:
        string = "dx"
    elif num == 11:
        string = "eh"
    elif num == 12:
        string = "l"
    elif num == 13:
        string = "n"
    elif num == 14:
        string = "er"
    elif num == 15:
        string = "ey"
    elif num == 16:
        string = "f"
    elif num == 17:
        string = "g"
    elif num == 18:
        string = "hh"
    elif num == 19:
        string = "ih"
    elif num == 20:
        string = "iy"
    elif num == 21:
        string = "jh"
    elif num == 22:
        string = "k"
    elif num == 23:
        string = "m"
    elif num == 24:
        string = "ng"
    elif num == 25:
        string = "ow"
    elif num == 26:
        string = "oy"
    elif num == 27:
        string = "p"
    elif num == 28:
        string = "r"
    elif num == 29:
        string = "sh"
    elif num == 30:
        string = "s"
    elif num == 31:
        string = "th"
    elif num == 32:
        string = "t"
    elif num == 33:
        string = "uh"
    elif num == 34:
        string = "uw"
    elif num == 35:
        string = "v"
    elif num == 36:
        string = "w"
    elif num == 37:
        string = "y"
    elif num == 38:
        string = "z"
    elif num == 39:
        string = "ao"
    elif num == 40:
        string = "ax"
    elif num == 41:
        string = "epi"
    elif num == 42:
        string = "cl"
    elif num == 43:
        string = "vcl"
    elif num == 44:
        string = "el"
    elif num == 45:
        string = "en"
    elif num == 46:
        string = "ix"
    elif num == 47:
        string = "zh"
    if string != "":
        return string
    else:
        print ("input number is not available!\n")
