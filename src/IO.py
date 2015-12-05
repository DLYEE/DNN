import numpy as np
import re
import cPickle as pickle

def readFile(f):
    print "Reading file..."
    inputData = {}
    keyOrder = []
    length = []
    length.append(0)
    my_file = open(f,"r+")
    count = 0
    for line in open(f):
        line = my_file.readline()
        s = re.split(" |\n",line)
        s.pop()
        s[1:] = [float(x) for x in s[1:]]
        inputData[s[0]] = np.asarray(s[1:])
        keyOrder.append(s[0])

        if len(keyOrder) > 1:
            s1 = re.split('_',keyOrder[-1])
            s2 = re.split('_',keyOrder[-2])
            if s1[:2] != s2[:2]:
                length.append(count)
        count += 1
    length.append(count)
    # print length
    my_file.close()
    # print "type of readFile inputData = ", type(imputData[keyOrder[0]][0])
    return inputData, keyOrder, length

def dnnReadFile(f1, f2):
    inputData = {}
    keyOrder = []
    my_file = open(f1,"r+")
    for line in open(f1):
        line = my_file.readline()
        s = re.split(" |\n",line)
        s.pop()
        s[1:] = [float(x) for x in s[1:]]
        inputData[s[0]] = s[1:]
        keyOrder.append(s[0])
    my_file.close()
    my_file = open(f2,"r+")
    for line in open(f2):
        line = my_file.readline()
        s = re.split(" |\n",line)
        s.pop()
        s[1:] = [float(x) for x in s[1:]]
        inputData[s[0]] = np.asarray(inputData[s[0]] + s[1:])
    return inputData, keyOrder


def readLabel(f, featureSize):
    label = {}
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
        label[s[0]] = np.asarray(labelElement)
    my_file.close()
    return label


def writeFile(f1, f2, possibilityVectors, outputData, keyOrder):

    print "Writing file..."
    file = open(f1,"w")
    file.write('Id,Prediction' + '\n')
    for index in range(len(keyOrder)):
        if index != 0 and index != len(keyOrder) - 1  :
            if outputData[keyOrder[index-1]] == outputData[keyOrder[index+1]] :
                outputData[keyOrder[index]] = outputData[keyOrder[index-1]]
            elif outputData[keyOrder[index-1]] != outputData[keyOrder[index]] and outputData[keyOrder[index]] != outputData[keyOrder[index+1]]:
                outputData[keyOrder[index]] = outputData[keyOrder[index-1]]
            elif index < len(keyOrder) - 2 and outputData[keyOrder[index]] == outputData[keyOrder[index+1]] and outputData[keyOrder[index]] != outputData[keyOrder[index+2]]:
                outputData[keyOrder[index]] = outputData[keyOrder[index-1]]
                outputData[keyOrder[index+1]] = outputData[keyOrder[index-1]]
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


def trimOutput(f1, f2):

    my_file = open(f1,"r+")
    frames = []
    for line in open(f1):
        line = my_file.readline()
        if line[:2] == 'Id':
            continue
        else:
            phone = re.split("_|,|\n",line)
            phone.pop()
            frames.append(phone)
    my_file.close()
    file = open(f2,"w")
    file.write('id,phone_sequence' + '\n')
    file.write(frames[0][0] + "_" + frames[0][1] + ",")
    file.write(idx2chr(frames[0][3]))
    for index in range(len(frames)):
        if index == 0:
            continue
        if frames[index][:2] == frames[index-1][:2]:
            if frames[index][3] != frames[index-1][3]:
                file.write(idx2chr(frames[index][3]))
        else:
            file.write('\n' + frames[index][0] + "_" + frames[index][1] + "," + idx2chr(frames[index][3]))
            #file.write("\n")
    file.close()

def deleteSil(f):
    file = open(f, "r+")
    seqs = []
    for line in open(f):
        line = file.readline()
        if line[:2] == 'id':
            continue
        else:
            seq = re.split(",|\n", line)
            # print seq
            if seq[-1] == '':
                seq.pop()
            # print seq
            if seq[1][0] == 'L':
                seqTemp = seq[1][1]
                for i in range(2, len(seq[1])):
                    seqTemp += seq[1][i]
            seq[1] = seqTemp
            if seq[1][-1] == 'L':
                seqTemp = seq[1][0]
                for i in range(1, len(seq[1]) - 1):
                    seqTemp += seq[1][i]
            seq[1] = seqTemp
            # print seq
            seqs.append(seq)
    file.close()

    file = open(f, "w")
    file.write('id,phone_sequence' + '\n')
    for seq in seqs:
        file.write(seq[0] + ',' + seq[1] + '\n')
    file.close()


def idx2chr(string):
    value = -1
    if string == "aa":
        value = 'a'
    elif string == "ae":
        value = 'b'
    elif string == "ah":
        value = 'c'
    elif string == "aw":
        value = 'e'
    elif string == "ay":
        value = 'g'
    elif string == "b":
        value = 'h'
    elif string == "ch":
        value = 'i'
    elif string == "sil":
        value = 'L'
    elif string == "d":
        value = 'k'
    elif string == "dh":
        value = 'l'
    elif string == "dx":
        value = 'm'
    elif string == "eh":
        value = 'n'
    elif string == "l":
        value = 'B'
    elif string == "n":
        value = 'D'
    elif string == "er":
        value = 'r'
    elif string == "ey":
        value = 's'
    elif string == "f":
        value = 't'
    elif string == "g":
        value = 'u'
    elif string == "hh":
        value = 'v'
    elif string == "ih":
        value = 'w'
    elif string == "iy":
        value = 'y'
    elif string == "jh":
        value = 'z'
    elif string == "k":
        value = 'A'
    elif string == "m":
        value = 'C'
    elif string == "ng":
        value = 'E'
    elif string == "ow":
        value = 'F'
    elif string == "oy":
        value = 'G'
    elif string == "p":
        value = 'H'
    elif string == "r":
        value = 'I'
    elif string == "sh":
        value = 'K'
    elif string == "s":
        value = 'J'
    elif string == "th":
        value = 'N'
    elif string == "t":
        value = 'M'
    elif string == "uh":
        value = 'O'
    elif string == "uw":
        value = 'P'
    elif string == "v":
        value = 'Q'
    elif string == "w":
        value = 'S'
    elif string == "y":
        value = 'T'
    elif string == "z":
        value = 'U'
    elif string == "ao":
        value = 'd'
    elif string == "ax":
        value = 'f'
    elif string == "epi":
        value = 'q'
    elif string == "cl":
        value = 'j'
    elif string == "vcl":
        value = 'R'
    elif string == "el":
        value = 'o'
    elif string == "en":
        value = 'p'
    elif string == "ix":
        value = 'x'
    elif string == "zh":
        value = 'V'
    if value != -1:
        return value
    else:
        print ("input string is not fit!\n")


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
    elif string == "ao":
        value = 3
    elif string == "aw":
        value = 4
    elif string == "ax":
        value = 5
    elif string == "ay":
        value = 6
    elif string == "b":
        value = 7
    elif string == "ch":
        value = 8
    elif string == "cl":
        value = 9
    elif string == "d":
        value = 10
    elif string == "dh":
        value = 11
    elif string == "dx":
        value = 12
    elif string == "eh":
        value = 13
    elif string == "el":
        value = 14
    elif string == "en":
        value = 15
    elif string == "epi":
        value = 16
    elif string == "er":
        value = 17
    elif string == "ey":
        value = 18
    elif string == "f":
        value = 19
    elif string == "g":
        value = 20
    elif string == "hh":
        value = 21
    elif string == "ih":
        value = 22
    elif string == "ix":
        value = 23
    elif string == "iy":
        value = 24
    elif string == "jh":
        value = 25
    elif string == "k":
        value = 26
    elif string == "l":
        value = 27
    elif string == "m":
        value = 28
    elif string == "ng":
        value = 29
    elif string == "n":
        value = 30
    elif string == "ow":
        value = 31
    elif string == "oy":
        value = 32
    elif string == "p":
        value = 33
    elif string == "r":
        value = 34
    elif string == "sh":
        value = 35
    elif string == "sil":
        value = 36
    elif string == "s":
        value = 37
    elif string == "th":
        value = 38
    elif string == "t":
        value = 39
    elif string == "uh":
        value = 40
    elif string == "uw":
        value = 41
    elif string == "vcl":
        value = 42
    elif string == "v":
        value = 43
    elif string == "w":
        value = 44
    elif string == "y":
        value = 45
    elif string == "zh":
        value = 46
    elif string == "z":
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
        string = "ao"
    elif num == 4:
        string = "aw"
    elif num == 5:
        string = "ax"
    elif num == 6:
        string = "ay"
    elif num == 7:
        string = "b"
    elif num == 8:
        string = "ch"
    elif num == 9:
        string = "cl"
    elif num == 10:
        string = "d"
    elif num == 11:
        string = "dh"
    elif num == 12:
        string = "dx"
    elif num == 13:
        string = "eh"
    elif num == 14:
        string = "el"
    elif num == 15:
        string = "en"
    elif num == 16:
        string = "epi"
    elif num == 17:
        string = "er"
    elif num == 18:
        string = "ey"
    elif num == 19:
        string = "f"
    elif num == 20:
        string = "g"
    elif num == 21:
        string = "hh"
    elif num == 22:
        string = "ih"
    elif num == 23:
        string = "ix"
    elif num == 24:
        string = "iy"
    elif num == 25:
        string = "jh"
    elif num == 26:
        string = "k"
    elif num == 27:
        string = "l"
    elif num == 28:
        string = "m"
    elif num == 29:
        string = "ng"
    elif num == 30:
        string = "n"
    elif num == 31:
        string = "ow"
    elif num == 32:
        string = "oy"
    elif num == 33:
        string = "p"
    elif num == 34:
        string = "r"
    elif num == 35:
        string = "sh"
    elif num == 36:
        string = "sil"
    elif num == 37:
        string = "s"
    elif num == 38:
        string = "th"
    elif num == 39:
        string = "t"
    elif num == 40:
        string = "uh"
    elif num == 41:
        string = "uw"
    elif num == 42:
        string = "vcl"
    elif num == 43:
        string = "v"
    elif num == 44:
        string = "w"
    elif num == 45:
        string = "y"
    elif num == 46:
        string = "zh"
    elif num == 47:
        string = "z"
    if string != "":
        return string
    else:
        print ("input number is not available!\n")
