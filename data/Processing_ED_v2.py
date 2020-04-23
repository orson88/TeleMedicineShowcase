import csv

## Initialize Arrays
File_Name = []
Emotion = []
## Chroma
Chr_SD = []
Chr_M = []
## MEL
MEL_SD = []
MEL_M = []
## Spectral centroid feature
SCF = []
## MFCC
MFCC_SD = []
MFCC_M = []
MFCC_1D = []
MFCC_2D = []
## Root Mean Square
RMS_M = []
## Spectral rolloff
SR_M = []
SR_R = []
## Zero crossing rate
ZCR = []

## Index through the rows of the csv file and add values to the matrices
with open('Emotion_Data_v2.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        File_Name.append(row[0])
        Emotion.append(row[1])
        Chr_SD.append(row[2])
        Chr_M.append(row[3])
        MEL_SD.append(row[4])
        MEL_M.append(row[5])
        SCF.append(row[6])
        MFCC_SD.append(row[7])
        MFCC_M.append(row[8])
        MFCC_1D.append(row[9])
        MFCC_2D.append(row[10])
        RMS_M.append(row[11])
        SR_M.append(row[12])
        SR_R.append(row[13])
        ZCR.append(row[14])