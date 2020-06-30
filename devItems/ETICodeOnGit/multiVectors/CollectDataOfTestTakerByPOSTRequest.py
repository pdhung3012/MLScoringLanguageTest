import requests
from multiVectors.utils import createDir,readFile
from datetime import date, timedelta


folder = "../../../../resultETI/dataCollectionFromMarch1st/"

createDir(folder)



fpLogDays=folder+'logDays.txt'


sdate = date(2020, 3, 1)   # start date
edate = date.today()   # end date

delta = edate - sdate       # as timedelta
lstDays=[]
for i in range(delta.days + 1):
    day = sdate + timedelta(days=i)
    lstDays.append(str(day))

strLogInfo=readFile(fpLogDays)
dictStatusByDays={}
arrLogs=strLogInfo.split('\n')
for i in range(len(arrLogs)):
    arrItem=arrLogs[i].strip().split('\t')
    if len(arrItem)>=2:
        dictStatusByDays[arrItem[0]]=arrItem[1]


# 'Authorize-Token':'b5d8714e74b94adebe4d9cdffbcd3a27'

strUrl='https://tms.languagetesting.com/LTI.CoreServices/api/dashboard/MachineScorefile'
strAuthorizeToken='b5d8714e74b94adebe4d9cdffbcd3a27'



for i in range(len(lstDays)):
    strItemDate = lstDays[i] + 'T12:23:59.999Z'
    fpItemType1=folder+lstDays[i]+'_1.csv'
    fpItemType2 = folder + lstDays[i] + '_2.csv'
    isType1DownloadOK=False
    isType2DownloadOK = False

    try:
        if( (not lstDays[i] in dictStatusByDays.keys()) or dictStatusByDays[lstDays[i]]=='False'):
            newHeaders = {'Authorize-Token': strAuthorizeToken}
            response = requests.post(strUrl,
                                     data={
                                         'date': strItemDate,
                                         'type': '1'
                                     },
                                     headers=newHeaders)
            if(response.status_code == 200):
                fCode = open(fpItemType1, 'wb')
                fCode.write(response.content)
                fCode.close()
                isType1DownloadOK=True

            response = requests.post(strUrl,
                                     data={
                                         'date': strItemDate,
                                         'type': '2'
                                     },
                                     headers=newHeaders)
            if (response.status_code == 200):
                fCode = open(fpItemType2, 'wb')
                fCode.write(response.content)
                fCode.close()
                isType2DownloadOK = True
    except:
        print('some error happen with {}'.format(lstDays[i]))
    if isType1DownloadOK and isType2DownloadOK:
        dictStatusByDays[lstDays[i]]=True
        print('download ok with {}'.format(lstDays[i]))
    else:
        dictStatusByDays[lstDays[i]] = False

file = open(fpLogDays, 'w')
file.write('')
file.close()
for key in dictStatusByDays.keys():
    file = open(fpLogDays, 'a')
    file.write('{}\t{}\n'.format(key,dictStatusByDays[key]))
    file.close()


# print("Status code: ", response.status_code)
#
# data = response.content
# print("Printing Post JSON data")


# print(str(responseData))

# print("Content-Type is ", response_Json['headers']['Content-Type'])