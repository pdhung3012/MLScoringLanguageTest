import os

def createDir(folderName):
    try:
        # Create target Directory
        os.mkdir(folderName)
        print("Directory ", folderName, " Created ")
    except FileExistsError:
        print("Directory ", folderName, " already exists")
def readFile(fp):
    strResult=''
    try:
        file=open(fp,'r')
        strResult=file.read()
        file.close()
    except:
        print("File ", fp, " doesn't exist")
    return strResult