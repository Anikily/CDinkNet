#encoding=utf-8
import os
import shutil
# 旧地址、新地址、文件格式
def filemove(oldfile, newfile, fileformat):
    i = 0;
    # 列出文件下所有文件
    weblist = os.walk(oldfile)
    newpath = newfile
    for path, d, filelist in weblist:
        for filename in filelist:
            if fileformat in filename:
                i = i +1
                full_path = os.path.join(path, filename)  # 旧地址 + 文件名
                despath = newpath + str(i)+".jpg"  # 新地址 +文件名
                
                print(shutil.move(full_path, despath), '文件移动成功')  # 移动文件至新的文件夹
            else:
                print('文件不存在', filename)
    print(i)
filemove('/home/aniki/code/test/正常驾驶', '/home/aniki/code/test/正常_文件夹/', '.jpg')

