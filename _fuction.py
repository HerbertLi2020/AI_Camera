import time,sys,os,datetime,win32con, win32api,wmi, http.client
import winreg as wg

def MsgBox(str='ERROR'):
    win32api.MessageBox(0, str, '运行错误提示', win32con.MB_OK)

def dectry(p):  # 解密
    dec_str = ""
    for i,j in zip(p.split("#")[:-1],k):    # i 为加密字符，j为秘钥字符
        temp = chr(int(i) - ord(j))         # 解密字符 = (加密Unicode码字符 - 秘钥字符的Unicode码)的单字节字符
        dec_str = dec_str+temp
    return dec_str

def get_web_time(host): #获取网络时间,并设置系统时间
    try:
        conn=http.client.HTTPConnection(host,timeout=20)
        conn.request("GET", "/")
        r=conn.getresponse()
    except: MsgBox("运行错误：网络连接超时！"); sys.exit()
    #r.getheaders() #获取所有的http头
    ts=  r.getheader('date') #获取http头date部分
    ltime= time.strptime(ts[5:25], "%d %b %Y %H:%M:%S")   #将GMT时间转换成北京时间
    ttime=time.localtime(time.mktime(ltime)+8*60*60)
    dat="date %u-%02u-%02u"%(ttime.tm_year,ttime.tm_mon,ttime.tm_mday)
    tm="time %02u:%02u:%02u"%(ttime.tm_hour,ttime.tm_min,ttime.tm_sec)
    os.system(dat)
    #os.system(tm)
    date = "%u%02u%02u" % (ttime.tm_year, ttime.tm_mon, ttime.tm_mday)
    return date

def checkdate():
    use_date = datetime.datetime.strptime(date, "%Y%m%d")
    #now_date = datetime.datetime.strptime(time.strftime("%Y%m%d", time.localtime()), "%Y%m%d")
    web_date = get_web_time('www.baidu.com')
    now_date = datetime.datetime.strptime(web_date, "%Y%m%d")
    if use_date < now_date: MsgBox("运行错误：软件许可期限已到！");sys.exit()
    elif (use_date - now_date).days < 30: MsgBox("运行提示：软件许可期限(%s)即将到期！"%date)

def checkcount():
    """
    filename = 'C:\\Python38\\Lib\\__pycache__\\__after_code_i.pyc'
    if os.path.exists(filename):
        f = open(filename, 'rb')
        c = f.read(); f.close()
        num = int((c.decode(encoding="utf-8")))
        if num > basenum + int(count):
            MsgBox("ERROR3：软件非法使用！")
            sys.exit()
        num += 1
        f = open(filename, 'wb')
        f.write(str(num).encode(encoding="utf-8"))
        f.close()
    else:
        f = open(filename, 'wb')
        savebin = str(basenum).encode(encoding="utf-8")
        f.write(savebin)
        f.close()
    """
    basenum = '462100'
    reg_key = wg.OpenKey(wg.HKEY_CURRENT_USER, r"Software", wg.KEY_SET_VALUE)
    try: num = int(wg.QueryValue(reg_key, 'Herbert'))
    except:
        wg.CreateKey(reg_key, 'Herbert')
        wg.SetValue(reg_key, 'Herbert', wg.REG_SZ, basenum)
    num = int(wg.QueryValue(reg_key, 'Herbert'))
    if num > int(basenum) + int(count): MsgBox("运行错误：软件使用次数已到！"); wg.CloseKey(reg_key); sys.exit()
    num += 1; wg.SetValue(reg_key, 'Herbert', wg.REG_SZ, str(num))
    wg.CloseKey(reg_key)

def checkSN():
    c = wmi.WMI()
    for cpu in c.Win32_Processor(): cpu_id = cpu.ProcessorId.strip()
    for board in c.Win32_BaseBoard(): board_id = board.SerialNumber
    myid = cpu_id + board_id
    #print(myid+'\n'+id)
    if id == myid: pass; #print('SN File is OK！Going on...')
    else: MsgBox("运行错误：此电脑未获得使用授权！"); sys.exit()

# 读取文件，并解密
k = 'herbert028978dh759cgbpw9oqdcyhtu4oihct$%^&**(^&1herbert02891djq%5cu-jeq15abg$z9_i_w=$o88m!*apbedlbat8cr74sd'
if not os.path.exists('SN'):
    MsgBox('SN File Not Find!'); sys.exit()
f = open('SN', 'r'); decrypted = dectry(f.read()); f.close()

sn_e,date_e,count_e,count,date,id = decrypted.split("#", 5)[0:6]
#print(sn_e,date_e,count_e,count,date,id)

if sn_e   == '1': checkSN()
if date_e == '1':
    get_web_time('www.baidu.com')
    checkdate()
if count_e == '1': checkcount()

