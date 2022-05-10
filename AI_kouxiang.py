#AI视频背景替换工具
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os, sys, time, cv2, threading, GPUtil
import moviepy.editor as mpe
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget,QMessageBox,QLabel, QLineEdit,QFileDialog,QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QIntValidator
import _fuction

#检测显卡是否是Nvidia，并查看型号是否支持
gpus = GPUtil.getGPUs(); gpu_list = []; gpu_i = 0; gpu_available = 0
for g in gpus: gpu_list.append([g.name, g.memoryTotal])
GPU_nums = len(gpu_list)  #显卡数量
print("【显卡数量】检测出的Nvidia GPU数量：%d个：" %GPU_nums)
for GPU_Info in gpu_list:
    gpu_i += 1; GPU_name = GPU_Info[0]  #显卡名称
    GPU_memsize = GPU_Info[1]/1024  # 显卡总的显存大小
    print("【显卡：%d】型号：%s | 显存：%dG"%(gpu_i,GPU_name,GPU_memsize))
    if ('GTX' in GPU_name or 'RTX' in GPU_name): gpu_available += 1
if gpu_available > 0: if_use_gpu = True;print("【可用数量】AI处理的可用GPU：%d个：" %gpu_i)
else:if_use_gpu = False; print("您的GPU是本软件不支持的Nvidia显卡，本软件将使用CPU进行后续处理！"); GPU_memsize = 1.0
#GPU检测完毕

if_good_model = True  # 默认使用的AI模型

#读取配置文件AI_GPU.ini
try:
    config_file = open('AI_GPU.ini', 'r', encoding='utf-8')
    config_dict = {}
    for line in config_file:
        if line[0]== '#' or line[0]== ' ' or line[0]== '\n': continue
        for i in range(100):
            if line[i]=='#' or line[i]==' ' or line[i]== '\n': line=line[0:i]; break
        k = line.split("=")[0]
        v = line.split("=")[1]
        config_dict[k] = v
        #print(config_dict)
    config_file.close()
    gpu_no = config_dict['GPU_NO']
except:
    print("\n配置文件AI_GPU.ini不存在，使用默认配置。")
    gpu_no='0'

if if_use_gpu:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_no  # GPU环境变量
#   if GPU_memsize >= 3: if_good_model = True

work_path = os.getcwd()+'\kx'; files = []
input_path = work_path + '\mp4'; bg_file=work_path + '\\bk.jpg'; out_dir=work_path + '\mp4_1'

my_width = 900; my_height = 680
run_flag = 0; Box1_flag = True; Box2_flag = True; Box3_flag = False
bk_pix = [8,188,8]; bk_img = np.zeros((1080, 1920, 3), np.uint8)  # Creat a Image
bk_img[:] = bk_pix; filesnums = 1 ; stop_flag = False ; t0 = 0

my_title = "AI视频抠像工具"
pil_img = Image.open("start_img.jpg")
#ImageDraw.Draw(pil_img).text((350,190), "视频抠像、背景替换工具v1.0", (255,255,255),font=ImageFont.truetype("msyh.ttc", 18))
ImageDraw.Draw(pil_img).text((410,320), "正在加载AI模型，请稍后 ......", (255,255,255),font=ImageFont.truetype("msyh.ttc", 16))
img_s = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def showpic():  # 以下代码显示软件初始界面
    global ret, frame
    while run_flag == 0:
        cv2.imshow("AI Video Editing System", img_s)
        cv2.waitKey(100)
    cv2.destroyAllWindows()

t = threading.Thread(target=showpic)
t.start()

class Winshot(QWidget):
    def __init__(self):
        super(Winshot, self).__init__()
        global hwnd, run_flag

        import paddlehub as hub  # 导入模型很消耗时间，大概5秒
        self.humanseg1 = hub.Module(directory='modules\deeplabv3p_xception65_humanseg')  # 抠图复杂模型
        self.humanseg2 = hub.Module(directory='modules\humanseg_mobile')  # 抠图简易模型
        #self.humanseg1 = hub.Module(name='deeplabv3p_xception65_humanseg')  # 抠图复杂模型
        #self.humanseg2 = hub.Module(name='humanseg_mobile')  # 抠图简易模型

        palette1 = QtGui.QPalette()
        palette1.setColor(palette1.Background, QtGui.QColor(200, 200, 200))
        self.createLayout()
        self.stopButton.setEnabled(False)
        self.set_rgb_False()
        self.setPalette(palette1)
        self.setWindowTitle(my_title)
        self.setFixedSize(my_width, my_height)
        self.setWindowFlags(Qt.WindowMinimizeButtonHint)
        self.show(); run_flag = 1

    def CV2toPIL(self, img):  # cv2转PIL
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    def PILtoCV2(self, img):  # PIL转cv2
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    def two_pic_combine_PIL(self, back_img, fore_img): #2个图片合并
        back_img = self.CV2toPIL(back_img); fore_img = self.CV2toPIL(fore_img); r,g,b,alpha = fore_img.split()
        return cv2.cvtColor(self.PILtoCV2(Image.composite(fore_img, back_img, alpha)), cv2.COLOR_BGRA2BGR)

    def video_change_background(self, videofile, backfile):
        global bg_file, iii, stop_flag, t0
        try: cap = cv2.VideoCapture(videofile)  # 读取视频文件
        except: self.show_error('读取视频文件:'+videofile+'时，出现错误！'); return
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
        total_fr = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
        size_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频流的帧宽度
        size_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频流的帧高度
        len_m,len_s = divmod(total_fr/fps, 60)
        videoinfo = '【视频信息】 文件总数：%d | 正在处理(%d/%d)：'%(filesnums,iii,filesnums) + os.path.split(videofile)[1]+\
                    ' | 帧分辨率：%dx%d | 视频长度：%d分%d秒 | FPS：%.2f帧/秒'%(size_x,size_y,len_m,len_s,fps)
        self.txt11.setText(videoinfo)

        img1 = cv2.resize(backfile, (size_x, size_y))
        tempfile = os.path.split(bg_file)[0]+'\\out.mp4'
        out = cv2.VideoWriter(tempfile, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size_x, size_y))
        t1 = time.time(); next_fr = 0
        if size_x / size_y > 1.778: fx = 427 / size_x; fy = fx   # 计算16:9的比例，以便缩放不变形
        else: fx = 240 / size_y; fy = fx

        while (True):
            t2 = time.time()
            if stop_flag:
                cap.release();  out.release(); os.remove(tempfile)
                self.txt11.setText('【视频信息】 文件总数：%d个 | 处理完成：%d个' % (filesnums, iii))
                self.txt12.setText('【运行信息】 用户终止了正在运行的程序......')
                return
            next_fr += 1
            ret, frame = cap.read()
            if ret:
                img2 = self.koutu(frame)
                #img2 = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2BGRA)   #测试用语句
                img3 = self.two_pic_combine_PIL(img1,img2)
                out.write(img3)  # 帧转成视频
            else: break
            if Box1_flag: self.my_label1.setPixmap(self.CvMatToQImage(cv2.resize(frame,(0,0),fx=fx,fy=fy)))
            if Box2_flag: self.my_label2.setPixmap(self.CvMatToQImage(cv2.resize(img3,(0,0),fx=fx,fy=fy)))
            cv2.waitKey(1)
            t3 = time.time(); m1, s1 = divmod(t3-t0, 60); m2, s2 = divmod(t3-t1, 60)
            runinfo = '【运行信息】 当前视频处理进度：%d%% | 总耗时：%d分%d秒 | 当前视频耗时：%d分%d秒 | 当前帧耗时：%.2f秒 | 处理速度:%.1fFPS'\
                      %(100*next_fr/total_fr, m1, s1, m2, s2,(t3-t2),1/(t3-t2))
            self.txt12.setText(runinfo)

        cap.release(); out.release()
        self.txt12.setText('【运行信息】 正在分离、合成音轨(大概需要：%.1f分钟)，请稍后......'%(len_m/2))
        cv2.waitKey(1)
        audio = mpe.AudioFileClip(videofile)  # 分离声轨
        clip = mpe.VideoFileClip(tempfile)
        videoclip = clip.set_audio(audio)  # 写入声轨
        videoclip.write_videofile(out_dir+'/'+os.path.splitext(os.path.split(videofile)[1])[0]+'_1.mp4')
        os.remove(tempfile)
        t3 = time.time()
        self.txt12.setText('【运行信息】 处理完毕！总消耗时间：%d分%d秒'%(m1, s1))
        self.txt11.setText('【视频信息】 文件总数：%d个 | 处理完成：%d个'%(filesnums,iii))

    def CvMatToQImage(self, ptr):  # Converts an opencv MAT format into a QImage
        ptr = cv2.cvtColor(ptr, cv2.COLOR_BGRA2RGBA)  # 颜色格式转换
        QtImg = QtGui.QImage(ptr.data, ptr.shape[1], ptr.shape[0], QtGui.QImage.Format_RGBA8888)
        return QtGui.QPixmap.fromImage(QtImg)

    def koutu(self, mat_img):  # 抠图函数
        if if_good_model:  # 使用复杂抠像模型， 使用GPU
            results = self.humanseg1.segmentation(images=[mat_img], batch_size=4, use_gpu=if_use_gpu)
        else:  # 使用简易抠像模型，使用CPU
            results = self.humanseg2.segment(images=[mat_img], batch_size=4, use_gpu=if_use_gpu)
        return np.uint8(results[0]['data'])  # 取出图像的RGBA数据,并从浮点转成整数

    def show_error(self,str):
        r_button = QMessageBox.question(self, my_title,'\n\n'+str+'\n\n', QMessageBox.Ok)
    def set_False_Btn(self):
        self.filesButton.setEnabled(False);       self.outButton.setEnabled(False)
        self.bkfileButton.setEnabled(False);      self.checkBox3.setEnabled(False)
        self.startButton.setEnabled(False);       self.stopButton.setEnabled(True)
        self.quitButton.setEnabled(False)
    def set_True_Btn(self):
        self.filesButton.setEnabled(True);       self.outButton.setEnabled(True)
        self.bkfileButton.setEnabled(True);      self.checkBox3.setEnabled(True)
        self.startButton.setEnabled(True);       self.stopButton.setEnabled(False)
        self.quitButton.setEnabled(True)

    def startrun(self):
        global iii,stop_flag,t0
        iii = 0; stop_flag = False
        self.txt12.setText('【运行信息】 正在初始化AI模型......');cv2.waitKey(1)
        t0 = time.time()
        if files == []: self.show_error('请选择需要替换背景的视频文件！'); return
        if not os.path.exists(out_dir): self.show_error('输出目录不存在，请重新选择！'); return
        self.set_False_Btn()
        if not Box3_flag:
            #try: back_ground = cv2.imread(bg_file)  # 读取背景文件
            try: back_ground = cv2.imdecode(np.fromfile(bg_file, dtype=np.uint8), -1)
            except: self.show_error('读取背景文件时，出现错误！'); self.set_True_Btn(); return
            if back_ground is None: self.show_error('读取背景文件时，出现错误！\n原因：目录/文件名不能包含中文...... '); self.set_True_Btn(); return
        else: back_ground = bk_img
        for file in files:
            iii += 1
            if stop_flag: break
            self.video_change_background(file, back_ground)
        self.set_True_Btn()
        self.my_label1.setPixmap(QPixmap("start_img.jpg"))
        self.my_label2.setPixmap(QPixmap("start_img.jpg"))

    def stoprun(self):
        global stop_flag
        r_button = QMessageBox.question(self, my_title,
                                        "\n\n    确定要停止替换背景吗？\n\n", QMessageBox.Yes | QMessageBox.No)
        if r_button == QMessageBox.Yes: stop_flag = True

    def helpWin(self):
        str="\n\n\n1、【选择文件】选择需要替换背景的视频文件；\n2、【输出目录】替换后的文件目录，文件名：源文件_1.mp4；\n"+\
        "3、【背景文件】选择后，人物视频的背景都被替换成此背景；\n4、【纯色背景】点选后，所有视频背景替换成纯色的；\n"+\
        "5、如没有Nvidia系列GPU，就选CPU处理，AI需选【简易】；\n6、AI抠像算法有简易、复杂2种，可在软件设置栏目里面选择；\n\n\n"+\
        "      本软件著作权归属：???        网址：www.???.com\n\n"
        QMessageBox.question(self, my_title, str, QMessageBox.Ok)
    def quitWin(self):
        r_button = QMessageBox.question(self, my_title,
                                        "\n\n    退出将终止替换进程\n\n    确认退出吗？\n\n", QMessageBox.Yes | QMessageBox.No)
        if r_button == QMessageBox.Yes: sys.exit()

    def filesButton_fuc(self):
        global files,filesnums
        files, ok1 = QFileDialog.getOpenFileNames(self,'请选择视频文件[全选:Ctrl+A、多选:Ctrl/Shift+鼠标]',
                                                       input_path,"*.mp4;;*.avi;;*.mkv")
        filesnums = len(files)
        if files!=[]:
            txt='目录：'+os.path.split(files[0])[0]+' | 已选文件：'+str(filesnums)+'个 | 文件名：'
            for file in files: txt=txt+ os.path.split(file)[1]+'; '
            self.txt1.setText(txt)
        else: self.txt1.setText('请选择视频文件[全选:Ctrl+A、多选:Ctrl/Shift+鼠标]......')
    def outButton_fuc(self):
        global out_dir
        out_dir = QFileDialog.getExistingDirectory(self,'选择转换后的输出文件夹', work_path)
        if out_dir == '': self.txt2.setText('请选择背景替换后文件保存目录......')
        else: self.txt2.setText(out_dir)
    def bkfileButton_fuc(self):
        global bg_file
        bg_file, ok1 = QFileDialog.getOpenFileName(self,"选择背景图片文件",work_path,"*.jpg;;*.png;;*.gif")
        if bg_file == '': self.txt3.setText('请选择背景图片文件......')
        else: self.txt3.setText(bg_file)

    def click_comboBox1(self, text):
        global if_use_gpu,if_good_model
        if text == 'GPU':
            if GPU_memsize < 3 and self.comboBox2.currentIndex() == 1:
                self.show_error("\nGPU名称：%s\nGPU内存：%dG\n\n注意：\nGPU内存必须大于3G才能进行复杂AI模型，\n请选择简易模型！"
                                %(GPU_name,GPU_memsize))
                self.comboBox2.setCurrentIndex(0);if_good_model = False
            if_use_gpu = True
        else: if_use_gpu = False
    def click_comboBox2(self, text):
        global if_good_model,if_use_gpu
        if text == '复杂':
            if GPU_memsize < 3 and self.comboBox1.currentIndex() == 1:
                self.show_error("\nGPU名称：%s\nGPU内存：%dG\n\n注意：\nGPU内存必须大于3G才能进行复杂AI模型，\n请选择简易模型或CPU处理！"
                                % (GPU_name, GPU_memsize))
                self.comboBox1.setCurrentIndex(0);if_use_gpu = False
            if_good_model = True
        else: if_good_model = False

    def box_choose(self):
        global Box1_flag, Box2_flag
        if self.checkBox1.isChecked(): Box1_flag = True
        else:Box1_flag = False;  self.my_label1.setPixmap(QPixmap("start_img.jpg"))
        if self.checkBox2.isChecked(): Box2_flag = True
        else:Box2_flag = False;  self.my_label2.setPixmap(QPixmap("start_img.jpg"))

    def box_choose3(self):
        global Box3_flag
        if self.checkBox3.isChecked():
            self.txt3.setEnabled(False); self.bkfileButton.setEnabled(False)
            self.txt3.setText('已经选择纯色背景......')
            self.set_rgb_True()
            Box3_flag = True
        else:
            self.txt3.setEnabled(True);  self.bkfileButton.setEnabled(True)
            self.txt3.setText(bg_file)
            self.set_rgb_False()
            Box3_flag = False
    def set_rgb_False(self):
        self.red.setEnabled(False);     self.red_e.setEnabled(False); self.green.setEnabled(False)
        self.green_e.setEnabled(False); self.blue.setEnabled(False);  self.blue_e.setEnabled(False)
    def set_rgb_True(self):
        self.red.setEnabled(True);     self.red_e.setEnabled(True); self.green.setEnabled(True)
        self.green_e.setEnabled(True); self.blue.setEnabled(True);  self.blue_e.setEnabled(True)
    
    def red_e_fuc(self, text):
        if text =='': rgb = 0
        else: rgb = int(text)
        bk_pix[2] = rgb; bk_img[:] = bk_pix
        self.txt4.setPixmap(self.CvMatToQImage(cv2.resize(bk_img, (50, 18))))
    def green_e_fuc(self, text):
        if text =='': rgb = 0
        else: rgb = int(text)
        bk_pix[1] = rgb; bk_img[:] = bk_pix
        self.txt4.setPixmap(self.CvMatToQImage(cv2.resize(bk_img, (50, 18))))
    def blue_e_fuc(self, text):
        if text =='': rgb = 0
        else: rgb = int(text)
        bk_pix[0] = rgb; bk_img[:] = bk_pix
        self.txt4.setPixmap(self.CvMatToQImage(cv2.resize(bk_img, (50, 18))))

    def createLayout(self):
        mainLayout = QtWidgets.QVBoxLayout();topLayout1 = QtWidgets.QHBoxLayout();topLayout2 = QtWidgets.QHBoxLayout()
        topLayout3 = QtWidgets.QHBoxLayout();topLayout4 = QtWidgets.QHBoxLayout()

        self.my_label1 = QtWidgets.QLabel(); self.my_label2 = QtWidgets.QLabel()
        topLayout1.addWidget(self.my_label1); topLayout1.addWidget(self.my_label2)

        self.my_label1.setPixmap(QPixmap("start_img.jpg")); self.my_label2.setPixmap(QPixmap("start_img.jpg"))
        self.my_label1.setFixedSize(427, 240); self.my_label2.setFixedSize(427, 240)
        self.my_label1.setAlignment(Qt.AlignCenter); self.my_label2.setAlignment(Qt.AlignCenter)

        self.my_label1.setToolTip("本区域，显示的是原始视频缩略图..."); self.my_label2.setToolTip("本区域，显示的是替换后的缩略图...")

        self.GroupBox1 = QtWidgets.QGroupBox("软件设置")
        self.GroupBox1.setFixedSize(280, 60)
        self.lbl_1 = QtWidgets.QLabel("处理器：", self)
        self.lbl_1.setFixedSize(45, 25)
        self.comboBox1 = QtWidgets.QComboBox(self)
        self.comboBox1.setFixedSize(50, 25)
        self.comboBox1.addItem("CPU");  self.comboBox1.addItem("GPU")
        if if_use_gpu: self.comboBox1.setCurrentIndex(1)
        else: self.comboBox1.setEnabled(False)
        self.comboBox1.activated[str].connect(self.click_comboBox1)
        self.lbl_2 = QtWidgets.QLabel("AI算法：", self)
        self.lbl_2.setFixedSize(45, 25)
        self.comboBox2 = QtWidgets.QComboBox(self)
        self.comboBox2.setFixedSize(50, 25)
        self.comboBox2.addItem("简易"); self.comboBox2.addItem("复杂")
        if if_good_model:self.comboBox2.setCurrentIndex(1)
        self.comboBox2.activated[str].connect(self.click_comboBox2)
        GroupBox1Layout = QtWidgets.QHBoxLayout()
        GroupBox1Layout.addWidget(self.lbl_2)
        GroupBox1Layout.addWidget(self.comboBox2)
        GroupBox1Layout.addWidget(self.lbl_1)
        GroupBox1Layout.addWidget(self.comboBox1)
        self.GroupBox1.setLayout(GroupBox1Layout)
        #if not if_use_gpu: self.GroupBox1.setEnabled(False)

        self.GroupBox2 = QtWidgets.QGroupBox("预览设置")
        self.GroupBox2.setFixedSize(180, 60)
        self.checkBox1 = QtWidgets.QCheckBox("原始视频")
        self.checkBox2 = QtWidgets.QCheckBox("输出视频")
        GroupBox2Layout = QtWidgets.QHBoxLayout()
        GroupBox2Layout.addWidget(self.checkBox1)
        GroupBox2Layout.addWidget(self.checkBox2)
        self.GroupBox2.setLayout(GroupBox2Layout)
        self.checkBox1.stateChanged.connect(self.box_choose)
        self.checkBox2.stateChanged.connect(self.box_choose)
        self.checkBox1.setChecked(True); self.checkBox2.setChecked(True)

        self.GroupBox4 = QtWidgets.QGroupBox("文件设置")
        self.GroupBox4.setFixedSize(850, 160)
        self.filesButton = self.createButton("选择文件", self.filesButton_fuc)
        self.outButton = self.createButton("输出目录", self.outButton_fuc)
        self.bkfileButton = self.createButton("背景文件", self.bkfileButton_fuc)
        self.filesButton.setToolTip("选择即将被替换背景的视频文件，可单选、多选...")
        self.outButton.setToolTip("选择输出文件目录，替换后的文件将存在此目录...")
        self.bkfileButton.setToolTip("选择可用作背景的图片文件，建议分辨率：1920x1080...")
        self.filesButton.setFixedSize(80,23); self.outButton.setFixedSize(80,23)
        self.bkfileButton.setFixedSize(80,23)
        self.txt1 = QLabel('请选择视频文件[Ctrl+A全选、Ctrl/Shift+鼠标可多选]......', self); self.txt2 = QLabel('输出目录', self)
        self.txt3 = QLabel('背景文件', self); self.txt4 = QLabel('纯色文件', self)
        self.txt2.setText(out_dir);   self.txt3.setText(bg_file);
        self.txt4.setPixmap(self.CvMatToQImage(cv2.resize(bk_img, (50, 18))))
        self.checkBox3 = QtWidgets.QCheckBox("纯色背景")
        self.checkBox3.stateChanged.connect(self.box_choose3)

        self.red  = QLabel(' 红', self); self.green= QLabel('    绿', self); self.blue = QLabel('    蓝', self)
        self.red_e =   QLineEdit(self);    self.red_e.setText('8')
        self.green_e = QLineEdit(self);    self.green_e.setText('188')
        self.blue_e =  QLineEdit(self);    self.blue_e.setText('8')
        self.red_e.setValidator(QIntValidator(0, 254))
        self.green_e.setValidator(QIntValidator(0, 254))
        self.blue_e.setValidator(QIntValidator(0, 254))

        self.red_e.setFixedSize(28, 20);self.green_e.setFixedSize(28, 20);self.blue_e.setFixedSize(28, 20)
        self.red_e.textChanged[str].connect(self.red_e_fuc)
        self.green_e.textChanged[str].connect(self.green_e_fuc)
        self.blue_e.textChanged[str].connect(self.blue_e_fuc)

        layout_box1 = QtWidgets.QHBoxLayout()
        layout_box2 = QtWidgets.QHBoxLayout()
        layout_box3 = QtWidgets.QHBoxLayout()
        layout_box1.addWidget(self.filesButton, Qt.AlignLeft| Qt.AlignVCenter)
        layout_box1.addWidget(self.txt1, Qt.AlignLeft| Qt.AlignVCenter)
        layout_box2.addWidget(self.outButton)
        layout_box2.addWidget(self.txt2)
        layout_box2.addWidget(self.bkfileButton)
        layout_box2.addWidget(self.txt3)
        layout_box3.addWidget(self.checkBox3)
        layout_box3.addWidget(self.txt4)
        layout_box3.addWidget(self.red)
        layout_box3.addWidget(self.red_e)
        layout_box3.addWidget(self.green)
        layout_box3.addWidget(self.green_e)
        layout_box3.addWidget(self.blue)
        layout_box3.addWidget(self.blue_e)
        layout_box3.addStretch(1)
        layout11 = QWidget();  layout21 = QWidget(); layout31 = QWidget()
        layout11.setLayout(layout_box1);  layout21.setLayout(layout_box2); layout31.setLayout(layout_box3)

        GroupBoxmainLayout = QtWidgets.QVBoxLayout()
        GroupBoxmainLayout.addWidget(layout11, Qt.AlignLeft | Qt.AlignVCenter)
        GroupBoxmainLayout.addWidget(layout21, Qt.AlignLeft | Qt.AlignVCenter)
        GroupBoxmainLayout.addWidget(layout31, Qt.AlignLeft | Qt.AlignVCenter)
        self.GroupBox4.setLayout(GroupBoxmainLayout)

        self.GroupBox5 = QtWidgets.QGroupBox("信息统计")
        self.GroupBox5.setFixedSize(850, 90)
        self.txt11 = QLabel('【视频信息】', self)
        self.txt12 = QLabel('【运行信息】', self)
        GroupBox5Layout = QtWidgets.QGridLayout()
        GroupBox5Layout.addWidget(self.txt11, 0, 1)
        GroupBox5Layout.addWidget(self.txt12, 1, 1)
        self.GroupBox5.setLayout(GroupBox5Layout)

        self.startButton = self.createButton("开始处理", self.startrun)
        self.stopButton = self.createButton("停止", self.stoprun)
        self.helpButton = self.createButton("帮助", self.helpWin)
        self.quitButton = self.createButton("退出", self.quitWin)
        self.startButton.setFixedSize(80,25)
        self.stopButton.setFixedSize(55, 25)
        self.helpButton.setFixedSize(55,25)
        self.quitButton.setFixedSize(55,25)

        topLayout2.addWidget(self.GroupBox4)
        topLayout3.addWidget(self.GroupBox5)
        topLayout4.addWidget(self.GroupBox1)
        topLayout4.addWidget(self.GroupBox2)
        topLayout4.addWidget(self.startButton)
        topLayout4.addWidget(self.stopButton)
        topLayout4.addWidget(self.helpButton)
        topLayout4.addWidget(self.quitButton)
        topLayout4.setSpacing(20)

        layout1 = QWidget();  layout2 = QWidget()
        layout3 = QWidget();  layout4 = QWidget()
        layout1.setLayout(topLayout1);  layout2.setLayout(topLayout2)
        layout3.setLayout(topLayout3);  layout4.setLayout(topLayout4)

        mainLayout.addWidget(layout1, Qt.AlignLeft | Qt.AlignTop)
        mainLayout.addWidget(layout2, Qt.AlignLeft | Qt.AlignBottom)
        mainLayout.addWidget(layout3, Qt.AlignLeft | Qt.AlignBottom)
        mainLayout.addWidget(layout4, Qt.AlignLeft | Qt.AlignBottom)
        self.setLayout(mainLayout)

    def createButton(self, text, member):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(member)
        return button

#if __name__ == '__main__':
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
app = QtWidgets.QApplication(sys.argv)
Winshot = Winshot()
sys.exit(app.exec_())
