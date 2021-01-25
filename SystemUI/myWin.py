# encoding=utf-8

import sys
from SystemUI.startupWin import *
from SystemUI.mainWin import *
from PyQt5.QtWidgets import *
import datetime
from PyQt5 import QtCore
import time
from calibration import calibration
from reconstruction import rec_config
from reconstruction.spare import sfmui
from reconstruction.dense import PMVSui, Dense_filterui
import numpy as np
import vispy.scene
from vispy.scene import visuals


class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))


class StartWin(QMainWindow, Ui_MainWindow):
    # welcome_page = []
    def __init__(self, parent=None):
        super(StartWin, self).__init__(parent)
        self.setupUi(self)
        # self.welcome_page.append(self.label)
        # self.welcome_page.append(self.pushButton)

        self.addWidgets()

        # 设置窗体无边框
        # self.setWindowFlags(QtCore.Qt.CustomizeWindowHint)

        # 设置文本框及按钮

    def addWidgets(self):
        self.set_name_tip = QtWidgets.QLabel(self.centralwidget)
        self.set_name_tip.setText("请输入工程文件名")
        self.set_name_tip.setGeometry(QtCore.QRect(400, 350, 180, 40))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.set_name_tip.setFont(font)
        self.set_name_tip.setObjectName("setnametip")
        self.set_name_tip.close()

        self.line_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.line_edit.setGeometry(QtCore.QRect(350, 390, 250, 40))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.line_edit.setFont(font)
        self.line_edit.setObjectName("LineEdit")
        self.line_edit.close()

        self.name_sub_button = QtWidgets.QPushButton(self.centralwidget)
        self.name_sub_button.setGeometry(QtCore.QRect(400, 438, 70, 40))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.name_sub_button.setFont(font)
        self.name_sub_button.setObjectName("namesubButton")
        self.name_sub_button.setText("确认")
        self.name_sub_button.close()

        self.cancle_button = QtWidgets.QPushButton(self.centralwidget)
        self.cancle_button.setGeometry(QtCore.QRect(480, 438, 70, 40))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.cancle_button.setFont(font)
        self.cancle_button.setObjectName("cancleButton")
        self.cancle_button.setText("取消")
        self.cancle_button.close()

        # # 加到数组中
        # self.welcome_page.append(self.setnametip)
        # self.welcome_page.append(self.LineEdit)
        # self.welcome_page.append(self.namesubButton)
        # self.welcome_page.append(self.cancleButton)

        # 绑定事件
        self.push_button.clicked.connect(self.setName)
        self.cancle_button.clicked.connect(self.cancleSetName)
        self.name_sub_button.clicked.connect(self.subName)

    def getProjectName(self):
        return self.project_name

    def setName(self):
        self.push_button.close()
        self.cancle_button.show()
        self.name_sub_button.show()
        self.set_name_tip.show()
        self.line_edit.show()

    def cancleSetName(self):
        self.push_button.show()
        self.cancle_button.close()
        self.name_sub_button.close()
        self.set_name_tip.close()
        self.line_edit.close()

    def subName(self):
        Input_text = self.line_edit.text()
        if Input_text != "":
            self.project_name = Input_text
            # print(Input_text)
        else:
            self.project_name = "Project-" + str(datetime.datetime.now()).split()[0]  # 默认工程文件名
        print("project name:", self.project_name)

        # 保存文件名到txt文件中
        with open("../project_name.txt", "w") as f:
            f.write(self.project_name)
            f.close()
        Start.close()
        main_win.show()


class MainWin(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(MainWin, self).__init__(parent)
        self.setupUi(self)

        # 下面将输出重定向到textBrowser中
        sys.stdout = EmittingStr(textWritten=self.outputWritten)
        sys.stderr = EmittingStr(textWritten=self.outputWritten)

        self.push_button.clicked.connect(self.openCaliImages)
        self.text_browser.close()

        self.addWidgets()

    def addWidgets(self):
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(290, 230, 381, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_4.setText("相机标定完毕，请选择重建图片")
        self.label_4.close()

        self.reconsbtn = QtWidgets.QPushButton(self)
        self.reconsbtn.setGeometry(QtCore.QRect(340, 320, 240, 41))
        self.reconsbtn.setObjectName("reconsbtn")
        self.reconsbtn.setText("点击选择重建图片文件夹")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.reconsbtn.setFont(font)
        self.reconsbtn.clicked.connect(self.recons)
        self.reconsbtn.close()

        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(290, 230, 381, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_4")
        self.label_5.setText("稀疏重建完成，现在开始稠密重建")
        self.label_5.close()

        self.preview_sparse_btn = QtWidgets.QPushButton(self)
        self.preview_sparse_btn.setGeometry(QtCore.QRect(340, 380, 240, 41))
        self.preview_sparse_btn.setObjectName("sparsebtn")
        self.preview_sparse_btn.setText("点击显示稀疏重建结果")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.preview_sparse_btn.setFont(font)
        self.preview_sparse_btn.clicked.connect(self.showSparse)
        self.preview_sparse_btn.close()

        self.dense_recons_btn = QtWidgets.QPushButton(self)
        self.dense_recons_btn.setGeometry(QtCore.QRect(340, 320, 240, 41))
        self.dense_recons_btn.setObjectName("reconsbtn")
        self.dense_recons_btn.setText("点击开始稠密重建")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.dense_recons_btn.setFont(font)
        self.dense_recons_btn.clicked.connect(self.denseRecons)
        self.dense_recons_btn.close()

        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(290, 230, 381, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_4")
        self.label_6.setText("重建完成，还需过滤掉一些杂点")
        self.label_6.close()

        self.filter_btn = QtWidgets.QPushButton(self)
        self.filter_btn.setGeometry(QtCore.QRect(355, 320, 240, 41))
        self.filter_btn.setObjectName("filterbtn")
        self.filter_btn.setText("点击开始点云过滤")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.filter_btn.setFont(font)
        self.filter_btn.clicked.connect(self.pointFilter)
        self.filter_btn.close()

        self.preview_dense_back_btn = QtWidgets.QPushButton(self)
        self.preview_dense_back_btn.setGeometry(QtCore.QRect(355, 380, 240, 41))
        self.preview_dense_back_btn.setObjectName("previewdensebackbtn")
        self.preview_dense_back_btn.setText("点击显示稠密重建结果")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.preview_dense_back_btn.setFont(font)
        self.preview_dense_back_btn.clicked.connect(self.showDenseWithBack)
        self.preview_dense_back_btn.close()

        self.label_7 = QtWidgets.QLabel(self)
        self.label_7.setGeometry(QtCore.QRect(440, 230, 381, 51))
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_4")
        self.label_7.setText("重建完成！")
        self.label_7.close()

        self.finish_btn = QtWidgets.QPushButton(self)
        self.finish_btn.setGeometry(QtCore.QRect(370, 320, 240, 41))
        self.finish_btn.setObjectName("reconsbtn")
        self.finish_btn.setText("完成")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.finish_btn.setFont(font)
        self.finish_btn.clicked.connect(self.recFinish)
        self.finish_btn.close()

        self.show_final_btn = QtWidgets.QPushButton(self)
        self.show_final_btn.setGeometry(QtCore.QRect(370, 380, 240, 41))
        self.show_final_btn.setObjectName("showfinalbtn")
        self.show_final_btn.setText("点击显示足部重建结果")
        font = QtGui.QFont()
        font.setFamily("幼圆")
        font.setPointSize(10)
        self.show_final_btn.setFont(font)
        self.show_final_btn.clicked.connect(self.showFinal)
        self.show_final_btn.close()

    def outputWritten(self, text):
        cursor = self.text_browser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.text_browser.setTextCursor(cursor)
        self.text_browser.ensureCursorVisible()

    def showInVispy(self, path):
        # Make a canvas and add simple view
        canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()

        # 读取点云
        with open(path, "r") as f:
            lines = f.readlines()
            lines = lines[13:]
            points = np.ones((len(lines), 3))
            colors = []
            for i in range(len(lines)):
                points[i, :3] = list(map(float, lines[i].strip("\n").split(" ")[:3]))
                colors.append(tuple(list(map(float, lines[i].strip("\n").split(" ")[-3:]))))
            colors = np.array(colors) / 255

        # create scatter object and fill in the data
        scatter = visuals.Markers()
        scatter.set_data(points[:, :3], edge_color=None, face_color=colors, size=4)

        view.add(scatter)
        view.camera = 'turntable'  # or try 'arcball'

        # add a colored 3D axis for orientation
        axis = visuals.XYZAxis(parent=view.scene)

        if sys.flags.interactive != 1:
            vispy.app.run()

    def openCaliImages(self):
        self.cali_img_dir = QFileDialog.getExistingDirectory(self, "选取文件夹", "../")  # 起始路径

        calibration.cali_img_dir = self.cali_img_dir
        self.cali_thead = cali()
        self.cali_thead.signal.connect(self.callback1)

        self.cali_thead.start()
        self.label_3.setText("正在进行相机标定，请等候片刻")
        self.label_3.setGeometry(QtCore.QRect(290, 230, 381, 51))
        self.text_browser.show()
        self.push_button.close()

    def recons(self):
        self.rec_img_dir = QFileDialog.getExistingDirectory(self, "选取文件夹", "../")  # 起始路径

        rec_config.image_dir = self.rec_img_dir
        self.rect_head = rec()
        self.rect_head.signal.connect(self.callback2)
        self.rect_head.start()
        self.text_browser.show()
        # self.label_4.setGeometry(QtCore.QRect(190, 270, 381, 51))
        self.label_4.setText("正在进行稀疏重建，请等候片刻...")
        self.reconsbtn.close()

    def denseRecons(self):
        PMVSui.CMVS.image_dir = self.rec_img_dir
        # print(self.rec_img_dir)
        self.denserecthead = DenseRec()
        self.denserecthead.signal.connect(self.callback3)
        self.denserecthead.start()
        # self.textBrowser.show()
        # self.label_5.setGeometry(QtCore.QRect(190, 270, 381, 51))
        self.label_5.setText("正在进行稠密重建，请等候片刻...")
        self.dense_recons_btn.close()
        self.preview_sparse_btn.close()

    def pointFilter(self):
        self.pfilt = PointFilt()
        self.model_path = QFileDialog.getExistingDirectory(self, "选取路径", "../")  # 起始路径
        self.pfilt.signal.connect(self.callback4)
        self.pfilt.start()
        # self.text_browser.show()
        # self.label_6.setGeometry(QtCore.QRect(190, 270, 381, 51))
        self.label_6.setText("正在进行过滤多余点云，请等候片刻")
        self.filter_btn.close()
        self.preview_dense_back_btn.close()

    def recFinish(self):
        main_win.close()
        Start.show()
        Start.push_button.show()
        Start.set_name_tip.close()
        Start.cancle_button.close()
        Start.name_sub_button.close()
        Start.line_edit.close()

    def showSparse(self):
        sfmui.save_sparse()
        self.showInVispy(self.rec_img_dir + '/sparse.ply')

    def showDenseWithBack(self):
        # # 将ply转换成txt
        # with open("../reconstruction/dense/pmvs/models/option-0000.ply", "r+") as f:
        #     lines = f.readlines()
        # with open('../reconstruction/dense/pmvs/models/dense.txt', "w") as f:
        #     for i in range(13, len(lines)):
        #         if lines[i] != '':
        #             f.write(" ".join(lines[i].split()[:3]) + "\n")
        self.showInVispy('../reconstruction/dense/pmvs/models/option-0000.ply')

    def showFinal(self):
        self.showInVispy(self.model_path + '/' + Start.getProjectName() + '.ply')

    def callback1(self, i):
        # self.label_3.setText("aa")
        print("=============相机标定完成=============")
        # self.mythead.terminate()
        main_win.close()
        main_win.show()

        self.label_3.close()
        self.push_button.close()
        # self.textBrowser.close()
        self.label_4.show()
        self.reconsbtn.show()
        self.label_2.setText("*------稀疏重建--------*")

    def callback2(self, i):
        # self.label_3.setText("aa")
        print("=============稀疏重建完成=============")
        main_win.close()
        main_win.show()

        self.label_4.close()
        # self.reconsbtn.close()
        # self.textBrowser.close()
        self.label_5.show()
        self.dense_recons_btn.show()
        self.preview_sparse_btn.show()
        self.label_2.setText("*------稠密重建--------*")

    def callback3(self, i):
        # self.label_3.setText("aa")
        print("=============稠密重建完成=============")
        main_win.close()
        main_win.show()

        self.label_5.close()
        # self.densereconsbtn.close()
        # self.previewsparsebtn.close()
        # self.textBrowser.close()
        self.label_2.setText("*------过滤点云--------*")
        self.label_6.show()
        self.filter_btn.show()
        self.preview_dense_back_btn.show()

    def callback4(self, i):
        # self.label_3.setText("aa")
        print("=============点云过滤完成=============")
        main_win.close()
        main_win.show()

        self.label_6.close()
        # self.filterbtn.close()
        # self.previewdensebackbtn.close()
        # self.textBrowser.close()
        self.label_2.setText("*------完成--------*")
        self.label_7.show()
        self.finish_btn.show()
        self.show_final_btn.show()

    def getModelPath(self):
        return self.model_path


class cali(QtCore.QThread):  # 建立相机标定子线程
    signal = QtCore.pyqtSignal(bool)

    def __init__(self):
        super(cali, self).__init__()

    def run(self):
        print("===========现在开始相机标定===========")
        calibration.Zhang()
        # time.sleep(2)
        self.signal.emit(True)


class rec(QtCore.QThread):
    signal = QtCore.pyqtSignal(bool)

    def __init__(self):
        super(rec, self).__init__()

    def run(self):
        print("===========现在开始稀疏重建===========")
        sfmui.SFM()
        # time.sleep(2)
        self.signal.emit(True)


class DenseRec(QtCore.QThread):
    signal = QtCore.pyqtSignal(bool)

    def __init__(self):
        super(DenseRec, self).__init__()

    def run(self):
        print("===========现在开始稠密重建===========")
        PMVSui.denserec()
        time.sleep(3)
        self.signal.emit(True)


class PointFilt(QtCore.QThread):
    signal = QtCore.pyqtSignal(bool)

    def __init__(self):
        super(PointFilt, self).__init__()

    def run(self):
        print("===========现在开始点云过滤===========")
        # Dense_filterui.pointfilter(mainwin.getmodelpath(),Start.getprojectname())
        # time.sleep(2)
        self.signal.emit(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Start = StartWin()
    Start.show()
    main_win = MainWin()

    sys.exit(app.exec_())
