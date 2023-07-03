from PyQt5 import QtCore, QtGui, QtWidgets
import Config
import requests
import sys
import datetime
import Controller
import GetCurrent


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1044, 893)
        Form.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        Form.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        Form.setStyleSheet("")
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(220, 30, 541, 801))
        self.widget.setStyleSheet("background:url(:/Img/Image/may.jpg);\n"
                                  "border-radius: 25px;")
        self.widget.setObjectName("widget")
        self.label_x = QtWidgets.QLabel(self.widget)
        self.label_x.setGeometry(QtCore.QRect(500, 10, 32, 32))
        self.label_x.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.label_x.setStyleSheet("background-image:url(:Img/Image/icons8-close-32.png);\n"
                                   " background-repeat: no-repeat;\n"
                                   "                                    background-position: center;\n"
                                   "                                    background-size: cover;\n"
                                   "")
        self.label_x.setText("")
        self.label_x.mousePressEvent = self.close
        self.label_x.setObjectName("label")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(40, 550, 141, 101))
        self.label.setStyleSheet("background: rgba(0,0,0,0.5);\n"
                                 "font: 8pt;\n"
                                 "color:rgb(200,200,200);")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(200, 550, 141, 101))
        self.label_2.setStyleSheet("background: rgba(0,0,0,0.5);\n"
                                   "font: 8pt;\n"
                                   "color:rgb(200,200,200);")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(360, 550, 141, 101))
        self.label_3.setStyleSheet("background: rgba(0,0,0,0.5);\n"
                                   "font: 8pt;\n"
                                   "color:rgb(200,200,200);")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(40, 670, 141, 101))
        self.label_4.setStyleSheet("background: rgba(0,0,0,0.5);\n"
                                   "font: 8pt;\n"
                                   "color:rgb(200,200,200);")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setGeometry(QtCore.QRect(200, 670, 141, 101))
        self.label_5.setStyleSheet("background: rgba(0,0,0,0.5);\n"
                                   "font: 8pt;\n"
                                   "color:rgb(200,200,200);")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setGeometry(QtCore.QRect(360, 670, 141, 101))
        self.label_6.setStyleSheet("background: rgba(0,0,0,0.5);\n"
                                   "font: 8pt;\n"
                                   "color:rgb(200,200,200);")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setGeometry(QtCore.QRect(40, 460, 461, 81))
        self.label_7.setStyleSheet("background: rgba(0,0,0,0.6);\n"
                                   "font: 8pt;\n"
                                   "color:rgb(200,200,200);")
        self.label_7.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        # set margin
        self.label_7.setMargin(13)
        self.label_7.setIndent(0)
        self.label_7.setOpenExternalLinks(False)
        self.label_7.setObjectName("label_7")
        self.label_9 = QtWidgets.QLabel(self.widget)
        self.label_9.setGeometry(QtCore.QRect(300, 560, 31, 31))
        self.label_9.setStyleSheet("background-image: url(:/Img/Image/win.png);\n"
                                   "background-repeat: no-repeat;\n"
                                   "background-position: center;\n"
                                   "background-size: cover;\n"
                                   "")
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.label_8 = QtWidgets.QLabel(self.widget)
        self.label_8.setGeometry(QtCore.QRect(460, 560, 31, 31))
        self.label_8.setStyleSheet("background-image:url(:/Img/Image/icons8-sun-25.png);\n"
                                   "background-repeat: no-repeat;\n"
                                   "background-position: center;\n"
                                   "background-size: cover;\n"
                                   "")
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.label_10 = QtWidgets.QLabel(self.widget)
        self.label_10.setGeometry(QtCore.QRect(460, 680, 31, 31))
        self.label_10.setStyleSheet("background-image:url(:/Img/Image/icons8-atmospheric-pressure-25.png);\n"
                                    "background-repeat: no-repeat;\n"
                                    "background-position: center;\n"
                                    "background-size: cover;\n"
                                    "")
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.widget)
        self.label_11.setGeometry(QtCore.QRect(300, 680, 31, 31))
        self.label_11.setStyleSheet("background-image:url(:/Img/Image/icons8-eye-25.png);\n"
                                    "background-repeat: no-repeat;\n"
                                    "background-position: center;\n"
                                    "background-size: cover;\n"
                                    "")
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.widget)
        self.label_12.setGeometry(QtCore.QRect(140, 680, 31, 31))
        self.label_12.setStyleSheet("background-image:url(:/Img/Image/icons8-thermometer-25.png);\n"
                                    "background-repeat: no-repeat;\n"
                                    "background-position: center;\n"
                                    "background-size: cover;\n"
                                    "")
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.widget)
        self.label_13.setGeometry(QtCore.QRect(140, 560, 31, 31))
        self.label_13.setStyleSheet("background-image:url(:/Img/Image/icons8-hygrometer-25.png);\n"
                                    "background-repeat: no-repeat;\n"
                                    "background-position: center;\n"
                                    "background-size: cover;\n"
                                    "")
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.label_15 = QtWidgets.QLabel(self.widget)
        self.label_15.setGeometry(QtCore.QRect(40, 470, 51, 21))
        self.label_15.setStyleSheet("background-image:url(:/Img/Image/icons8-leaf-25.png);\n"
                                    "background-repeat: no-repeat;\n"
                                    "background-position: center;\n"
                                    "background-size: cover;\n"
                                    "")
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.Tem = QtWidgets.QLabel(self.widget)
        self.Tem.setGeometry(QtCore.QRect(50, 730, 91, 21))
        self.Tem.setStyleSheet("background: rgba(0,0,0,0);\n"
                               "font: 11pt;\n"
                               "color:white;")
        self.Tem.setObjectName("Tem")
        self.Vision = QtWidgets.QLabel(self.widget)
        self.Vision.setGeometry(QtCore.QRect(210, 730, 91, 21))
        self.Vision.setStyleSheet("background: rgba(0,0,0,0);\n"
                                  "font: 11pt;\n"
                                  "color:white;")
        self.Vision.setObjectName("Vision")
        self.Press = QtWidgets.QLabel(self.widget)
        self.Press.setGeometry(QtCore.QRect(370, 730, 91, 21))
        self.Press.setStyleSheet("background: rgba(0,0,0,0);\n"
                                 "font: 11pt;\n"
                                 "color:white;")
        self.Press.setObjectName("Press")
        self.Hum = QtWidgets.QLabel(self.widget)
        self.Hum.setGeometry(QtCore.QRect(50, 610, 91, 21))
        self.Hum.setStyleSheet("background: rgba(0,0,0,0);\n"
                               "font: 11pt;\n"
                               "color:white;")
        self.Hum.setObjectName("Hum")
        self.Win = QtWidgets.QLabel(self.widget)
        self.Win.setGeometry(QtCore.QRect(210, 610, 91, 21))
        self.Win.setStyleSheet("background: rgba(0,0,0,0);\n"
                               "font: 11pt;\n"
                               "color:white;")
        self.Win.setObjectName("Win")
        self.UV = QtWidgets.QLabel(self.widget)
        self.UV.setGeometry(QtCore.QRect(370, 610, 110, 30))
        self.UV.setStyleSheet("background: rgba(0,0,0,0);\n"
                              "font: 11pt;\n"
                              "color:white;")
        self.UV.setObjectName("UV")
        self.Air = QtWidgets.QLabel(self.widget)
        self.Air.setGeometry(QtCore.QRect(80, 490, 150, 30))
        self.Air.setStyleSheet("background: rgba(0,0,0,0);\n"
                               "font: 10pt;\n"
                               "color:white;")
        self.Air.setObjectName("Air")
        self.line = QtWidgets.QFrame(self.widget)
        self.line.setGeometry(QtCore.QRect(60, 520, 421, 4))
        self.line.setStyleSheet("border-radius: 25px;\n"
                                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00ff00, stop:0.5 #F1E421, stop:1 #ff0000);\n"
                                "\n"
                                "\n"
                                "")
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.MucKK = QtWidgets.QLabel(self.widget)
        self.MucKK.setGeometry(QtCore.QRect(120, 515, 4, 15))
        self.MucKK.setStyleSheet("background: white;\n"
                                 "boder-radius:50%;")
        self.MucKK.setText("")
        self.MucKK.setObjectName("MucKK")
        self.label_14 = QtWidgets.QLabel(self.widget)
        self.label_14.setGeometry(QtCore.QRect(45, 60, 291, 71))
        self.label_14.setStyleSheet("font: 40px;\n"
                                    "color: white;\n"
                                    "background: rgba(0,0,0,0);")
        self.label_14.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label_14.setObjectName("label_14")
        self.label_16 = QtWidgets.QLabel(self.widget)
        self.label_16.setGeometry(QtCore.QRect(40, 94, 31, 41))
        self.label_16.setStyleSheet("background-image: url(:/Img/Image/icons8-visit-25.png);\n"
                                    "background-repeat: no-repeat;\n"
                                    "background-position: center;\n"
                                    "background-size: cover;")
        self.label_16.setText("")
        self.label_16.setObjectName("label_16")
        self.Temcur = QtWidgets.QLabel(self.widget)
        self.Temcur.setGeometry(QtCore.QRect(50, 180, 90, 81))
        self.Temcur.setStyleSheet("font: 45px;\n"
                                  "color: white;\n"
                                  "background: rgba(0,0,0,0);")
        self.Temcur.setObjectName("Temcur")
        self.label_18 = QtWidgets.QLabel(self.widget)
        self.label_18.setGeometry(QtCore.QRect(100, 190, 21, 21))
        self.label_18.setStyleSheet(
            "background-image: url(:/Img/Image/icons8-circle-25.png);")
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.weatertext = QtWidgets.QLabel(self.widget)
        self.weatertext.setGeometry(QtCore.QRect(130, 210, 271, 31))
        self.weatertext.setStyleSheet("background: rgba(0,0,0,0);\n"
                                      "font: 13pt;\n"
                                      "color:white;")
        self.weatertext.setObjectName("weatertext")
        self.label_17 = QtWidgets.QLabel(self.widget)
        self.label_17.setGeometry(QtCore.QRect(40, 250, 461, 51))
        self.label_17.setStyleSheet("background: rgba(0,0,0,0);\n"
                                    "font: 11pt;\n"
                                    "color:white;")
        self.label_17.setObjectName("label_17")
        self.icontomo = QtWidgets.QLabel(self.widget)
        self.icontomo.setGeometry(QtCore.QRect(150, 250, 51, 41))
        self.icontomo.setStyleSheet("background-image: url(:/Img/Image/icons8-partly-cloudy-day-48.png);\n"
                                    "background-repeat: no-repeat;\n"
                                    "background-position: center;\n"
                                    "background-size: cover;")
        self.icontomo.setText("")
        self.icontomo.setObjectName("icontomo")
        self.texttomo = QtWidgets.QLabel(self.widget)
        self.texttomo.setGeometry(QtCore.QRect(220, 260, 201, 31))
        self.texttomo.setStyleSheet("background: rgba(0,0,0,0);\n"
                                    "font: 11pt;\n"
                                    "color:white;")
        self.texttomo.setObjectName("texttomo")
        self.temtomo = QtWidgets.QLabel(self.widget)
        self.temtomo.setGeometry(QtCore.QRect(430, 260, 61, 31))
        self.temtomo.setStyleSheet("background: rgba(0,0,0,0);\n"
                                   "font: 11pt;\n"
                                   "color:white;")
        self.temtomo.setAlignment(QtCore.Qt.AlignCenter)
        self.temtomo.setObjectName("temtomo")
        self.now = QtWidgets.QLabel(self.widget)
        self.now.setGeometry(QtCore.QRect(60, 350, 51, 21))
        self.now.setStyleSheet("background: rgba(0,0,0,0);\n"
                               "font: 9pt;\n"
                               "color:white;")
        self.now.setObjectName("now")
        self.iconnow = QtWidgets.QLabel(self.widget)
        self.iconnow.setGeometry(QtCore.QRect(60, 370, 51, 41))
        self.iconnow.setStyleSheet("background-image: url(:/Img/Image/icons8-partly-cloudy-day-48.png);\n"
                                   "background-repeat: no-repeat;\n"
                                   "background-position: center;\n"
                                   "background-size: cover;")
        self.iconnow.setText("")
        self.iconnow.setObjectName("iconnow")
        self.temnow = QtWidgets.QLabel(self.widget)
        self.temnow.setGeometry(QtCore.QRect(60, 420, 51, 21))
        self.temnow.setStyleSheet("background: rgba(0,0,0,0);\n"
                                  "font: 9pt;\n"
                                  "color:white;")
        self.temnow.setAlignment(QtCore.Qt.AlignCenter)
        self.temnow.setObjectName("temnow")
        self.tem1h = QtWidgets.QLabel(self.widget)
        self.tem1h.setGeometry(QtCore.QRect(140, 420, 51, 21))
        self.tem1h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                 "font: 9pt;\n"
                                 "color:white;")
        self.tem1h.setAlignment(QtCore.Qt.AlignCenter)
        self.tem1h.setObjectName("tem1h")
        self.icon1h = QtWidgets.QLabel(self.widget)
        self.icon1h.setGeometry(QtCore.QRect(140, 370, 51, 41))
        self.icon1h.setStyleSheet("background-image: url(:/Img/Image/icons8-partly-cloudy-day-48.png);\n"
                                  "background-repeat: no-repeat;\n"
                                  "background-position: center;\n"
                                  "background-size: cover;")
        self.icon1h.setText("")
        self.icon1h.setObjectName("icon1h")
        self.text1h = QtWidgets.QLabel(self.widget)
        self.text1h.setGeometry(QtCore.QRect(140, 350, 51, 21))
        self.text1h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                  "font: 9pt;\n"
                                  "color:white;")
        self.text1h.setAlignment(QtCore.Qt.AlignCenter)
        self.text1h.setObjectName("text1h")
        self.tem2h = QtWidgets.QLabel(self.widget)
        self.tem2h.setGeometry(QtCore.QRect(210, 420, 51, 21))
        self.tem2h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                 "font: 9pt;\n"
                                 "color:white;")
        self.tem2h.setAlignment(QtCore.Qt.AlignCenter)
        self.tem2h.setObjectName("tem2h")
        self.icon2h = QtWidgets.QLabel(self.widget)
        self.icon2h.setGeometry(QtCore.QRect(210, 370, 51, 41))
        self.icon2h.setStyleSheet("background-image: url(:/Img/Image/icons8-partly-cloudy-day-48.png);\n"
                                  "background-repeat: no-repeat;\n"
                                  "background-position: center;\n"
                                  "background-size: cover;")
        self.icon2h.setText("")
        self.icon2h.setObjectName("icon2h")
        self.text2h = QtWidgets.QLabel(self.widget)
        self.text2h.setGeometry(QtCore.QRect(210, 350, 51, 21))
        self.text2h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                  "font: 9pt;\n"
                                  "color:white;")
        self.text2h.setAlignment(QtCore.Qt.AlignCenter)
        self.text2h.setObjectName("text2h")
        self.tem4h = QtWidgets.QLabel(self.widget)
        self.tem4h.setGeometry(QtCore.QRect(370, 420, 51, 21))
        self.tem4h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                 "font: 9pt;\n"
                                 "color:white;")
        self.tem4h.setAlignment(QtCore.Qt.AlignCenter)
        self.tem4h.setObjectName("tem4h")
        self.icon4h = QtWidgets.QLabel(self.widget)
        self.icon4h.setGeometry(QtCore.QRect(370, 370, 51, 41))
        self.icon4h.setStyleSheet("background-image: url(:/Img/Image/icons8-partly-cloudy-day-48.png);\n"
                                  "background-repeat: no-repeat;\n"
                                  "background-position: center;\n"
                                  "background-size: cover;")
        self.icon4h.setText("")
        self.icon4h.setObjectName("icon4h")
        self.text4h = QtWidgets.QLabel(self.widget)
        self.text4h.setGeometry(QtCore.QRect(370, 350, 51, 21))
        self.text4h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                  "font: 9pt;\n"
                                  "color:white;")
        self.text4h.setAlignment(QtCore.Qt.AlignCenter)
        self.text4h.setObjectName("text4h")
        self.tem3h = QtWidgets.QLabel(self.widget)
        self.tem3h.setGeometry(QtCore.QRect(290, 420, 51, 21))
        self.tem3h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                 "font: 9pt;\n"
                                 "color:white;")
        self.tem3h.setAlignment(QtCore.Qt.AlignCenter)
        self.tem3h.setObjectName("tem3h")
        self.icon3h = QtWidgets.QLabel(self.widget)
        self.icon3h.setGeometry(QtCore.QRect(290, 370, 51, 41))
        self.icon3h.setStyleSheet("background-image: url(:/Img/Image/icons8-partly-cloudy-day-48.png);\n"
                                  "background-repeat: no-repeat;\n"
                                  "background-position: center;\n"
                                  "background-size: cover;")
        self.icon3h.setText("")
        self.icon3h.setObjectName("icon3h")
        self.text3h = QtWidgets.QLabel(self.widget)
        self.text3h.setGeometry(QtCore.QRect(290, 350, 51, 21))
        self.text3h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                  "font: 9pt;\n"
                                  "color:white;")
        self.text3h.setAlignment(QtCore.Qt.AlignCenter)
        self.text3h.setObjectName("text3h")
        self.text5h = QtWidgets.QLabel(self.widget)
        self.text5h.setGeometry(QtCore.QRect(440, 350, 51, 21))
        self.text5h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                  "font: 9pt;\n"
                                  "color:white;")
        self.text5h.setAlignment(QtCore.Qt.AlignCenter)
        self.text5h.setObjectName("text5h")
        self.icon5h = QtWidgets.QLabel(self.widget)
        self.icon5h.setGeometry(QtCore.QRect(440, 370, 51, 41))
        self.icon5h.setStyleSheet("background-image: url(:/Img/Image/icons8-partly-cloudy-day-48.png);\n"
                                  "background-repeat: no-repeat;\n"
                                  "background-position: center;\n"
                                  "background-size: cover;")
        self.icon5h.setText("")
        self.icon5h.setObjectName("icon5h")
        self.tem5h = QtWidgets.QLabel(self.widget)
        self.tem5h.setGeometry(QtCore.QRect(440, 420, 51, 21))
        self.tem5h.setStyleSheet("background: rgba(0,0,0,0);\n"
                                 "font: 9pt;\n"
                                 "color:white;")
        self.tem5h.setAlignment(QtCore.Qt.AlignCenter)
        self.tem5h.setObjectName("tem5h")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "  Độ ẩm"))
        self.label_2.setText(_translate("Form", "  Gió "))
        self.label_3.setText(_translate("Form", "   UV"))
        self.label_4.setText(_translate("Form", "  Nhiệt độ cảm nhận"))
        self.label_5.setText(_translate("Form", "  Tầm nhìn"))
        self.label_6.setText(_translate("Form", "  Áp suất không khí"))
        self.label_7.setText(_translate("Form", "      Chất lượng không khí"))
        self.Tem.setText(_translate("Form", "37 C"))
        self.Vision.setText(_translate("Form", "9 km"))
        self.Press.setText(_translate("Form", "1.003 hPa "))
        self.Hum.setText(_translate("Form", "84 %"))
        self.Win.setText(_translate("Form", "cấp: 3"))
        self.UV.setText(_translate("Form", "0"))
        self.Air.setText(_translate("Form", "Tốt  84"))
        self.label_14.setText(_translate("Form", "Hà Nội"))
        self.Temcur.setText(_translate("Form", "38"))
        self.weatertext.setText(_translate("Form", "Chủ yếu là nắng"))
        self.label_17.setText(_translate("Form", "  Ngày mai"))
        self.texttomo.setText(_translate("Form", "Có mây"))
        self.temtomo.setText(_translate("Form", "30/40 C"))
        self.now.setText(_translate("Form", "Bây giờ"))
        self.temnow.setText(_translate("Form", "38 C"))
        self.tem1h.setText(_translate("Form", "38 C"))
        self.text1h.setText(_translate("Form", "Bây giờ"))
        self.tem2h.setText(_translate("Form", "38 C"))
        self.text2h.setText(_translate("Form", "Bây giờ"))
        self.tem4h.setText(_translate("Form", "38 C"))
        self.text4h.setText(_translate("Form", "Bây giờ"))
        self.tem3h.setText(_translate("Form", "38 C"))
        self.text3h.setText(_translate("Form", "Bây giờ"))
        self.text5h.setText(_translate("Form", "Bây giờ"))
        self.tem5h.setText(_translate("Form", "38 C"))

    # close app when click close button
    def close(self, event):
        sys.exit()



app = QtWidgets.QApplication(sys.argv)
Form = QtWidgets.QWidget()
ui = Ui_Form()
ui.setupUi(Form)
url = "https://api.weatherbit.io/v2.0/current?&city=Hanoi&country=VN&key=f7f4fb4d3ad146e4862e400a8038631b&include=minutely"

response = requests.get(url)
data = response.json()


tem = round(data['data'][0]['temp'])
app_temp = round(data['data'][0]['app_temp'])
uv = int(data['data'][0]['uv'])
win_spd = data['data'][0]['wind_spd']
weather_des = data['data'][0]['weather']['description']
hum = data['data'][0]['rh']
pres = data['data'][0]['pres']
vis = data['data'][0]['vis']
aqi = data['data'][0]['aqi']

# data_now = GetCurrent.data_now

# tem = round(data_now[8] - 273.15)
# app_temp = round(data_now[5] - 273.15)
# uv = int(data_now[10])
# win_spd = data_now[1]
# weather_des = data_now[9]
# hum = data_now[0]
# pres = data_now[4]
# vis = data_now[-2]
# aqi = 100


weather_de = GetCurrent.get_weather(weather_des.lower())

if weather_de == 0:
    weather_de = "Nhiều mây"
elif weather_de == 1:
    weather_de = "Nắng"
elif weather_de == 2:
    weather_de = "Mưa"


now = datetime.datetime.now()
current_time = now.strftime("%H")
current_time = int(current_time)

# nomalize time
time = [current_time+1, current_time+2,
        current_time+3, current_time+4, current_time+5]
for i in range(len(time)):
    if time[i] > 23:
        time[i] = time[i] - 24

# nomalize aqi
if aqi < 90:
    aqi_text = "Tốt"
elif aqi < 120:
    aqi_text = "Trung bình"
elif aqi < 150:
    aqi_text = "Có hại"
else:
    aqi_text = "Nguy hại"

# nomalize UV
if uv < 3:
    uv_text = "Yếu"
elif uv < 6:
    uv_text = "Trung bình"
elif uv < 8:
    uv_text = "Cao"
else:
    uv_text = "Rất cao"

# set x for MucKK by aqi

x = int(aqi/430*360 + 60)


# set data to ui
ui.Tem.setText(str(app_temp) + " C")
ui.Temcur.setText(str(tem))
ui.temnow.setText(str(tem)+" C")

ui.Vision.setText(str(vis)+" km")
ui.Press.setText(str(pres)+" hPa")
ui.Hum.setText(str(hum)+" %")
ui.Win.setText(str(win_spd) + " m/s" )
ui.UV.setText(str(uv)+" "+uv_text)
ui.weatertext.setText(str(weather_de))
ui.Air.setText(aqi_text+" "+str(aqi))
ui.MucKK.setGeometry(QtCore.QRect(x, 515, 4, 15))

# set time
ui.now.setText("Bây giờ")
ui.text1h.setText(str(time[0])+" giờ")
ui.text2h.setText(str(time[1])+" giờ")
ui.text3h.setText(str(time[2])+" giờ")
ui.text4h.setText(str(time[3])+" giờ")
ui.text5h.setText(str(time[4])+" giờ")

# set tem predict from Controller

temp = Controller.predict_temp()


ui.tem1h.setText(str(temp[0])+" C")
ui.tem2h.setText(str(temp[1])+" C")
ui.tem3h.setText(str(temp[2])+" C")
ui.tem4h.setText(str(temp[3])+" C")
ui.tem5h.setText(str(temp[4])+" C")


# # set icon weather to background

ui.iconnow.setStyleSheet("background-image:" + Controller.get_icon_url(GetCurrent.get_weather(weather_des.lower())))
ui.icon1h.setStyleSheet("background-image:" + Controller.predict_weather_1h())
ui.icon2h.setStyleSheet("background-image:" + Controller.predict_weather_2h())
ui.icon3h.setStyleSheet("background-image:" + Controller.predict_weather_3h())
ui.icon4h.setStyleSheet("background-image:" + Controller.predict_weather_4h())
ui.icon5h.setStyleSheet("background-image:" + Controller.predict_weather_5h())

# set tommorow weather

ui.temtomo.setText(str(Controller.predict_temp_day())+" C")
ui.texttomo.setText(Controller.predict_weather_day()[0])
ui.icontomo.setStyleSheet("background-image:" + Controller.predict_weather_day()[1])


if __name__ == "__main__":

    Form.show()
    sys.exit(app.exec_())
