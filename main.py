import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudio
from PyQt5 import QtMultimedia
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from scipy.signal import filtfilt
from numpy import nonzero, diff


import pyqtgraph as pg
from soundcardlib import SoundCardDataSource
from gui import Ui_MainWindow
import librosa
import math
import pygame

b_canvas = False


input_audio_deviceInfos = QAudioDeviceInfo.availableDevices(QAudio.AudioInput)

audio_path = r"C:\Users\52551\Downloads\AssignmentC\mariachi.mid"
# mixer config
freq = 44100  # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024   # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)

# optional volume 0 to 1.0
pygame.mixer.music.set_volume(0.8)

musicNote = ""
A4 = 440
C0 = A4*pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


#function to convert frequency to notes as defined in name variable
def freq_to_note(freq):
    h = round(12*math.log2(freq/C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)



def rfftfreq(n, d=1.0):
    if not isinstance(n, int):
        raise ValueError("n cannot be decimal")
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val


def fft_buffer(x):
    window = np.hanning(x.shape[0])
    # Calculate FFT
    fx = np.fft.rfft(window * x)
    # Convert to normalised PSD
    Pxx = abs(fx) ** 2 / (np.abs(window) ** 2).sum()
    # Scale for one-sided
    Pxx[1:-1] *= 2
    return Pxx ** 0.5



def find_peaks(Pxx):
    # filter parameters
    b, a = [0.01], [1, -0.99]
    Pxx_smooth = filtfilt(b, a, abs(Pxx))
    peakedness = abs(Pxx) / Pxx_smooth


    peaky_regions = nonzero(peakedness > 1)[0]
    edge_indices = nonzero(diff(peaky_regions) > 10)[0]  
    edges = [0] + [(peaky_regions[i] + 5) for i in edge_indices]
    if len(edges) < 2:
        edges += [len(Pxx) - 1]

    peaks = []
    for i in range(len(edges) - 1):
        j, k = edges[i], edges[i+1]
        peaks.append(j + np.argmax(peakedness[j:k]))
    return peaks


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        global b_canvas
        title = "Remove background noise"
        self.setWindowTitle(title)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # list available mic
        self.devices_list = []
        for device in input_audio_deviceInfos:
            self.devices_list.append(device.deviceName())
        
        # add all the available device name to the combo box
        self.ui.comboBox.addItems(self.devices_list)
        # when the combobox selection changes run the function update_now
        self.ui.comboBox.currentIndexChanged["QString"].connect(self.update_Mic)
        self.ui.comboBox.setCurrentIndex(0)

        self.ui.pbPlay.setCheckable(True)

        self.ui.pbMic.clicked.connect(self.add_mpl)
        # this cause the program hang 
        # self.ui.pbPlay.clicked.connect(self.playAudio)
        self.ui.pbPause.clicked.connect(self.rm_mpl)
        # self.add_mpl()


    def playAudio(self):
        if self.ui.pbPlay.isChecked():
            clock = pygame.time.Clock()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                clock.tick(30) # check if playback has finished
        else:
            pygame.mixer.pause()

    def update_Mic(self, value):
        self.device = self.devices_list.index(value)

    def add_mpl(self):
        global b_canvas
        self.canvas = LiveFFTWindow(self)
        self.canvas.move(25,90)
        self.ui.verticalLayoutLeft.addWidget(self.canvas)

        b_canvas = True
        
    def rm_mpl(self,):
        global b_canvas
        self.ui.verticalLayoutLeft.removeWidget(self.canvas)
        self.canvas.close()

        b_canvas = False


class LiveFFTWindow(pg.GraphicsLayoutWidget):
    FS = 44000
    recorder = SoundCardDataSource(num_chunks=3,
                               sampling_rate=FS,
                               chunk_size=4*1024)

    def __init__(self, parent=None):
        pg.GraphicsLayoutWidget.__init__(self, parent)

        self.recorder = self.recorder
        self.paused = False
        self.logScale = False
        self.showPeaks = True
        self.downsample = True

        # preparing plots
        self.p1 = self.addPlot()
        self.p1.setLabel('bottom', 'Time', 's')
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setTitle("")
        self.p1.setLimits(xMin=0, yMin=-1, yMax=1)
        self.ts = self.p1.plot(pen='y')
        self.nextRow()
        self.p2 = self.addPlot()
        self.p2.setLabel('bottom', 'Frequency', 'Hz')
        self.p2.setLimits(xMin=0, yMin=0)
        self.spec = self.p2.plot(pen=(50, 100, 200),
                                 brush=(50,100,200),
                                 fillLevel=-100)

        # Show the lines for the notes
        A = 440.0
        notePen = pg.mkPen((0, 200, 50, 50))
        while A < (self.recorder.fs / 2):
            self.p2.addLine(x=A, pen=notePen)
            A *= 2

        # denoting pikes with lines (not always showing)
        self.peakMarkers = []

        # rearrange the interval
        self.resetRanges()

        # timer for plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        interval_ms = 1000 * (self.recorder.chunk_size / self.recorder.fs)
        print("Updating graphs every %.1f ms" % interval_ms)
        self.timer.start(int(interval_ms))


    def plotPeaks(self, Pxx):
        # fire if peak is bigger than define threshold (0.3)
        peaks = [p for p in find_peaks(Pxx) if Pxx[p] > 0.3]

        if self.logScale:
            Pxx = 20*np.log10(Pxx)

        # labels for the musical notes
        old = self.peakMarkers
        self.peakMarkers = []
        hz = ""
        for p in peaks:
            if old:
                t = old.pop()
            else:
                t = pg.TextItem(color=(150, 150, 150, 150))
                self.p2.addItem(t)
            self.peakMarkers.append(t)
            hz = self.freqValues[p]
            t.setText(str(freq_to_note(hz)))
            t.setPos(self.freqValues[p], Pxx[p])
        for t in old:
            self.p2.removeItem(t)
            del t
        
        if hz != "":
            print(freq_to_note(hz))

    def update(self):
        if self.paused:
            return
        data = self.recorder.get_buffer()
        weighting = np.exp(self.timeValues / self.timeValues[-1])
        Pxx = fft_buffer(weighting * data[:, 0])

        if self.downsample:
            downsample_args = dict(autoDownsample=False,
                                   downsampleMethod='subsample',
                                   downsample=10)
        else:
            downsample_args = dict(autoDownsample=True)

        self.ts.setData(x=self.timeValues, y=data[:, 0], **downsample_args)
        self.spec.setData(x=self.freqValues,
                          y=(20*np.log10(Pxx) if self.logScale else Pxx))

        if self.showPeaks:
            self.plotPeaks(Pxx)

    def resetRanges(self):
        self.timeValues = self.recorder.timeValues
        self.freqValues = rfftfreq(len(self.timeValues),
                                   1./self.recorder.fs)

        self.p1.setRange(xRange=(0, self.timeValues[-1]), yRange=(-1, 1))
        self.p1.setLimits(xMin=0, xMax=self.timeValues[-1], yMin=-1, yMax=1)
        if self.logScale:
            self.p2.setRange(xRange=(0, self.freqValues[-1] / 2),
                             yRange=(-60, 20))
            self.p2.setLimits(xMax=self.freqValues[-1], yMin=-60, yMax=20)
            self.spec.setData(fillLevel=-100)
            self.p2.setLabel('left', 'PSD', 'dB / Hz')
        else:
            self.p2.setRange(xRange=(0, self.freqValues[-1] / 5),
                             yRange=(0, 3))
            self.p2.setLimits(xMax=self.freqValues[-1], yMax=5)
            self.spec.setData(fillLevel=0)
            self.p2.setLabel('left', 'PSD', '1 / Hz')
    

app = QApplication(sys.argv)
window = Window()
window.show()
app.exec()