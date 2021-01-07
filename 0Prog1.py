from tkinter import *
from tkinter.ttk import *
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import Checkbutton

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scipy import signal
from math import log10, ceil
import numpy as np


#from rtlsdr import *
import rtlsdr

class Application(Frame):
	"""Implements the spectrum analyzer application"""
	def __init__(self, master=None):
		Frame.__init__(self, master)
		self.pack(side='top', fill='both', expand=True)
		self.master.protocol('WM_DELETE_WINDOW', self.mainWindowOnClose)
		
		self.ax1_lines = []
		self.Data = []
		self.maxHoldData = []
		self.minHoldData = []
		self.averageData = []
		self.hm_im_line = 500
		self.ImshowData = np.zeros((self.hm_im_line,100))
		self.ifft_Data = []
		
		# spectrum analyzer default settings
		self.center_freq = 97.97 # MHz
		self.span = 2 # MHz
		self.numsamples = 4096
		self.avgsamples = 8
		self.rbw = 0.5 # kHz
		self.ampMax = -20
		self.ampMin = -100
		self.spanMin = self.center_freq - self.span / 2
		self.spanMax = self.center_freq + self.span / 2 
		
		# configure SDR
		self.sdr = rtlsdr.RtlSdr()
		self.sdr.sample_rate = 2.048e6 # Hz
		self.sdr.center_freq = self.center_freq * 1e6 # Hz
		self.sdr.gain = 'auto'
		
		
		self.createMainWindow()
		self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.animation = ani.FuncAnimation(self.fig, self.animate, repeat=False, interval=5)   
	
	def onclick(self, event):
		# print("onclick",'%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
			# ('double' if event.dblclick else 'single', event.button,
			# event.x, event.y, event.xdata, event.ydata))
	
		if event.inaxes==None: return
		if event.dblclick:
			colour_line="#990000"
			if len(event.inaxes.lines)>=3:
				#event.inaxes.get_ylim()
				#event.inaxes.lines = [event.inaxes.lines[0], event.inaxes.lines[2]]
				event.inaxes.lines[1].set_data(event.inaxes.lines[2].get_xdata(),[event.inaxes.lines[0].get_xydata().min(),event.inaxes.lines[0].get_xydata().max()])
				event.inaxes.lines[1].set_color(colour_line)
				event.inaxes.lines[2].set_data([event.xdata,event.xdata],[event.inaxes.lines[0].get_xydata().min(),event.inaxes.lines[0].get_xydata().max()])
				event.inaxes.lines[2].set_color(colour_line)
			else:
				event.inaxes.axvline(event.xdata, color=colour_line)
				event.inaxes.axvline(event.xdata+10, color=colour_line)
			event.canvas.draw()
			self.ax1_lines=event.inaxes.lines[-2:]
	
	def createMainWindow(self):
		"""creates the main application window"""
		# create embedded plot window
		# self.fig = Figure(figsize=(10,5), dpi=100)
		# self.ax[0] = self.fig.add_subplot(211)
		# self.ax[1] = self.fig.add_subplot(212)
		self.fig, self.ax = plt.subplots(2)
		
		plotFrame = Frame(self)
		plotFrame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky='n')
		
		plotCanvas = FigureCanvasTkAgg(self.fig, plotFrame)
		plotCanvas.draw()
		plotCanvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
		
		plotCanvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)
		
		# Spectrum Analyzer Controls		
		controlFrame = LabelFrame(self, text='Display', relief='groove')
		controlFrame.grid(row=0, column=1, padx=5, pady=5, sticky='n')
		
		Label(controlFrame, text='Freq. center (MHz): ').grid(row=1, column=0, padx=5, pady=5, sticky='e')
		self.fcButVar = StringVar()
		self.fcButVar.set(str(self.center_freq))
		self.fcButton = Button(controlFrame, textvariable=self.fcButVar, command=self.setFreqCentre)
		self.fcButton.grid(row=1, column=1, padx=5, pady=5, sticky='w')
		
		# Label(controlFrame, text='Freq. start (MHz): ').grid(row=2, column=0, padx=5, pady=5, sticky='e')
		# self.fStartButVar = StringVar()
		# self.fStartButVar.set(str(self.spanMin))
		# self.fStartButton = Button(controlFrame, textvariable=self.fStartButVar, command=self.setFreqStart, state='disabled')
		# self.fStartButton.grid(row=2, column=1, padx=5, pady=5, sticky='w')
		
		# Label(controlFrame, text='Freq. stop (MHz): ').grid(row=3, column=0, padx=5, pady=5, sticky='e')
		# self.fStopButVar = StringVar()
		# self.fStopButVar.set(str(self.spanMax))
		# self.fStopButton = Button(controlFrame, textvariable=self.fStopButVar, command=self.setFreqStop, state='disabled')
		# self.fStopButton.grid(row=3, column=1, padx=5, pady=5, sticky='w')
		
		Label(controlFrame, text='Span (MHz): ').grid(row=4, column=0, padx=5, pady=5, sticky='e')
		self.spanButVar = StringVar()
		self.spanButVar.set(str(self.span))
		self.spanButton = Button(controlFrame, textvariable=self.spanButVar, command=self.setFreqSpan)
		self.spanButton.grid(row=4, column=1, padx=5, pady=5, sticky='w')
		
		# Label(controlFrame, text='RBW (kHz): ').grid(row=5, column=0, padx=5, pady=5, sticky='e')
		# self.rbwButVar = StringVar()
		# self.rbwButVar.set(str(self.rbw))
		# self.rbwButton = Button(controlFrame, textvariable=self.rbwButVar, command=self.setRbw)
		# self.rbwButton.grid(row=5, column=1, padx=5, pady=5, sticky='w')
		
		Label(controlFrame, text='Amp max (dB/Hz): ').grid(row=6, column=0, padx=5, pady=5, sticky='e')
		self.ampMaxButVar = StringVar()
		self.ampMaxButVar.set(str(self.ampMax))
		self.ampMaxButton = Button(controlFrame, textvariable=self.ampMaxButVar, command=self.setAmpMax)
		self.ampMaxButton.grid(row=6, column=1, padx=5, pady=5, sticky='w')
		
		Label(controlFrame, text='Amp min (dB/Hz): ').grid(row=7, column=0, padx=5, pady=5, sticky='e')
		self.ampMinButVar = StringVar()
		self.ampMinButVar.set(str(self.ampMin))
		self.ampMinButton = Button(controlFrame, textvariable=self.ampMinButVar, command=self.setAmpMin)
		self.ampMinButton.grid(row=7, column=1, padx=5, pady=5, sticky='w')
		
		# Label(controlFrame, text='sample averages: ').grid(row=8, column=0, padx=5, pady=5, sticky='e')
		# self.sampleAvgButVar = StringVar()
		# self.sampleAvgButVar.set(str(self.avgsamples))
		# self.sampleAvgButton = Button(controlFrame, textvariable=self.sampleAvgButVar, command=self.setSampleAvg)
		# self.sampleAvgButton.grid(row=8, column=1, padx=5, pady=5, sticky='w')
		
		# button to pause/resume updating the plot
		self.plotCtrlButVar = StringVar()
		self.plotCtrlButVar.set('Pause')
		self.plotCtrlButton = Button(controlFrame, textvariable=self.plotCtrlButVar, command=self.setPlotStatus)
		self.plotCtrlButton.grid(row=9, column=0, columnspan=2, padx=5, pady=5)
		
		# selections for type of plots to display
		plotControlFrame = LabelFrame(self, text='Plots', relief='groove')
		plotControlFrame.grid(row=1, column=1, padx=5, pady=5, sticky='nwes')
		
		self.clearWriteVar = IntVar()
		self.clearWriteVar.set(1)
		Checkbutton(plotControlFrame, text='[ax[0]] Data', variable=self.clearWriteVar, fg='blue').grid(row=0, column=0, padx=5, pady=5, sticky='w')
		
		self.maxHoldVar = IntVar()
		self.maxHoldVar.set(0)
		Checkbutton(plotControlFrame, text='[ax[0]] Max hold', variable=self.maxHoldVar, command=self.clearMaxHold, fg='red').grid(row=1, column=0, padx=5, pady=5, sticky='w')
		
		self.minHoldVar = IntVar()
		self.minHoldVar.set(0)
		Checkbutton(plotControlFrame, text='[ax[0]] Min hold', variable=self.minHoldVar, command=self.clearMinHold, fg='green').grid(row=2, column=0, padx=5, pady=5, sticky='w')
		
		# self.avgPlotVar = IntVar()
		# self.avgPlotVar.set(0)
		# Checkbutton(plotControlFrame, text='[ax[0]] Average', variable=self.avgPlotVar, command=self.clearAverageHold, fg='gold').grid(row=3, column=0, padx=5, pady=5, sticky='w')
		
		self.ImshowVar = IntVar()
		self.ImshowVar.set(1)
		Checkbutton(plotControlFrame, text='[ax[1]] Imshow', variable=self.ImshowVar, command=self.clearImshow, fg='black').grid(row=3, column=0, padx=5, pady=5, sticky='w')
		
	def mainWindowOnClose(self):
		"""function to call when the main window closes"""
		self.quit()
		
	def clearMaxHold(self):
		"""This function clears the max hold array"""
		self.maxHoldData = []
		
	def clearMinHold(self):
		"""This function clears the min hold array"""
		self.minHoldData = []
		
	def clearAverageHold(self):
		"""This function clears the average hold array"""
		self.averageData = []
		
	def clearImshow(self):
		"""This function clears the imshow hold array"""
		# if self.ImshowVar.get():
			# self.ImshowData = []
		# else:
			# self.ax[0].clear()

	def setFreqStart(self):
		"""sets the start frequency of the analyzer"""
		newFreq = simpledialog.askfloat("Start Frequency", "Please enter the start frequency in MHz", parent=self, minvalue=25.0, maxvalue=1750.0)
		
		if newFreq is not None:
			if newFreq > self.spanMax:
				messagebox.showerror('Invalid Frequency', 'Start frequency must be less than stop frequency', parent=self.master)
				return
			self.spanMin = newFreq
			self.fStartButVar.set(str(self.spanMin))
			self.span = self.spanMax - self.spanMin
			self.spanButVar.set("{:.4f}".format(self.span))
			self.center_freq = (self.spanMin + self.spanMax) / 2
			self.fcButVar.set("{:.4f}".format(self.center_freq))
			self.sdr.center_freq = self.center_freq * 1e6
		
	def setFreqStop(self):
		"""sets the stop frequency of the analyzer"""
		newFreq = simpledialog.askfloat("Stop Frequency", "Please enter the stop frequency in MHz", parent=self, minvalue=25.0, maxvalue=1750.0)
		
		if newFreq is not None:
			if newFreq < self.spanMin:
				messagebox.showerror('Invalid Frequency', 'Stop frequency must be greater than start frequency', parent=self.master)
				return
			self.spanMax = newFreq
			self.fStopButVar.set(str(self.spanMax))
			self.span = self.spanMax - self.spanMin
			self.spanButVar.set("{:.4f}".format(self.span))
			self.center_freq = (self.spanMin + self.spanMax) / 2
			self.fcButVar.set("{:.4f}".format(self.center_freq))
			self.sdr.center_freq = self.center_freq * 1e6
		
	def setPlotStatus(self):
		"""function to control the status of the plot animation"""
		if self.plotCtrlButVar.get() == 'Pause':
			self.animation.event_source.stop()
			self.plotCtrlButVar.set('Resume')
		else:
			self.animation.event_source.start()
			self.plotCtrlButVar.set('Pause')
		
	def setFreqCentre(self):
		"""sets the center frequency of the analyzer"""
		newFreq = simpledialog.askfloat("Centre Frequency", "Please enter the centre frequency in MHz", parent=self, minvalue=50.0, maxvalue=1700.0)
		
		if newFreq is not None:
			self.center_freq = newFreq
			self.fcButVar.set(str(self.center_freq))
			self.sdr.center_freq = self.center_freq * 1e6
			self.spanMin = self.center_freq - self.span / 2
			self.fStartButVar.set("{:.4f}".format(self.spanMin))
			self.spanMax = self.center_freq + self.span / 2
			self.fStopButVar.set("{:.4f}".format(self.spanMax))
		
	def setFreqSpan(self):
		"""sets the span of the analyzer"""
		newSpan = simpledialog.askfloat("Frequency Span", "Please enter the frequency span in MHz", parent=self, minvalue=0.005, maxvalue=100)
		
		if newSpan is not None:
			self.span = newSpan
			self.spanButVar.set(str(newSpan))
			self.spanMin = self.center_freq - self.span / 2
			self.fStartButVar.set("{:.4f}".format(self.spanMin))
			self.spanMax = self.center_freq + self.span / 2
			self.fStopButVar.set("{:.4f}".format(self.spanMax))
		
	# def setRbw(self):
		# """sets the resolution bandwidth of the analyzer"""
		# newRbw = simpledialog.askfloat("Resolution bandwidth", "Please enter the resolution bandwidth in kHz", parent=self, minvalue=0.1, maxvalue=1000.0)
		
		# if newRbw is not None:
			# if newRbw > self.span*1e3:
				# messagebox.showerror('Invalid RBW', 'Resolution bandwidth cannot exceed span!', parent=self.master)
				# return
			# # calculate the number of sample points based on the chosen rbw
			# self.numsamples = int(round(self.sdr.sample_rate / (newRbw * 1e3)))
			# # make the number of samples even
			# if not self.numsamples % 2 == 0:
				# self.numsamples = self.numsamples + 1
			# # update the rbw
			# self.rbw = self.sdr.sample_rate / 1e3 / self.numsamples
			# self.rbwButVar.set("{:.2f}".format(self.rbw))
			
	def setAmpMax(self):
		"""sets the maximum amplitude of the analyzer"""
		newAmpMax = simpledialog.askfloat("Amplitude Max", "Please enter the maximum amplitude in dB/Hz", parent=self, minvalue=-140, maxvalue=30.0)
		
		if newAmpMax is not None:
			if self.ampMin >= newAmpMax:
				messagebox.showerror('Invalid Amplitude', 'Amp Max must be greater than Amp Min', parent=self.master)
				return
			self.ampMax = newAmpMax
			self.ampMaxButVar.set(str(self.ampMax))
		
	def setAmpMin(self):
		"""sets the minimum amplitude of the analyzer"""
		newAmpMin = simpledialog.askfloat("Amplitude Min", "Please enter the minimum amplitude in dB/Hz", parent=self, minvalue=-140, maxvalue=30.0)
		
		if newAmpMin is not None:
			if self.ampMax <= newAmpMin:
				messagebox.showerror('Invalid Amplitude', 'Amp Min must be less than Amp Max', parent=self.master)
				return
			self.ampMin = newAmpMin
			self.ampMinButVar.set(str(self.ampMin))
			
	# def setSampleAvg(self):
		# """sets the number of samples to be averaged"""
		# newSampleAvg = simpledialog.askinteger("Sample averages", "Please enter the number of samples to be averaged", parent=self, minvalue=1, maxvalue=100)
		
		if newSampleAvg is not None:
			self.avgsamples = newSampleAvg
			self.sampleAvgButVar.set(str(self.avgsamples))
		
	def animate(self, i):
		"""function that gets called repeatedly to display the live spectrum"""
		xData = []
		yData = []
		trimRatio = 0.75 # this is the ratio of the FFT bins taken to remove FFT edge effects 
		requestedFc = self.sdr.center_freq
		# read samples that covers the required frequency span
		self.sdr.center_freq = self.spanMin * 1e6 + (self.sdr.sample_rate * trimRatio) / 2
		while self.sdr.center_freq < (self.spanMax * 1e6 + (self.sdr.sample_rate * trimRatio) / 2):
			# read samples from SDR
			samples = self.sdr.read_samples(self.avgsamples*self.numsamples)
			# calculate power spectral density
			f, pxx = signal.welch(samples, fs=self.sdr.sample_rate, nperseg=self.numsamples)
			# rotate the arrays so the plot values are continuous and also trim the edges
			f = list(f)
			pxx = list(pxx)
			f = f[int(self.numsamples/2 + self.numsamples*(1-trimRatio)/2):] + f[:int(self.numsamples/2 - self.numsamples*(1-trimRatio)/2)]
			pxx = pxx[int(self.numsamples/2 + self.numsamples*(1-trimRatio)/2):] + pxx[:int(self.numsamples/2 - self.numsamples*(1-trimRatio)/2)]
			# adjust the format of the values to be plotted and add to plot arrays
			xData = xData + [(x+self.sdr.center_freq)/1e6 for x in f]
			yData = yData + [10*np.log10(np.abs(float(y))) for y in pxx]
			# calculate the next center frequency
			self.sdr.center_freq = self.sdr.center_freq + (self.sdr.sample_rate * trimRatio)
		# reset the sdr center frequency to requested frequency
		self.sdr.center_freq = requestedFc
		# plot the power spectral density
		self.ax[0].clear()
		self.ax[1].clear()
		
		self.Data=[xData,yData]
		
		if len(self.ImshowData[0])!=len(yData):
			self.ImshowData = []
			print(max(yData))
		self.ImshowData.append(np.float64(yData))
		self.ImshowData=self.ImshowData[-self.hm_im_line:]
		
		if self.clearWriteVar.get():
			self.ax[0].set_ylim(self.ampMin, self.ampMax)
			self.ax[0].set_xlim(self.spanMin, self.spanMax)
			#self.ax[0].xaxis.get_major_formatter().set_useOffset(True)
			self.ax[0].set_xlabel('Frequency (MHz)')
			self.ax[0].set_ylabel('Spectral Power Density (dB/Hz)')
			self.ax[0].plot(self.Data[0],self.Data[1], color="blue")
			if len(self.ax1_lines)==2:
				self.ax[0].lines.append(self.ax1_lines[0])
				self.ax[0].lines.append(self.ax1_lines[1])
		# if enabled display maxHoldData
		if self.maxHoldVar.get():
			if not len(self.maxHoldData) == len(yData):
				self.maxHoldData = yData
				return
			else:
				for idx in range(0, len(self.maxHoldData)):
					if yData[idx] > self.maxHoldData[idx]:
						self.maxHoldData[idx] = yData[idx] 
			self.ax[0].set_ylim(self.ampMin, self.ampMax)
			self.ax[0].set_xlim(self.spanMin, self.spanMax)
			self.ax[0].xaxis.get_major_formatter().set_useOffset(False)
			self.ax[0].set_xlabel('Frequency (MHz)')
			self.ax[0].set_ylabel('Spectral Power Density (dB/Hz)')
			self.ax[0].plot(self.Data[0], self.maxHoldData, color='red')
		# if enabled display minHoldData
		if self.minHoldVar.get():
			if not len(self.minHoldData) == len(yData):
				self.minHoldData = yData
				return
			else:
				for idx in range(0, len(self.minHoldData)):
					if yData[idx] < self.minHoldData[idx]:
						self.minHoldData[idx] = yData[idx] 
			self.ax[0].set_ylim(self.ampMin, self.ampMax)
			self.ax[0].set_xlim(self.spanMin, self.spanMax)
			self.ax[0].xaxis.get_major_formatter().set_useOffset(False)
			self.ax[0].set_xlabel('Frequency (MHz)')
			self.ax[0].set_ylabel('Spectral Power Density (dB/Hz)')
			self.ax[0].plot(self.Data[0], self.minHoldData, color='green')
		# if enabled display average data	
		# if self.avgPlotVar.get():
			# if not len(self.averageData) == len(yData):
				# self.averageData = yData
				# return
			# else:
				# for idx in range(0, len(self.averageData)):
					# self.averageData[idx] = (self.averageData[idx] + yData[idx]) / 2 
			# self.ax[0].set_ylim(self.ampMin, self.ampMax)
			# self.ax[0].set_xlim(self.spanMin, self.spanMax)
			# self.ax[0].xaxis.get_major_formatter().set_useOffset(False)
			# self.ax[0].set_xlabel('Frequency (MHz)')
			# self.ax[0].set_ylabel('Spectral Power Density (dB/Hz)')
			# self.ax[0].plot(self.Data[0], self.averageData, color='gold')
		if self.ImshowVar.get():
			#self.ax[1].clear()
			self.ax[1].lines=[]
			#self.ax[0].xaxis.get_major_formatter().set_useOffset(False)
			self.ax[1].imshow(self.ImshowData)#,cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0, vmax=1)
			#self.ax[1].lines=self.ax[1].lines[-1:]

import argparse
parser = argparse.ArgumentParser(description="Denial-of-service ToolKit")
parser.add_argument(
    "-m", "--method", 
	type=int, 
	metavar="0/1/2", 
	help="0 - GUI, 1 - cmd save file, 2 - cmd load file"
)
parser.add_argument(
    "--sample_rate", "-Fs",
    type=int,
	default=2400000,
    metavar="2400000",
    help="default: 2400000",
)
parser.add_argument(
    "--center_freq",  "-Fc",
	type=int, 
	default=97970000, 
	metavar="97970000", 
	help="default: 97970000"
)
parser.add_argument(
    "--gain",  "-g",
	type=str, 
	default="auto", 
	metavar="\"auto\" or 1", 
	help="default: \"auto\""
)
parser.add_argument(
    "--freq_correction",  "-c",
	type=int, 
	default=60, 
	metavar="60", 
	help="default: 60"
)
parser.add_argument(
    "--set_bandwidth",  "-bw",
	type=int, 
	default=1024, 
	metavar="1024", 
	help="default: 1024"
)
parser.add_argument(
    "--time",  "-t",
	type=int, 
	default=1024*5, 
	metavar="2048", 
	help="default: 1024*5, if t=5 sec time=t*sample_rate"
)
parser.add_argument(
    "--name_file",  "-nf",
	type=str, 
	default="file", 
	metavar="\"file\"", 
	help="default: \"file\""
)
parser.add_argument(
    "--NFFT_size",  "-ns",
	type=int, 
	default=1024, 
	metavar="1024", 
	help="default: 1024"
)
parser.add_argument(
    "--f_let",
	type=int,  
	metavar="100", 
	help="load_file[f_let:f_right]"
)
parser.add_argument(
    "--f_right",
	type=int, 
	metavar="1000", 
	help="load_file[f_let:f_right]"
)

def Print_SDR_Statistic(sdr):
	name = None
	gains = []
	gain = 0
	calibration = 0
	

	name = rtlsdr.librtlsdr.rtlsdr_get_device_name(0)
	gains = sdr.valid_gains_db+["auto"]
		

	print("sdr.name: ", name)
	print("sdr.gains: ", gains)
	print("sdr.gain: ", sdr.gain)
	print("sdr.sample_rate: ", sdr.sample_rate)
	print("sdr.center_freq: ", sdr.center_freq)
	print("sdr.freq_correction: ", sdr.freq_correction)


if __name__ == "__main__":
	
	
	
	sdr = rtlsdr.RtlSdr(0)
	Print_SDR_Statistic(sdr)
	sdr.close()

	

	# Get args
	args = parser.parse_args()
	method = args.method
	print("method ",method)
	# print help
	if method==None:
		print("exit")
		parser.print_help()
		sys.exit(1)
	elif method==0:
		
		root = Tk()
		root.resizable(width=False, height=False)
		app = Application(master=root)
		app.master.title('SDR - analyze')
		app.mainloop()
		try:
			root.destroy()
		except TclError:
			pass
	
	elif method==1:
		Fs = sample_rate = args.sample_rate
		Fc = center_freq = args.center_freq
		if args.gain=="auto":
			gain = args.gain
		else:
			gain = int(args.gain)
		freq_correction = args.freq_correction
		set_bandwidth = args.set_bandwidth
		
		chunk_size = args.time #sec
		name_file = args.name_file
		NFFT_size = args.NFFT_size
		
		sdr = rtlsdr.RtlSdr(0)
		sdr.sample_rate = sample_rate
		sdr.center_freq = center_freq
		sdr.gain = gain
		sdr.freq_correction = freq_correction
		sdr.set_bandwidth = set_bandwidth
		
		Print_SDR_Statistic(sdr)
		
		samples = sdr.read_samples(chunk_size)
		sdr.close()
		
		from scipy.io import wavfile
		wavfile.write(str(name_file)+".wav", Fs, np.float64(samples))
		
		import class_math
		val_psd, freq_psd=class_math.mypsd(samples, NFFT=NFFT_size, Fs=sample_rate/1e6)
		
		fig, ax = plt.subplots(2)
		ax[0].plot(samples)
		ax[1].plot(freq_psd*Fc,np.log10(np.abs(val_psd)))
		plt.show()
		
		
	elif method==2:
		name_file = args.name_file
		f_let = args.f_let
		f_right = args.f_right
		NFFT_size = args.NFFT_size
		if args.center_freq==None:
			Fc = input('center freq:')
		else:
			Fc = args.center_freq
		
		from scipy.io import wavfile
		samplerate, data = wavfile.read(str(name_file)+".wav")
		if f_let!=None and f_right!=None:
			data=data[f_let:f_right]
		
		import class_math
		val_psd, freq_psd=class_math.mypsd(data, NFFT=NFFT_size, Fs=samplerate/1e6)
		
		fig, ax = plt.subplots(2)
		ax[0].plot(data)
		ax[1].plot(freq_psd*Fc,np.log10(np.abs(val_psd)))
		plt.show()
		
		
		
		
	

