from tkinter import *
from tkinter.ttk import *
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import Checkbutton

import os

import time

import threading

import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft

import rtlsdr

from scipy.io import wavfile
import sounddevice

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as ani

import class_sdr

import re

class Application(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master)
		print("[__init__] init")
		self.pack(side='top', fill='both', expand=True)
		self.master.protocol('WM_DELETE_WINDOW', self.__del__)
		
		# default settings
		# SDR
		self.center_freq = 103500000 # MHz
		self.sample_rate = 2048000 # Hz
		self.gain = 'auto' #db
		self.freq_correction = 1
		self.set_bandwidth = 250000
		self.SDR=None
		# Spectral
		self.step = 2*1e6
		self.span = 2 # MHz
		self.numsamples = 4096
		self.avgsamples = 8
		self.rbw = 0.5 # kHz
		self.ampMax = -20
		self.ampMin = -100
		self.spanMin = self.center_freq - self.span / 2
		self.spanMax = self.center_freq + self.span / 2 
		self.nfft = 2048
		
		
		# File record
		self.name_dir = str(self.center_freq)
		self.time_record = 0.5 #sec
		self.new_sample_rate = 600000
		
		
		# Data for work
		self.Data = []
		self.hm_im_line = 600
		self.WaterFall = np.zeros((self.hm_im_line,100))
		self.stay_time = 0.2
		
		self.p_printer=True
		self.do_print=False
		self.p_worker=False
		self.p_worker_pause=[False, False]
		self.Constructor()
		
	# /-CONSTRUCTOR-\
	# Entry Labels Buttons
	def construct_EntryButtonLabel(self, a_Frame, a_struct_name, a_struct_data, N=0):
		i=N
		sdr_entry_var={name : StringVar() for name in a_struct_name}
		sdr_label={name : StringVar() for name in a_struct_name}
		#sdr_entry_var=[]
		for name in a_struct_name:
			sdr_entry_var[name].set(str(a_struct_data[name][0]))
			#sdr_entry_var.append(StringVar())
			#sdr_entry_var[i].set(str(a_struct_data[name][0]))
			ent = Entry(a_Frame, textvariable=sdr_entry_var[name])
			#message_entry.pack(side=LEFT, padx=0, pady=0, ipadx=0, ipady=0)
			ent.grid(row=i, column=0, padx=0, pady=0, ipadx=0, ipady=0, sticky='news')
			#name_entry.insert(0, "Tom")
			#name_entry.delete(0, END) #clear
			
			fcButton = Button(a_Frame, text=str(name), command=a_struct_data[name][2])
			#fcButton.pack(side=LEFT, padx=0, pady=0, ipadx=0, ipady=0)
			fcButton.grid(row=i, column=1, padx=0, pady=0, ipadx=0, ipady=0, sticky='news')
			
			sdr_label[name].set(str(a_struct_data[name][1]))
			label=Label(a_Frame, textvariable=sdr_label[name])
			#label.pack(side=LEFT, padx=0, pady=0, ipadx=0, ipady=0)
			label.grid(row=i, column=2, padx=0, pady=0, ipadx=0, ipady=0, sticky='w')
			
			i+=1
		
		return sdr_entry_var,sdr_label
	# Label Button
	def construct_LabelButton(self, a_Frame, a_struct_name, a_struct_data, N=0):
		i=N
		sdr_entry_var={name : StringVar() for name in a_struct_name}
		for name in a_struct_name:
			sdr_entry_var[name].set(str(a_struct_data[name][0]))
			
			label=Label(a_Frame, textvariable=sdr_entry_var[name])
			#label.pack(side=LEFT, padx=0, pady=0, ipadx=0, ipady=0)
			label.grid(row=i, column=0, padx=0, pady=0, ipadx=0, ipady=0, sticky='e')
			
			fcButton = Button(a_Frame, text=str(name), command=a_struct_data[name][1])
			#fcButton.pack(side=LEFT, padx=0, pady=0, ipadx=0, ipady=0)
			fcButton.grid(row=i, column=1, padx=0, pady=0, ipadx=0, ipady=0, sticky='news')
			i+=1
		
		return sdr_entry_var
	# GRAPHICs
	def construct_graph(self, a_Frame, a_name_newFrame, a_hm_subplots=1,a_figsize=(6,2)):
		#fig.set_size_inches(18.5, 10.5)
		fig, ax = plt.subplots(a_hm_subplots,figsize=a_figsize)
		
		newFrame = tk.LabelFrame(a_Frame, text=a_name_newFrame, relief='groove')
		newFrame.pack(side=TOP, fill=BOTH, expand=True)#grid(row=0, column=1, padx=5, pady=5, sticky='news')
		newFrame.config(bg ="#FFFFFF",bd=0 , fg ='#000000', borderwidth=2)#, height=100, width=200)
		
		plotCanvas = FigureCanvasTkAgg(fig, newFrame)
		plotCanvas.draw()
		plotCanvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
		plotCanvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)
		#plotCanvas.config(width=100, height=200)
		#plotCanvas.scale("all",0,0,100,200)
		
		toolbar = NavigationToolbar2Tk(plotCanvas, newFrame)
		toolbar.update()
		plotCanvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)
		
		return fig, ax
	# TABLE
	def construct_table(self):
		tableframe = tk.Frame(top)
		tableframe.pack(side='left',fill='y')

		COLUMNS=['name','value']

		table=ttk.Treeview(tableframe, columns=COLUMNS, show='headings')
		table.pack(side='left',fill='y')

		for column in COLUMNS:
			table.heading(column,text=column)

		scroll=tk.Scrollbar(tableframe,command=table.yview)
		scroll.pack(side='left',fill='y')
		table.configure(yscrollcommand=scroll.set)
	
	def Constructor(self):
		print("[Constructor] init")
		plotFrame = Frame(self)
		plotFrame.grid(row=0, column=0, padx=5, pady=5, sticky='news')
		
		self.fig, self.ax = self.construct_graph(plotFrame, "ax", 2, (6,4))
		
		
		controlFrame_settings = LabelFrame(self, text='SETTINGS', relief='groove')
		controlFrame_settings.grid(row=0, column=1, padx=5, pady=5, sticky='news')
		
		# Settings SDR
		controlFrame_sdr = LabelFrame(controlFrame_settings, text='SDR', relief='groove')
		controlFrame_sdr.grid(row=0, column=0, padx=5, pady=5, sticky='news')
		
		struct_names=[
					"center_freq",
					"sample_rate",
					"gain",
					"freq_correction",
					"bandwidth"
					]
		
		struct_data={
					"center_freq":[self.center_freq, "Hz", self.set_center_freq],
					"sample_rate":[self.sample_rate, "Hz", self.set_sample_rate],
					"gain":[self.gain, "db", self.set_gain],
					"freq_correction":[self.freq_correction, "Hz", self.set_freq_correction],
					"bandwidth":[self.set_bandwidth, "Hz", self.set_set_bandwidth]
					}
		
		self.var_sdr_entrys_EBL, self.var_sdr_labels_EBL = self.construct_EntryButtonLabel(controlFrame_sdr, struct_names, struct_data)
		
		# Control
		controlFrame_control = LabelFrame(controlFrame_settings, text='Control', relief='groove')
		controlFrame_control.grid(row=1, column=0, padx=5, pady=5, sticky='news')
		
		controlFrame_control_step = LabelFrame(controlFrame_control, text='Step', relief='groove')
		controlFrame_control_step.grid(row=0, column=0, padx=0, pady=0, sticky='news')
		fcButton = Button(controlFrame_control_step, text="Prev", command=self.set_def)
		fcButton.grid(row=0, column=0, padx=0, pady=0, ipadx=0, ipady=0, sticky='news')
		fcButton = Button(controlFrame_control_step, text="Next", command=self.set_def)
		fcButton.grid(row=0, column=1, padx=0, pady=0, ipadx=0, ipady=0, sticky='news')
		
		controlFrame_control_record = LabelFrame(controlFrame_control, text='Record', relief='groove')
		controlFrame_control_record.grid(row=1, column=0, padx=5, pady=5, sticky='news')
		struct_names=[
					"Name_dir",
					"time_record",
					"new_sample_rate"
					]
		
		struct_data={
					"Name_dir":[self.name_dir, "[]", self.set_name_dir],
					"time_record":[self.time_record, "[]", self.set_time_record],
					"new_sample_rate":[self.new_sample_rate, "[]", self.set_new_sample_rate]
					}
		
		self.var_control_record_entrys_EBL, self.var_control_record_labels_EBL = self.construct_EntryButtonLabel(controlFrame_control_record, struct_names, struct_data, 3)
		struct_names=[
					"Record",
					"Open"
					]
		
		struct_data={
					"Record":["[off]", self.do_Record],
					"Open":["[off]", self.do_Open]
					}
		
		self.var_control_record_L_LB = self.construct_LabelButton(controlFrame_control_record, struct_names, struct_data)
		
		
		
		controlFrame_control_other = LabelFrame(controlFrame_control, text='Other', relief='groove')
		controlFrame_control_other.grid(row=3, column=0, padx=0, pady=0, sticky='news')
		controlFrame_control_other_left = LabelFrame(controlFrame_control_other, relief='groove')
		controlFrame_control_other_left.grid(row=0, column=0, padx=0, pady=0, sticky='news')
		controlFrame_control_other_right = LabelFrame(controlFrame_control_other, relief='groove')
		controlFrame_control_other_right.grid(row=0, column=1, padx=0, pady=0, sticky='news')
		struct_names=[
					"Record",
					"Open",
					"SDR",
					"Draw",
					"Draw_pause"
					]
		
		struct_data={
					"Record":["[off]", self.do_Record],
					"Open":["[off]", self.do_Open],
					"SDR":["[off]", self.do_SDR],
					"Draw":["[off]", self.do_Draw],
					"Draw_pause":["[off]", self.do_Draw_pause]
					}
		
		self.var_control_other_L_LB = self.construct_LabelButton(controlFrame_control_other_left, struct_names, struct_data)
		
		self.setting_save_nsr = IntVar()
		self.setting_save_nsr.set(0)
		Checkbutton(controlFrame_control_other_right, text='[save] new_sample_rate', variable=self.setting_save_nsr, command=self.settings_save, fg='black').grid(row=0, column=0, padx=5, pady=5, sticky='w')
		
		self.setting_save_fil = IntVar()
		self.setting_save_fil.set(0)
		Checkbutton(controlFrame_control_other_right, text='[save] filtr', variable=self.setting_save_fil, command=self.settings_save, fg='black').grid(row=1, column=0, padx=5, pady=5, sticky='w')
		
		self.setting_save_i = IntVar()
		self.setting_save_i.set(0)
		Checkbutton(controlFrame_control_other_right, text='[save] int', variable=self.setting_save_i, command=self.settings_save, fg='black').grid(row=2, column=0, padx=5, pady=5, sticky='w')
		
		self.setting_save_f = IntVar()
		self.setting_save_f.set(0)
		Checkbutton(controlFrame_control_other_right, text='[save] float', variable=self.setting_save_f, command=self.settings_save, fg='black').grid(row=3, column=0, padx=5, pady=5, sticky='w')
		
		
		self.clearWriteVar = IntVar()
		self.clearWriteVar.set(1)
		#Checkbutton(plotControlFrame, text='[ax[0]] Data', variable=self.clearWriteVar, fg='black').grid(row=0, column=0, padx=5, pady=5, sticky='w')
		
		
		
		
		
		self.animation = ani.FuncAnimation(self.fig, self.animate, repeat=False)#, interval=5)
		time.sleep(1)
		# thread_printer=threading.Thread(target=self.printer, daemon=True)
		# thread_printer.start()
		
		#self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		#self.c.bind('<Configure>', self.resize)
	
	# \-END CONSTRUCTOR-/
	
	def settings_save(self):
		if self.setting_save_i.get() and self.setting_save_f.get():
			self.setting_save_i.set(0)
			self.setting_save_f.set(0)
		
	
	# Setter funcs	DELETE
	def set_def(self):
		print("def")
		pass
	
	# /-SETTER FUNCTIONS-\
	def set_center_freq(self):
		print("set_center_freq")
		val=int(self.var_sdr_entrys_EBL["center_freq"].get())
		if val!=self.center_freq:
			self.center_freq=val
			self.set_worker_pause(0)
			self.SDR.dev.center_freq = self.center_freq
			self.set_worker_pause(1)
			self.center_freq=self.SDR.dev.center_freq
			self.var_sdr_entrys_EBL["center_freq"].set(str(self.center_freq))
			self.SDR.info()
		print("ok")
	def set_sample_rate(self):
		print("set_sample_rate")
		val=self.var_sdr_entrys_EBL["sample_rate"].get()
		if val!=self.sample_rate:
			self.sample_rate=int(val)
			self.set_worker_pause(0)
			self.SDR.dev.sample_rate = self.sample_rate
			self.set_worker_pause(1)
			self.sample_rate=self.SDR.dev.sample_rate
			self.var_sdr_entrys_EBL["sample_rate"].set(str(self.sample_rate))
			self.SDR.info()
		print("ok")
	def set_gain(self):
		print("set_gain")
		val=self.var_sdr_entrys_EBL["gain"].get()
		if val!=self.gain:
			if val=="auto":
				self.gain="auto"
			else:
				self.gain=int(val)
			self.set_worker_pause(0)
			self.SDR.dev.gain = self.gain
			self.set_worker_pause(1)
			self.gain=self.SDR.dev.gain
			self.var_sdr_entrys_EBL["gain"].set(str(self.gain))
			self.SDR.info()
		print("ok")
	def set_freq_correction(self):
		print("set_freq_correction")
		val=int(self.var_sdr_entrys_EBL["freq_correction"].get())
		if val!=self.freq_correction:
			self.freq_correction=val
			self.set_worker_pause(0)
			self.SDR.dev.freq_correction = self.freq_correction
			self.set_worker_pause(1)
			self.freq_correction=self.SDR.dev.freq_correction
			self.var_sdr_entrys_EBL["freq_correction"].set(str(self.freq_correction))
			self.SDR.info()
		print("ok")
	def set_set_bandwidth(self):
		print("set_bandwidth")
		val=int(self.var_sdr_entrys_EBL["bandwidth"].get())
		if val!=self.set_bandwidth:
			self.set_bandwidth=val
			self.set_worker_pause(0)
			self.SDR.dev.set_bandwidth = self.set_bandwidth
			self.set_worker_pause(1)
			self.set_bandwidth=self.SDR.dev.set_bandwidth
			self.var_sdr_entrys_EBL["bandwidth"].set(str(self.set_bandwidth))
			self.SDR.info()
		print("ok")
	
	#	delete
	def set_worker_pause(self, val):
		if self.p_worker==True:
			if val==0:
				self.p_worker_pause[0]=True
				while self.p_worker_pause[1]!=True:
					pass
				print("set_worker_pause: worker-pause")
				#name_entry.insert(0, "Tom")
				#name_entry.delete(0, END) #clear
				self.var_control_other_L_LB["Draw_pause"].set("[on]")
			elif val==1:
				self.p_worker_pause[0]=False
				while self.p_worker_pause[1]!=False:
					pass
				print("set_worker_pause: worker-play")
			self.var_control_other_L_LB["Draw_pause"].set("[off]")
	
	def set_name_dir(self):
		self.name_dir = str(self.var_control_record_entrys_EBL["Name_dir"].get())
		try:
			os.mkdir(self.name_dir,True)
		except OSError:
			pass
		print("name_dir:",self.name_dir)
	def set_time_record(self):
		self.time_record = float(self.var_control_record_entrys_EBL["time_record"].get())
		print("time_record:",self.time_record)
	def set_new_sample_rate(self):
		self.new_sample_rate = int(self.var_control_record_entrys_EBL["new_sample_rate"].get())
		print("new_sample_rate:",self.new_sample_rate)
	
	# \-END SETTER FUNCTIONS-/
	
	# /-DO FUNCTIONS-\
	def do_SDR(self):
		self.var_control_other_L_LB["SDR"].set("[..]")
		if self.SDR==None:
			print("do_SDR: sdr open")
			self.SDR=class_sdr.SDR()
			self.SDR.dev.center_freq = self.center_freq
			self.SDR.dev.sample_rate = self.sample_rate
			self.SDR.dev.gain = self.gain
			self.SDR.dev.freq_correction = self.freq_correction
			self.SDR.dev.set_bandwidth = self.set_bandwidth
			self.SDR.info()
		else:
			self.SDR=None
			print("do_SDR: sdr close")
		self.var_control_other_L_LB["SDR"].set("[on]")
		#self.SDR.get_sample(1024)
	def do_Draw(self):
		if self.p_worker==True:
			print("[do_Draw]: worker-stop")
			self.p_worker=False
			#self.animation.event_source.stop()
			self.var_control_other_L_LB["Draw"].set("[off]")
		elif self.p_worker==False:
			print("[do_Draw]: worker-start")
			self.p_worker=True
			thread_worker=threading.Thread(target=self.worker, daemon=True)
			thread_worker.start()
			self.var_control_other_L_LB["Draw"].set("[on]")
	
	# delete
	def do_Draw_pause(self):
		if self.p_worker_pause[0]==False:
			self.p_worker_pause[0]=True
			print("do_Draw_pause: worker-pause-true")
			self.var_control_other_L_LB["Draw_pause"].set("[on]")
		elif self.p_worker_pause[0]==True:
			self.p_worker_pause[0]=False
			print("do_Draw_pause: worker-pause-false")
			self.var_control_other_L_LB["Draw_pause"].set("[off]")
			
		
		self.p_worker_pause[0]=True
		self.set_worker_pause(0)
	
	def do_Record(self):
		self.var_control_other_L_LB["Record"].set("[on]")
		
		self.set_worker_pause(0)
		
		
		Fs=self.sample_rate
		t_time=self.time_record
		chunk=int(Fs*t_time)
		print("Read..", end=' ')
		data=self.SDR.get_sample(chunk)
		print("ok")
		
		print("Reshape:")
		if self.setting_save_i.get():
			resh_data, resh_Fs = self.ResizeBandwidth(data, Fs, new_Fs=self.new_sample_rate, Bandwidth=self.set_bandwidth, new_type="int64")
		elif self.setting_save_f.get():
			resh_data, resh_Fs = self.ResizeBandwidth(data, Fs, new_Fs=self.new_sample_rate, Bandwidth=self.set_bandwidth, new_type="float64")
		else:
			resh_data, resh_Fs = self.ResizeBandwidth(data, Fs, new_Fs=self.new_sample_rate, Bandwidth=self.set_bandwidth, new_type="complex64")
		
		
		
		#[center frequency][sample rate] name .dat
		print("Save..", end=' ')
		count=len(os.listdir(os.getcwd()+"/"+self.name_dir))
		print(count)
		
		namefile = "Fc["+str(self.center_freq)+"]Fs["+str(resh_Fs)+'] '+str(count)
		#np.savetxt(namefile, data.view(float).reshape(-1, 2))
		np.savetxt(self.name_dir+"/"+namefile, resh_data)
		print("ok")
		
		self.set_worker_pause(1)
		self.var_control_other_L_LB["Record"].set("[off]")
		
		# fig, ax = plt.subplots(2,2)
		# ax[0,0].psd(data, NFFT=4096, Fs=Fs)
		# ax[1,0].specgram(data, NFFT=4096, Fs=Fs)
		# ax[0,1].psd(resh_data, NFFT=4096, Fs=resh_Fs)
		# ax[1,1].specgram(resh_data, NFFT=4096, Fs=resh_Fs)
		# plt.show()
		
		"""
		fig, ax = plt.subplots(2)
		f, pxx = mypsd(data, Fs=Fs, NFFT=10000)#,fs=Fs nperseg=self.numsamples)
		ax[0].plot(f)
		
		wavfile.write(str(self.name_dir)+".wav", Fs, datax7.astype("float64"))#np.float64(data))#float64
		
		array = np.loadtxt('outfile.txt').view(complex).reshape(-1)
		
		samplerate, data = wavfile.read(str(self.name_dir)+".wav")
		f, pxx = mypsd(array, Fs=Fs, NFFT=10000)#,fs=Fs nperseg=self.numsamples)
		ax[1].plot(f)
		plt.show()
		"""
		
		
	def do_Open(self):
		namefile = "["+str(self.center_freq)+"]["+str(self.sample_rate)+'] '+str(self.name_dir)
		#array = np.loadtxt(namefile).view(complex).reshape(-1)
		array = np.loadtxt(namefile)
		#samplerate, data = wavfile.read(namefile)
		
		Fc=int(re.findall(r'\[(\d+)\]\[\d+\].dat',namefile)[0])
		Fs=int(re.findall(r'\[\d+\]\[(\d+)\]',namefile)[0])
		
		print(Fc,Fs)
		
		
		
		fig, ax = plt.subplots(3)
		ax[0].psd(array, NFFT=4096, Fs=Fs)
		ax[1].specgram(array, NFFT=4096, Fs=Fs)
		ax[2].plot(array)
		plt.show()
		
		# f, pxx = signal.welch(array, fs=Fs)
		# plt.plot(pxx)
		# plt.show()
	
	# \-END DO FUNCTIONS-/
	
	# /-Other helpfull functions-\
	def ResizeBandwidth(self, data, Fs, new_type="complex64", new_Fs=300000, Bandwidth=300000):
		print("change type...", end=' ')
		x1 = np.array(data).astype(new_type)
		print("ok")
		
		
		if self.setting_save_fil.get():
			print("filtration(",Bandwidth,")...", end=' ')
			n_taps = 32
			# Рассчет минимаксного оптимальный фильтр, используя алгоритм обмена Ремеза
			#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.remez.html
			#lpf = signal.remez(n_taps, [0, Bandwidth, Bandwidth+(Fs/2-Bandwidth)/4, Fs/2], [1,0], Hz=Fs)
			lpf = signal.remez(n_taps, [0, Bandwidth, Bandwidth+(Fs/2-Bandwidth)/4, Fs/2], [1,0], Hz=Fs)
			# фильтрация
			#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
			x1 = signal.lfilter(lpf, 1.0, x1)
			print("ok")
		
		if self.setting_save_nsr.get():
			# Изменение частоты дискретизации
			print("change sample rate...", end=' ')
			dec_audio = int(Fs/new_Fs)  
			new_Fs = Fs / dec_audio
			x3 = signal.decimate(x1, dec_audio) 
			print("ok")
			print(new_Fs)

			x3 *= 10000 / np.max(np.abs(x3))  

			
			print("change type...", end=' ')
			x1 = np.array(x3).astype(new_type)
			print("ok")
			return x1, new_Fs
		
		return x1, Fs
	
	# \-END Other helpfull functions-/
	
	
	# /-Main loop thread-\
	
	def printer(self):
		while self.p_printer==True:
			if len(self.Data)!=0 and self.do_print==True:
				self.ax[0].clear()
				self.ax[1].clear()
				
				self.ax[0].plot(self.Data[0],self.Data[1], color="blue")
				data=np.array(self.WaterFall)
				self.ax[1].imshow(data)
				
				# if self.clearWriteVar.get():
					# #self.ax[0].xaxis.get_major_formatter().set_useOffset(True)
					# self.ax[0].set_xlabel('Frequency (MHz)')
					# self.ax[0].set_ylabel('Spectral Power Density (dB/Hz)')
					# self.ax[0].plot(self.Data[0],self.Data[1], color="blue")
					# if len(self.ax1_lines)>0:
						# self.ax[0].lines.append(self.ax1_lines[0])
						# self.ax[0].lines.append(self.ax1_lines[1])
				self.do_print=False
	
	def worker(self):
		print("[worker]: init")
		start_time=0
		while self.p_worker==True:
			if self.p_worker_pause[0]==True and self.p_worker_pause[1]==False:
				self.p_worker_pause[1]=True
			elif self.p_worker_pause[0]==True and self.p_worker_pause[1]==True:
				while self.p_worker_pause[0]==True:
					time.sleep(1)
					print("[worker]: pause")
					pass
				self.p_worker_pause[1]=False
			
			else:
			
			
			
				xData = []
				yData = []
				trimRatio = 1#0.75#0.75 
				requestedFc = self.center_freq
				Fs = self.SDR.dev.sample_rate
				self.spanMin = requestedFc-self.span*1e6/2
				self.spanMax = requestedFc+self.span*1e6/2
				# read samples that covers the required frequency span
				self.SDR.dev.center_freq = self.spanMin + (Fs * trimRatio) / 2
				while self.SDR.dev.center_freq < (self.spanMax + (Fs * trimRatio) / 2):
					# read samples from SDR
					samples = self.SDR.get_sample(self.avgsamples*self.numsamples)
					# calculate power spectral density
					f, pxx = signal.welch(samples, fs=Fs, nperseg=self.numsamples)
					# rotate the arrays so the plot values are continuous and also trim the edges
					f = list(f)
					pxx = list(pxx)
					f = f[int(self.numsamples/2 + self.numsamples*(1-trimRatio)/2):] + f[:int(self.numsamples/2 - self.numsamples*(1-trimRatio)/2)]
					pxx = pxx[int(self.numsamples/2 + self.numsamples*(1-trimRatio)/2):] + pxx[:int(self.numsamples/2 - self.numsamples*(1-trimRatio)/2)]
					# adjust the format of the values to be plotted and add to plot arrays
					xData = xData + [(x+self.SDR.dev.center_freq)/1e6 for x in f]
					#yData = yData + [float(y) for y in pxx]
					yData = yData + [10*np.log10(np.abs(float(y))) for y in pxx]
					# calculate the next center frequency
					self.SDR.dev.center_freq = self.SDR.dev.center_freq + (Fs * trimRatio)
				# reset the sdr center frequency to requested frequency
				self.SDR.dev.center_freq = requestedFc
				# plot the power spectral density
				#self.ax[0].clear()
				#self.ax[1].clear()
				
				self.Data=[xData,yData]
				
				if len(self.WaterFall[0])!=len(yData):
					self.WaterFall = []
				self.WaterFall.append(np.float64(yData))
				self.WaterFall=self.WaterFall[-self.hm_im_line:]
				#time.sleep(0.15)
			
			if time.time()-start_time>self.stay_time:
				self.do_print=True
				start_time=time.time()
			
		print("[worker]: stop")
	
	def animate(self, i):
		"""function that gets called repeatedly to display the live spectrum"""
		
		if len(self.Data)!=0 and self.do_print==True:
			self.ax[0].clear()
			self.ax[1].clear()
			
			self.ax[0].plot(self.Data[0],self.Data[1], color="blue")
			data=np.array(self.WaterFall)
			self.ax[1].imshow(data)
			
			# if self.clearWriteVar.get():
				# #self.ax[0].xaxis.get_major_formatter().set_useOffset(True)
				# self.ax[0].set_xlabel('Frequency (MHz)')
				# self.ax[0].set_ylabel('Spectral Power Density (dB/Hz)')
				# self.ax[0].plot(self.Data[0],self.Data[1], color="blue")
				# if len(self.ax1_lines)>0:
					# self.ax[0].lines.append(self.ax1_lines[0])
					# self.ax[0].lines.append(self.ax1_lines[1])
			self.do_print=False
		
	
	#\-END Main loop thread-/
	
	# 
	def __del__(self):
		print("[__del__] init")
		self.quit()


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

if __name__ == "__main__":
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
		root.resizable(width=True, height=True)
		app = Application(master=root)
		app.master.title('SDR - analyze 2.0')
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

