#https://xn--90aeniddllys.xn--p1ai/python-nero-seti-chast-1/

#что должен собой представлять класс нейронной сети
#• инициализация — задание количества входных, скрытых и выходных узлов;
#• тренировка — уточнение весовых коэффициентов в процессе обработки предоставленных для обучения сети тренировочных примеров;
#• опрос — получение значений сигналов с выходны

#Весовые коэффициенты связей (веса). Они используются для расчета распространения сигналов в прямом направлении, а также обратного распространения ошибок, и именно весовые коэффициенты уточняются в попытке улучшить характеристики сети
#• матрицу весов для связей между входным и скрытым слоями, Wвходной_скрытый, размерностью hidden_nodes х input_nodes;
#• другую матрицу для связей между скрытым и выходным слоями, Wскрытый_выжодной, размерностью output_nodes х hidden_nodes.

#https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork
import numpy
#Библиотека scipy содержит набор специальных функций - для функций активации
import scipy.special
# библиотека для графического отображения массивов
import matplotlib.pyplot as plt


# определение класса нейронной сети
class neuralNetwork:
	# инициализировать нейронную сеть
	def __init__(self, 
					inputnodes,		#Количество узлов - Входной слой
					hiddennodes, 	#Количество узлов - выходной слой
					outputnodes, 	#Количество узлов - Скрытый слой
					learningrate):	#Коэффициент обучения
		# задать количество узлов во входном, скрытом и выходном слое
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		
		# Матрицы весовых коэффициентов связей wih (между входным и скрытым слоями) и who (между скрытым и выходным слоями).
		# Весовые коэффициенты связей между узлом i и узлом j следующего слоя
		# обозначены как w_i_j:
		# wll w21
		# wl2 w22 и т.д.
		# numpy.random.rand(rows, columns)
		self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
		self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
		"""
		#некоторые предпочитают несколько усовершенствованный подход к созданию случайных начальных значений весов
		#Для этого весовые коэффициенты выбираются из нормального распределения с центром в нуле и со стандартным отклонением, величина которого обратно пропорциональна корню квадратному из количества входящих связей на узел.
		#параметрами numpy.random.normal являются центр распределения, стандартное отклонение и размер массива numpy, если нам нужна матрица случайных чисел, а не одиночное число.
		#Как видите, центр нормального распределения установлен здесь в 0,0
		#Стандартное отклонение вычисляется по количеству узлов в следующем слое с помощью функции pow(self.hnodes, -0.5), которая просто возводит количество узлов в степень -0,5. Последний параметр определяет конфигурацию массива numpy.
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))
		"""
		
		
		# коэффициент обучения
		self.lr = learningrate
		
		#Для получения выходных сигналов скрытого слоя мы просто применяем к каждому из них сигмоиду
		# scipy
		#	expit() - Сигмоида
		#	logit() - Обратная функция Сигмоиды
		# Создание функции активации - сигмоида
		self.activation_function = lambda x: scipy.special.expit(x)
		self.inverse_activation_function = lambda x: scipy.special.logit(x)
		
		pass
	
	# опрос нейронной сети
	# Расчет выходного сигнала
	# Функция принимает в качестве аргумента входные данные нейронной сети и возвращает ее выходные данные
	def query(self, inputs_list):
		# преобразовать список входных значений
		# в двухмерный массив
		inputs = numpy.array(inputs_list, ndmin=2).T
			
		#numpy.dot - функция скалярного произведения (матрица весов (между входным и скрытым слоями) и ходные сигналы)
		# рассчитать входящие сигналы для скрытого слоя
		hidden_inputs = numpy.dot(self.wih, inputs)
		# рассчитать исходящие сигналы для скрытого слоя
		hidden_outputs = self.activation_function(hidden_inputs)
		
		# рассчитать входящие сигналы для выходного слоя
		final_inputs = numpy. dot (self. who, hidden_outputs)
		# рассчитать исходящие сигналы для выходного слоя
		final_outputs = self.activation_function(final_inputs)
		
		
		
		return final_outputs
	
	def backquery(self, targets_list):
		# transpose the targets list to a vertical array
		final_outputs = numpy.array(targets_list, ndmin=2).T

		# calculate the signal into the final output layer
		final_inputs = self.inverse_activation_function(final_outputs)

		# calculate the signal out of the hidden layer
		hidden_outputs = numpy.dot(self.who.T, final_inputs)
		# scale them back to 0.01 to .99
		hidden_outputs -= numpy.min(hidden_outputs)
		hidden_outputs /= numpy.max(hidden_outputs)
		hidden_outputs *= 0.98
		hidden_outputs += 0.01

		# calculate the signal into the hidden layer
		hidden_inputs = self.inverse_activation_function(hidden_outputs)

		# calculate the signal out of the input layer
		inputs = numpy.dot(self.wih.T, hidden_inputs)
		# scale them back to 0.01 to .99
		inputs -= numpy.min(inputs)
		inputs /= numpy.max(inputs)
		inputs *= 0.98
		inputs += 0.01

		return inputs
	
	# тренировка нейронной сети
	# Обратное распространение ошибок, информирующее о том, каковы должны быть поправки к весовым коэффициента
	#сравнение рассчитанных выходных сигналов с желаемым ответом и обновление весовых коэффициентов связей между узлами на основе найденных различий
	def train(self, inputs_list, targets_list):
		#Рсчет выходных сигналов для заданного тренировочного примера
		# преобразовать список входных значений в двухмерный массив
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T
		
		# рассчитать входящие сигналы для скрытого слоя
		hidden_inputs = numpy.dot(self.wih, inputs)
		# рассчитать исходящие сигналы для скрытого слоя
		hidden_outputs = self.activation_function(hidden_inputs)
		
		# рассчитать входящие сигналы для выходного слоя
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# рассчитать исходящие сигналы для выходного слоя
		final_outputs = self.activation_function(final_inputs)
		
		# Вычисление ошибки - разность между желаемым целевым выходным значением и фактическим выходным значением
		# ошибка = целевое значение - фактическое значение 
		# Вычисление обратного ошибки
		# Для выходного слоя
		output_errors = targets - final_outputs
		# Для скрытого слоя
		# ошибки скрытого слоя - это ошибки output_errors, распределенные пропорционально весовым коэффициентам связей и рекомбинированные на скрытых узлах
		hidden_errors = numpy.dot(self.who.T, output_errors)
		# Для выходного слоя
		
		
		# Уточнение есовых коэффициентов
		# Обновление веса связи между узлом j и узлом k следующего слоя в матричной форме
		#dWij=a*Ek*sigm(Ok)*(1-sigm(Ok))[*]Oj^T
		#Величина а — это коэффициент обучения, а сигмоида — это функция активации, с которой вы уже знакомы. Вспомните, что символ "*" означает обычное поэлементное умножение, а символ "[*]" — скалярное произведение матриц. Последний член выражения — это транспонированная (т) матрица исходящих сигналов предыдущего слоя.
		
		# обновить весовые коэффициенты связей между СКрытым и ВЫходным слоями
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
		# обновить весовые коэффициенты связей между ВХодным и СКрытым слоями
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
		
		
		
		pass
	

class Dog:
	def __init__(self, petname, temp):
		pass
	







# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
# коэффициент обучения
learning_rate = 0.8
# создать экземпляр нейронной сети
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open('mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# переменная epochs указывает, сколько раз тренировочный
# набор данных используется для тренировки сети
epochs = 10
for е in range(epochs):
	print("epoch=",е)
	# перебрать все записи в тренировочном наборе данных
	for record in training_data_list[1:10000]:
		#Подготовка входных данных
		# получить список значений, используя символы запятой (1,1)
		# в качестве разделителей
		all_values = record.split(',')
		
		# print(all_values[0])
		# image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
		# fig, ax = plt.subplots()
		# ax.imshow(image_array, cmap='Greys', interpolation='None')
		# plt.show()
		# cou-=1
		
		
		# масштабировать и сместить входные значения
		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		# создать целевые выходные значения (все равны 0,01, за исключением
		# желаемого маркерного значения, равного 0,99)
		targets = numpy.zeros(output_nodes) + 0.01
		# all_values[0] - целевое маркерное значение для данной записи
		targets[int(all_values[0])] =0.99
		n.train(inputs, targets)
		#outp=n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)
		#print(outp)

"""
A2=n.wih
A1=n.who

fig, ax = plt.subplots()
ax.imshow(A2, cmap='Greys', interpolation='None')
plt.show()

fig = plt.figure()#subplots(10)
for i in range(10):
	im=numpy.asfarray(A2[i]).reshape((28,28))
	ax = fig.add_subplot(3, 4, i+1)
	ax.imshow(im, cmap='Greys', interpolation='None')
plt.show()
"""

# загрузить в список тестовый набор данных CSV-файла набора MNIST
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# журнал оценок работы сети, первоначально пустой
scorecard = []
# перебрать все записи в тестовом наборе данных 
for record in test_data_list[1:1000]:
	# получить список значений из записи, используя символы
	# запятой (*,1) в качестве разделителей
	all_values = record.split(',')
	# правильный ответ - первое значение
	correct_label = int(all_values[0])
	#print(correct_label, "истинный маркер")
	# масштабировать и сместить входные значения
	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	# опрос сети
	outputs = n.query(inputs)
	# индекс наибольшего значения является маркерным значением
	label = numpy.argmax(outputs)
	#print(label, "ответ сети")
	# присоединить оценку ответа сети к концу списка
	if (label == correct_label):
		# в случае правильного ответа сети присоединить
		# к списку значение 1
		scorecard.append(1)
	else:
		# в случае неправильного ответа сети присоединить
		# к списку значение 0
		scorecard.append(0)

# рассчитать показатель эффективности в виде
# доли правильных ответов
scorecard_array = numpy.asarray(scorecard)
print ("эффективность = ", scorecard_array.sum()/scorecard_array.size)
print("\n\n\n")
# label to test
label = 4
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99
print(targets)

# get image data
image_data = n.backquery(targets)

# plot image data
plt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
plt.show()
#----------------------------
#Иногда генерируют новые данные - например поворотом изображения
# создание повернутых на некоторый угол вариантов изображений
# повернуть на 10 градусов против часовой стрелки
#inputs_pluslO_img = scipy.ndimage.interpolation.rotate(scaled_input.reshape(28,28), 10, cval=0.01, reshape=False)
# повернуть на 10 градусов по часовой стрелке
#inputs_minuslO_img = scipy.ndimage.interpolation.rotate(scaled_input.reshape (28,28), -10, cval=0.01, reshape=False)


