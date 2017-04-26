import numpy as nm;
import csv;
from math import log;

def sigmoid(z) :
	return 1 / (1 + nm.exp(-z));

def h_theta(X, theta) :
	return sigmoid(X * theta);

def initial_theta(m, n) :
	epsilon = 0.12;
	return nm.mat((1 - nm.random.rand(m, n) * 2) * epsilon);

def normalize(y) :
	max_ = y.max();
	min_ = y.min();
	return [(y - min_) / (max_ - min_), max_, min_];

def NNcostFunction(X, Theta1, Theta2, y, m, lda) :

	J = 0;
	Theta1_grad = nm.mat(nm.zeros(Theta1.shape));
	Theta2_grad = nm.mat(nm.zeros(Theta2.shape));

	m = X.shape[0];
	h_theta = nm.mat([0]);
	# forward popation

	for i in range(0, m) :
		a_1 = nm.vstack((nm.mat(nm.ones(1)), X[i, :].T));
		z_2 = Theta1.T * a_1;
		a_2 = sigmoid(z_2);
		a_2 = nm.vstack((nm.mat(nm.ones(1)), a_2));
		z_3 = Theta2.T * a_2;
		a_3 = sigmoid(z_3);  # this is hypothesis(X)
		h_theta = nm.vstack((h_theta, a_3));

		J = J + nm.sum(- y[i] * log(a_3) - (1 - y[i]) * log(1 - a_3));
		# print(J);
		
		delta_3 = a_3 - y[i];
		print(str(a_3) + ' ' + str(y[i]));
		delta_2 = nm.multiply(Theta2 * delta_3, nm.multiply(a_2, 1 - a_2));

		Theta2_grad = Theta2_grad + a_2 * delta_3.T;
		Theta1_grad = Theta1_grad + a_1 * delta_2[1:].T;

	temp1 = Theta1.copy();
	temp1[:,0] = 0;
	Theta1_grad = (Theta1_grad + lda * temp1) / m; 
	temp2 = Theta2.copy();
	temp2[:,0] = 0;
	Theta2_grad = (Theta2_grad + lda * temp2) / m;
	J = J / m + nm.sum(nm.square(temp1)) / (2*m) + nm.sum(nm.square(temp2)) / (2*m);

	return [J, Theta1_grad, Theta2_grad, h_theta[1:]];


csvfile = file('save_train.csv', 'rb');
reader = csv.DictReader(csvfile);

vectors = [];
y = [];
theta = [];
for line in reader :
	vector = [1];
	for num in range(0, 384) :
		vector.append(float(line['value' + str(num)]));
	vector = nm.array(vector);
	vectors.append(vector);
	y.append(float(line['reference']));
csvfile.close();

# csvfile = file('theta.csv', 'rb');
# reader = csv.DictReader(csvfile);
# for line in reader:
# 	theta.append(float(line['theta']));
# csvfile.close();

X = nm.mat(vectors);
m, n = X.shape;
y = nm.mat(nm.array(y));
y = y.T;
train = m * .7;
X_train = X.copy()[:train];
Theta1 = initial_theta(n + 1, 10);
Theta2 = initial_theta(11, 1);

n_y, max_y, min_y = normalize(y);
y_train = n_y.copy()[:train];

alpha = 0.1;
max_iterate = 50;
lda = 2;

for i in range(0, max_iterate) :
	J, grad_1, grad_2, h_theta = NNcostFunction(X_train, Theta1, Theta2, y_train, m, lda);
	Theta1 = Theta1 - alpha * grad_1;
	Theta2 = Theta2 - alpha * grad_2;
	alpha = alpha * 1.05;
	print(J);
	
print(J);

X_cv = X.copy()[train:];
y_cv = n_y.copy()[train:];
J, grad_1, grad_2, h_theta = NNcostFunction(X_cv, Theta1, Theta2, y_cv, m, lda);
y_cv = h_theta * (max_y - min_y) + min_y;

poper = 0;
for inx in range(0, y_cv.size):
	poper = poper + abs(y_cv[inx, 0] - y[train + inx, 0]);
	print(str(h_theta[inx, 0]) + ' ' + str(y_cv[inx, 0]) + ' ' + str(y[train + inx, 0]));
print(poper);

'''
test_file = file('save_test.csv', 'rb');
reader = csv.DictReader(test_file);
test_vectors = [];
for line in reader :
	vector = [1];
	for num in range(0, 384) :
		vector.append(float(line['value' + str(num)]));
	vector = nm.array(vector);
	test_vectors.append(vector);

test_file.close();

test_X = nm.mat(test_vectors);
predict = hypothesis(test_X, theta);

# print(predict);
writefile = file("prediction.csv", 'wb');
writer = csv.writer(writefile);
writer.writerow(['Id', 'reference']);
datas = [];
for i in range(0, predict.shape[0]):
	datas.append((i, predict[i,0]));
writer.writerows(datas);
writefile.close();

thetafile = file("theta.csv", 'wb');
writer = csv.writer(thetafile);
writer.writerow(['Id', 'theta']);
datas = [];
for i in range(0, theta.shape[0]):
	datas.append((i, theta[i, 0]));
writer.writerows(datas);
thetafile.close();
'''
