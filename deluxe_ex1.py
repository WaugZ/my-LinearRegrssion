import numpy;
import csv;

def hypothesis(X, theta) :
	return X * theta;

def costFunction(X, theta, y, m, lda) :
	temp = theta.copy();
	temp[0, 0] = 0;
	J = numpy.sum(numpy.square(hypothesis(X, theta) - y)) / (2 * m) + lda / (2 * m) * numpy.sum(numpy.square(temp));
	return J;

def grad(X, theta, y, m, lda) :
	temp = theta.copy();
	temp[0, 0] = 0;
	dJ = X.T * (hypothesis(X, theta) - y) / m + lda / m * temp;
	return dJ;

csvfile = file('save_train.csv', 'rb');
reader = csv.DictReader(csvfile);

vectors = [];
y = [];
theta = [];
for line in reader :
	vector = [1];
	for num in range(0, 384) :
		vector.append(float(line['value' + str(num)]));
	vector = numpy.array(vector);
	vectors.append(vector);
	y.append(float(line['reference']));
csvfile.close();

csvfile = file('theta.csv', 'rb');
reader = csv.DictReader(csvfile);
for line in reader:
	theta.append(float(line['theta']));
csvfile.close();

X = numpy.mat(vectors);
m = len(y);
theta = numpy.mat(numpy.array(theta));
theta = theta.T;
y = numpy.mat(numpy.array(y));
y = y.T
# print((X * theta).shape);
# print(costFunction(X, theta, y, m));
# print(grad(X, theta, y, m));

alpha = 0.05; #it used to be .05
max_iterate = 3000;
lda = 0.5;

for i in range(0, max_iterate) :
	theta = theta - alpha * grad(X, theta, y, m, lda);
	# print(costFunction(X, theta, y, m, lda));
	
print(costFunction(X, theta, y, m, lda));

test_file = file('save_test.csv', 'rb');
reader = csv.DictReader(test_file);
test_vectors = [];
for line in reader :
	vector = [1];
	for num in range(0, 384) :
		vector.append(float(line['value' + str(num)]));
	vector = numpy.array(vector);
	test_vectors.append(vector);

test_file.close();

test_X = numpy.mat(test_vectors);
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
