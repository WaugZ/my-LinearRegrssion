import numpy;
import csv;

def hypothesis(X, theta) :
	return X * theta;

def costFunction(X, theta, y, m) :
	J = numpy.sum(numpy.square(hypothesis(X, theta) - y)) / (2 * m);
	return J;

def grad(X, theta, y, m) :
	dJ = X.T * (hypothesis(X, theta) - y) / m;
	return dJ;

train_data = numpy.genfromtxt("save_train.csv", delimiter = ",");
row, col = train_data.shape;
X = train_data[1:, :-1];
y = train_data[1:, -1:];

m = row - 1;

theta = numpy.mat(numpy.zeros((col - 1, 1)));

alpha = 0.00001;
max_iterate = 2;

for i in range(0, max_iterate) :
	theta = theta - alpha * grad(X, theta, y, m);
	print(grad(X, theta, y, m));
	# print(X * theta);
	# print(costFunction(X, theta, y, m));
	
print(costFunction(X, theta, y, m));

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
writefile = file("predict.csv", 'wb');
writer = csv.writer(writefile);
writer.writerow(['Id', 'reference']);
datas = [];
for i in range(0, predict.shape[0]):
	datas.append((i, predict[i,0]));
writer.writerows(datas);
writefile.close();
