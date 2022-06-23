import os, cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUILD_DATA = False
DATA_FOLDER = "E:\\Docs\\nn\\pytorch"
LABELS = {"Cat": 0, "Dog": 1}
IMG_SIZE = 64
BATCH_SIZE = 100
EPOCHS = 5
MODEL_IS_TRAINED = True
TRAIN_MORE = False
MODEL_PATH = "network.model"


class catsdogs():

	training_data = []
	catcount = dogcount = 0

	def __init__(self, f, s, l):
		self.FOLDER = f
		self.IMG_SIZE = s
		self.LABELS = l

	def create_training_data(self):
		print("Loading Data ..")
		for label in self.LABELS:
			for f in tqdm(os.listdir(os.path.join(self.FOLDER, label)), desc=label):
				try:
					path = os.path.join(self.FOLDER, label, f)
					img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
					img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
					self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
					if label == self.CATS: self.catcount += 1
					else: self.dogcount += 1
				except:
					continue
		np.random.shuffle(self.training_data)
		np.save(os.path.join(self.FOLDER, "training_data.npy"), self.training_data)
		print(f"Cats: {self.catcount} Dogs: {self.dogcount}")


if BUILD_DATA:
	cd = catsdogs(DATA_FOLDER, IMG_SIZE, LABELS)
	cd.create_training_data()

training_data = np.load(os.path.join(DATA_FOLDER, "training_data.npy"), allow_pickle=True)
print(f"Total data : {len(training_data)}")


class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, 5)  # (input, output, kernelsize)
		self.conv2 = nn.Conv2d(32, 64, 5)
		self.conv3 = nn.Conv2d(64, 128, 5)

		x = torch.randn(IMG_SIZE, IMG_SIZE).view(-1, 1, IMG_SIZE, IMG_SIZE)  # this is a workaround
		self._to_linear = None
		self.convs(x)  # until here

		self.fc1 = nn.Linear(self._to_linear, 256)
		self.fc2 = nn.Linear(256, 2)

	def convs(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

		if self._to_linear is None:
			self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
		return x

	def forward(self, x):
		x = self.convs(x)
		x = x.view(-1, self._to_linear)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)


def train(net, X, Y):
	print("Training Started ..")
	for epoch in range(EPOCHS):
		print(f"Epoch {epoch+1}")
		for i in tqdm(range(0, len(X), BATCH_SIZE)):
			batch_X = X[i:i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
			batch_Y = Y[i:i + BATCH_SIZE]

			net.zero_grad()
			output = net(batch_X)
			loss = loss_function(output, batch_Y)
			loss.backward()
			optimizer.step()
	print("Training finished ..")
	print(f"Loss = {loss}")


print("Creating the network ..")
net = Network()
if MODEL_IS_TRAINED:
	print("Loading state ..")
	net.load_state_dict(torch.load(MODEL_PATH))
if not MODEL_IS_TRAINED or TRAIN_MORE:
	optimizer = optim.Adam(net.parameters(), lr=0.001)
	loss_function = nn.MSELoss()

	print("Scaling data ..")
	X = torch.Tensor([i[0] for i in training_data]).view(-1, IMG_SIZE, IMG_SIZE)
	X = X / 255.0  # scaling value between 0 and 1
	Y = torch.Tensor([i[1] for i in training_data])

	train(net, X, Y)

	torch.save(net.state_dict(), MODEL_PATH)
	print(f"Model saved as {MODEL_PATH}")


def pass_img(img):
	img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	img = np.array(img)
	img = torch.tensor(img).view(-1, 1, IMG_SIZE, IMG_SIZE)
	img = img / 255.0
	output = net(img)
	thing = torch.argmax(output)
	print(f"This is a {list(LABELS.keys())[thing]}")


while True:
	img = input("Your image : ")
	if img == "q" or img == "Q":
		break
	pass_img(img)
