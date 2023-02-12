import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

import torchvision
from Experiments.exp_tools import *
from tools.build_PNN import MultiPNN
from tools.models import MLPNet, AlexClassifier
from Experiments.ImageNet.ImageNetTools import imagenet_test_acc, read_imagenet_images, imagenet_test_diff

data_path='/projectnb/pnn/data/ImageNet'
n_labels = 10
alexnet = torchvision.models.alexnet(pretrained=True)
train_inputs, train_labels = read_imagenet_images(parent=data_path+'/train')

acc = imagenet_test_acc(data_path, alexnet)
print(f'Acc: {acc}')

alexnet_classifier = AlexClassifier(train_labels)
mimic_model = MLPNet([4096, 512, 256, 256, n_labels])

def preprocess(inputs):
    x = alexnet.features(inputs).view(-1, 256 * 6 * 6)
    x = alexnet.classifier[:4](x)
    return x

optimizer = torch.optim.SGD(mimic_model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
total = len(train_inputs)
batch_size = 128
epochs = 200
loss = 0.0
max_acc = 0.7
for i in range(epochs):
    print(f'Epoch:{i+1}.')
    for k in range(total//batch_size):
        inputs, labels = train_inputs[k*batch_size: ((k+1)*batch_size)].float(), torch.tensor(train_labels[k*batch_size: ((k+1)*batch_size)])
        latent = preprocess(inputs)
        optimizer.zero_grad()
        output = mimic_model(latent)
        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()
        loss = batch_loss.item()
        print('Loss:', loss)
        if k%10 == 0:
            print(f'{k}/{total//batch_size}')
            acc = imagenet_test_acc(data_path, mimic_model, preprocess=preprocess)
            print(f'Acc: {acc}')
            if acc >= max_acc+0.01:
                max_acc = acc
                torch.save(mimic_model.state_dict(), f'AlexNet_MiMic{acc}.pt')
    scheduler.step()
torch.save(mimic_model.state_dict(), 'AlexNet_MiMic.pt')
imagenet_test_acc(data_path, mimic_model, preprocess=preprocess)

