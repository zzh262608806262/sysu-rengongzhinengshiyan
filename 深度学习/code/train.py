import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ==================== 配置参数 ====================
TRAIN_DIR = '../cnn图片/train'
TEST_DIR = '../cnn图片/test'
BATCH_SIZE = 32
TOTAL_EPOCHS = 30
EPOCHS_PER_RUN = 5  # Run 5 epochs at a time
LEARNING_RATE = 0.001
IMAGE_SIZE = 128
NUM_CLASSES = 5
CLASS_NAMES = ['baihe', 'dangshen', 'gouqi', 'huaihua', 'jinyinhua']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据预处理 ====================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==================== 加载训练集 ====================
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ==================== 自定义测试集 ====================
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for filename in sorted(os.listdir(test_dir)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(test_dir, filename))
                for i, class_name in enumerate(CLASS_NAMES):
                    if filename.startswith(class_name):
                        self.labels.append(i)
                        break
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

test_dataset = TestDataset(test_dir=TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==================== 搭建CNN模型 ====================
class ChineseHerbCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ChineseHerbCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = self.fc_layers(x)
        return x

# ==================== 训练和测试函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, correct / total

# ==================== 主程序 ====================
checkpoint_path = 'checkpoint.pth'
model = ChineseHerbCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 加载检查点
start_epoch = 0
train_losses, train_accs, test_losses, test_accs = [], [], [], []
best_test_acc = 0.0

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']
    train_losses = ckpt['train_losses']
    train_accs = ckpt['train_accs']
    test_losses = ckpt['test_losses']
    test_accs = ckpt['test_accs']
    best_test_acc = ckpt['best_acc']
    print(f"Resuming from epoch {start_epoch}")

end_epoch = min(start_epoch + EPOCHS_PER_RUN, TOTAL_EPOCHS)

for epoch in range(start_epoch, end_epoch):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    scheduler.step()
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    # 保存检查点
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_acc': best_test_acc,
    }, checkpoint_path)
    
    print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

# 如果训练完成，绘制曲线
if end_epoch >= TOTAL_EPOCHS:
    print(f"\n训练完成! 最佳测试准确率: {best_test_acc:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss', marker='o', markersize=4)
    ax1.plot(range(1, len(test_losses)+1), test_losses, 'r-', label='Test Loss', marker='s', markersize=4)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training and Testing Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, len(train_accs)+1), train_accs, 'b-', label='Train Accuracy', marker='o', markersize=4)
    ax2.plot(range(1, len(test_accs)+1), test_accs, 'r-', label='Test Accuracy', marker='s', markersize=4)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Testing Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("曲线图已保存为 training_curves.png")
    
    # 测试集预测详情
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().tolist()
            for pred, label in zip(preds, labels):
                total_preds += 1
                status = "OK" if pred == label else "FAIL"
                print(f"{status} true={CLASS_NAMES[label]}, pred={CLASS_NAMES[pred]}")
                if pred == label:
                    correct_preds += 1
    print(f"\n测试集准确率: {correct_preds}/{total_preds} = {correct_preds/total_preds:.4f}")
    
    # 保留检查点以便后续恢复训练
    print(f"检查点已保留: {checkpoint_path}")
else:
    print(f"\n已运行到第 {end_epoch} 轮，请再次运行继续训练")
