import csv
from sklearn.model_selection import train_test_split

def load_csv_data(file_path):
    X = []
    y = []
    with open(file_path, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            if len(row) >= 2:
                label = row[0]
                text = row[1]
                if label and text.strip():
                    X.append(text)
                    y.append(label)
    return X, y

def save_data(X, y, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for text, label in zip(X, y):
            f.write(f"{label}\t{text}\n")

def main():
    print("=== 数据预处理 ===")
    
    X, y = load_csv_data('archive/spam.csv')
    print(f"总样本数: {len(X)}")
    
    spam_count = sum(1 for label in y if label == 'spam')
    ham_count = len(y) - spam_count
    print(f"垃圾短信(spam): {spam_count} ({spam_count/len(y):.2%})")
    print(f"正常短信(ham): {ham_count} ({ham_count/len(y):.2%})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\n训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    
    save_data(X_train, y_train, 'data/spam_train.txt')
    save_data(X_test, y_test, 'data/spam_test.txt')
    
    print("\n数据已保存到 data/spam_train.txt 和 data/spam_test.txt")

if __name__ == '__main__':
    main()