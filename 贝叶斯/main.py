from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from preprocessing import load_data

def main():
    print("=== 朴素贝叶斯分类器 - SMS垃圾短信分类实验 ===")
    print("数据集来源: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    print()
    
    X, y = load_data('data/spam_train.txt')
    
    spam_count = sum(1 for label in y if label == 'spam')
    ham_count = len(y) - spam_count
    print(f"数据集统计:")
    print(f"  总样本数: {len(X)}")
    print(f"  垃圾短信(spam): {spam_count} ({spam_count/len(X):.2%})")
    print(f"  正常短信(ham): {ham_count} ({ham_count/len(X):.2%})")
    print()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"数据集划分:")
    print(f"  训练集: {len(X_train)} 条")
    print(f"  测试集: {len(X_test)} 条")
    
    train_spam = sum(1 for label in y_train if label == 'spam')
    test_spam = sum(1 for label in y_test if label == 'spam')
    print(f"  训练集 spam比例: {train_spam/len(y_train):.2%}")
    print(f"  测试集 spam比例: {test_spam/len(y_test):.2%}")
    print()
    
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"特征提取:")
    print(f"  词汇表大小: {len(vectorizer.vocabulary_)}")
    print(f"  训练集特征维度: {X_train_vec.shape}")
    print(f"  测试集特征维度: {X_test_vec.shape}")
    print()
    
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_vec, y_train)
    
    y_pred = clf.predict(X_test_vec)
    
    accuracy = (y_pred == y_test).mean()
    print(f"模型评估:")
    print(f"  准确率: {accuracy:.4f}")
    print()
    
    print("=== 混淆矩阵 ===")
    cm = confusion_matrix(y_test, y_pred)
    print("           预测标签")
    print("           ham   spam")
    print(f"真实标签 ham  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
    print(f"        spam [{cm[1][0]:4d}  {cm[1][1]:4d}]")
    print()
    
    print("=== 分类报告 ===")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    print("=== 类别不平衡分析 ===")
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)
    spam_recall = report['spam']['recall']
    print(f"垃圾短信(正例)召回率: {spam_recall:.4f}")
    if spam_recall < 0.9:
        print(f"警告: 垃圾短信召回率偏低({spam_recall:.4f}), 在实际应用中可能导致漏检垃圾邮件，")
        print("      可能带来用户体验下降、信息安全风险等问题。")
    else:
        print("垃圾短信召回率较高，漏检风险较低。")
    print()
    
    print("=== 测试新样本 ===")
    test_samples = [
        "Free money! Claim now!",
        "Hey, are you coming to the party?",
        "URGENT! You have won a holiday!",
        "Call me tomorrow morning."
    ]
    
    test_vec = vectorizer.transform(test_samples)
    predictions = clf.predict(test_vec)
    
    for sample, pred in zip(test_samples, predictions):
        print(f"文本: {sample}")
        print(f"预测类别: {pred}")
        print()

if __name__ == '__main__':
    main()