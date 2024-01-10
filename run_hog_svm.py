import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib  # 如果使用 scikit-learn 版本 0.22 之前的版本

# Step 1: 提取HOG特征
def extract_hog_features(image):
    win_size = (64, 64)
    cell_size = (8, 8)
    block_size = (2, 2)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, cell_size, (8, 8), nbins)
    features = hog.compute(image)
    return features.flatten()

# Step 2: 加载训练数据和标签
def load_data():
    # TODO: 加载包含正类别（路标）和负类别（非路标）的图像数据
    # 此处需要根据你的实际情况实现数据加载
    # 返回数据和相应的标签
    pass

# Step 3: 训练SVM模型
def train_svm(X_train, y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

# Step 4: 测试模型
def test_model(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

# Step 5: 保存模型
def save_model(clf, model_path="svm_model.pkl"):
    joblib.dump(clf, model_path)

# Step 6: 加载模型并进行预测
def load_and_predict(model_path, test_image):
    clf = joblib.load(model_path)
    features = extract_hog_features(test_image)
    prediction = clf.predict([features])
    return prediction

# 主程序
if __name__ == "__main__":
    # Step 2
    X, y = load_data()

    # Step 2: 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: 训练模型
    clf = train_svm(X_train, y_train)

    # Step 4: 测试模型
    test_model(clf, X_test, y_test)

    # Step 5: 保存模型
    save_model(clf)

    # Step 6: 加载模型并进行预测
    test_image_path = "path/to/your/test/image.jpg"
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    prediction = load_and_predict("svm_model.pkl", test_image)

    print("Prediction:", prediction)
