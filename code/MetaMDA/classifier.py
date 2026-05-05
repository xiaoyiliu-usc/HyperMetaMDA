from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from xgboost import XGBClassifier

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def train_nn(X_train, Y_train, X_test):
    # print('NN Classifier')
    model = Sequential()
    model.add(Dense(2048, input_dim=1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(512, input_dim=128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 设置回调函数
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # 训练模型并记录历史
    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=128,
        verbose=0
    )#validation_split=0.1,  # 20% 的数据作为验证集

    # 可视化训练和验证准确率曲线
    # plt.plot(history.history['accuracy'], label='Train Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    y_prob = model.predict(X_test)
    return y_prob



def train_RF(X_train, Y_train, X_test):

    print("==========================================")
    print("RandomForest Classifier")
    RF = RandomForestClassifier(n_estimators=100, random_state=11)
    RF.fit(X_train, Y_train)
    # predictions = RF.predict(X_test)
    predictions = RF.predict_proba(X_test)[:, 1]
    return predictions


def train_GBDT(X_train, Y_train, X_test):
    print("==========================================")
    print("GBDT Classifier")
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(X_train, Y_train)
    # predictions = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)[:, 1]
    return predictions


def train_AdaBoost(X_train, Y_train, X_test):
    print("==========================================")
    print("AdaBoost Classifier")
    clf = AdaBoostClassifier()
    clf.fit(X_train, Y_train)
    # predictions = clf.predict(X_test)
    predictions = clf.predict_proba(X_test)[:, 1]
    return predictions

def train_SVM(X_train, Y_train, X_test):
    print("==========================================")
    print("SVM Classifier ")
    clf = SVC()
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions


def train_xgboost(X_train, Y_train, X_test):
    print("==========================================")
    print("xgboost Classifier ")
    clf = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, eval_metric='rmse')
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions


###############################
def train_LR(X_train, Y_train, X_test):
    print("==========================================")
    print("Logistic Regression Classifier")
    clf = LogisticRegression(penalty='l2')
    clf.fit(X_train, Y_train)
    predictions = clf.predict_proba(X_test)[:, 1]
    return predictions


def train_DT(X_train, Y_train, X_test):
    print("==========================================")
    print("Decision Tree Classifier")
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    predictions = clf.predict_proba(X_test)[:, 1]
    return predictions
