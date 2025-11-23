from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.svm import SVC
from tensorflow.keras import layers, models


def build_knn():
    return KNeighborsClassifier(n_neighbors=19)


def build_decision_tree():
    return DecisionTreeClassifier(
        random_state=42,
        max_depth=None,         
        min_samples_split=2,
        min_samples_leaf=1
    )


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )


def build_xgboost(num_classes=None):
    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softmax",
        "random_state": 42,
        "eval_metric": "mlogloss",
        "n_jobs": -1,
    }

    if num_classes is not None:
        params["num_class"] = num_classes

    return XGBClassifier(**params)

def build_svm():
    return SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,  
        random_state=42
    )


def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model