# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


# ------------------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------------------
INPUT_TRAIN = "files/input/train_data.csv.zip"
INPUT_TEST = "files/input/test_data.csv.zip"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]


# ------------------------------------------------------------------------------
# Limpieza
# ------------------------------------------------------------------------------
def clean_dataset(df):

    df = df.copy()

    df = df.rename(
        columns={"default payment next month": "default"}
    )

    df = df.drop(columns=["ID"])

    df = df.dropna()

    df = df[
        (df["EDUCATION"] != 0)
        & (df["MARRIAGE"] != 0)
    ]

    df["EDUCATION"] = df["EDUCATION"].apply(
        lambda x: 4 if x > 4 else x
    )

    return df


def load_and_clean_data():

    train_df = pd.read_csv(
        INPUT_TRAIN,
        compression="zip",
    )

    test_df = pd.read_csv(
        INPUT_TEST,
        compression="zip",
    )

    return (
        clean_dataset(train_df),
        clean_dataset(test_df),
    )


# ------------------------------------------------------------------------------
# Split
# ------------------------------------------------------------------------------
def split_data(df):

    X = df.drop(columns=["default"])
    y = df["default"]

    return X, y


# ------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------
def make_pipeline(x_train):

    numerical_features = [
        col for col in x_train.columns
        if col not in CATEGORICAL_FEATURES
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", SVC()),
        ]
    )

    return pipeline


# ------------------------------------------------------------------------------
# Grid Search
# ------------------------------------------------------------------------------
def make_grid_search(pipeline: Pipeline) -> GridSearchCV:

    param_grid = {
        "pca__n_components": [15, 20, None],
        "feature_selection__k": [15, "all"],
        "classifier__kernel": ["rbf"],
     "classifier__C": [1, 10, 100],
        "classifier__gamma": ["scale", "auto"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )

    return grid_search


# ------------------------------------------------------------------------------
# Guardar modelo
# ------------------------------------------------------------------------------
def save_model(model):

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    with gzip.open(
        MODEL_PATH,
        "wb",
    ) as file:

        pickle.dump(model, file)


# ------------------------------------------------------------------------------
# Métricas
# ------------------------------------------------------------------------------
def compute_metrics(dataset, y_true, y_pred):

    return {

        "type": "metrics",

        "dataset": dataset,

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

    }


def compute_confusion_matrix(dataset, y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return {

        "type": "cm_matrix",

        "dataset": dataset,

        "true_0": {

            "predicted_0": int(cm[0][0]),

            "predicted_1": int(cm[0][1]),

        },

        "true_1": {

            "predicted_0": int(cm[1][0]),

            "predicted_1": int(cm[1][1]),

        },

    }


def save_metrics(records):

    os.makedirs(
        os.path.dirname(METRICS_PATH),
        exist_ok=True,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(record) + "\n"
            )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():

    print("1. Iniciando")

    train_df, test_df = load_and_clean_data()

    print("2. Datos cargados")

    x_train, y_train = split_data(train_df)
    x_test, y_test = split_data(test_df)

    print("3. Datos divididos")

    pipeline = make_pipeline(x_train)

    print("4. Pipeline creado")

    model = make_grid_search(pipeline)

    print("5. GridSearch creado")

    model.fit(x_train, y_train)

    print("6. Entrenamiento terminado")

    print(model.best_params_)
    print(model.best_score_)

    save_model(model)

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    save_metrics(
        [
            compute_metrics("train", y_train, y_train_pred),
            compute_metrics("test", y_test, y_test_pred),
            compute_confusion_matrix("train", y_train, y_train_pred),
            compute_confusion_matrix("test", y_test, y_test_pred),
        ]
    )


if __name__ == "__main__":
    main()
