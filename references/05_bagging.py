import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35)

    # Implementación del clasificador KNN
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # Definimos nuestro clasificador
    knn_classifier = KNeighborsClassifier()

    # Entrenamos el clasificador con los datos de entrenamiento
    knn_classifier.fit(X_train, y_train)

    # Realizamos predicciones con el clasificador KNN
    knn_pred = knn_classifier.predict(X_test)

    # Evaluamos la precisión del clasificador
    accuracy_knn = accuracy_score(y_test, knn_pred)
    print(f"Precisión del clasificador KNN: {accuracy_knn}")

    # Implementación del clasificador de ensamble Bagging con KNN
    from sklearn.ensemble import BaggingClassifier

    # Definimos el clasificador de ensamble
    bagging_classifier = BaggingClassifier(base_estimator=knn_classifier, n_estimators=50)

    # Entrenamos el clasificador de ensamble
    bagging_classifier.fit(X_train, y_train)

    # Realizamos predicciones utilizando el clasificador de ensamble
    bagging_pred = bagging_classifier.predict(X_test)

    # Evaluamos la precisión del clasificador de ensamble
    accuracy_bagging = accuracy_score(y_test, bagging_pred)
    print(f"Precisión del clasificador de ensamble: {accuracy_bagging}")