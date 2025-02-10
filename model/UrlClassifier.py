import pickle
import re
import pandas as pd

class UrlClassifier:

    models = {
        "Random Forest":'modelo_random_forest1',
        "Regresión Logística": 'modelo_regresion_logistica1',
        "SVM": 'modelo_SVM1'
    }

    df_classification = pd.read_csv('https://raw.githubusercontent.com/JorgeQuille/test-verify-url/main/classification.csv') 
    
    def __init__(self):
        with open('vectorizer_param.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def get_models(self):
        return list(self.models.keys())
    
    def normalize_url(self, url):
        url = re.sub(r'http[s]?://', '', url)
        url = re.sub(r'^www\.', '', url)
        url = url.lower()
        url = re.sub(r'[^a-z0-9/_-]', '', url)
        return url

    def predic(self, url, model_name):
        
        with open(f'{self.models.get(model_name)}.pkl', 'rb') as f:
            self.model = pickle.load(f)

        url = self.normalize_url(url)
        url_vectorizada = self.vectorizer.transform([url])
        prediction = self.model.predict(url_vectorizada)
        probability = self.model.predict_proba(url_vectorizada)[0][1] * 100
        category = self.df_classification.loc[self.df_classification['num_type'] == prediction[0], 'type']
        if not category.empty:
            category = f"{prediction} {category.values[0]}"
        else:
            category = f'No se encontró categoría para la predicción {prediction}.'
        return category, probability
    
    def predict_csv(self, model_name, df, column_name):
        with open(f'{self.models.get(model_name)}.pkl', 'rb') as f:
            self.model = pickle.load(f)

        df[column_name] = df[column_name].apply(self.normalize_url)
        url_vectorizada = self.vectorizer.transform(df[column_name])

        probability = self.model.predict_proba(url_vectorizada)  # Devuelve las probabilidades
        prediction = self.model.predict(url_vectorizada)  # Devuelve la clase predicha

        df['prediccion'] = prediction
        df['probabilidad'] = probability[:, 1] * 100

        df = df.merge(self.df_classification,
                    left_on='prediccion',  # Columna de predicción en el DataFrame original
                    right_on='num_type',
                    how='left')
        df.drop(columns=['num_type'], inplace=True)

        return df