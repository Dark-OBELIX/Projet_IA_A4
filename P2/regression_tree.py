import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class HousePricePredictor:
    def __init__(self, housing):
        self.housing = housing.copy()  # freeze des data
        print("Colonnes initiales du dataset :", self.housing.columns)

        if "median_house_value" not in self.housing.columns:
            raise ValueError("ERREUR : La colonne 'median_house_value' est absente du dataset.")

        self.prepare_data()
        self.train_models()

    def prepare_data(self):

        print("Étape 1 : Vérification des colonnes avant transformation...")
        num_attribs = self.housing.select_dtypes(include=[np.number]).columns.tolist()
        if "median_house_value" in num_attribs:
            num_attribs.remove("median_house_value")  # On ne veut pas la transformer

        cat_attribs = ["ocean_proximity"] if "ocean_proximity" in self.housing.columns else []

        # Pipeline 
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

        self.pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        # Transformation
        X = self.housing.drop(columns=["median_house_value"])
        self.housing_prepared = self.pipeline.fit_transform(X)
        self.housing_labels = self.housing["median_house_value"].copy()

    def train_models(self):
        
        self.models = {
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "LinearRegression": LinearRegression(),
            "SVR": SVR(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "NaiveBayes": GaussianNB()
        }

        self.predictions = {}
        self.errors = {}

        for name, model in self.models.items():
            model.fit(self.housing_prepared, self.housing_labels)
            preds = model.predict(self.housing_prepared)
            self.predictions[name] = preds
            self.errors[name] = mean_squared_error(self.housing_labels, preds)

    def predict(self, house_features):

        # Si dico converti en dataframe
        if isinstance(house_features, dict):
            house_features = pd.DataFrame([house_features])

        house_features_prepared = self.pipeline.transform(house_features)

        
        predictions = {name: model.predict(house_features_prepared) for name, model in self.models.items()}
        return predictions

housing = pd.read_csv("C:/Users/hugol/OneDrive/Documents/4_CESI/1-Projet IA/housing.csv")

predictor = HousePricePredictor(housing)

house_features = {
    'longitude': -118.5,
    'latitude': 34.0,
    'housing_median_age': 10,
    'total_rooms': 6,
    'total_bedrooms': 2,
    'population': 3,
    'households': 150,
    'median_income': 3.0,
    'ocean_proximity': 'NEAR BAY'
}

predictions = predictor.predict(house_features)
print("Prédictions pour la maison donnée :", predictions)
