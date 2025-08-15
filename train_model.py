
import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()


np.random.seed(42)
random.seed(42)

# Generate dummy investors
def generate_investors(n=100):
    investors = []
    sectors = ['Tech', 'Healthcare', 'Fintech', 'Consumer', 'Enterprise', 'AI/ML', 'CleanTech']
    stages = ['Pre-seed', 'Seed', 'Series A', 'Series B', 'Growth']

    for i in range(n):
        investor = {
            'id': f'inv_{i}',
            'name': fake.company(),
            'type': random.choice(['VC', 'Angel', 'Corporate', 'PE']),
            'avg_check_size': random.randint(50, 500) * 1000,
            'preferred_sectors': random.sample(sectors, k=random.randint(1, 3)),
            'preferred_stages': random.sample(stages, k=random.randint(1, 3)),
            'min_roi': random.randint(2, 10),
            'risk_appetite': random.randint(1, 5),
            'years_active': random.randint(1, 20),
            'thesis': fake.paragraph(),
            'location': fake.country(),
            'total_investments': random.randint(5, 100)
        }
        investors.append(investor)
    return pd.DataFrame(investors)

# Generate dummy startups
def generate_startups(n=500):
    startups = []
    sectors = ['Tech', 'Healthcare', 'Fintech', 'Consumer', 'Enterprise', 'AI/ML', 'CleanTech']
    stages = ['Pre-seed', 'Seed', 'Series A', 'Series B', 'Growth']

    for i in range(n):
        founded_date = fake.date_between(start_date='-5y', end_date='today')
        startup = {
            'id': f'stp_{i}',
            'name': fake.company(),
            'sector': random.choice(sectors),
            'stage': random.choice(stages),
            'founding_date': founded_date,
            'employees': random.randint(1, 200),
            'mrr': random.randint(0, 500) * 1000 if random.random() > 0.3 else 0,
            'growth_rate': random.uniform(0, 2.0),
            'burn_rate': random.randint(10, 100) * 1000,
            'funding_to_date': random.randint(50, 5000) * 1000,
            'description': fake.paragraph(),
            'location': fake.country(),
            'last_valuation': random.randint(1, 50) * 1000000
        }
        startups.append(startup)
    return pd.DataFrame(startups)

# Generate interaction history
def generate_interactions(investors, startups, n=1000):
    interactions = []
    for _ in range(n):
        investor = random.choice(investors['id'].values)
        startup = random.choice(startups['id'].values)

        # Make more successful matches more likely for compatible pairs
        inv_sectors = investors[investors['id'] == investor]['preferred_sectors'].values[0]
        stp_sector = startups[startups['id'] == startup]['sector'].values[0]
        sector_match = 1 if stp_sector in inv_sectors else 0

        # Base probability of positive interaction
        base_prob = 0.3 + (0.4 * sector_match)

        interacted = random.random() < base_prob
        invested = interacted and (random.random() < 0.2)

        interactions.append({
            'investor_id': investor,
            'startup_id': startup,
            'interacted': int(interacted),
            'invested': int(invested),
            'date': fake.date_between(start_date='-2y', end_date='today')
        })
    return pd.DataFrame(interactions)

# Generate all data
investors_df = generate_investors(150)
startups_df = generate_startups(1000)
interactions_df = generate_interactions(investors_df, startups_df, 2000)

# Save to CSV for later use
investors_df.to_csv('dummy_investors.csv', index=False)
startups_df.to_csv('dummy_startups.csv', index=False)
interactions_df.to_csv('dummy_interactions.csv', index=False)

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Preprocess investor data
def preprocess_investors(investors):
    # One-hot encode categoricals
    investor_features = pd.get_dummies(investors[['type', 'location']])

    # Add numerical features
    numerical = ['avg_check_size', 'min_roi', 'risk_appetite', 'years_active', 'total_investments']
    investor_features[numerical] = investors[numerical]

    # Normalize numerical features
    scaler = MinMaxScaler()
    investor_features[numerical] = scaler.fit_transform(investor_features[numerical])

    # Add sector preferences as binary columns
    all_sectors = ['Tech', 'Healthcare', 'Fintech', 'Consumer', 'Enterprise', 'AI/ML', 'CleanTech']
    for sector in all_sectors:
        investor_features[f'sector_{sector}'] = investors['preferred_sectors'].apply(lambda x: 1 if sector in x else 0)

    # Add stage preferences
    all_stages = ['Pre-seed', 'Seed', 'Series A', 'Series B', 'Growth']
    for stage in all_stages:
        investor_features[f'stage_{stage}'] = investors['preferred_stages'].apply(lambda x: 1 if stage in x else 0)

    # Add text features (investment thesis)
    thesis_vectorizer = TfidfVectorizer(max_features=50)
    thesis_vectors = thesis_vectorizer.fit_transform(investors['thesis'])
    thesis_df = pd.DataFrame(thesis_vectors.toarray(), columns=[f'thesis_{i}' for i in range(thesis_vectors.shape[1])])

    investor_features = pd.concat([investor_features, thesis_df], axis=1)
    return investor_features, thesis_vectorizer

# Preprocess startup data
def preprocess_startups(startups):
    # One-hot encode categoricals
    startup_features = pd.get_dummies(startups[['sector', 'stage', 'location']])

    # Calculate age
    startup_features['age_days'] = (pd.to_datetime('today') - pd.to_datetime(startups['founding_date'])).dt.days

    # Add numerical features
    numerical = ['employees', 'mrr', 'growth_rate', 'burn_rate', 'funding_to_date', 'last_valuation']
    startup_features[numerical] = startups[numerical]

    # Handle missing MRR (pre-revenue)
    startup_features['mrr'] = startup_features['mrr'].fillna(0)

    # Normalize numerical features
    scaler = MinMaxScaler()
    startup_features[numerical + ['age_days']] = scaler.fit_transform(startup_features[numerical + ['age_days']])

    # Add text features (description)
    description_vectorizer = TfidfVectorizer(max_features=50)
    desc_vectors = description_vectorizer.fit_transform(startups['description'])
    desc_df = pd.DataFrame(desc_vectors.toarray(), columns=[f'desc_{i}' for i in range(desc_vectors.shape[1])])

    startup_features = pd.concat([startup_features, desc_df], axis=1)
    return startup_features, description_vectorizer

# Create labeled dataset for compatibility model
def create_labeled_dataset(investors_df, startups_df, investor_features, startup_features, interactions):
    # Get positive and negative examples
    positives = interactions[interactions['interacted'] == 1]
    negatives = interactions[interactions['interacted'] == 0].sample(len(positives))

    # Combine and shuffle
    labeled = pd.concat([positives, negatives]).sample(frac=1).reset_index(drop=True)

    # Merge with original dataframes to get IDs
    labeled = labeled.merge(investors_df[['id']], left_on='investor_id', right_on='id', suffixes=('', '_inv_orig')).drop('id', axis=1)
    labeled = labeled.merge(startups_df[['id']], left_on='startup_id', right_on='id', suffixes=('', '_stp_orig')).drop('id', axis=1)

    # Create features and labels
    X = []
    for index, row in labeled.iterrows():
        # Get features from preprocessed dataframes using original IDs
        inv_id = row['investor_id']
        stp_id = row['startup_id']

        inv_features = investor_features.loc[investors_df[investors_df['id'] == inv_id].index].iloc[0]
        stp_features = startup_features.loc[startups_df[startups_df['id'] == stp_id].index].iloc[0]

        combined = np.concatenate([inv_features.values, stp_features.values])
        X.append(combined)

    y = labeled['interacted'].values
    return np.array(X), y

# Preprocess all data
investor_features, thesis_vectorizer = preprocess_investors(investors_df)
startup_features, description_vectorizer = preprocess_startups(startups_df)
X, y = create_labeled_dataset(investors_df, startups_df, investor_features, startup_features, interactions_df)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

class CompatibilityModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        print(f"Model AUC: {auc:.3f}")
        return auc

# Train and evaluate
compat_model = CompatibilityModel()
compat_model.train(X_train, y_train)
compat_model.evaluate(X_test, y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class InvestmentHistoryModel:
    def __init__(self, input_shape):
        self.model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=20, batch_size=32):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        return history

    def predict(self, X):
        return self.model.predict(X).flatten()

# Create sequential investment history data
def create_sequences(investors, interactions, seq_length=5):
    sequences = []
    labels = []

    for investor in investors['id'].unique():
        inv_interactions = interactions[interactions['investor_id'] == investor].sort_values('date')

        for i in range(len(inv_interactions) - seq_length):
            seq = inv_interactions.iloc[i:i+seq_length]
            next_interaction = inv_interactions.iloc[i+seq_length]

            # Get features for each interaction in sequence
            seq_features = []
            for _, row in seq.iterrows():
                startup = startups_df[startups_df['id'] == row['startup_id']].iloc[0]
                features = [
                    1 if startup['sector'] in investors[investors['id'] == investor]['preferred_sectors'].values[0] else 0,
                    1 if startup['stage'] in investors[investors['id'] == investor]['preferred_stages'].values[0] else 0,
                    startup['mrr'] / 1e6 if not np.isnan(startup['mrr']) else 0,
                    startup['growth_rate'],
                    row['interacted']
                ]
                seq_features.append(features)

            sequences.append(seq_features)
            labels.append(next_interaction['interacted'])

    return np.array(sequences), np.array(labels)

# Prepare sequence data
sequences, seq_labels = create_sequences(investors_df, interactions_df)
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(sequences, seq_labels, test_size=0.2, random_state=42)

# Train history model
history_model = InvestmentHistoryModel(input_shape=(X_seq_train.shape[1], X_seq_train.shape[2]))
history_model.train(X_seq_train, y_seq_train)

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

class TractionModel:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Traction Model MSE: {mse:.4f}")
        return mse

# Create traction labels (composite score)
def create_traction_labels(startups):
    # Simple composite score based on growth, funding, and MRR
    scores = (
        0.4 * startups['growth_rate'].fillna(0) +
        0.3 * np.log1p(startups['funding_to_date']) +
        0.3 * np.log1p(startups['mrr'].fillna(0))
    )
    return MinMaxScaler().fit_transform(scores.values.reshape(-1, 1)).flatten()

# Prepare traction data
traction_labels = create_traction_labels(startups_df)
X_trac = startup_features.values
X_trac_train, X_trac_test, y_trac_train, y_trac_test = train_test_split(X_trac, traction_labels, test_size=0.2, random_state=42)

# Train traction model
traction_model = TractionModel()
traction_model.train(X_trac_train, y_trac_train)
traction_model.evaluate(X_trac_test, y_trac_test)

import networkx as nx
from node2vec import Node2Vec

class IndustryCompatibilityModel:
    def __init__(self):
        self.graph = None
        self.model = None

    def build_graph(self, startups):
        G = nx.Graph()

        # Add nodes (sectors)
        all_sectors = startups['sector'].unique()
        for sector in all_sectors:
            G.add_node(sector)

        # Add edges based on co-occurrence in investor preferences
        investor_sectors = investors_df['preferred_sectors'].explode().value_counts()
        for sector1 in all_sectors:
            for sector2 in all_sectors:
                if sector1 != sector2:
                    # Simple co-occurrence weight
                    weight = 0
                    for _, inv in investors_df.iterrows():
                        if sector1 in inv['preferred_sectors'] and sector2 in inv['preferred_sectors']:
                            weight += 1
                    if weight > 0:
                        G.add_edge(sector1, sector2, weight=weight)

        self.graph = G
        return G

    def train_embeddings(self, dimensions=8):
        # Generate walks
        node2vec = Node2Vec(self.graph, dimensions=dimensions, walk_length=10, num_walks=100, workers=4)

        # Learn embeddings
        self.model = node2vec.fit(window=10, min_count=1)
        return self.model

    def get_sector_similarity(self, sector1, sector2):
        if sector1 not in self.model.wv or sector2 not in self.model.wv:
            return 0.5  # Default neutral score

        # Cosine similarity between sector embeddings
        return (1 + self.model.wv.similarity(sector1, sector2)) / 2  # Scale to 0-1

# Train industry model
industry_model = IndustryCompatibilityModel()
industry_model.build_graph(startups_df)
industry_model.train_embeddings()

# Example usage
print(f"Tech-Fintech compatibility: {industry_model.get_sector_similarity('Tech', 'Fintech'):.2f}")
print(f"Tech-Healthcare compatibility: {industry_model.get_sector_similarity('Tech', 'Healthcare'):.2f}")




import networkx as nx
from node2vec import Node2Vec

class IndustryCompatibilityModel:
    def __init__(self):
        self.graph = None
        self.model = None

    def build_graph(self, startups):
        G = nx.Graph()

        # Add nodes (sectors)
        all_sectors = startups['sector'].unique()
        for sector in all_sectors:
            G.add_node(sector)

        # Add edges based on co-occurrence in investor preferences
        investor_sectors = investors_df['preferred_sectors'].explode().value_counts()
        for sector1 in all_sectors:
            for sector2 in all_sectors:
                if sector1 != sector2:
                    # Simple co-occurrence weight
                    weight = 0
                    for _, inv in investors_df.iterrows():
                        if sector1 in inv['preferred_sectors'] and sector2 in inv['preferred_sectors']:
                            weight += 1
                    if weight > 0:
                        G.add_edge(sector1, sector2, weight=weight)

        self.graph = G
        return G

    def train_embeddings(self, dimensions=8):
        # Generate walks
        node2vec = Node2Vec(self.graph, dimensions=dimensions, walk_length=10, num_walks=100, workers=4)

        # Learn embeddings
        self.model = node2vec.fit(window=10, min_count=1)
        return self.model

    def get_sector_similarity(self, sector1, sector2):
        if sector1 not in self.model.wv or sector2 not in self.model.wv:
            return 0.5  # Default neutral score

        # Cosine similarity between sector embeddings
        return (1 + self.model.wv.similarity(sector1, sector2)) / 2  # Scale to 0-1

# Train industry model
industry_model = IndustryCompatibilityModel()
industry_model.build_graph(startups_df)
industry_model.train_embeddings()

# Example usage
print(f"Tech-Fintech compatibility: {industry_model.get_sector_similarity('Tech', 'Fintech'):.2f}")
print(f"Tech-Healthcare compatibility: {industry_model.get_sector_similarity('Tech', 'Healthcare'):.2f}")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np

class SuggestionEngine:
    def __init__(self):
        self.investor_pca = PCA(n_components=8)
        self.startup_pca = PCA(n_components=8)
        self.startup_cluster = KMeans(n_clusters=10, random_state=42, n_init=10) # Added n_init for KMeans
        self.startup_nn = NearestNeighbors(n_neighbors=5)

    def train(self, investor_features, startup_features):
        # Train PCA separately for investor and startup features
        self.investor_pca.fit(investor_features)
        investor_reduced = self.investor_pca.transform(investor_features)

        self.startup_pca.fit(startup_features)
        startup_reduced = self.startup_pca.transform(startup_features)

        # Cluster and fit nearest neighbors on startup features in PCA space
        self.startup_cluster.fit(startup_reduced)
        self.startup_nn.fit(startup_reduced)

    def get_novel_suggestions(self, investor_vec, startup_vecs, startup_ids, n=5):
        # Transform to PCA space
        investor_pca = self.investor_pca.transform(investor_vec.reshape(1, -1))
        startup_pcas = self.startup_pca.transform(startup_vecs)

        # Find distance to nearest cluster center for each startup
        cluster_dists = []
        for vec in startup_pcas:
            # Predict cluster for the current startup vector
            cluster_label = self.startup_cluster.predict(vec.reshape(1, -1))[0]
            # Get the centroid of the predicted cluster
            centroid = self.startup_cluster.cluster_centers_[cluster_label]
            # Calculate distance from the startup vector to its cluster centroid
            dist = np.linalg.norm(vec - centroid)
            cluster_dists.append(dist)

        # Find startups far from investor in their respective PCA feature spaces
        # Calculate distance between investor_pca and each startup_pca
        investor_dist = np.linalg.norm(startup_pcas - investor_pca, axis=1)

        # Combine scores (higher is more novel)
        # Using a weighted sum of distance to cluster centroid and distance from investor
        # The weights can be tuned based on desired balance between diversity and investor fit
        novelty_scores = 0.6 * np.array(cluster_dists) + 0.4 * investor_dist

        # Get top novel startups
        novel_indices = np.argsort(-novelty_scores)[:n]
        # Return the IDs of the novel startups and their scores
        return [startup_ids[i] for i in novel_indices], novelty_scores[novel_indices]


# Train suggestion engine
suggestion_engine = SuggestionEngine()
suggestion_engine.train(investor_features.values, startup_features.values)

# Example of getting novel suggestions for a specific investor (e.g., the first investor)
# Need to pass investor_features and startup_features as numpy arrays or similar
# Also need startup_ids to return meaningful suggestions
investor_index_to_suggest = 0 # Example: Suggest for the first investor
investor_vector_for_suggestion = investor_features.iloc[investor_index_to_suggest].values
startup_vectors_for_suggestion = startup_features.values
startup_ids_for_suggestion = startups_df['id'].values # Assuming startups_df is available

novel_startup_indices, novelty_scores = suggestion_engine.get_novel_suggestions(
    investor_vector_for_suggestion,
    startup_vectors_for_suggestion,
    startup_ids_for_suggestion,
    n=5
)

print(f"Novel suggestions for investor {investors_df.loc[investor_index_to_suggest, 'id']}:")
for i, startup_id in enumerate(novel_startup_indices):
    print(f"  Startup ID: {startup_id}, Novelty Score: {novelty_scores[i]:.4f}")

import joblib

print("Saving models and vectorizers...")
# Save your models (just the internal sklearn models)
joblib.dump(compat_model.model, 'compatibility_model.joblib')
joblib.dump(history_model.model, 'history_model.joblib')
joblib.dump(traction_model.model, 'traction_model.joblib')
joblib.dump(industry_model.model, 'industry_model.joblib')

# Save vectorizers
joblib.dump(thesis_vectorizer, 'thesis_vectorizer.joblib')
joblib.dump(description_vectorizer, 'description_vectorizer.joblib')

# Save dummy data (optional, for later use)
investors_df.to_json('dummy_investors.json', orient='records')
startups_df.to_json('dummy_startups.json', orient='records')
print("All models and data saved successfully!")






