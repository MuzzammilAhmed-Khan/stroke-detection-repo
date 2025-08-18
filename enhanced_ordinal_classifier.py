#!/usr/bin/env python3
"""
Enhanced ordinal classifier with improved MODERATE risk detection
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class EnhancedOrdinalStrokeClassifier:
    def __init__(self):
        self.classifiers = []
        self.moderate_classifier = None  # Special classifier for MODERATE detection
        self.scaler = StandardScaler()
        self.risk_labels = ['LOW', 'MODERATE', 'ELEVATED', 'HIGH', 'EXTREME']
        self.feature_names = None
        
    def create_ordinal_labels(self, X, y):
        """Enhanced ordinal label creation with better MODERATE detection"""
        df = pd.DataFrame(X, columns=self.feature_names)
        df['stroke'] = y
        
        risk_scores = []
        age_factors = []
        
        for idx, row in df.iterrows():
            score = 0
            age = row['age']
            hypertension = row['hypertension']
            heart_disease = row['heart_disease']
            glucose = row['avg_glucose_level']
            bmi = row['bmi']
            smoking = row['smoking_status']
            
            # Enhanced age scoring
            if age >= 80: score += 5
            elif age >= 75: score += 4
            elif age >= 65: score += 3
            elif age >= 55: score += 2
            elif age >= 45: score += 1  # Added middle-age factor
            
            # Conditions
            if hypertension: score += 2
            if heart_disease: score += 3  # Increased weight
            
            # Enhanced glucose scoring
            if glucose >= 300: score += 4
            elif glucose >= 250: score += 3
            elif glucose >= 200: score += 2
            elif glucose >= 140: score += 1
            elif glucose >= 110: score += 0.5  # Added pre-diabetic range
            
            # Enhanced BMI scoring
            if bmi >= 40: score += 3
            elif bmi >= 35: score += 2
            elif bmi >= 30: score += 1
            elif bmi >= 27: score += 0.5  # Added overweight range
            
            # Smoking
            if smoking == 3: score += 3
            elif smoking == 1: score += 2
            elif smoking == 0: score += 1
            
            risk_scores.append(score)
            age_factors.append(age)
        
        # Enhanced ordinal labeling with more nuanced thresholds
        ordinal_labels = []
        for score, stroke, age in zip(risk_scores, y, age_factors):
            if stroke == 1:  # Actual stroke cases
                if score >= 14:
                    ordinal_labels.append(4)  # EXTREME
                elif score >= 10:
                    ordinal_labels.append(3)  # HIGH
                else:
                    ordinal_labels.append(2)  # ELEVATED
            else:  # No stroke cases
                if score >= 12:
                    ordinal_labels.append(2)  # ELEVATED
                elif score >= 7:  # Lowered threshold for MODERATE
                    ordinal_labels.append(1)  # MODERATE
                elif score >= 4 and age >= 45:  # Age-based MODERATE
                    ordinal_labels.append(1)  # MODERATE
                else:
                    ordinal_labels.append(0)  # LOW
        
        return np.array(ordinal_labels)
    
    def fit(self, X, y):
        """Train enhanced ordinal classifier with special MODERATE detection"""
        print("🎯 Training Enhanced Ordinal Stroke Risk Classifier...")
        
        X_scaled = self.scaler.fit_transform(X)
        ordinal_y = self.create_ordinal_labels(X, y)
        
        print(f"📊 Enhanced ordinal label distribution:")
        unique, counts = np.unique(ordinal_y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"   {self.risk_labels[label]}: {count} samples")
        
        # Train main ordinal classifiers
        thresholds = [0, 1, 2, 3]
        
        for i, threshold in enumerate(thresholds):
            print(f"\n🔄 Training classifier {i+1}/4: {self.risk_labels[threshold]} vs higher...")
            
            binary_y = (ordinal_y > threshold).astype(int)
            unique_classes = np.unique(binary_y)
            
            if len(unique_classes) < 2:
                from sklearn.dummy import DummyClassifier
                clf = DummyClassifier(strategy='constant', constant=unique_classes[0])
                clf.fit(X_scaled, binary_y)
                self.classifiers.append(clf)
                continue
            
            # Use gradient boosting for better performance
            clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            
            # Apply SMOTE if needed
            class_counts = np.bincount(binary_y)
            minority_size = min(class_counts)
            majority_size = max(class_counts)
            
            if majority_size / minority_size > 3 and minority_size >= 6:
                try:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, minority_size-1))
                    X_balanced, y_balanced = smote.fit_resample(X_scaled, binary_y)
                    clf.fit(X_balanced, y_balanced)
                except:
                    clf.fit(X_scaled, binary_y)
            else:
                clf.fit(X_scaled, binary_y)
            
            self.classifiers.append(clf)
            print(f"✅ Trained successfully")
        
        # Train special MODERATE risk classifier
        print(f"\n🎯 Training special MODERATE risk classifier...")
        moderate_binary_y = (ordinal_y == 1).astype(int)
        
        if np.sum(moderate_binary_y) > 10:  # Enough MODERATE samples
            try:
                # Use ensemble approach for MODERATE detection
                self.moderate_classifier = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    class_weight='balanced',
                    random_state=42
                )
                
                # Apply SMOTE for MODERATE class
                smote = SMOTE(random_state=42)
                X_mod_balanced, y_mod_balanced = smote.fit_resample(X_scaled, moderate_binary_y)
                self.moderate_classifier.fit(X_mod_balanced, y_mod_balanced)
                print("✅ MODERATE classifier trained successfully")
            except Exception as e:
                print(f"⚠️ MODERATE classifier failed: {e}")
                self.moderate_classifier = None
        else:
            self.moderate_classifier = None
            print("⚠️ Not enough MODERATE samples for dedicated classifier")
        
        print("\n🎉 Enhanced ordinal classifier training completed!")
        return self
    
    def predict_proba(self, X):
        """Enhanced probability prediction with MODERATE boost"""
        X_scaled = self.scaler.transform(X)
        
        # Get base probabilities
        threshold_probs = []
        for clf in self.classifiers:
            try:
                proba = clf.predict_proba(X_scaled)
                if proba.shape[1] == 2:
                    threshold_probs.append(proba[:, 1])
                else:
                    threshold_probs.append(np.zeros(X.shape[0]))
            except:
                threshold_probs.append(np.zeros(X.shape[0]))
        
        threshold_probs = np.array(threshold_probs).T
        
        # Convert to ordinal probabilities
        n_samples = X.shape[0]
        ordinal_probs = np.zeros((n_samples, 5))
        
        for i in range(n_samples):
            probs = threshold_probs[i]
            cum_probs = np.clip(probs, 0, 1)
            
            # Base probabilities
            ordinal_probs[i, 0] = 1 - cum_probs[0]  # LOW
            ordinal_probs[i, 1] = cum_probs[0] - cum_probs[1]  # MODERATE
            ordinal_probs[i, 2] = cum_probs[1] - cum_probs[2]  # ELEVATED
            ordinal_probs[i, 3] = cum_probs[2] - cum_probs[3]  # HIGH
            ordinal_probs[i, 4] = cum_probs[3]  # EXTREME
            
            # Apply MODERATE boost if classifier available
            if self.moderate_classifier is not None:
                try:
                    moderate_prob = self.moderate_classifier.predict_proba(X_scaled[i:i+1])[0, 1]
                    
                    # Boost MODERATE probability if conditions met
                    if moderate_prob > 0.3:  # MODERATE classifier is confident
                        # Redistribute probabilities to favor MODERATE
                        current_moderate = ordinal_probs[i, 1]
                        boost_factor = min(2.0, moderate_prob / max(current_moderate, 0.1))
                        
                        # Boost MODERATE, reduce LOW and ELEVATED proportionally
                        ordinal_probs[i, 1] = min(0.7, current_moderate * boost_factor)
                        
                        # Renormalize
                        remaining = 1 - ordinal_probs[i, 1]
                        other_sum = ordinal_probs[i, 0] + ordinal_probs[i, 2] + ordinal_probs[i, 3] + ordinal_probs[i, 4]
                        if other_sum > 0:
                            scale_factor = remaining / other_sum
                            ordinal_probs[i, 0] *= scale_factor
                            ordinal_probs[i, 2] *= scale_factor
                            ordinal_probs[i, 3] *= scale_factor
                            ordinal_probs[i, 4] *= scale_factor
                except:
                    pass  # Continue with base probabilities
            
            # Ensure valid probabilities
            ordinal_probs[i] = np.maximum(ordinal_probs[i], 0)
            prob_sum = ordinal_probs[i].sum()
            if prob_sum > 0:
                ordinal_probs[i] /= prob_sum
            else:
                ordinal_probs[i, 0] = 1.0
        
        return ordinal_probs
    
    def predict(self, X):
        """Enhanced prediction"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def predict_risk_labels(self, X):
        """Enhanced risk label prediction"""
        predictions = self.predict(X)
        return [self.risk_labels[pred] for pred in predictions]

def train_enhanced_classifier():
    """Train and test enhanced classifier"""
    print("🚀 Loading enhanced stroke dataset...")
    
    df = pd.read_csv('model/enhanced_stroke_dataset.csv')
    feature_columns = ['gender', 'age', 'hypertension', 'heart_disease', 
                      'ever_married', 'work_type', 'Residence_type', 
                      'avg_glucose_level', 'bmi', 'smoking_status']
    
    X = df[feature_columns].values
    y = df['stroke'].values
    
    # Initialize enhanced classifier
    classifier = EnhancedOrdinalStrokeClassifier()
    classifier.feature_names = feature_columns
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train classifier
    classifier.fit(X_train, y_train)
    
    # Test classifier
    print("\n📊 Testing Enhanced Ordinal Classifier...")
    ordinal_test_labels = classifier.create_ordinal_labels(X_test, y_test)
    predictions = classifier.predict(X_test)
    
    accuracy = accuracy_score(ordinal_test_labels, predictions)
    print(f"🎯 Enhanced Ordinal Classification Accuracy: {accuracy:.3f}")
    
    print("\n📋 Enhanced Classification Report:")
    print(classification_report(ordinal_test_labels, predictions, 
                              target_names=classifier.risk_labels, zero_division=0))
    
    # Save enhanced classifier
    with open('model/enhanced_ordinal_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    print("💾 Enhanced classifier saved to model/enhanced_ordinal_classifier.pkl")
    
    return classifier

if __name__ == "__main__":
    classifier = train_enhanced_classifier()