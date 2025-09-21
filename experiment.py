
import sys
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoSESFramework:
    """
    Simplified implementation of the MoSEs (Mixture of Stylistic Experts) framework
    for AI-generated text detection based on the research paper.
    """
    
    def __init__(self, n_prototypes=5, n_neighbors=3, pca_components=32):
        """
        Initialize the MoSEs framework components.
        
        Args:
            n_prototypes: Number of prototypes per style category
            n_neighbors: Number of nearest prototypes to consider
            pca_components: Number of PCA components for semantic feature compression
        """
        self.n_prototypes = n_prototypes
        self.n_neighbors = n_neighbors
        self.pca_components = pca_components
        
        # Core components
        self.srr_prototypes = {}  # Stylistics Reference Repository prototypes
        self.sar_router = None    # Stylistics-Aware Router
        self.cte_estimator = None # Conditional Threshold Estimator
        self.pca = PCA(n_components=pca_components)
        self.scaler = StandardScaler()
        
        logger.info("MoSEs framework initialized with %d prototypes and %d PCA components", 
                   n_prototypes, pca_components)
    
    def extract_linguistic_features(self, texts):
        """
        Extract linguistic features from texts (simulated for demonstration).
        In a real implementation, these would be actual linguistic features.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of linguistic features
        """
        logger.info("Extracting linguistic features from %d texts", len(texts))
        
        # Simulated feature extraction (replace with real implementation)
        features = []
        for text in texts:
            # Simulated features: text_length, log_prob_mean, log_prob_var, 
            # ngram_repetition_2, ngram_repetition_3, type_token_ratio
            text_length = len(text.split())
            log_prob_mean = np.random.normal(0, 1)
            log_prob_var = np.random.normal(1, 0.5)
            ngram_rep_2 = np.random.uniform(0, 0.2)
            ngram_rep_3 = np.random.uniform(0, 0.1)
            ttr = np.random.uniform(0.3, 0.7)
            
            features.append([text_length, log_prob_mean, log_prob_var, 
                           ngram_rep_2, ngram_rep_3, ttr])
        
        return np.array(features)
    
    def extract_semantic_embeddings(self, texts):
        """
        Extract semantic embeddings from texts (simulated for demonstration).
        In a real implementation, use BGE-M3 or similar embedding model.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of semantic embeddings
        """
        logger.info("Extracting semantic embeddings from %d texts", len(texts))
        
        # Simulated embeddings (replace with real BGE-M3 embeddings)
        embedding_dim = 384  # Typical embedding dimension
        embeddings = np.random.randn(len(texts), embedding_dim)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def build_srr(self, texts, labels, styles):
        """
        Build Stylistics Reference Repository with prototype-based approximation.
        
        Args:
            texts: List of reference texts
            labels: List of labels (0=human, 1=AI)
            styles: List of style categories
        """
        logger.info("Building Stylistics Reference Repository with %d samples", len(texts))
        
        try:
            # Extract features
            linguistic_features = self.extract_linguistic_features(texts)
            semantic_embeddings = self.extract_semantic_embeddings(texts)
            
            # Compress semantic features with PCA
            semantic_compressed = self.pca.fit_transform(semantic_embeddings)
            
            # Combine features
            combined_features = np.hstack([linguistic_features, semantic_compressed])
            
            # Store reference data
            self.srr_data = {
                'texts': texts,
                'labels': labels,
                'styles': styles,
                'linguistic_features': linguistic_features,
                'semantic_embeddings': semantic_embeddings,
                'combined_features': combined_features
            }
            
            # Create prototypes for each style using K-means clustering
            unique_styles = np.unique(styles)
            self.srr_prototypes = {}
            
            for style in unique_styles:
                style_mask = np.array(styles) == style
                style_features = combined_features[style_mask]
                
                if len(style_features) >= self.n_prototypes:
                    kmeans = KMeans(n_clusters=self.n_prototypes, random_state=42, n_init=10)
                    kmeans.fit(style_features)
                    self.srr_prototypes[style] = kmeans.cluster_centers_
                else:
                    # If not enough samples, use all samples as prototypes
                    self.srr_prototypes[style] = style_features
            
            logger.info("SRR built successfully with %d style categories", len(unique_styles))
            
        except Exception as e:
            logger.error("Error building SRR: %s", str(e))
            raise
    
    def sar_route(self, text):
        """
        Stylistics-Aware Router: Find nearest prototypes for input text.
        
        Args:
            text: Input text to route
            
        Returns:
            Indices of nearest reference samples
        """
        try:
            # Extract features from input text
            linguistic_features = self.extract_linguistic_features([text])
            semantic_embeddings = self.extract_semantic_embeddings([text])
            semantic_compressed = self.pca.transform(semantic_embeddings)
            input_features = np.hstack([linguistic_features, semantic_compressed])[0]
            
            # Find nearest prototypes across all styles
            all_prototypes = []
            for style, prototypes in self.srr_prototypes.items():
                for prototype in prototypes:
                    all_prototypes.append((style, prototype))
            
            # Calculate distances to all prototypes
            distances = []
            for i, (style, prototype) in enumerate(all_prototypes):
                distance = np.linalg.norm(input_features - prototype)
                distances.append((i, distance, style))
            
            # Sort by distance and select nearest neighbors
            distances.sort(key=lambda x: x[1])
            nearest_prototypes = distances[:self.n_neighbors]
            
            # Find reference samples associated with these prototypes
            activated_indices = []
            for proto_idx, _, style in nearest_prototypes:
                # For simplicity, return all samples from the nearest style categories
                style_mask = np.array(self.srr_data['styles']) == style
                style_indices = np.where(style_mask)[0]
                activated_indices.extend(style_indices)
            
            logger.info("SAR activated %d reference samples from %d nearest prototypes", 
                       len(activated_indices), self.n_neighbors)
            return activated_indices
            
        except Exception as e:
            logger.error("Error in SAR routing: %s", str(e))
            raise
    
    def train_cte(self):
        """
        Train Conditional Threshold Estimator using logistic regression.
        """
        logger.info("Training Conditional Threshold Estimator")
        
        try:
            # Use all reference data for training in this simplified version
            X = self.srr_data['combined_features']
            y = self.srr_data['labels']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train logistic regression model
            self.cte_estimator = LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
            self.cte_estimator.fit(X_scaled, y)
            
            logger.info("CTE trained successfully with accuracy: %.2f", 
                       self.cte_estimator.score(X_scaled, y))
            
        except Exception as e:
            logger.error("Error training CTE: %s", str(e))
            raise
    
    def predict(self, text):
        """
        Predict whether text is AI-generated using the MoSEs framework.
        
        Args:
            text: Input text to classify
            
        Returns:
            prediction: 0 (human) or 1 (AI)
            confidence: Prediction confidence score
        """
        try:
            # Route to find relevant reference samples
            activated_indices = self.sar_route(text)
            
            if not activated_indices:
                logger.warning("No reference samples activated, using default prediction")
                return 0, 0.5  # Default to human with medium confidence
            
            # Extract features from input text
            linguistic_features = self.extract_linguistic_features([text])
            semantic_embeddings = self.extract_semantic_embeddings([text])
            semantic_compressed = self.pca.transform(semantic_embeddings)
            input_features = np.hstack([linguistic_features, semantic_compressed])
            
            # Scale features
            input_scaled = self.scaler.transform(input_features)
            
            # Predict using CTE
            prediction = self.cte_estimator.predict(input_scaled)[0]
            confidence = self.cte_estimator.predict_proba(input_scaled)[0].max()
            
            logger.info("Prediction: %s with confidence: %.3f", 
                       "AI-generated" if prediction == 1 else "Human-written", confidence)
            
            return prediction, confidence
            
        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            raise

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic data for demonstration purposes.
    In a real implementation, use actual datasets.
    
    Returns:
        texts, labels, styles
    """
    logger.info("Generating synthetic data with %d samples", n_samples)
    
    texts = []
    labels = []
    styles = []
    
    # Define some example styles
    style_categories = ['news', 'academic', 'dialogue', 'story']
    
    for i in range(n_samples):
        # Randomly assign style and label
        style = np.random.choice(style_categories)
        is_ai = np.random.choice([0, 1], p=[0.5, 0.5])  # 50/50 distribution
        
        # Generate simple text based on style and label
        if style == 'news':
            text = "In recent developments, experts have observed significant changes in the market trends."
        elif style == 'academic':
            text = "The experimental results demonstrate a clear correlation between the observed phenomena."
        elif style == 'dialogue':
            text = "I think we should consider all options before making a final decision on this matter."
        else:  # story
            text = "As the sun set behind the mountains, she realized this was just the beginning of her journey."
        
        # Add some variation
        variation = f" Sample {i} with additional context."
        text += variation
        
        if is_ai:
            # Make AI text slightly different
            text = text.replace("significant", "notable").replace("clear", "evident")
        
        texts.append(text)
        labels.append(is_ai)
        styles.append(style)
    
    return texts, labels, styles

def main():
    """
    Main function to demonstrate the MoSEs framework.
    """
    logger.info("Starting MoSEs framework demonstration")
    
    try:
        # Initialize MoSEs framework
        meses = MoSESFramework(n_prototypes=3, n_neighbors=2, pca_components=16)
        
        # Generate synthetic training data
        train_texts, train_labels, train_styles = generate_synthetic_data(200)
        
        # Build Stylistics Reference Repository
        meses.build_srr(train_texts, train_labels, train_styles)
        
        # Train Conditional Threshold Estimator
        meses.train_cte()
        
        # Generate test data
        test_texts, test_labels, test_styles = generate_synthetic_data(50)
        
        # Test the framework
        predictions = []
        confidences = []
        
        logger.info("Testing MoSEs framework on %d samples", len(test_texts))
        
        for i, text in enumerate(test_texts):
            pred, conf = meses.predict(text)
            predictions.append(pred)
            confidences.append(conf)
            
            if i % 10 == 0:
                logger.info("Processed %d/%d test samples", i, len(test_texts))
        
        # Calculate performance metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        avg_confidence = np.mean(confidences)
        
        logger.info("Test Results:")
        logger.info("Accuracy: %.3f", accuracy)
        logger.info("F1 Score: %.3f", f1)
        logger.info("Average Confidence: %.3f", avg_confidence)
        
        # Print final summary
        print("\n" + "="*50)
        print("MoSEs Framework Implementation Summary")
        print("="*50)
        print(f"Training samples: {len(train_texts)}")
        print(f"Test samples: {len(test_texts)}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print("="*50)
        print("Note: This is a simplified implementation with synthetic data.")
        print("For real applications, use actual linguistic features and")
        print("pre-trained embeddings like BGE-M3 for better performance.")
        
    except Exception as e:
        logger.critical("Critical error in MoSEs framework: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
