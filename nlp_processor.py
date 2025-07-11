import spacy
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class NLPProcessor:
    """
    A class for processing documents and extracting meaningful information using NLP techniques.
    """
    
    def __init__(self, model: str = "en_core_web_md", max_keywords: int = 10):
        """
        Initialize the NLP processor with specified model.
        
        Args:
            model (str): The spaCy model to use for NLP tasks
            max_keywords (int): Default maximum number of keywords to extract
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            # If model isn't installed, download it
            import subprocess
            print(f"Downloading spaCy model: {model}")
            subprocess.run(["python", "-m", "spacy", "download", model], check=True)
            self.nlp = spacy.load(model)
        
        self.max_keywords = max_keywords
        
        # Custom list of stopwords to extend spaCy's default list
        self.custom_stopwords = {
            "figure", "table", "section", "chapter", "page", "et", "al",
            "etc", "ie", "eg", "vs", "fig", "ref", "refs"
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing unnecessary characters and formatting.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep alphanumeric and whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, 
                         text: str, 
                         max_keywords: Optional[int] = None,
                         min_word_length: int = 3,
                         include_entities: bool = True,
                         use_tfidf: bool = True,
                         ngram_range: Tuple[int, int] = (1, 2),
                         pos_tags: Optional[Set[str]] = None,
                         min_freq: int = 2,
                         tfidf_threshold: float = 0.1) -> List[str]:
        """
        Extract important keywords from text using advanced NLP techniques.
        
        Args:
            text (str): The text to extract keywords from
            max_keywords (int, optional): Maximum number of keywords to return
            min_word_length (int): Minimum length of words to consider
            include_entities (bool): Whether to include named entities
            use_tfidf (bool): Whether to use TF-IDF weighting
            ngram_range (Tuple[int, int]): Range of n-gram sizes to consider
            pos_tags (Set[str], optional): POS tags to include (if None, uses default set)
            min_freq (int): Minimum frequency for a term to be considered
            tfidf_threshold (float): Minimum TF-IDF score to include a term
            
        Returns:
            List[str]: List of extracted keywords, sorted by importance
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Set default POS tags if not provided
        if pos_tags is None:
            pos_tags = {'NOUN', 'PROPN', 'ADJ', 'VERB'}
        
        # Set max_keywords to instance default if not provided
        if max_keywords is None:
            max_keywords = self.max_keywords
            
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Process with spaCy
        doc = self.nlp(processed_text)
        
        # Extract candidate terms based on POS tags
        candidates = []
        for token in doc:
            if (token.pos_ in pos_tags and 
                len(token.text) >= min_word_length and 
                not token.is_stop and 
                token.text.lower() not in self.custom_stopwords and
                not token.is_punct and 
                not token.is_space):
                candidates.append(token.lemma_.lower())
        
        # Extract named entities if requested
        entities = []
        if include_entities:
            for ent in doc.ents:
                if ent.label_ in {'ORG', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'EVENT', 'PERSON'}:
                    entities.append(ent.text.lower())
        
        # Combine candidates and entities
        all_terms = candidates + entities
        
        # Filter by frequency
        term_freq = Counter(all_terms)
        frequent_terms = [term for term, count in term_freq.items() if count >= min_freq]
        
        # Apply TF-IDF weighting if requested
        if use_tfidf and len(frequent_terms) > 0:
            # Split text into sentences for TF-IDF
            sentences = [sent.text for sent in doc.sents]
            if len(sentences) <= 1:
                # If there's only one sentence, create artificial sentence splits
                sentences = [processed_text[i:i+100] for i in range(0, len(processed_text), 100)]
            
            # Calculate TF-IDF
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                stop_words='english',
                max_features=max_keywords * 3  # Get more than we need to filter later
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Sum TF-IDF scores across sentences
                tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
                
                # Create a dictionary of terms and their scores
                term_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
                
                # Filter by threshold
                tfidf_keywords = [term for term, score in term_scores.items() 
                                if score >= tfidf_threshold and term in frequent_terms]
                
                # Sort by score
                tfidf_keywords.sort(key=lambda x: term_scores.get(x, 0), reverse=True)
                
                # Take top keywords
                keywords = tfidf_keywords[:max_keywords]
            except:
                # Fallback if TF-IDF fails
                keywords = [term for term, count in term_freq.most_common(max_keywords)]
        else:
            # Get most common terms if not using TF-IDF
            keywords = [term for term, count in term_freq.most_common(max_keywords)]
        
        # Combine multi-word phrases that appear in the original text
        if ngram_range[1] > 1:
            final_keywords = []
            for i, keyword in enumerate(keywords):
                if ' ' in keyword and keyword in processed_text.lower():
                    final_keywords.append(keyword)
                elif ' ' not in keyword:
                    final_keywords.append(keyword)
            keywords = final_keywords
        
        return keywords[:max_keywords]
    
    def analyze_document(self, text: str) -> Dict:
        """
        Perform a comprehensive analysis of a document.
        
        Args:
            text (str): The document text to analyze
            
        Returns:
            Dict: Dictionary containing analysis results
        """
        doc = self.nlp(text)
        
        # Extract basic statistics
        stats = {
            "word_count": len([token for token in doc if not token.is_punct and not token.is_space]),
            "sentence_count": len(list(doc.sents)),
            "keywords": self.extract_keywords(text),
            "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        }
        
        return stats
