import requests
import logging
import arxiv
from typing import List, Dict, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def ensure_nltk_resources():
    """Ensure all required NLTK resources are downloaded."""
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    # Set custom NLTK data path in user's home directory
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    nltk.data.path.append(nltk_data_dir)
    
    for resource, resource_path in required_resources:
        try:
            # Check if resource exists
            nltk.data.find(resource_path)
        except LookupError:
            try:
                # Download missing resource
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"Successfully downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.error(f"Failed to download NLTK resource {resource}: {str(e)}")
                raise RuntimeError(f"Could not download required NLTK resource: {resource}")

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """Extract keywords from text using NLTK."""
    try:
        # Ensure NLTK resources are available
        ensure_nltk_resources()
        
        # Tokenize and get parts of speech
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        
        # Get English stop words
        stop_words = set(stopwords.words('english'))
        
        # Extract nouns and adjectives that aren't stop words
        keywords = [word for word, tag in tagged 
                   if word.isalnum() and word not in stop_words 
                   and len(word) > 2
                   and tag in ('NN', 'NNS', 'NNP', 'NNPS', 'JJ')]
        
        # Count frequencies
        keyword_freq = Counter(keywords)
        
        # Get top keywords
        top_keywords = [word for word, _ in keyword_freq.most_common(num_keywords)]
        
        return top_keywords
    except Exception as e:
        logger.error(f"Error in keyword extraction: {str(e)}")
        # Fallback to simple word frequency if NLTK fails
        try:
            words = text.lower().split()
            word_freq = Counter(words)
            # Filter out short words and common stop words
            filtered_words = [word for word, _ in word_freq.most_common(num_keywords * 2)
                            if len(word) > 2 and word not in {'the', 'and', 'for', 'with', 'that', 'this'}]
            return filtered_words[:num_keywords]
        except Exception as fallback_error:
            logger.error(f"Fallback keyword extraction failed: {str(fallback_error)}")
            return []

def create_arxiv_client(page_size: int = 100, delay_seconds: float = 3.0, num_retries: int = 3) -> arxiv.Client:
    """Create an arXiv client with custom settings."""
    return arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=num_retries
    )

def search_papers(keywords: List[str], max_results: int = 10, client: Optional[arxiv.Client] = None) -> List[Dict]:
    """Search papers on arXiv using the official arxiv package."""
    try:
        if not keywords:
            logger.warning("No keywords provided for paper search")
            return []
        
        if client is None:
            client = create_arxiv_client()
        
        # Construct query using OR operator
        query = ' OR '.join(keywords)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in client.results(search):
            try:
                paper_dict = {
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'url': paper.entry_id,
                    'pdf_url': paper.pdf_url,
                    'abstract': paper.summary,
                    'published': paper.published.strftime('%Y-%m-%d'),
                    'doi': paper.doi,
                    'source': 'arXiv',
                    'arxiv_id': paper.get_short_id()
                }
                results.append(paper_dict)
            except Exception as entry_error:
                logger.error(f"Error processing paper entry: {str(entry_error)}")
                continue
        
        return clean_paper_data(results)
    except Exception as e:
        logger.error(f"Error in search_papers: {str(e)}")
        return []

def download_paper(paper: Dict, output_dir: str = "./papers", client: Optional[arxiv.Client] = None) -> Optional[str]:
    """Download PDF for a specific paper."""
    try:
        if not paper.get('arxiv_id'):
            logger.error("No arXiv ID provided for paper download")
            return None

        if client is None:
            client = create_arxiv_client()

        os.makedirs(output_dir, exist_ok=True)

        search = arxiv.Search(id_list=[paper['arxiv_id']])
        paper_result = next(client.results(search))

        safe_title = "".join(c for c in paper['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_title}_{paper['arxiv_id']}.pdf"
        filepath = os.path.join(output_dir, filename)

        paper_result.download_pdf(dirpath=output_dir, filename=filename)
        logger.info(f"Successfully downloaded paper to: {filepath}")
        
        return filepath
    except Exception as e:
        logger.error(f"Error downloading paper: {str(e)}")
        return None

def clean_paper_data(papers: List[Dict]) -> List[Dict]:
    """Clean and validate paper data."""
    cleaned_papers = []
    for paper in papers:
        try:
            cleaned_paper = {
                'title': paper.get('title', 'Untitled').strip(),
                'authors': [author.strip() for author in paper.get('authors', ['Unknown'])],
                'url': paper.get('url', '').strip(),
                'pdf_url': paper.get('pdf_url', '').strip(),
                'abstract': paper.get('abstract', 'No abstract available.').strip(),
                'published': paper.get('published', '').strip(),
                'doi': paper.get('doi', '').strip(),
                'source': paper.get('source', 'arXiv').strip(),
                'arxiv_id': paper.get('arxiv_id', '').strip()
            }
            if cleaned_paper['title'] != 'Untitled' and (cleaned_paper['url'] or cleaned_paper['abstract']):
                cleaned_papers.append(cleaned_paper)
        except Exception as e:
            logger.error(f"Error cleaning paper data: {str(e)}")
            continue
    
    return cleaned_papers
