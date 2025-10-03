import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch_geometric.data import download_url, extract_gz, extract_tar, extract_zip

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the root directory path (one level up from src/)
ROOT_DIR = Path(__file__).resolve().parent.parent
print(f"ROOT_DIR for data download: {ROOT_DIR}")
DATA_PATH = ROOT_DIR / 'data'
STUDIES_CONFIG_PATH = DATA_PATH / 'studies.json'

def get_download_directory() -> Path:
    """
    Get the default download directory path.
    
    :return: Path to the temp directory where files will be downloaded.
    """
    return DATA_PATH

def load_studies_config() -> Dict[str, Any]:
    """
    Load the studies configuration from the JSON file.

    :return: A dictionary containing the study data.
    """
    try:
        with open(STUDIES_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Studies configuration file not found at {STUDIES_CONFIG_PATH}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {STUDIES_CONFIG_PATH}")
        return {}

def url_search(name: Optional[str] = None, keywords: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    """
    Function with dictionary with url guides from selected studies.
    
    :param name: The name of the study or a list of strings with the names.
    :param keywords: A list of strings that correspond to keywords.
    :return: A tuple containing a list of URLs and a list of sources.
    """
    study_dict = load_studies_config()
    if not study_dict:
        return [], []

    urls, sources = set(), set()

    if name:
        names = [name] if isinstance(name, str) else name
        for n in names:
            if n in study_dict:
                urls.add(study_dict[n]['url'])
                sources.add(study_dict[n]['source'])
    elif keywords:
        keyword_map = defaultdict(list)
        for study, data in study_dict.items():
            for keyword in data.get('keywords', []):
                keyword_map[keyword].append(study)
        
        keywords_list = [keywords] if isinstance(keywords, str) else keywords
        for keyword in keywords_list:
            for study in keyword_map.get(keyword, []):
                urls.add(study_dict[study]['url'])
                sources.add(study_dict[study]['source'])
    else:
        for data in study_dict.values():
            urls.add(data['url'])
            sources.add(data['source'])

    if not urls:
        logging.warning("No studies found for the given criteria.")

    return list(urls), list(sources)

def download_and_extract(url: str, folder: Path, extract_func) -> Path:
    """
    Helper function to download and extract a file.
    
    :param url: The URL of the file to download.
    :param folder: The directory to download and extract the file to.
    :param extract_func: The function to use for extraction (e.g., extract_zip, extract_tar).
    :return: The path to the extracted file or directory.
    """
    path = download_url(url, str(folder))
    logging.info(f"Extracting {path} to {folder}")
    extract_func(path, str(folder))
    # Determine the correct suffix to remove
    suffix = ''
    if url.endswith('.zip'):
        suffix = '.zip'
    elif url.endswith('.tar.gz'):
        suffix = '.tar.gz'
    
    return Path(path.replace(suffix, ''))

def download_study(name: Optional[str] = None, keywords: Optional[List[str]] = None, folder: Optional[Path] = None) -> Tuple[List[Path], List[str], List[str]]:
    """
    Function to download a file from a URL and extract it.
    
    :param name: The name of the study or a list of names.
    :param keywords: A list of strings that correspond to keywords.
    :param folder: The folder where the file will be downloaded. If None, uses the default temp directory.
    :return: A tuple containing a list of local file paths, a list of sources, and a list of URLs.
    """
    if folder is None:
        folder = get_download_directory()
    
    folder.mkdir(parents=True, exist_ok=True)
    
    urls, sources = url_search(name, keywords)
    paths = []

    for url in urls:
        try:
            logging.info(f"Downloading from {url}")
            if url.endswith('.zip'):
                paths.append(download_and_extract(url, folder, extract_zip))
            elif url.endswith('.tar.gz'):
                paths.append(download_and_extract(url, folder, extract_tar))
            elif url.endswith('.gz'):
                path = download_url(url, str(folder))
                extract_gz(path, str(folder))
                paths.append(Path(path).with_suffix(''))
            elif url.endswith('.txt') or url.endswith('.obo'):
                path = download_url(url, str(folder))
                paths.append(Path(path))
            else:
                logging.warning(f"Unsupported file type for URL: {url}")
        except Exception as e:
            logging.error(f"Failed to download or extract {url}: {e}")

    return paths, sources, urls
