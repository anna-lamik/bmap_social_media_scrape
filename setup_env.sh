#!/usr/bin/env bash
# Setup environment for the Social Media Intelligence Tool
set -e

python -m venv content_intel_env
source content_intel_env/bin/activate

# Core Agno framework and AI tools
pip install agno openai exa-py firecrawl

# Data processing and analysis
pip install pandas numpy scikit-learn
pip install plotly matplotlib seaborn

# NLP and text processing
pip install nltk spacy textblob
python -m spacy download en_core_web_sm

# Platform-specific tools
pip install yt-dlp
pip install pyktok
pip install instagrapi
pip install google-api-python-client
pip install TikTokApi

# Additional utilities
pip install python-dotenv requests beautifulsoup4
pip install selenium webdriver-manager

