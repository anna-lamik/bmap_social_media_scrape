"""
Advanced Content Analysis and Generation System
Analyzes viral patterns and generates platform-optimized content
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import logging

# NLP and ML imports
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# OpenAI for advanced analysis
import openai
from openai import OpenAI

# Import our data structures
from platform_extractors import VideoContent, Comment

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContentPattern:
    """Represents a successful content pattern"""
    pattern_type: str  # hook, structure, emotional, topic
    pattern_value: str
    frequency: int
    avg_engagement: float
    examples: List[str]
    

@dataclass
class AudienceInsight:
    """Represents insights about the audience"""
    primary_emotions: List[str]
    pain_points: List[str]
    desires: List[str]
    objections: List[str]
    language_patterns: List[str]
    demographic_hints: Dict[str, any]
    

@dataclass
class ContentStrategy:
    """Complete content strategy for a topic/niche"""
    topic: str
    platforms: List[str]
    target_audience: AudienceInsight
    content_patterns: List[ContentPattern]
    optimal_length: Dict[str, int]  # platform -> seconds/words
    best_times: Dict[str, List[str]]  # platform -> times
    trending_formats: Dict[str, List[str]]  # platform -> formats
    

class ViralPatternAnalyzer:
    """Analyzes viral content to extract success patterns"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_viral_patterns(self, videos: List[VideoContent], 
                             comments: Dict[str, List[Comment]]) -> ContentStrategy:
        """Comprehensive analysis of viral content patterns"""
        
        # Separate high-performing content (top 20%)
        sorted_videos = sorted(videos, key=lambda x: x.engagement_rate, reverse=True)
        viral_threshold = sorted_videos[int(len(sorted_videos) * 0.2)].engagement_rate
        viral_videos = [v for v in videos if v.engagement_rate >= viral_threshold]
        
        logger.info(f"Analyzing {len(viral_videos)} viral videos out of {len(videos)} total")
        
        # Extract patterns
        hook_patterns = self._extract_hook_patterns(viral_videos)
        structural_patterns = self._extract_structural_patterns(viral_videos)
        emotional_patterns = self._extract_emotional_patterns(viral_videos, comments)
        topic_patterns = self._extract_topic_patterns(viral_videos)
        
        # Analyze audience
        audience_insights = self._analyze_audience(comments, viral_videos)
        
        # Analyze timing and format
        optimal_length = self._analyze_optimal_length(viral_videos)
        best_times = self._analyze_posting_times(viral_videos)
        trending_formats = self._identify_trending_formats(viral_videos)
        
        # Compile strategy
        all_patterns = hook_patterns + structural_patterns + emotional_patterns + topic_patterns
        
        return ContentStrategy(
            topic=self._identify_main_topic(viral_videos),
            platforms=list(set(v.platform for v in viral_videos)),
            target_audience=audience_insights,
            content_patterns=all_patterns,
            optimal_length=optimal_length,
            best_times=best_times,
            trending_formats=trending_formats
        )
    
    def _extract_hook_patterns(self, videos: List[VideoContent]) -> List[ContentPattern]:
        """Extract successful hook patterns from titles and openings"""
        hook_patterns = []
        
        # Common hook formulas
        hook_templates = [
            (r'^(\d+)\s+\w+\s+that', 'Numbered list hook'),
            (r'^how\s+to\s+', 'How-to hook'),
            (r'^why\s+\w+\s+', 'Why explanation hook'),
            (r'^stop\s+\w+ing', 'Stop doing X hook'),
            (r'^the\s+\w+\s+secret', 'Secret reveal hook'),
            (r'^i\s+tried\s+', 'Personal experience hook'),
            (r'^this\s+is\s+why', 'Explanation hook'),
            (r'^\w+\s+in\s+\d+\s+seconds?', 'Time-based hook'),
            (r'^you\'ve\s+been\s+\w+ing\s+wrong', 'Correction hook'),
            (r'^\$[\d,]+', 'Money hook'),
        ]
        
        hook_matches = defaultdict(list)
        
        for video in videos:
            title_lower = video.title.lower()
            for pattern, hook_type in hook_templates:
                if re.search(pattern, title_lower):
                    hook_matches[hook_type].append({
                        'video': video,
                        'title': video.title
                    })
        
        # Calculate effectiveness of each hook type
        for hook_type, matches in hook_matches.items():
            if len(matches) >= 2:  # Need multiple examples
                avg_engagement = np.mean([m['video'].engagement_rate for m in matches])
                examples = [m['title'] for m in matches[:3]]
                
                pattern = ContentPattern(
                    pattern_type='hook',
                    pattern_value=hook_type,
                    frequency=len(matches),
                    avg_engagement=avg_engagement,
                    examples=examples
                )
                hook_patterns.append(pattern)
        
        # Use GPT to identify additional patterns
        if self.client and len(videos) > 5:
            titles = [v.title for v in videos[:20]]
            gpt_patterns = self._extract_patterns_with_gpt(titles, "hook")
            hook_patterns.extend(gpt_patterns)
        
        return sorted(hook_patterns, key=lambda x: x.avg_engagement, reverse=True)
    
    def _extract_structural_patterns(self, videos: List[VideoContent]) -> List[ContentPattern]:
        """Extract content structure patterns"""
        structural_patterns = []
        
        # Analyze video lengths
        length_bins = {
            'micro': (0, 15),
            'short': (15, 60),
            'medium': (60, 180),
            'long': (180, 600),
            'extra_long': (600, float('inf'))
        }
        
        length_performance = defaultdict(list)
        
        for video in videos:
            for label, (min_len, max_len) in length_bins.items():
                if min_len <= video.duration < max_len:
                    length_performance[label].append(video)
                    break
        
        # Find best performing lengths
        for length_type, vids in length_performance.items():
            if vids:
                avg_engagement = np.mean([v.engagement_rate for v in vids])
                pattern = ContentPattern(
                    pattern_type='structure',
                    pattern_value=f'{length_type}_form_content',
                    frequency=len(vids),
                    avg_engagement=avg_engagement,
                    examples=[f"{v.duration}s video" for v in vids[:3]]
                )
                structural_patterns.append(pattern)
        
        return structural_patterns
    
    def _extract_emotional_patterns(self, videos: List[VideoContent], 
                                  comments: Dict[str, List[Comment]]) -> List[ContentPattern]:
        """Extract emotional triggers and patterns"""
        emotional_patterns = []
        emotion_counts = defaultdict(list)
        
        # Analyze emotions in high-performing content
        for video in videos:
            video_comments = comments.get(video.video_id, [])
            
            # Analyze title and description emotions
            title_emotion = self._detect_emotion(video.title + " " + video.description)
            
            # Analyze comment emotions
            comment_emotions = []
            for comment in video_comments[:50]:  # Sample comments
                emotion = self._detect_emotion(comment.text)
                comment_emotions.append(emotion)
            
            # Most common emotion in comments
            if comment_emotions:
                most_common = Counter(comment_emotions).most_common(1)[0][0]
                emotion_counts[most_common].append(video)
        
        # Create patterns from emotions
        for emotion, vids in emotion_counts.items():
            if len(vids) >= 2:
                avg_engagement = np.mean([v.engagement_rate for v in vids])
                pattern = ContentPattern(
                    pattern_type='emotional',
                    pattern_value=emotion,
                    frequency=len(vids),
                    avg_engagement=avg_engagement,
                    examples=[v.title for v in vids[:3]]
                )
                emotional_patterns.append(pattern)
        
        return emotional_patterns
    
    def _detect_emotion(self, text: str) -> str:
        """Detect primary emotion in text"""
        # Use TextBlob for sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        # Use VADER for more nuanced analysis
        scores = self.sia.polarity_scores(text)
        
        # Map to emotions
        if scores['compound'] >= 0.5:
            return 'joy'
        elif scores['compound'] <= -0.5:
            return 'anger'
        elif scores['pos'] > 0.5:
            return 'excitement'
        elif scores['neg'] > 0.5:
            return 'frustration'
        elif 'surprise' in text.lower() or '!' in text:
            return 'surprise'
        elif '?' in text:
            return 'curiosity'
        else:
            return 'neutral'
    
    def _extract_topic_patterns(self, videos: List[VideoContent]) -> List[ContentPattern]:
        """Extract successful topic angles"""
        topic_patterns = []
        
        # Extract topics using TF-IDF
        texts = [v.title + " " + v.description for v in videos]
        
        if len(texts) < 5:
            return topic_patterns
        
        # TF-IDF Analysis
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Get top terms
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        top_indices = tfidf_scores.argsort()[-20:][::-1]
        
        top_terms = [feature_names[i] for i in top_indices]
        
        # Group videos by top terms
        term_videos = defaultdict(list)
        for i, video in enumerate(videos):
            video_tfidf = tfidf_matrix[i].toarray()[0]
            top_term_idx = video_tfidf.argsort()[-1]
            top_term = feature_names[top_term_idx]
            term_videos[top_term].append(video)
        
        # Create patterns from top terms
        for term, vids in term_videos.items():
            if len(vids) >= 2 and term in top_terms:
                avg_engagement = np.mean([v.engagement_rate for v in vids])
                pattern = ContentPattern(
                    pattern_type='topic',
                    pattern_value=term,
                    frequency=len(vids),
                    avg_engagement=avg_engagement,
                    examples=[v.title for v in vids[:3]]
                )
                topic_patterns.append(pattern)
        
        return sorted(topic_patterns, key=lambda x: x.avg_engagement, reverse=True)[:10]
    
    def _analyze_audience(self, comments: Dict[str, List[Comment]], 
                         videos: List[VideoContent]) -> AudienceInsight:
        """Deep audience analysis from comments and engagement"""
        
        all_comments_text = []
        for video_comments in comments.values():
            all_comments_text.extend([c.text for c in video_comments])
        
        # Extract emotions
        emotions = []
        for text in all_comments_text[:500]:  # Sample
            emotion = self._detect_emotion(text)
            emotions.append(emotion)
        
        primary_emotions = [e[0] for e in Counter(emotions).most_common(3)]
        
        # Extract pain points and desires using GPT
        pain_points = []
        desires = []
        objections = []
        
        if self.client and all_comments_text:
            # Sample comments for analysis
            sample_comments = all_comments_text[:100]
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a viral content strategist who creates engaging, platform-native content ideas."
                        },
                        {
                            "role": "user",
                            "content": f"""Generate {num_ideas} viral content ideas based on this strategy:

Topic: {strategy.topic}
Target Emotions: {strategy.target_audience.primary_emotions}
Pain Points: {strategy.target_audience.pain_points}
Desires: {strategy.target_audience.desires}

Use these proven patterns:
- Hook styles: {[h.pattern_value for h in top_hooks]}
- Topics that work: {[t.pattern_value for t in top_topics]}
- Emotional angles: {[e.pattern_value for e in top_emotions]}

For each idea, provide:
1. Title/Hook
2. Content angle
3. Platform recommendations
4. Key points to cover
5. Emotional trigger

Return as JSON array."""
                        }
                    ],
                    temperature=0.8
                )
                
                ideas = json.loads(response.choices[0].message.content)
                
            except Exception as e:
                logger.error(f"Error generating content ideas: {e}")
                ideas = self._generate_fallback_ideas(strategy, num_ideas)
        else:
            ideas = self._generate_fallback_ideas(strategy, num_ideas)
        
        return ideas
    
    def generate_platform_content(self, idea: Dict, platform: str, 
                                strategy: ContentStrategy) -> Dict:
        """Generate platform-specific content from an idea"""
        spec = self.platform_specs.get(platform, {})
        
        if platform == 'youtube':
            return self._generate_youtube_content(idea, strategy, spec)
        elif platform == 'tiktok':
            return self._generate_tiktok_content(idea, strategy, spec)
        elif platform == 'instagram':
            return self._generate_instagram_content(idea, strategy, spec)
        else:
            return {}
    
    def _generate_youtube_content(self, idea: Dict, strategy: ContentStrategy, 
                                spec: Dict) -> Dict:
        """Generate YouTube-optimized content"""
        content = {
            'platform': 'youtube',
            'title': '',
            'description': '',
            'tags': [],
            'thumbnail_text': '',
            'script_outline': [],
            'cta': ''
        }
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a YouTube content expert who creates engaging, SEO-optimized content."
                        },
                        {
                            "role": "user",
                            "content": f"""Create YouTube content based on:
Idea: {json.dumps(idea)}
Audience desires: {strategy.target_audience.desires}
Language patterns: {strategy.target_audience.language_patterns}

Generate:
1. SEO-optimized title (max {spec['title_max']} chars)
2. Description with timestamps (max {spec['description_max']} chars)
3. 15 relevant tags
4. Thumbnail text overlay
5. Video script outline with hooks
6. Strong CTA

Make it {spec['style']}. Return as JSON."""
                        }
                    ],
                    temperature=0.7
                )
                
                content = json.loads(response.choices[0].message.content)
                
            except Exception as e:
                logger.error(f"Error generating YouTube content: {e}")
                content = self._generate_fallback_youtube_content(idea)
        else:
            content = self._generate_fallback_youtube_content(idea)
        
        return content
    
    def _generate_tiktok_content(self, idea: Dict, strategy: ContentStrategy, 
                               spec: Dict) -> Dict:
        """Generate TikTok-optimized content"""
        content = {
            'platform': 'tiktok',
            'hook': '',
            'caption': '',
            'hashtags': [],
            'script': [],
            'transitions': [],
            'music_suggestions': [],
            'cta': ''
        }
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a TikTok content expert who creates viral, trend-aware content."
                        },
                        {
                            "role": "user",
                            "content": f"""Create TikTok content based on:
Idea: {json.dumps(idea)}
Trending formats: {strategy.trending_formats.get('tiktok', [])}
Audience emotions: {strategy.target_audience.primary_emotions}

Generate:
1. Attention-grabbing hook (first 3 seconds)
2. Caption (max {spec['caption_max']} chars)
3. 8 viral hashtags (mix of broad and niche)
4. Quick script with timestamps
5. Transition ideas
6. Trending audio suggestions
7. Clear CTA

Make it {spec['style']}. Return as JSON."""
                        }
                    ],
                    temperature=0.8
                )
                
                content = json.loads(response.choices[0].message.content)
                
            except Exception as e:
                logger.error(f"Error generating TikTok content: {e}")
                content = self._generate_fallback_tiktok_content(idea)
        else:
            content = self._generate_fallback_tiktok_content(idea)
        
        return content
    
    def _generate_instagram_content(self, idea: Dict, strategy: ContentStrategy, 
                                  spec: Dict) -> Dict:
        """Generate Instagram-optimized content"""
        content = {
            'platform': 'instagram',
            'caption': '',
            'hashtags': [],
            'reel_script': [],
            'carousel_slides': [],
            'story_sequence': [],
            'cta': ''
        }
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an Instagram content expert who creates aesthetic, engaging content."
                        },
                        {
                            "role": "user",
                            "content": f"""Create Instagram content based on:
Idea: {json.dumps(idea)}
Audience pain points: {strategy.target_audience.pain_points}
Language patterns: {strategy.target_audience.language_patterns}

Generate:
1. Engaging caption with line breaks (max {spec['caption_max']} chars)
2. 30 hashtags (mix of sizes)
3. Reel script with visual directions
4. Carousel slide ideas (if applicable)
5. Story sequence for promotion
6. Strong CTA

Make it {spec['style']}. Return as JSON."""
                        }
                    ],
                    temperature=0.7
                )
                
                content = json.loads(response.choices[0].message.content)
                
            except Exception as e:
                logger.error(f"Error generating Instagram content: {e}")
                content = self._generate_fallback_instagram_content(idea)
        else:
            content = self._generate_fallback_instagram_content(idea)
        
        return content
    
    def create_content_calendar(self, ideas: List[Dict], strategy: ContentStrategy, 
                              days: int = 30) -> pd.DataFrame:
        """Create a content calendar with optimal posting times"""
        calendar_data = []
        
        # Distribute content across days
        posts_per_platform = days // len(strategy.platforms)
        
        current_date = datetime.now()
        
        for platform in strategy.platforms:
            best_times = strategy.best_times.get(platform, ['Monday 12:00'])
            ideas_for_platform = [i for i in ideas if platform in i.get('platform_recommendations', [])]
            
            if not ideas_for_platform:
                ideas_for_platform = ideas[:posts_per_platform]
            
            for i, idea in enumerate(ideas_for_platform[:posts_per_platform]):
                # Calculate posting date
                days_ahead = (i * len(strategy.platforms)) + strategy.platforms.index(platform)
                post_date = current_date + timedelta(days=days_ahead)
                
                # Select best time for that day
                day_name = post_date.strftime('%A')
                matching_times = [t for t in best_times if day_name in t]
                post_time = matching_times[0] if matching_times else best_times[0]
                
                calendar_data.append({
                    'date': post_date.strftime('%Y-%m-%d'),
                    'time': post_time.split()[-1] if ' ' in post_time else '12:00',
                    'platform': platform,
                    'content_title': idea.get('title', idea.get('Title/Hook', '')),
                    'content_type': self._determine_content_type(idea, platform),
                    'status': 'planned'
                })
        
        calendar_df = pd.DataFrame(calendar_data)
        calendar_df['datetime'] = pd.to_datetime(calendar_df['date'] + ' ' + calendar_df['time'])
        calendar_df = calendar_df.sort_values('datetime')
        
        return calendar_df
    
    def _determine_content_type(self, idea: Dict, platform: str) -> str:
        """Determine content type based on idea and platform"""
        platform_types = {
            'youtube': ['shorts', 'video', 'live'],
            'tiktok': ['reel', 'story', 'live'],
            'instagram': ['reel', 'post', 'story', 'carousel']
        }
        
        # Simple logic - would be more sophisticated in practice
        return platform_types.get(platform, ['post'])[0]
    
    def _generate_fallback_ideas(self, strategy: ContentStrategy, num_ideas: int) -> List[Dict]:
        """Generate basic ideas without GPT"""
        ideas = []
        
        templates = [
            "{number} Ways to {action} {topic}",
            "How I {achievement} in {timeframe}",
            "Stop {bad_habit} - Do This Instead",
            "The {adjective} Truth About {topic}",
            "Why {audience} Need to {action}"
        ]
        
        for i in range(num_ideas):
            template = templates[i % len(templates)]
            idea = {
                'title': template.format(
                    number=str(3 + i),
                    action="master",
                    topic=strategy.topic,
                    achievement="succeeded",
                    timeframe="30 days",
                    bad_habit="struggling",
                    adjective="hidden",
                    audience="you"
                ),
                'content_angle': f"Address {strategy.target_audience.pain_points[0] if strategy.target_audience.pain_points else 'common problem'}",
                'platform_recommendations': strategy.platforms,
                'key_points': ["Point 1", "Point 2", "Point 3"],
                'emotional_trigger': strategy.target_audience.primary_emotions[0] if strategy.target_audience.primary_emotions else 'curiosity'
            }
            ideas.append(idea)
        
        return ideas
    
    def _generate_fallback_youtube_content(self, idea: Dict) -> Dict:
        """Basic YouTube content without GPT"""
        return {
            'platform': 'youtube',
            'title': idea.get('title', 'Video Title'),
            'description': f"In this video, we explore {idea.get('content_angle', 'this topic')}...",
            'tags': ['educational', 'howto', 'tutorial'],
            'thumbnail_text': 'MUST WATCH',
            'script_outline': ['Hook', 'Problem', 'Solution', 'Examples', 'CTA'],
            'cta': 'Subscribe for more content like this!'
        }
    
    def _generate_fallback_tiktok_content(self, idea: Dict) -> Dict:
        """Basic TikTok content without GPT"""
        return {
            'platform': 'tiktok',
            'hook': 'Wait for it...',
            'caption': idea.get('title', 'Check this out!'),
            'hashtags': ['fyp', 'viral', 'trending'],
            'script': ['0-3s: Hook', '3-10s: Main content', '10-15s: CTA'],
            'transitions': ['Quick cut', 'Zoom in', 'Text overlay'],
            'music_suggestions': ['Trending sound 1', 'Trending sound 2'],
            'cta': 'Follow for more!'
        }
    
    def _generate_fallback_instagram_content(self, idea: Dict) -> Dict:
        """Basic Instagram content without GPT"""
        return {
            'platform': 'instagram',
            'caption': f"{idea.get('title', 'Amazing content')}\\n\\n{idea.get('content_angle', 'Learn more')}",
            'hashtags': ['instagood', 'viral', 'trending'],
            'reel_script': ['Hook shot', 'Main content', 'Call to action'],
            'carousel_slides': ['Slide 1: Hook', 'Slide 2: Problem', 'Slide 3: Solution'],
            'story_sequence': ['Teaser', 'Behind the scenes', 'Swipe up'],
            'cta': 'Save this for later!'
        }


# Export functions
def export_content_playbook(strategy: ContentStrategy, content_ideas: List[Dict], 
                          calendar: pd.DataFrame, filename: str = "content_playbook"):
    """Export complete content playbook"""
    
    # Create JSON export
    playbook_data = {
        'strategy': asdict(strategy),
        'content_ideas': content_ideas,
        'calendar': calendar.to_dict('records'),
        'generated_at': datetime.now().isoformat()
    }
    
    with open(f"{filename}.json", 'w') as f:
        json.dump(playbook_data, f, indent=2, default=str)
    
    # Create Markdown report
    markdown_report = f"""# Content Playbook - {strategy.topic}

## Target Audience Insights

### Primary Emotions
{', '.join(strategy.target_audience.primary_emotions)}

### Pain Points
{chr(10).join('- ' + p for p in strategy.target_audience.pain_points)}

### Desires
{chr(10).join('- ' + d for d in strategy.target_audience.desires)}

### Common Objections
{chr(10).join('- ' + o for o in strategy.target_audience.objections)}

## Winning Content Patterns

### Top Hooks
"""
    
    for pattern in [p for p in strategy.content_patterns if p.pattern_type == 'hook'][:5]:
        markdown_report += f"\\n**{pattern.pattern_value}** (Avg engagement: {pattern.avg_engagement:.2%})\\n"
        markdown_report += f"Examples: {', '.join(pattern.examples[:2])}\\n"
    
    markdown_report += "\\n### Optimal Content Length\\n"
    for platform, length in strategy.optimal_length.items():
        markdown_report += f"- {platform}: {length} seconds\\n"
    
    markdown_report += "\\n### Best Posting Times\\n"
    for platform, times in strategy.best_times.items():
        markdown_report += f"\\n**{platform}**: {', '.join(times[:3])}\\n"
    
    markdown_report += "\\n## Content Ideas\\n"
    for i, idea in enumerate(content_ideas[:10], 1):
        markdown_report += f"\\n### {i}. {idea.get('title', 'Idea ' + str(i))}\\n"
        markdown_report += f"**Angle**: {idea.get('content_angle', 'N/A')}\\n"
        markdown_report += f"**Platforms**: {', '.join(idea.get('platform_recommendations', []))}\\n"
    
    with open(f"{filename}.md", 'w') as f:
        f.write(markdown_report)
    
    # Export calendar as CSV
    calendar.to_csv(f"{filename}_calendar.csv", index=False)
    
    logger.info(f"Exported playbook to {filename}.json, {filename}.md, and {filename}_calendar.csv")


# Example usage
if __name__ == "__main__":
    # Sample data for testing
    sample_videos = [
        VideoContent(
            platform='youtube',
            video_id='123',
            url='https://youtube.com/123',
            title='How to Use AI for Productivity - 5 Game Changing Tools',
            description='Learn the best AI tools...',
            creator='TechGuru',
            creator_followers=100000,
            views=50000,
            likes=5000,
            comments_count=500,
            shares=1000,
            duration=480,
            upload_date=datetime.now() - timedelta(days=5),
            hashtags=['ai', 'productivity', 'tools'],
            mentions=[],
            engagement_rate=0.12,
            thumbnail_url='thumb.jpg'
        )
    ]
    
    sample_comments = {
        '123': [
            Comment(
                comment_id='c1',
                text='This is exactly what I needed! Been struggling with time management',
                author='user1',
                likes=50,
                replies_count=5,
                timestamp=datetime.now() - timedelta(days=4)
            )
        ]
    }
    
    # Initialize analyzer
    analyzer = ViralPatternAnalyzer()
    
    # Analyze patterns
    strategy = analyzer.analyze_viral_patterns(sample_videos, sample_comments)
    
    # Generate content
    generator = SmartContentGenerator()
    content_ideas = generator.generate_content_ideas(strategy, num_ideas=5)
    
    # Create calendar
    calendar = generator.create_content_calendar(content_ideas, strategy, days=14)
    
    # Export playbook
    export_content_playbook(strategy, content_ideas, calendar)
                            "content": "You are an expert at analyzing audience psychology from social media comments."
                        },
                        {
                            "role": "user",
                            "content": f"""Analyze these comments and extract:
                            1. Top 5 pain points or frustrations
                            2. Top 5 desires or goals
                            3. Top 5 objections or concerns
                            
                            Comments: {json.dumps(sample_comments[:50])}
                            
                            Return as JSON with keys: pain_points, desires, objections"""
                        }
                    ],
                    temperature=0.7
                )
                
                analysis = json.loads(response.choices[0].message.content)
                pain_points = analysis.get('pain_points', [])
                desires = analysis.get('desires', [])
                objections = analysis.get('objections', [])
                
            except Exception as e:
                logger.error(f"Error analyzing audience with GPT: {e}")
        
        # Extract language patterns
        language_patterns = self._extract_language_patterns(all_comments_text)
        
        # Demographic hints
        demographic_hints = self._infer_demographics(all_comments_text, videos)
        
        return AudienceInsight(
            primary_emotions=primary_emotions,
            pain_points=pain_points,
            desires=desires,
            objections=objections,
            language_patterns=language_patterns,
            demographic_hints=demographic_hints
        )
    
    def _extract_language_patterns(self, texts: List[str]) -> List[str]:
        """Extract common language patterns and phrases"""
        # Simple n-gram extraction
        from collections import Counter
        
        all_bigrams = []
        all_trigrams = []
        
        for text in texts[:200]:  # Sample
            words = word_tokenize(text.lower())
            words = [w for w in words if w not in self.stop_words and w.isalnum()]
            
            # Bigrams
            for i in range(len(words) - 1):
                all_bigrams.append(' '.join(words[i:i+2]))
            
            # Trigrams
            for i in range(len(words) - 2):
                all_trigrams.append(' '.join(words[i:i+3]))
        
        # Get most common
        common_bigrams = [b[0] for b in Counter(all_bigrams).most_common(5)]
        common_trigrams = [t[0] for t in Counter(all_trigrams).most_common(5)]
        
        return common_bigrams + common_trigrams
    
    def _infer_demographics(self, comments: List[str], 
                           videos: List[VideoContent]) -> Dict[str, any]:
        """Infer audience demographics from language and engagement patterns"""
        demographics = {
            'age_indicators': [],
            'interest_categories': [],
            'language_complexity': 'medium',
            'engagement_style': 'active'
        }
        
        # Simple heuristics
        young_indicators = ['fr', 'ngl', 'bussin', 'no cap', 'slay', 'periodt']
        mature_indicators = ['furthermore', 'however', 'nevertheless', 'regarding']
        
        young_count = sum(1 for c in comments if any(ind in c.lower() for ind in young_indicators))
        mature_count = sum(1 for c in comments if any(ind in c.lower() for ind in mature_indicators))
        
        if young_count > mature_count * 2:
            demographics['age_indicators'].append('younger_audience')
        elif mature_count > young_count * 2:
            demographics['age_indicators'].append('mature_audience')
        else:
            demographics['age_indicators'].append('mixed_age')
        
        # Interest categories from hashtags
        all_hashtags = []
        for video in videos:
            all_hashtags.extend(video.hashtags)
        
        common_hashtags = [h[0] for h in Counter(all_hashtags).most_common(10)]
        demographics['interest_categories'] = common_hashtags
        
        return demographics
    
    def _analyze_optimal_length(self, videos: List[VideoContent]) -> Dict[str, int]:
        """Determine optimal content length per platform"""
        platform_lengths = defaultdict(list)
        
        for video in videos:
            platform_lengths[video.platform].append({
                'duration': video.duration,
                'engagement': video.engagement_rate
            })
        
        optimal_lengths = {}
        
        for platform, data in platform_lengths.items():
            if data:
                # Find duration with highest average engagement
                duration_buckets = defaultdict(list)
                
                for item in data:
                    bucket = (item['duration'] // 30) * 30  # 30-second buckets
                    duration_buckets[bucket].append(item['engagement'])
                
                # Find best bucket
                best_bucket = max(duration_buckets.items(), 
                                key=lambda x: np.mean(x[1]))[0]
                
                optimal_lengths[platform] = best_bucket
        
        return optimal_lengths
    
    def _analyze_posting_times(self, videos: List[VideoContent]) -> Dict[str, List[str]]:
        """Analyze best posting times per platform"""
        platform_times = defaultdict(list)
        
        for video in videos:
            hour = video.upload_date.hour
            day = video.upload_date.strftime('%A')
            platform_times[video.platform].append({
                'time': f"{day} {hour}:00",
                'engagement': video.engagement_rate
            })
        
        best_times = {}
        
        for platform, data in platform_times.items():
            if data:
                # Group by time slot
                time_performance = defaultdict(list)
                
                for item in data:
                    time_performance[item['time']].append(item['engagement'])
                
                # Sort by average engagement
                sorted_times = sorted(
                    time_performance.items(),
                    key=lambda x: np.mean(x[1]),
                    reverse=True
                )
                
                best_times[platform] = [t[0] for t in sorted_times[:5]]
        
        return best_times
    
    def _identify_trending_formats(self, videos: List[VideoContent]) -> Dict[str, List[str]]:
        """Identify trending content formats per platform"""
        formats = {
            'youtube': ['tutorial', 'reaction', 'compilation', 'shorts', 'documentary'],
            'tiktok': ['dance', 'comedy', 'educational', 'transformation', 'storytime'],
            'instagram': ['reels', 'carousel', 'behind-the-scenes', 'tutorial', 'aesthetic']
        }
        
        # This would need more sophisticated analysis
        # For now, return common formats
        return formats
    
    def _identify_main_topic(self, videos: List[VideoContent]) -> str:
        """Identify the main topic from video content"""
        all_text = ' '.join([v.title + ' ' + v.description for v in videos])
        
        # Simple keyword extraction
        words = word_tokenize(all_text.lower())
        words = [w for w in words if w not in self.stop_words and w.isalnum() and len(w) > 3]
        
        most_common = Counter(words).most_common(1)
        return most_common[0][0] if most_common else 'general'
    
    def _extract_patterns_with_gpt(self, texts: List[str], pattern_type: str) -> List[ContentPattern]:
        """Use GPT to extract additional patterns"""
        patterns = []
        
        if not self.client:
            return patterns
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert at identifying viral {pattern_type} patterns in social media content."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze these successful titles/content and identify common {pattern_type} patterns:

{json.dumps(texts, indent=2)}

Return top 3 patterns as JSON array with structure:
[
  {{
    "pattern_name": "Name of the pattern",
    "description": "How this pattern works",
    "examples": ["example1", "example2"]
  }}
]"""
                    }
                ],
                temperature=0.7
            )
            
            gpt_patterns = json.loads(response.choices[0].message.content)
            
            for p in gpt_patterns:
                pattern = ContentPattern(
                    pattern_type=pattern_type,
                    pattern_value=p['pattern_name'],
                    frequency=len(p.get('examples', [])),
                    avg_engagement=0.0,  # Would need to calculate
                    examples=p.get('examples', [])
                )
                patterns.append(pattern)
                
        except Exception as e:
            logger.error(f"Error extracting patterns with GPT: {e}")
        
        return patterns


class SmartContentGenerator:
    """Generates platform-optimized content based on viral patterns"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
        
        # Platform-specific specs
        self.platform_specs = {
            'youtube': {
                'title_max': 100,
                'description_max': 5000,
                'hashtag_limit': 15,
                'style': 'informative and engaging'
            },
            'tiktok': {
                'caption_max': 150,
                'hashtag_limit': 8,
                'style': 'fast-paced and trendy'
            },
            'instagram': {
                'caption_max': 2200,
                'hashtag_limit': 30,
                'style': 'visual and lifestyle-focused'
            }
        }
    
    def generate_content_ideas(self, strategy: ContentStrategy, 
                             num_ideas: int = 10) -> List[Dict]:
        """Generate content ideas based on strategy"""
        ideas = []
        
        # Use top patterns
        top_hooks = [p for p in strategy.content_patterns if p.pattern_type == 'hook'][:3]
        top_topics = [p for p in strategy.content_patterns if p.pattern_type == 'topic'][:3]
        top_emotions = [p for p in strategy.content_patterns if p.pattern_type == 'emotional'][:2]
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            