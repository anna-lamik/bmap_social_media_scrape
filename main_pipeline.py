"""
Complete Social Content Intelligence Pipeline
Main integration script that orchestrates the entire system
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv

# Import our modules
from platform_extractors import (
    MultiPlatformExtractor, VideoContent, Comment
)
from content_analyzer import (
    ViralPatternAnalyzer, SmartContentGenerator, 
    ContentStrategy, export_content_playbook
)

# Import Agno components
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools
from agno.tools.firecrawl import FirecrawlTools

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('content_intelligence.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ContentIntelligencePipeline:
    """Main orchestrator for the content intelligence system"""
    
    def __init__(self, platforms: List[str] = ['youtube', 'instagram', 'tiktok']):
        logger.info(f"Initializing Content Intelligence Pipeline for platforms: {platforms}")
        
        # Initialize components
        self.platforms = platforms
        self.extractor = MultiPlatformExtractor(platforms)
        self.analyzer = ViralPatternAnalyzer()
        self.generator = SmartContentGenerator()
        
        # Initialize Agno agent for supplementary research
        self.research_agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[
                ExaTools(start_published_date=self._get_start_date(30)),
                FirecrawlTools(scrape=True)
            ],
            description="Content research and trend analysis expert"
        )
        
        # Results storage
        self.results = {
            'extraction_time': None,
            'videos_extracted': {},
            'comments_extracted': {},
            'strategy': None,
            'content_ideas': [],
            'calendar': None
        }
    
    def _get_start_date(self, days_back: int) -> str:
        """Calculate start date for research"""
        return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    def run_full_pipeline(self, 
                         keywords: List[str],
                         videos_per_platform: int = 30,
                         comments_per_video: int = 50,
                         num_content_ideas: int = 20,
                         calendar_days: int = 30) -> Dict:
        """Run the complete content intelligence pipeline"""
        
        logger.info("="*50)
        logger.info("STARTING CONTENT INTELLIGENCE PIPELINE")
        logger.info("="*50)
        
        # Step 1: Extract trending content
        logger.info(f"\\n📊 STEP 1: Extracting trending content for keywords: {keywords}")
        self.results['extraction_time'] = datetime.now()
        
        all_videos = {}
        for keyword in keywords:
            logger.info(f"\\n🔍 Searching for: {keyword}")
            platform_results = self.extractor.search_all_platforms(
                keyword, 
                videos_per_platform
            )
            
            # Merge results
            for platform, videos in platform_results.items():
                if platform not in all_videos:
                    all_videos[platform] = []
                all_videos[platform].extend(videos)
        
        self.results['videos_extracted'] = all_videos
        
        # Log extraction summary
        total_videos = sum(len(videos) for videos in all_videos.values())
        logger.info(f"\\n✅ Extracted {total_videos} total videos")
        for platform, videos in all_videos.items():
            if videos:
                avg_engagement = sum(v.engagement_rate for v in videos) / len(videos)
                logger.info(f"  - {platform}: {len(videos)} videos, avg engagement: {avg_engagement:.2%}")
        
        # Step 2: Extract comments from top videos
        logger.info(f"\\n💬 STEP 2: Extracting comments from top performing videos")
        
        # Get top videos from each platform
        top_videos = []
        for platform, videos in all_videos.items():
            sorted_videos = sorted(videos, key=lambda x: x.engagement_rate, reverse=True)
            top_videos.extend(sorted_videos[:10])  # Top 10 from each platform
        
        # Extract comments
        comments = self.extractor.extract_comments_batch(
            top_videos, 
            comments_per_video
        )
        self.results['comments_extracted'] = comments
        
        total_comments = sum(len(c) for c in comments.values())
        logger.info(f"✅ Extracted {total_comments} comments from {len(comments)} videos")
        
        # Step 3: Analyze patterns
        logger.info(f"\\n🧠 STEP 3: Analyzing viral patterns and audience insights")
        
        # Flatten all videos for analysis
        all_videos_list = []
        for videos in all_videos.values():
            all_videos_list.extend(videos)
        
        # Run analysis
        strategy = self.analyzer.analyze_viral_patterns(
            all_videos_list, 
            comments
        )
        self.results['strategy'] = strategy
        
        logger.info(f"✅ Analysis complete:")
        logger.info(f"  - Identified {len(strategy.content_patterns)} content patterns")
        logger.info(f"  - Primary emotions: {', '.join(strategy.target_audience.primary_emotions)}")
        logger.info(f"  - Found {len(strategy.target_audience.pain_points)} pain points")
        logger.info(f"  - Found {len(strategy.target_audience.desires)} audience desires")
        
        # Step 4: Generate content ideas
        logger.info(f"\\n💡 STEP 4: Generating {num_content_ideas} content ideas")
        
        content_ideas = self.generator.generate_content_ideas(
            strategy, 
            num_content_ideas
        )
        self.results['content_ideas'] = content_ideas
        
        logger.info(f"✅ Generated {len(content_ideas)} content ideas")
        
        # Step 5: Create content for each platform
        logger.info(f"\\n🎨 STEP 5: Creating platform-specific content")
        
        platform_content = {}
        for platform in self.platforms:
            platform_content[platform] = []
            
            # Generate content for top 5 ideas on each platform
            for idea in content_ideas[:5]:
                content = self.generator.generate_platform_content(
                    idea, 
                    platform, 
                    strategy
                )
                platform_content[platform].append(content)
        
        self.results['platform_content'] = platform_content
        
        # Step 6: Create content calendar
        logger.info(f"\\n📅 STEP 6: Creating {calendar_days}-day content calendar")
        
        calendar = self.generator.create_content_calendar(
            content_ideas, 
            strategy, 
            calendar_days
        )
        self.results['calendar'] = calendar
        
        logger.info(f"✅ Created calendar with {len(calendar)} scheduled posts")
        
        # Step 7: Supplementary research
        logger.info(f"\\n🔬 STEP 7: Conducting supplementary research")
        
        supplementary_insights = self._conduct_supplementary_research(keywords, strategy)
        self.results['supplementary_insights'] = supplementary_insights
        
        logger.info("\\n✨ PIPELINE COMPLETE!")
        
        return self.results
    
    def _conduct_supplementary_research(self, keywords: List[str], 
                                      strategy: ContentStrategy) -> str:
        """Use Agno agent for additional research"""
        research_prompt = f"""
        Analyze current trends and provide additional insights for:
        Keywords: {', '.join(keywords)}
        Main topic: {strategy.topic}
        
        Focus on:
        1. Emerging sub-trends not captured in social media
        2. Industry news and developments
        3. Competitor strategies
        4. Future predictions
        
        Provide actionable insights for content creation.
        """
        
        try:
            response = self.research_agent.run(research_prompt)
            return response
        except Exception as e:
            logger.error(f"Supplementary research error: {e}")
            return "Supplementary research unavailable"
    
    def export_results(self, output_dir: str = "output"):
        """Export all results in multiple formats"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export extracted videos
        if self.results['videos_extracted']:
            self.extractor.export_to_csv(
                self.results['videos_extracted'],
                os.path.join(output_dir, f"extracted_videos_{timestamp}.csv")
            )
        
        # Export content playbook
        if self.results['strategy'] and self.results['content_ideas']:
            export_content_playbook(
                self.results['strategy'],
                self.results['content_ideas'],
                self.results['calendar'],
                os.path.join(output_dir, f"content_playbook_{timestamp}")
            )
        
        # Export platform-specific content
        if self.results.get('platform_content'):
            for platform, contents in self.results['platform_content'].items():
                filename = os.path.join(output_dir, f"{platform}_content_{timestamp}.json")
                with open(filename, 'w') as f:
                    json.dump(contents, f, indent=2)
        
        # Create summary report
        self._create_summary_report(output_dir, timestamp)
        
        logger.info(f"\\n📁 All results exported to: {output_dir}/")
    
    def _create_summary_report(self, output_dir: str, timestamp: str):
        """Create executive summary report"""
        report = f"""# Content Intelligence Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

### Content Extraction
- Total videos analyzed: {sum(len(v) for v in self.results['videos_extracted'].values())}
- Total comments analyzed: {sum(len(c) for c in self.results['comments_extracted'].values())}
- Platforms covered: {', '.join(self.platforms)}

### Key Insights
"""
        
        if self.results['strategy']:
            strategy = self.results['strategy']
            report += f"""
#### Target Audience
- **Primary Emotions**: {', '.join(strategy.target_audience.primary_emotions)}
- **Top Pain Points**: 
{chr(10).join('  - ' + p for p in strategy.target_audience.pain_points[:3])}
- **Top Desires**:
{chr(10).join('  - ' + d for d in strategy.target_audience.desires[:3])}

#### Winning Patterns
"""
            
            # Top patterns by type
            for pattern_type in ['hook', 'emotional', 'topic']:
                patterns = [p for p in strategy.content_patterns if p.pattern_type == pattern_type]
                if patterns:
                    top_pattern = patterns[0]
                    report += f"\\n**Best {pattern_type}**: {top_pattern.pattern_value} "
                    report += f"(avg engagement: {top_pattern.avg_engagement:.2%})\\n"
        
        report += f"""

### Content Strategy
- Generated content ideas: {len(self.results['content_ideas'])}
- Calendar days planned: {len(self.results['calendar']) if self.results['calendar'] is not None else 0}

### Next Steps
1. Review the content playbook for detailed strategies
2. Customize content ideas to your brand voice
3. Begin creating content using the provided templates
4. Monitor performance and iterate based on results

### Files Generated
- `extracted_videos_{timestamp}.csv` - Raw video data
- `content_playbook_{timestamp}.json` - Complete strategy and ideas
- `content_playbook_{timestamp}.md` - Human-readable playbook
- `content_playbook_{timestamp}_calendar.csv` - Posting schedule
- Platform-specific content JSON files
"""
        
        if self.results.get('supplementary_insights'):
            report += f"""

### Supplementary Research Insights
{self.results['supplementary_insights'][:1000]}...
"""
        
        # Save report
        report_path = os.path.join(output_dir, f"summary_report_{timestamp}.md")
        with open(report_path, 'w') as f