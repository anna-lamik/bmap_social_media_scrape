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
        with open(report_path, 'w') as f:
            f.write(report)


def main():
    """Main entry point for the content intelligence system"""
    parser = argparse.ArgumentParser(
        description="Social Content Intelligence System - Extract viral patterns and generate content"
    )
    
    parser.add_argument(
        'keywords',
        nargs='+',
        help='Keywords to search for (e.g., "AI tools" "productivity hacks")'
    )
    
    parser.add_argument(
        '--platforms',
        nargs='+',
        choices=['youtube', 'instagram', 'tiktok'],
        default=['youtube', 'instagram', 'tiktok'],
        help='Platforms to analyze'
    )
    
    parser.add_argument(
        '--videos-per-platform',
        type=int,
        default=30,
        help='Number of videos to extract per platform (default: 30)'
    )
    
    parser.add_argument(
        '--comments-per-video',
        type=int,
        default=50,
        help='Number of comments to extract per video (default: 50)'
    )
    
    parser.add_argument(
        '--content-ideas',
        type=int,
        default=20,
        help='Number of content ideas to generate (default: 20)'
    )
    
    parser.add_argument(
        '--calendar-days',
        type=int,
        default=30,
        help='Number of days for content calendar (default: 30)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results (default: output)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode - fewer videos and comments for testing'
    )
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.videos_per_platform = 5
        args.comments_per_video = 10
        args.content_ideas = 5
        args.calendar_days = 7
        logger.info("Running in QUICK MODE - reduced data collection")
    
    # Check for required API keys
    required_keys = ['OPENAI_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        logger.warning(f"Missing API keys: {missing_keys}")
        logger.warning("Some features may be limited. Add keys to .env file for full functionality.")
    
    # Initialize and run pipeline
    try:
        pipeline = ContentIntelligencePipeline(platforms=args.platforms)
        
        results = pipeline.run_full_pipeline(
            keywords=args.keywords,
            videos_per_platform=args.videos_per_platform,
            comments_per_video=args.comments_per_video,
            num_content_ideas=args.content_ideas,
            calendar_days=args.calendar_days
        )
        
        # Export results
        pipeline.export_results(args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("✅ CONTENT INTELLIGENCE PIPELINE COMPLETE!")
        print("="*50)
        print(f"\n📁 Results saved to: {args.output}/")
        print("\nKey files generated:")
        print("  - summary_report_*.md - Executive summary")
        print("  - content_playbook_*.json - Complete playbook data")
        print("  - content_playbook_*.md - Human-readable strategies")
        print("  - *_content_*.json - Platform-specific content")
        print("  - content_playbook_*_calendar.csv - Posting schedule")
        
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)


def demo_mode():
    """Run a demonstration with sample data"""
    logger.info("Running in DEMO MODE with sample data")
    
    # Create sample data
    sample_videos = [
        VideoContent(
            platform='youtube',
            video_id='demo1',
            url='https://youtube.com/demo1',
            title='5 AI Tools That 10x My Productivity',
            description='Discover game-changing AI tools...',
            creator='ProductivityGuru',
            creator_followers=250000,
            views=100000,
            likes=8000,
            comments_count=500,
            shares=2000,
            duration=480,
            upload_date=datetime.now() - timedelta(days=3),
            hashtags=['ai', 'productivity', 'tools', 'automation'],
            mentions=['@openai'],
            engagement_rate=0.105,
            thumbnail_url='thumb.jpg'
        ),
        VideoContent(
            platform='tiktok',
            video_id='demo2',
            url='https://tiktok.com/demo2',
            title='You\'ve been using ChatGPT wrong this whole time #ai #productivity',
            description='Mind-blowing ChatGPT hack...',
            creator='TechTips',
            creator_followers=500000,
            views=500000,
            likes=50000,
            comments_count=3000,
            shares=10000,
            duration=30,
            upload_date=datetime.now() - timedelta(days=1),
            hashtags=['ai', 'chatgpt', 'productivity', 'hack'],
            mentions=[],
            engagement_rate=0.126,
            thumbnail_url='thumb.jpg'
        ),
        VideoContent(
            platform='instagram',
            video_id='demo3',
            url='https://instagram.com/demo3',
            title='Stop wasting time! Here are 3 AI automation tricks',
            description='Save hours every day with these AI tricks...',
            creator='AILifeHacks',
            creator_followers=150000,
            views=75000,
            likes=9000,
            comments_count=800,
            shares=1500,
            duration=60,
            upload_date=datetime.now() - timedelta(days=2),
            hashtags=['aiautomation', 'productivity', 'lifehacks'],
            mentions=['@anthropic'],
            engagement_rate=0.15,
            thumbnail_url='thumb.jpg'
        )
    ]
    
    sample_comments = {
        'demo1': [
            Comment('c1', 'This changed my workflow completely!', 'user1', 50, 2, datetime.now()),
            Comment('c2', 'Which tool is best for writing?', 'user2', 20, 1, datetime.now()),
            Comment('c3', 'I wish I knew about these sooner', 'user3', 35, 0, datetime.now())
        ],
        'demo2': [
            Comment('c4', 'Mind = blown 🤯', 'user4', 100, 5, datetime.now()),
            Comment('c5', 'Does this work with GPT-4?', 'user5', 45, 2, datetime.now())
        ],
        'demo3': [
            Comment('c6', 'Saving this for later!', 'user6', 80, 1, datetime.now()),
            Comment('c7', 'The automation tip is gold', 'user7', 60, 3, datetime.now())
        ]
    }
    
    # Run analysis
    analyzer = ViralPatternAnalyzer()
    strategy = analyzer.analyze_viral_patterns(sample_videos, sample_comments)
    
    # Generate content
    generator = SmartContentGenerator()
    ideas = generator.generate_content_ideas(strategy, num_ideas=5)
    
    # Create sample outputs
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export playbook
    calendar = generator.create_content_calendar(ideas, strategy, days=7)
    export_content_playbook(strategy, ideas, calendar, 
                          os.path.join(output_dir, "demo_playbook"))
    
    print("\n✅ Demo complete! Check demo_output/ directory for results.")


def validate_environment():
    """Validate the environment setup"""
    print("🔍 Validating environment setup...\n")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python {python_version.major}.{python_version.minor} - OK")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor} - Requires 3.8+")
    
    # Check required packages
    required_packages = [
        'agno', 'openai', 'pandas', 'numpy', 'nltk', 
        'textblob', 'sklearn', 'yt_dlp', 'instagrapi'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Not installed")
    
    # Check API keys
    api_keys = {
        'OPENAI_API_KEY': 'OpenAI (Required)',
        'YOUTUBE_API_KEY': 'YouTube (Optional)',
        'INSTAGRAM_USERNAME': 'Instagram (Optional)',
        'EXA_API_KEY': 'Exa (Optional)',
        'FIRECRAWL_API_KEY': 'Firecrawl (Optional)'
    }
    
    print("\n🔑 API Keys:")
    for key, name in api_keys.items():
        if os.getenv(key):
            print(f"✅ {name} - Found")
        else:
            print(f"❌ {name} - Not found")
    
    print("\n💡 To add API keys, create a .env file with:")
    print("OPENAI_API_KEY=your_key_here")
    print("YOUTUBE_API_KEY=your_key_here")
    print("# etc...")


if __name__ == "__main__":
    # Special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo_mode()
        elif sys.argv[1] == '--validate':
            validate_environment()
        else:
            main()
    else:
        # Show help if no arguments
        print("Social Content Intelligence System")
        print("=" * 40)
        print("\nUsage:")
        print("  python main.py [keywords] [options]")
        print("\nExamples:")
        print('  python main.py "AI productivity" --quick')
        print('  python main.py "fitness tips" "workout hacks" --platforms youtube tiktok')
        print("\nSpecial commands:")
        print("  python main.py --demo     # Run with sample data")
        print("  python main.py --validate # Check environment setup")
        print("\nFor full options:")
        print("  python main.py --help")
