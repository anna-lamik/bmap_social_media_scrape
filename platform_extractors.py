"""
Platform-specific content extractors for YouTube, Instagram, and TikTok
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
import logging

# YouTube imports
import yt_dlp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Instagram imports
from instagrapi import Client
from instagrapi.exceptions import LoginRequired, ClientError

# TikTok imports
import pyktok as pyk
from TikTokApi import TikTokApi

# Utilities
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VideoContent:
    """Unified video content structure"""
    platform: str
    video_id: str
    url: str
    title: str
    description: str
    creator: str
    creator_followers: int
    views: int
    likes: int
    comments_count: int
    shares: int
    duration: int  # seconds
    upload_date: datetime
    hashtags: List[str]
    mentions: List[str]
    engagement_rate: float
    thumbnail_url: str
    download_url: Optional[str] = None
    captions: Optional[str] = None
    music: Optional[Dict] = None
    

@dataclass
class Comment:
    """Comment structure"""
    comment_id: str
    text: str
    author: str
    likes: int
    replies_count: int
    timestamp: datetime
    is_pinned: bool = False
    sentiment: Optional[str] = None
    

class YouTubeExtractor:
    """Extract content from YouTube"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        self.youtube = None
        if self.api_key:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        
        # yt-dlp options
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
        }
    
    def search_videos(self, query: str, max_results: int = 50) -> List[VideoContent]:
        """Search for videos by keyword"""
        videos = []
        
        if self.youtube:
            # Use YouTube Data API
            try:
                search_response = self.youtube.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=min(max_results, 50),
                    order='viewCount',
                    type='video',
                    publishedAfter=(datetime.now() - timedelta(days=30)).isoformat() + 'Z'
                ).execute()
                
                video_ids = [item['id']['videoId'] for item in search_response['items']]
                
                # Get video statistics
                stats_response = self.youtube.videos().list(
                    part='statistics,contentDetails,snippet',
                    id=','.join(video_ids)
                ).execute()
                
                for item in stats_response['items']:
                    video = self._parse_youtube_api_response(item)
                    videos.append(video)
                    
            except HttpError as e:
                logger.error(f"YouTube API error: {e}")
                # Fallback to yt-dlp
                videos = self._search_with_ytdlp(query, max_results)
        else:
            # Use yt-dlp
            videos = self._search_with_ytdlp(query, max_results)
            
        return sorted(videos, key=lambda x: x.engagement_rate, reverse=True)
    
    def _search_with_ytdlp(self, query: str, max_results: int) -> List[VideoContent]:
        """Search using yt-dlp (no API needed)"""
        videos = []
        search_url = f"ytsearch{max_results}:{query}"
        
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            try:
                search_results = ydl.extract_info(search_url, download=False)
                
                for entry in search_results.get('entries', []):
                    if entry:
                        video = self._parse_ytdlp_response(entry)
                        videos.append(video)
                        
            except Exception as e:
                logger.error(f"yt-dlp search error: {e}")
                
        return videos
    
    def get_video_details(self, video_id: str) -> Optional[VideoContent]:
        """Get detailed information about a video"""
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return self._parse_ytdlp_response(info)
            except Exception as e:
                logger.error(f"Error extracting video {video_id}: {e}")
                return None
    
    def get_comments(self, video_id: str, max_comments: int = 100) -> List[Comment]:
        """Extract comments from a video"""
        comments = []
        
        if self.youtube:
            # Use YouTube API
            try:
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=min(max_comments, 100),
                    order='relevance'
                )
                
                while request and len(comments) < max_comments:
                    response = request.execute()
                    
                    for item in response['items']:
                        comment = self._parse_youtube_comment(item)
                        comments.append(comment)
                        
                        # Get replies if any
                        if item['snippet']['totalReplyCount'] > 0:
                            replies = self._get_comment_replies(
                                item['snippet']['topLevelComment']['id']
                            )
                            comments.extend(replies[:5])  # Limit replies
                    
                    request = self.youtube.commentThreads().list_next(
                        request, response
                    )
                    
            except HttpError as e:
                logger.error(f"Error fetching comments: {e}")
        else:
            # Use yt-dlp with comments
            opts = self.ydl_opts.copy()
            opts['getcomments'] = True
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                try:
                    info = ydl.extract_info(
                        f"https://www.youtube.com/watch?v={video_id}", 
                        download=False
                    )
                    
                    for comment_data in info.get('comments', [])[:max_comments]:
                        comment = Comment(
                            comment_id=comment_data.get('id', ''),
                            text=comment_data.get('text', ''),
                            author=comment_data.get('author', ''),
                            likes=comment_data.get('like_count', 0),
                            replies_count=0,
                            timestamp=datetime.fromtimestamp(
                                comment_data.get('timestamp', 0)
                            ),
                            is_pinned=comment_data.get('is_pinned', False)
                        )
                        comments.append(comment)
                        
                except Exception as e:
                    logger.error(f"Error extracting comments with yt-dlp: {e}")
                    
        return comments
    
    def _parse_youtube_api_response(self, item: Dict) -> VideoContent:
        """Parse YouTube API response"""
        snippet = item['snippet']
        stats = item.get('statistics', {})
        
        views = int(stats.get('viewCount', 0))
        likes = int(stats.get('likeCount', 0))
        comments_count = int(stats.get('commentCount', 0))
        
        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', snippet.get('description', ''))
        
        # Calculate engagement rate
        engagement_rate = (likes + comments_count) / views if views > 0 else 0
        
        return VideoContent(
            platform='youtube',
            video_id=item['id'],
            url=f"https://www.youtube.com/watch?v={item['id']}",
            title=snippet['title'],
            description=snippet.get('description', ''),
            creator=snippet['channelTitle'],
            creator_followers=0,  # Would need separate API call
            views=views,
            likes=likes,
            comments_count=comments_count,
            shares=0,  # YouTube doesn't provide share count
            duration=self._parse_duration(item.get('contentDetails', {}).get('duration', '')),
            upload_date=datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
            hashtags=hashtags,
            mentions=[],
            engagement_rate=engagement_rate,
            thumbnail_url=snippet['thumbnails']['high']['url']
        )
    
    def _parse_ytdlp_response(self, info: Dict) -> VideoContent:
        """Parse yt-dlp response"""
        views = info.get('view_count', 0) or 0
        likes = info.get('like_count', 0) or 0
        comments_count = info.get('comment_count', 0) or 0
        
        # Extract hashtags from description
        description = info.get('description', '') or ''
        hashtags = re.findall(r'#(\w+)', description)
        
        # Calculate engagement rate
        engagement_rate = (likes + comments_count) / views if views > 0 else 0
        
        return VideoContent(
            platform='youtube',
            video_id=info.get('id', ''),
            url=info.get('webpage_url', ''),
            title=info.get('title', ''),
            description=description,
            creator=info.get('uploader', ''),
            creator_followers=info.get('channel_follower_count', 0) or 0,
            views=views,
            likes=likes,
            comments_count=comments_count,
            shares=0,
            duration=info.get('duration', 0) or 0,
            upload_date=datetime.fromtimestamp(info.get('upload_date', 0)) if info.get('upload_date') else datetime.now(),
            hashtags=hashtags,
            mentions=[],
            engagement_rate=engagement_rate,
            thumbnail_url=info.get('thumbnail', ''),
            download_url=info.get('url', '')
        )
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        import isodate
        try:
            duration = isodate.parse_duration(duration_str)
            return int(duration.total_seconds())
        except:
            return 0
    
    def _parse_youtube_comment(self, item: Dict) -> Comment:
        """Parse YouTube API comment response"""
        snippet = item['snippet']['topLevelComment']['snippet']
        
        return Comment(
            comment_id=item['id'],
            text=snippet['textDisplay'],
            author=snippet['authorDisplayName'],
            likes=snippet['likeCount'],
            replies_count=item['snippet']['totalReplyCount'],
            timestamp=datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
            is_pinned=False  # Need additional logic to detect
        )


class InstagramExtractor:
    """Extract content from Instagram"""
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.client = Client()
        self.username = username or os.getenv("INSTAGRAM_USERNAME")
        self.password = password or os.getenv("INSTAGRAM_PASSWORD")
        self.logged_in = False
        
        if self.username and self.password:
            self._login()
    
    def _login(self):
        """Login to Instagram"""
        try:
            self.client.login(self.username, self.password)
            self.logged_in = True
            logger.info("Successfully logged in to Instagram")
        except Exception as e:
            logger.error(f"Instagram login failed: {e}")
            self.logged_in = False
    
    def search_hashtag(self, hashtag: str, amount: int = 50) -> List[VideoContent]:
        """Search for videos by hashtag"""
        videos = []
        
        if not self.logged_in:
            logger.error("Not logged in to Instagram")
            return videos
        
        try:
            # Get hashtag info
            hashtag_info = self.client.hashtag_info(hashtag)
            
            # Get recent media
            medias = self.client.hashtag_medias_recent(
                hashtag_info.id, 
                amount=amount
            )
            
            for media in medias:
                if media.media_type == 2:  # Video/Reel
                    video = self._parse_instagram_media(media)
                    videos.append(video)
                    
        except Exception as e:
            logger.error(f"Error searching Instagram hashtag: {e}")
            
        return sorted(videos, key=lambda x: x.engagement_rate, reverse=True)
    
    def get_user_reels(self, username: str, amount: int = 20) -> List[VideoContent]:
        """Get reels from a specific user"""
        videos = []
        
        if not self.logged_in:
            return videos
            
        try:
            user_id = self.client.user_id_from_username(username)
            reels = self.client.user_clips(user_id, amount=amount)
            
            for reel in reels:
                video = self._parse_instagram_media(reel)
                videos.append(video)
                
        except Exception as e:
            logger.error(f"Error getting user reels: {e}")
            
        return videos
    
    def get_comments(self, media_id: str, amount: int = 100) -> List[Comment]:
        """Get comments from a post"""
        comments = []
        
        if not self.logged_in:
            return comments
            
        try:
            media_comments = self.client.media_comments(media_id, amount=amount)
            
            for comment in media_comments:
                parsed_comment = Comment(
                    comment_id=str(comment.pk),
                    text=comment.text,
                    author=comment.user.username,
                    likes=comment.comment_like_count,
                    replies_count=comment.child_comment_count,
                    timestamp=comment.created_at_utc,
                    is_pinned=comment.has_liked_comment
                )
                comments.append(parsed_comment)
                
        except Exception as e:
            logger.error(f"Error getting Instagram comments: {e}")
            
        return comments
    
    def _parse_instagram_media(self, media) -> VideoContent:
        """Parse Instagram media object"""
        # Extract hashtags and mentions
        caption = media.caption_text or ""
        hashtags = re.findall(r'#(\w+)', caption)
        mentions = re.findall(r'@(\w+)', caption)
        
        # Calculate engagement rate
        views = media.view_count or media.play_count or 0
        likes = media.like_count or 0
        comments = media.comment_count or 0
        
        engagement_rate = (likes + comments) / views if views > 0 else 0
        
        return VideoContent(
            platform='instagram',
            video_id=media.id,
            url=f"https://www.instagram.com/p/{media.code}/",
            title=caption[:100] + "..." if len(caption) > 100 else caption,
            description=caption,
            creator=media.user.username,
            creator_followers=media.user.follower_count if hasattr(media.user, 'follower_count') else 0,
            views=views,
            likes=likes,
            comments_count=comments,
            shares=0,  # Instagram doesn't provide share count
            duration=int(media.video_duration) if hasattr(media, 'video_duration') else 0,
            upload_date=media.taken_at,
            hashtags=hashtags,
            mentions=mentions,
            engagement_rate=engagement_rate,
            thumbnail_url=media.thumbnail_url if hasattr(media, 'thumbnail_url') else "",
            download_url=media.video_url if hasattr(media, 'video_url') else None,
            music=media.clips_metadata.get('music_info') if hasattr(media, 'clips_metadata') else None
        )


class TikTokExtractor:
    """Extract content from TikTok"""
    
    def __init__(self):
        self.api = None
        try:
            # Try TikTok-Api first
            self.api = TikTokApi()
        except:
            logger.info("TikTok-Api not available, using pyktok")
    
    def search_videos(self, keyword: str, num_videos: int = 50) -> List[VideoContent]:
        """Search for videos by keyword"""
        videos = []
        
        if self.api:
            # Use TikTok-Api
            try:
                search_results = self.api.search.search_for_videos(
                    keyword, 
                    count=num_videos
                )
                
                for video in search_results:
                    parsed = self._parse_tiktokapi_video(video)
                    videos.append(parsed)
                    
            except Exception as e:
                logger.error(f"TikTok-Api search error: {e}")
                # Fallback to pyktok
                videos = self._search_with_pyktok(keyword, num_videos)
        else:
            # Use pyktok
            videos = self._search_with_pyktok(keyword, num_videos)
            
        return sorted(videos, key=lambda x: x.engagement_rate, reverse=True)
    
    def _search_with_pyktok(self, keyword: str, num_videos: int) -> List[VideoContent]:
        """Search using pyktok"""
        videos = []
        
        try:
            # Search for videos
            video_data = pyk.search(keyword, max_results=num_videos)
            
            for video in video_data:
                parsed = self._parse_pyktok_video(video)
                videos.append(parsed)
                
        except Exception as e:
            logger.error(f"PyKTok search error: {e}")
            
        return videos
    
    def get_user_videos(self, username: str, num_videos: int = 20) -> List[VideoContent]:
        """Get videos from a specific user"""
        videos = []
        
        try:
            if self.api:
                user = self.api.user(username)
                user_videos = user.videos(count=num_videos)
                
                for video in user_videos:
                    parsed = self._parse_tiktokapi_video(video)
                    videos.append(parsed)
            else:
                # Use pyktok
                user_data = pyk.get_user_videos(username, max_results=num_videos)
                
                for video in user_data:
                    parsed = self._parse_pyktok_video(video)
                    videos.append(parsed)
                    
        except Exception as e:
            logger.error(f"Error getting TikTok user videos: {e}")
            
        return videos
    
    def get_comments(self, video_id: str, count: int = 100) -> List[Comment]:
        """Get comments from a video"""
        comments = []
        
        try:
            if self.api:
                video = self.api.video(id=video_id)
                video_comments = video.comments(count=count)
                
                for comment in video_comments:
                    parsed = Comment(
                        comment_id=comment.id,
                        text=comment.text,
                        author=comment.author.username,
                        likes=comment.likes_count,
                        replies_count=comment.reply_comment_total,
                        timestamp=datetime.fromtimestamp(comment.create_time),
                        is_pinned=False
                    )
                    comments.append(parsed)
            else:
                # PyKTok doesn't have direct comment access
                logger.info("Comment extraction not available with pyktok")
                
        except Exception as e:
            logger.error(f"Error getting TikTok comments: {e}")
            
        return comments
    
    def _parse_tiktokapi_video(self, video) -> VideoContent:
        """Parse TikTok-Api video object"""
        stats = video.stats
        
        # Extract hashtags and mentions
        desc = video.desc or ""
        hashtags = re.findall(r'#(\w+)', desc)
        mentions = re.findall(r'@(\w+)', desc)
        
        # Calculate engagement
        views = stats.playCount
        likes = stats.diggCount
        comments = stats.commentCount
        shares = stats.shareCount
        
        engagement_rate = (likes + comments + shares) / views if views > 0 else 0
        
        return VideoContent(
            platform='tiktok',
            video_id=video.id,
            url=f"https://www.tiktok.com/@{video.author.username}/video/{video.id}",
            title=desc[:100] + "..." if len(desc) > 100 else desc,
            description=desc,
            creator=video.author.username,
            creator_followers=video.author.stats.followerCount,
            views=views,
            likes=likes,
            comments_count=comments,
            shares=shares,
            duration=video.video.duration,
            upload_date=datetime.fromtimestamp(video.createTime),
            hashtags=hashtags,
            mentions=mentions,
            engagement_rate=engagement_rate,
            thumbnail_url=video.video.cover,
            download_url=video.video.playAddr,
            music={"title": video.music.title, "author": video.music.authorName} if video.music else None
        )
    
    def _parse_pyktok_video(self, video: Dict) -> VideoContent:
        """Parse pyktok video data"""
        # PyKTok returns different structure
        stats = video.get('stats', {})
        
        desc = video.get('desc', '')
        hashtags = re.findall(r'#(\w+)', desc)
        mentions = re.findall(r'@(\w+)', desc)
        
        views = stats.get('playCount', 0)
        likes = stats.get('diggCount', 0)
        comments = stats.get('commentCount', 0)
        shares = stats.get('shareCount', 0)
        
        engagement_rate = (likes + comments + shares) / views if views > 0 else 0
        
        return VideoContent(
            platform='tiktok',
            video_id=video.get('id', ''),
            url=video.get('video_url', ''),
            title=desc[:100] + "..." if len(desc) > 100 else desc,
            description=desc,
            creator=video.get('author', {}).get('uniqueId', ''),
            creator_followers=video.get('author', {}).get('followerCount', 0),
            views=views,
            likes=likes,
            comments_count=comments,
            shares=shares,
            duration=video.get('video', {}).get('duration', 0),
            upload_date=datetime.fromtimestamp(video.get('createTime', 0)),
            hashtags=hashtags,
            mentions=mentions,
            engagement_rate=engagement_rate,
            thumbnail_url=video.get('video', {}).get('cover', ''),
            music=video.get('music')
        )


# Unified extractor manager
class MultiPlatformExtractor:
    """Manages all platform extractors"""
    
    def __init__(self, platforms: List[str] = ['youtube', 'instagram', 'tiktok']):
        self.extractors = {}
        
        if 'youtube' in platforms:
            self.extractors['youtube'] = YouTubeExtractor()
            
        if 'instagram' in platforms:
            self.extractors['instagram'] = InstagramExtractor()
            
        if 'tiktok' in platforms:
            self.extractors['tiktok'] = TikTokExtractor()
    
    def search_all_platforms(self, query: str, videos_per_platform: int = 20) -> Dict[str, List[VideoContent]]:
        """Search across all platforms"""
        results = {}
        
        for platform, extractor in self.extractors.items():
            logger.info(f"Searching {platform} for: {query}")
            
            try:
                if platform == 'youtube':
                    videos = extractor.search_videos(query, videos_per_platform)
                elif platform == 'instagram':
                    # Instagram searches by hashtag
                    hashtag = query.replace(' ', '').replace('#', '')
                    videos = extractor.search_hashtag(hashtag, videos_per_platform)
                elif platform == 'tiktok':
                    videos = extractor.search_videos(query, videos_per_platform)
                
                results[platform] = videos
                logger.info(f"Found {len(videos)} videos on {platform}")
                
            except Exception as e:
                logger.error(f"Error searching {platform}: {e}")
                results[platform] = []
        
        return results
    
    def get_trending_by_creator(self, creators: Dict[str, str], videos_per_creator: int = 10) -> Dict[str, List[VideoContent]]:
        """Get videos from specific creators on each platform"""
        results = {}
        
        for platform, creator_name in creators.items():
            if platform not in self.extractors:
                continue
                
            try:
                if platform == 'youtube':
                    # Would need channel ID for YouTube
                    logger.info(f"Creator search not implemented for YouTube")
                    results[platform] = []
                elif platform == 'instagram':
                    videos = self.extractors[platform].get_user_reels(creator_name, videos_per_creator)
                    results[platform] = videos
                elif platform == 'tiktok':
                    videos = self.extractors[platform].get_user_videos(creator_name, videos_per_creator)
                    results[platform] = videos
                    
            except Exception as e:
                logger.error(f"Error getting creator content from {platform}: {e}")
                results[platform] = []
        
        return results
    
    def extract_comments_batch(self, videos: List[VideoContent], comments_per_video: int = 50) -> Dict[str, List[Comment]]:
        """Extract comments from multiple videos"""
        all_comments = {}
        
        for video in videos:
            if video.platform not in self.extractors:
                continue
                
            logger.info(f"Extracting comments from {video.platform} video: {video.video_id}")
            
            try:
                comments = self.extractors[video.platform].get_comments(
                    video.video_id, 
                    comments_per_video
                )
                all_comments[video.video_id] = comments
                
            except Exception as e:
                logger.error(f"Error extracting comments: {e}")
                all_comments[video.video_id] = []
        
        return all_comments
    
    def export_to_csv(self, videos: Dict[str, List[VideoContent]], filename: str = "extracted_content.csv"):
        """Export extracted content to CSV"""
        all_videos = []
        
        for platform, platform_videos in videos.items():
            for video in platform_videos:
                video_dict = {
                    'platform': video.platform,
                    'video_id': video.video_id,
                    'url': video.url,
                    'title': video.title,
                    'creator': video.creator,
                    'creator_followers': video.creator_followers,
                    'views': video.views,
                    'likes': video.likes,
                    'comments': video.comments_count,
                    'shares': video.shares,
                    'engagement_rate': video.engagement_rate,
                    'upload_date': video.upload_date,
                    'hashtags': ','.join(video.hashtags),
                    'duration_seconds': video.duration
                }
                all_videos.append(video_dict)
        
        df = pd.DataFrame(all_videos)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(all_videos)} videos to {filename}")
        
        return df


# Example usage and testing
if __name__ == "__main__":
    # Initialize multi-platform extractor
    extractor = MultiPlatformExtractor(['youtube', 'instagram', 'tiktok'])
    
    # Search across all platforms
    search_query = "AI productivity tools"
    results = extractor.search_all_platforms(search_query, videos_per_platform=10)
    
    # Print summary
    for platform, videos in results.items():
        print(f"\n{platform.upper()} Results:")
        print(f"Found {len(videos)} videos")
        
        if videos:
            top_video = videos[0]
            print(f"Top video: {top_video.title}")
            print(f"Views: {top_video.views:,}")
            print(f"Engagement rate: {top_video.engagement_rate:.2%}")
            print(f"Creator: {top_video.creator} ({top_video.creator_followers:,} followers)")
    
    # Extract comments from top videos
    all_top_videos = []
    for platform_videos in results.values():
        if platform_videos:
            all_top_videos.extend(platform_videos[:3])  # Top 3 from each platform
    
    comments = extractor.extract_comments_batch(all_top_videos, comments_per_video=20)
    
    print(f"\n\nExtracted comments from {len(comments)} videos")
    
    # Export to CSV
    extractor.export_to_csv(results, "trending_ai_content.csv")