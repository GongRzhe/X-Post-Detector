#!/usr/bin/env python3
import os
import time
import json
import asyncio
import requests
import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import AutoGen components
from autogen_core import FunctionCall, MessageContext, RoutedAgent, message_handler
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams


# Custom JSON Encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Define Pydantic models for our data structures
class XPost(BaseModel):
    id: str
    text: str
    created_at: str
    author_id: str
    html_url: str
    media_urls: List[str] = Field(default_factory=list)


class AccountConfig(BaseModel):
    username: str
    latest_check: Optional[str] = None  # Store as string instead of datetime
    posts: List[XPost] = Field(default_factory=list)


class PostHistory(BaseModel):
    accounts: List[AccountConfig] = Field(default_factory=list)


class EmailPayload(BaseModel):
    to: List[str]
    subject: str
    body: str
    htmlBody: Optional[str] = None
    mimeType: str = "multipart/alternative"
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None


# X Post Agent to monitor posts
class XPostAgent(RoutedAgent):
    def __init__(self, history_file: str = "post_history.json", 
                 bearer_token: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_key_secret: Optional[str] = None,
                 access_token: Optional[str] = None,
                 access_token_secret: Optional[str] = None):
        super().__init__("An X (Twitter) post monitoring agent")
        
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_key_secret = api_key_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        
        # Initialize headers
        self.headers = {"User-Agent": "X-Post-Detector/1.0"}
        
        # Check if bearer token is available
        if bearer_token:
            self.headers["Authorization"] = f"Bearer {bearer_token}"
            print("Bearer token authentication configured")
        else:
            print("WARNING: No bearer token provided. X API requests will likely fail.")
        
        # Debug info about API credentials
        print(f"API credentials status:")
        print(f"- Bearer token: {'Configured' if bearer_token else 'Missing'}")
        print(f"- API key: {'Configured' if api_key else 'Missing'}")
        print(f"- API key secret: {'Configured' if api_key_secret else 'Missing'}")
        print(f"- Access token: {'Configured' if access_token else 'Missing'}")
        print(f"- Access token secret: {'Configured' if access_token_secret else 'Missing'}")
            
        self.history_file = history_file
        self.history = self._load_history()
    
    def _load_history(self) -> PostHistory:
        """Load post history from file or create new if not exists."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                return PostHistory(**data)
            except Exception as e:
                print(f"Error loading history: {e}")
                return PostHistory()
        return PostHistory()
    
    def _save_history(self):
        """Save post history to file."""
        with open(self.history_file, 'w') as f:
            # Use the custom encoder to handle datetime objects
            json.dump(self.history.model_dump(), f, indent=2, cls=DateTimeEncoder)
    
    def _get_account_index(self, username: str) -> int:
        """Get index of account in history or -1 if not found."""
        for i, account_config in enumerate(self.history.accounts):
            if account_config.username == username:
                return i
        return -1
    
    def _ensure_account_exists(self, username: str) -> int:
        """Ensure account exists in history and return its index."""
        index = self._get_account_index(username)
        if index == -1:
            # Account doesn't exist, add it
            self.history.accounts.append(AccountConfig(username=username))
            index = len(self.history.accounts) - 1
        return index
    
    async def _get_user_id(self, username: str) -> Optional[str]:
        """Get user ID from username using the X API."""
        api_url = f"https://api.twitter.com/2/users/by/username/{username}"
        
        try:
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            if 'data' in data and 'id' in data['data']:
                return data['data']['id']
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting user ID for {username}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Response: {e.response.text}")
            return None
    
    async def get_recent_posts(self, username: str) -> List[Dict[str, Any]]:
        """Fetch recent posts from X API for a specific account."""
        user_id = await self._get_user_id(username)
        if not user_id:
            print(f"Could not find user ID for {username}")
            return []
        
        # API endpoint for user tweets
        api_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
        
        # Parameters for the request
        params = {
            "max_results": 10,  # Fetch up to 10 recent tweets
            "tweet.fields": "created_at,entities",
            "exclude": "retweets,replies"
        }
        
        try:
            response = requests.get(api_url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data:
                return data['data']
            return []
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching posts for {username}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Status code: {e.response.status_code}")
                print(f"Response: {e.response.text}")
            return []
    
    async def check_for_new_posts(self, username: str) -> List[XPost]:
        """
        Check for new posts from a specific account.
        
        Returns:
            List of XPost objects for new posts.
        """
        posts_data = await self.get_recent_posts(username)
        if not posts_data:
            return []
        
        # Ensure account exists in history
        account_index = self._ensure_account_exists(username)
        
        # Update latest check time - store as string
        self.history.accounts[account_index].latest_check = datetime.now().isoformat()
        
        # Extract IDs of known posts
        known_post_ids = [post.id for post in self.history.accounts[account_index].posts]
        
        new_posts = []
        for post_data in posts_data:
            if post_data['id'] not in known_post_ids:
                # Extract media URLs if available
                media_urls = []
                if 'entities' in post_data and 'urls' in post_data['entities']:
                    for url_entity in post_data['entities']['urls']:
                        if 'media_key' in url_entity:
                            media_urls.append(url_entity['expanded_url'])
                
                post = XPost(
                    id=post_data['id'],
                    text=post_data['text'],
                    created_at=post_data.get('created_at', datetime.now().isoformat()),
                    author_id=post_data.get('author_id', ''),
                    html_url=f"https://twitter.com/{username}/status/{post_data['id']}",
                    media_urls=media_urls
                )
                new_posts.append(post)
                self.history.accounts[account_index].posts.append(post)
        
        if new_posts:
            self._save_history()
            
        return new_posts
    
    async def check_all_accounts(self, usernames: List[str]) -> Dict[str, List[XPost]]:
        """
        Check for new posts across all specified accounts.
        
        Args:
            usernames: List of X usernames to check
            
        Returns:
            Dictionary mapping usernames to lists of new posts
        """
        all_new_posts = {}
        
        for username in usernames:
            new_posts = await self.check_for_new_posts(username)
            if new_posts:
                all_new_posts[username] = new_posts
        
        return all_new_posts


# Content Analysis Agent using LLM to extract key information
class ContentAnalysisAgent(RoutedAgent):
    def __init__(self, model_client: OpenAIChatCompletionClient):
        super().__init__("A post content analysis agent")
        self._model_client = model_client
        self._system_message = SystemMessage(
            content=(
                "You are an expert at analyzing X (Twitter) posts and extracting the most important "
                "information. Your task is to analyze posts and identify:\n"
                "1. Key topics or themes\n"
                "2. Hashtags and mentions\n"
                "3. Links to external resources\n"
                "4. Overall summary and significance\n\n"
                "Format your response as a structured summary with HTML formatting for better readability "
                "in email notifications. Use proper HTML tags like <h2>, <h3>, <ul>, <li>, <p>, etc. "
                "DO NOT use Markdown or code block syntax like ```html or ```. "
                "Provide ONLY valid HTML that can be directly embedded in an email."
            )
        )
    
    async def analyze_post(self, username: str, post: XPost) -> str:
        """Analyze post content and extract key information."""
        prompt = (
            f"X Username: @{username}\n"
            f"Post ID: {post.id}\n"
            f"Posted at: {post.created_at}\n\n"
            f"Post Text:\n{post.text}\n\n"
            "Please analyze this X post and provide a structured summary with HTML formatting. "
            "Identify key topics, hashtags, mentions, and any links to external resources. "
            "Explain the significance of this post in the context of AI and tech news. "
            "DO NOT use Markdown syntax or code block delimiters like ```html or ```."
        )
        
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source="user")],
        )
        
        response = llm_result.content
        assert isinstance(response, str)
        
        # Clean up any markdown code block delimiters that might have been included
        response = re.sub(r'```html', '', response)
        response = re.sub(r'```', '', response)
        
        return response


# Email Notification Agent using Gmail MCP Server
class EmailNotificationAgent(RoutedAgent):
    def __init__(self, workbench: McpWorkbench):
        super().__init__("An email notification agent")
        self._workbench = workbench
    
    async def send_notification(self, recipient: str, username: str, post: XPost, content_analysis: str) -> bool:
        """Send email notification with post information."""
        # Format subject
        subject = f"New X Post: @{username}"
        
        # Clean up any markdown that might have been included
        clean_content = re.sub(r'```html', '', content_analysis)
        clean_content = re.sub(r'```', '', clean_content)
        
        # Basic text content for text/plain part
        text_body = f"""
New X Post from @{username}
Posted at: {post.created_at}
URL: {post.html_url}

Post Content:
{post.text}

Post Analysis:
{clean_content.replace('<h2>', '').replace('</h2>', '').replace('<h3>', '').replace('</h3>', '').replace('<p>', '').replace('</p>', '').replace('<br>', '\n').replace('<ul>', '').replace('</ul>', '').replace('<li>', '- ').replace('</li>', '')}
"""
        
        # HTML content for text/html part
        html_body = f"""
<html>
<body>
  <h1>New X Post from @{username}</h1>
  <p><strong>Posted at:</strong> {post.created_at}</p>
  <p><strong>URL:</strong> <a href="{post.html_url}">{post.html_url}</a></p>
  
  <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px; margin: 15px 0; border-left: 4px solid #1DA1F2;">
    <p style="font-size: 16px;">{post.text}</p>
  </div>
  
  <hr>
  <h2>Post Analysis</h2>
  {clean_content}
</body>
</html>
"""
        
        # Create email payload
        email_payload = EmailPayload(
            to=[recipient],
            subject=subject,
            body=text_body,
            htmlBody=html_body,
            mimeType="multipart/alternative"
        )
        
        try:
            # Call the Gmail MCP server to send the email
            result = await self._workbench.call_tool(
                "send_email", 
                arguments=email_payload.model_dump(exclude_none=True)
            )
            
            if result.is_error:
                print(f"Error sending email: {result.to_text()}")
                return False
            
            print(f"Email notification sent to {recipient} successfully")
            return True
        except Exception as e:
            print(f"Failed to send email notification: {e}")
            return False


# Main X Post Monitor Orchestrator
class XPostMonitorOrchestrator:
    def __init__(
        self,
        usernames: List[str],
        recipient_email: str,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
        check_interval: int = 900,  # 15 minutes by default
        history_file: str = "post_history.json"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            usernames: List of X usernames to monitor
            recipient_email: Email address to send notifications to
            bearer_token: X API bearer token
            api_key, api_key_secret, access_token, access_token_secret: X API OAuth credentials
            check_interval: Time interval between checks in seconds
            history_file: Path to the file for storing post history
        """
        self.usernames = usernames
        self.recipient_email = recipient_email
        self.check_interval = check_interval
        
        # Setup OpenAI client
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4.1-mini", 
            temperature=0.3
        )
        
        # Create agents
        self.x_agent = XPostAgent(
            history_file, 
            bearer_token, 
            api_key, 
            api_key_secret,
            access_token,
            access_token_secret
        )
        self.analysis_agent = ContentAnalysisAgent(self.model_client)
        
        # Gmail MCP server configuration
        self.gmail_mcp_server = StdioServerParams(
            command="npx",
            args=["@gongrzhe/server-gmail-autoauth-mcp"]
        )
    
    async def start_monitoring(self):
        """Start monitoring for new posts from specified X accounts."""
        accounts_list = ", ".join([f"@{username}" for username in self.usernames])
        print(f"Starting to monitor {len(self.usernames)} X accounts: {accounts_list}")
        print(f"Checking every {self.check_interval} seconds")
        print(f"Email notifications will be sent to: {self.recipient_email}")
        
        # Check if essential credentials are available
        if not self.x_agent.bearer_token:
            print("\nWARNING: You need to set the X_BEARER_TOKEN environment variable!")
            print("Create a .env file with the following content or set environment variables:")
            print("X_BEARER_TOKEN=your_bearer_token")
            print("X_API_KEY=your_api_key")
            print("X_API_KEY_SECRET=your_api_key_secret")
            print("X_ACCESS_TOKEN=your_access_token")
            print("X_ACCESS_TOKEN_SECRET=your_access_token_secret")
            print("\nFollow these steps to get your X API credentials:")
            print("1. Go to https://developer.twitter.com/")
            print("2. Create a new project and app")
            print("3. Apply for Elevated access (Basic access won't work for this tool)")
            print("4. Generate tokens in the 'Keys and Tokens' section")
        
        # Start the workbench in a context manager
        async with McpWorkbench(self.gmail_mcp_server) as workbench:
            # Create email agent with workbench
            email_agent = EmailNotificationAgent(workbench)
            
            try:
                while True:
                    print(f"\n[{datetime.now()}] Checking for new posts...")
                    
                    # Check for new posts across all accounts
                    new_posts_by_account = await self.x_agent.check_all_accounts(self.usernames)
                    
                    total_new_posts = sum(len(posts) for posts in new_posts_by_account.values())
                    
                    if total_new_posts > 0:
                        print(f"Found {total_new_posts} new post(s)!")
                        
                        for username, posts in new_posts_by_account.items():
                            for post in posts:
                                print(f"New post from @{username}: {post.id}")
                                print(f"Posted at: {post.created_at}")
                                print(f"URL: {post.html_url}")
                                print("-" * 40)
                                
                                # Analyze post content
                                content_analysis = await self.analysis_agent.analyze_post(username, post)
                                
                                # Send email notification
                                await email_agent.send_notification(
                                    self.recipient_email, username, post, content_analysis
                                )
                    else:
                        print(f"No new posts found from any monitored account")
                    
                    print(f"Next check in {self.check_interval} seconds")
                    await asyncio.sleep(self.check_interval)
            
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
            finally:
                # Close the model client
                await self.model_client.close()


# Command-line interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor X accounts for new posts')
    parser.add_argument('--accounts', required=True, nargs='+', help='X usernames to monitor (without @)')
    parser.add_argument('--email', required=True, help='Email address to receive notifications')
    parser.add_argument('--interval', type=int, default=900, help='Check interval in seconds (default: 900)')
    parser.add_argument('--history-file', default='post_history.json', help='File to store post history')
    
    args = parser.parse_args()
    
    # Clean usernames (remove @ if present)
    usernames = [username.lstrip('@') for username in args.accounts]
    
    if not usernames:
        print("Error: No valid X accounts specified")
        return
    
    # Get X API credentials from environment variables
    bearer_token = os.environ.get('X_BEARER_TOKEN')
    api_key = os.environ.get('X_API_KEY')
    api_key_secret = os.environ.get('X_API_KEY_SECRET')
    access_token = os.environ.get('X_ACCESS_TOKEN')
    access_token_secret = os.environ.get('X_ACCESS_TOKEN_SECRET')
    
    # Set up the orchestrator
    orchestrator = XPostMonitorOrchestrator(
        usernames=usernames,
        recipient_email=args.email,
        bearer_token=bearer_token,
        api_key=api_key,
        api_key_secret=api_key_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
        check_interval=args.interval,
        history_file=args.history_file
    )
    
    # Start monitoring
    await orchestrator.start_monitoring()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
