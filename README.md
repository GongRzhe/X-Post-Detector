# X Account Post Detector

A tool that automatically monitors X (formerly Twitter) accounts for new posts and sends email notifications when they're detected. Stay up-to-date with your favorite accounts, like @aigclink, without constantly checking for updates.

![X Post Detector](https://img.shields.io/badge/status-active-brightgreen)
![MCP](https://img.shields.io/badge/MCP-enabled-blue)
![AutoGen](https://img.shields.io/badge/AutoGen-compatible-orange)
![X API](https://img.shields.io/badge/X-API-1DA1F2?logo=x)
![Agent](https://img.shields.io/badge/Agent-powered-yellow)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-workflow-2088FF?logo=github-actions)

## Overview

X Account Post Detector is an agent-based monitoring system that leverages the Model Context Protocol (MCP) architecture to create an autonomous pipeline for tracking and notifying users of new posts from specified X accounts. The system implements a multi-layered service architecture connecting X's API with Gmail's SMTP services through OAuth2 authentication.

![image](https://github.com/user-attachments/assets/a66a0d95-4a60-483c-a7c0-1645c319a7d1)

![image](https://github.com/user-attachments/assets/fc7df911-e736-485a-afd8-537edf45b63f)


## Features

- 🔍 Monitor multiple X accounts for new posts (including @aigclink)
- 📧 Receive email notifications when new posts are detected
- 🤖 Includes AI-powered analysis of post content
- ⏱️ Configurable checking intervals
- 🔄 Easy setup with GitHub Actions for continuous monitoring
- 🔐 Secure Gmail authentication

## Prerequisites

- Python 3.12+
- Node.js 20+
- Gmail account
- Google Cloud Platform OAuth credentials
- X (Twitter) API credentials with **Elevated access**

## Installation

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GongRzhe/X-Post-Detector.git
   cd X-Post-Detector
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the Gmail MCP tool:
   ```bash
   npm install -g @gongrzhe/server-gmail-autoauth-mcp
   ```

4. Set up OAuth credentials:
   - Create a project in [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Gmail API
   - Create OAuth credentials (Desktop app)
   - Download the credentials as `gcp-oauth.keys.json`

5. Set up X API credentials:
   - Sign up for an X developer account at [developer.twitter.com](https://developer.twitter.com/)
   - Create a new project and app
   - **Apply for Elevated access** (Basic access won't work for this tool)
   - Generate API keys, access tokens, and bearer token in the "Keys and Tokens" section
   - Add all credentials to your `.env` file (see Configuration section below)

> **Important**: The X API requires Elevated access (v2) for the endpoints used by this tool. Basic access is not sufficient. Make sure to apply for Elevated access when setting up your developer account.

### GitHub Actions Setup

1. Fork this repository
2. Add the following secrets to your repository:
   - `OPENAI_API_KEY`: Your OpenAI API key (for AI analysis features)
   - `X_ACCOUNTS_TO_MONITOR`: Space-separated list of X usernames (e.g., "aigclink elonmusk")
   - `RECIPIENT_EMAIL`: Email address to receive notifications
   - `X_BEARER_TOKEN`: Your X API bearer token
   - `X_API_KEY`: Your X API key
   - `X_API_KEY_SECRET`: Your X API key secret
   - `X_ACCESS_TOKEN`: Your X access token
   - `X_ACCESS_TOKEN_SECRET`: Your X access token secret
   - `GCP_OAUTH_B64`: Base64-encoded OAuth credentials (see below)
   - `GMAIL_AUTH_B64`: Base64-encoded Gmail credentials (see below)

#### Encoding Secrets for GitHub Actions

In PowerShell:
```powershell
# For OAuth credentials
$fileContent = Get-Content -Path "gcp-oauth.keys.json" -Raw
$encodedContent = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($fileContent))
$encodedContent | Out-File -FilePath "gcp-oauth.keys.json.b64"

# For Gmail credentials (if you have them already)
$fileContent = Get-Content -Path "credentials.json" -Raw
$encodedContent = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($fileContent))
$encodedContent | Out-File -FilePath "credentials.json.b64"
```

## Configuration

Create a `.env` file in the project directory with your API credentials:

```bash
# .env file example
OPENAI_API_KEY=sk-XXX
X_BEARER_TOKEN=XXX
X_API_KEY=XXX
X_API_KEY_SECRET=XXX
X_ACCESS_TOKEN=XXX
X_ACCESS_TOKEN_SECRET=XXX
```

All API credentials are automatically loaded from environment variables, making it more secure to manage sensitive information.

## Usage

### Running Locally

```bash
python x_post_detector.py --accounts aigclink --email recipient@example.com --interval 300
```

Make sure your `.env` file contains all the required X API credentials.

### Using GitHub Actions

The included workflow will run automatically every 30 minutes or can be triggered manually.

To manually trigger the workflow:
1. Go to the Actions tab in your repository
2. Select "X Post Detector" workflow
3. Click "Run workflow"

## How It Works

1. The script periodically checks the X API for new posts from the specified accounts
2. When a new post is detected, it uses AI to analyze the content
3. The Gmail MCP tool authenticates with your Gmail account and sends the notification
4. The system keeps track of which posts have already been reported to avoid duplicates

## System Architecture

The X Post Detector utilizes an agent-based architecture with the following components:

1. **XPostAgent**: Interfaces with the X API to monitor accounts and detect new posts
2. **ContentAnalysisAgent**: Uses AI (via OpenAI's API) to analyze post content and extract key information
3. **EmailNotificationAgent**: Sends notifications via Gmail using the MCP protocol
4. **XPostMonitorOrchestrator**: Coordinates the entire monitoring process

Data is stored in a simple JSON file to track post history and prevent duplicate notifications.

## Troubleshooting

### Common Issues

1. **X API Authentication Errors (401 Unauthorized)**:
   - Make sure you have **Elevated access** for your X developer account (Basic access is not sufficient)
   - Verify that your bearer token is correct and not expired
   - Check your `.env` file or environment variables to ensure all credentials are properly set
   - Remember that X API access requires an approved developer account

2. **X API Rate Limiting (429 Too Many Requests)**:
   - The X API has rate limits. If you hit them, increase the check interval
   - Elevated access provides higher rate limits than Basic access

3. **JSON Parsing Errors**:
   - Ensure your OAuth credentials are properly encoded when using GitHub Actions

4. **Gmail Authentication Failures**:
   - Run the Gmail MCP tool with `auth` parameter locally first:
   ```bash
   npx @gongrzhe/server-gmail-autoauth-mcp auth
   ```

5. **Testing X API Authentication**:
   - You can test your bearer token with a simple curl command:
   ```bash
   curl -X GET "https://api.twitter.com/2/users/by/username/twitter" -H "Authorization: Bearer YOUR_BEARER_TOKEN"
   ```
   - This should return information about the Twitter account if your token is valid

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- Add support for monitoring hashtags and keywords
- Implement sentiment analysis of posts
- Add filtering options for specific content types
- Create a web dashboard for monitoring
- Support for more notification channels (Slack, Discord, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- X API for providing access to post information
- Google's Gmail API for email functionality
- AutoGen for agent-based architecture
- OpenAI for content analysis capabilities

---

Made with ❤️ by [GongRzhe](https://github.com/GongRzhe)
