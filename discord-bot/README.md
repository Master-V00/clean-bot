# PanelCleaner Discord Bot

A Discord bot for cleaning manga panels via the PanelCleaner API.

## Setup

### 1. Install dependencies

```bash
cd discord-bot
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `DISCORD_TOKEN` | Bot token from the [Discord Developer Portal](https://discord.com/developers/applications) |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Your Supabase **service role** key (not the anon key) |
| `PANEL_CLEANER_API_URL` | URL of your PanelCleaner API server |
| `PANEL_CLEANER_API_KEY` | API key for the PanelCleaner server |
| `ADMIN_ROLE_ID` | Discord Role ID for admin commands (set to `0` to use server admin permissions) |

### 3. Discord Developer Portal settings

In the [Discord Developer Portal](https://discord.com/developers/applications):

- **Bot → Privileged Gateway Intents**: Enable **Message Content Intent**
- **OAuth2 → URL Generator**: Select scopes:
  - `bot`
  - `applications.commands`
- **Bot Permissions**: Select:
  - Send Messages
  - Embed Links
  - Use Slash Commands

### 4. Run the bot

```bash
python bot.py
```

---

## Commands

| Command | Description | Who |
|---|---|---|
| `/ping` | Check bot latency | Everyone |
| `/balance` | Check your credit balance | Everyone |
| `/profile` | View your full profile and stats | Everyone |
| `/history` | View last 10 transactions | Everyone |
| `/clean` | Submit a Google Drive folder for cleaning | Everyone |
| `/add_points` | Add credits to a user | Admin only |
| `/deduct` | Deduct credits from a user | Admin only |

---

## How Realtime Notifications Work

The bot polls Supabase every **10 seconds** for `completed` or `failed` requests and sends a DM to the user automatically. No Supabase Realtime websocket configuration is needed on your end.

---

## Credit System

| Total Pixels | Credits Required |
|---|---|
| ≤ 200,000 | 1 credit |
| ≤ 300,000 | 2 credits |
| ≤ 400,000 | 3 credits |
| ≤ 500,000 | 4 credits |
| > 500,000 | Error — split your images |

New users start with **5 free credits**.

---

## Suggested Future Commands

- `/status <request_id>` — Check the status of a specific request by ID
- `/leaderboard` — Show top users by credits used
- `/refund <request_id>` — Admin: refund credits for a failed request
- `/set_credits <user> <amount>` — Admin: set a user's balance directly
- `/stats` — Server-wide processing statistics
