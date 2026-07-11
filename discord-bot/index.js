import "dotenv/config";
import {
  Client,
  GatewayIntentBits,
  REST,
  Routes,
  SlashCommandBuilder,
  EmbedBuilder,
  ActivityType,
  ModalBuilder,
  TextInputBuilder,
  TextInputStyle,
  ActionRowBuilder,
  Events,
  PermissionsBitField,
} from "discord.js";
import { createClient } from "@supabase/supabase-js";

// ─── Config ───────────────────────────────────────────────────────────────────

const DISCORD_TOKEN = process.env.DISCORD_TOKEN;
const CLIENT_ID = process.env.CLIENT_ID;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_KEY;
const PANEL_CLEANER_API_URL = process.env.PANEL_CLEANER_API_URL?.replace(/\/$/, "");
const PANEL_CLEANER_API_KEY = process.env.PANEL_CLEANER_API_KEY;
const ADMIN_USER_ID = process.env.ADMIN_USER_ID ?? "YOUR_DISCORD_ID";

for (const [k, v] of Object.entries({ DISCORD_TOKEN, SUPABASE_URL, SUPABASE_KEY, PANEL_CLEANER_API_URL, PANEL_CLEANER_API_KEY })) {
  if (!v) { console.error(`Missing env var: ${k}`); process.exit(1); }
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

const COLORS = {
  primary: 0x5865F2,
  success: 0x57F287,
  error: 0xED4245,
  warning: 0xFEE75C,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function log(level, ...args) {
  const ts = new Date().toISOString().replace("T", " ").slice(0, 19);
  console[level === "error" ? "error" : "log"](`${ts} [${level.toUpperCase()}]`, ...args);
}

function isAdmin(userId) {
  return userId === ADMIN_USER_ID;
}

async function ensureUser(discordUserId) {
  const { data } = await supabase
    .from("users")
    .select("*")
    .eq("discord_user_id", discordUserId)
    .maybeSingle();
  if (data) return data;
  const { data: newUser } = await supabase
    .from("users")
    .insert({ discord_user_id: discordUserId, credits: 5 })
    .select()
    .single();
  return newUser;
}

// ─── Commands definition ──────────────────────────────────────────────────────

const commands = [
  new SlashCommandBuilder().setName("ping").setDescription("Check bot latency"),
  new SlashCommandBuilder()
    .setName("balance")
    .setDescription("Check your current credit balance")
    .addUserOption(o => o.setName("user").setDescription("User to check (Admin only)")),
  new SlashCommandBuilder()
    .setName("profile")
    .setDescription("View your profile and credit stats")
    .addUserOption(o => o.setName("user").setDescription("User to check (Admin only)")),
  new SlashCommandBuilder()
    .setName("history")
    .setDescription("View your last 10 transactions")
    .addUserOption(o => o.setName("user").setDescription("User to check (Admin only)")),
  new SlashCommandBuilder().setName("clean").setDescription("Submit a Google Drive folder for cleaning"),
  new SlashCommandBuilder()
    .setName("add_points")
    .setDescription("[Admin] Add credits to a user")
    .addUserOption(o => o.setName("user").setDescription("User to top up").setRequired(true))
    .addIntegerOption(o => o.setName("amount").setDescription("Credits to add").setRequired(true).setMinValue(1))
    .addStringOption(o => o.setName("reason").setDescription("Reason (optional)")),
  new SlashCommandBuilder()
    .setName("deduct")
    .setDescription("[Admin] Deduct credits from a user")
    .addUserOption(o => o.setName("user").setDescription("User to deduct from").setRequired(true))
    .addIntegerOption(o => o.setName("amount").setDescription("Credits to deduct").setRequired(true).setMinValue(1))
    .addStringOption(o => o.setName("reason").setDescription("Reason (optional)")),
].map(c => c.toJSON());

// ─── Register commands ────────────────────────────────────────────────────────

async function registerCommands(clientId) {
  const rest = new REST().setToken(DISCORD_TOKEN);
  try {
    await rest.put(Routes.applicationCommands(clientId), { body: commands });
    log("info", "Slash commands registered globally.");
  } catch (err) {
    log("error", "Failed to register commands:", err.message);
  }
}

// ─── Credit calculation helper ────────────────────────────────────────────────

function calculateCredits(pixelCount) {
  if (pixelCount > 500000) return -1;
  if (pixelCount <= 200000) return 1;
  if (pixelCount <= 300000) return 2;
  if (pixelCount <= 400000) return 3;
  if (pixelCount <= 500000) return 4;
  return -1;
}

// ─── Realtime poll ────────────────────────────────────────────────────────────

const notifiedIds = new Set();

async function startRealtimePoller(client) {
  log("info", "Starting realtime polling loop (every 10s).");
  setInterval(async () => {
    try {
      const { data: requests } = await supabase
        .from("requests")
        .select("id, discord_user_id, status, output_url, error_message, credits_deducted, pixel_count, channel_id")
        .in("status", ["completed", "failed"])
        .eq("notified", false);

      for (const req of requests ?? []) {
        // Skip if already notified locally (double safeguard)
        if (notifiedIds.has(req.id)) continue;
        await notifyUser(client, req);
      }
    } catch (err) {
      log("error", "Realtime poll error:", err.message);
    }
  }, 10_000);
}

async function notifyUser(client, req) {
  // Add to local cache immediately to prevent any re-processing
  notifiedIds.add(req.id);

  let user;
  try {
    user = await client.users.fetch(req.discord_user_id);
  } catch {
    log("warn", `Could not fetch user ${req.discord_user_id}`);
    // Still mark as notified in DB to prevent infinite loop
    try { await supabase.from("requests").update({ notified: true }).eq("id", req.id); } catch {}
    return;
  }

  const credits = req.credits_deducted || (req.pixel_count ? calculateCredits(req.pixel_count) : null);

  const mention = `<@${req.discord_user_id}>`;

  let embed;
  if (req.status === "completed") {
    embed = new EmbedBuilder()
      .setTitle("✅ Processing Complete!")
      .setDescription(`${mention} Your manga panels have been cleaned successfully.`)
      .setColor(COLORS.success)
      .addFields(
        { name: "🆔 Request ID", value: `\`${req.id}\``, inline: true },
        { name: "📦 Download Link", value: `[Click here to download](${req.output_url})`, inline: false }
      );
    if (req.pixel_count) embed.addFields({ name: "📐 Total Pixels", value: req.pixel_count.toLocaleString(), inline: true });
    if (credits !== null && credits > 0) embed.addFields({ name: "💳 Credits Deducted", value: `**${credits}** credit(s)`, inline: true });
    embed.addFields({
      name: "⚠️ Important",
      value: "The output file will be **deleted automatically after 20 minutes**. Please download it now.",
      inline: false,
    });
  } else {
    embed = new EmbedBuilder()
      .setTitle("❌ Processing Failed")
      .setDescription(`${mention} Unfortunately, your request could not be processed.`)
      .setColor(COLORS.error)
      .addFields(
        { name: "🆔 Request ID", value: `\`${req.id}\``, inline: false },
        { name: "Error Details", value: `\`\`\`\n${req.error_message || "Unknown error."}\n\`\`\``, inline: false },
        { name: "💡 Tip", value: "Make sure your Google Drive folder is shared correctly and the images are within the maximum limit of 500K pixels. If the problem is not from these two reasons, open the support server link in the bot's bio and send them the error that appeared to you.", inline: false }
      );
  }
  embed.setFooter({ text: "CLEAN-BOT" });

  // Try channel delivery first if channel_id exists, fall back to DM
  let delivered = false;

  if (req.channel_id) {
    try {
      const channel = await client.channels.fetch(req.channel_id);
      if (channel) {
        const botMember = channel.guild?.members.me;
        if (botMember) {
          const permissions = channel.permissionsFor(botMember);
          if (permissions && permissions.has(PermissionsBitField.Flags.SendMessages)) {
            await channel.send({
              content: mention,
              embeds: [embed],
              allowedMentions: { users: [req.discord_user_id] },
            });
            log("info", `Notified user ${req.discord_user_id} in channel ${req.channel_id} about request ${req.id} (${req.status})`);
            delivered = true;
          }
        }
      }
    } catch {
      log("warn", `Cannot send to channel ${req.channel_id}, falling back to DM.`);
    }
  }

  if (!delivered) {
    try {
      await user.send({ embeds: [embed] });
      log("info", `Notified user ${req.discord_user_id} via DM about request ${req.id} (${req.status})`);
      delivered = true;
    } catch {
      log("warn", `Cannot DM user ${req.discord_user_id} — DMs closed.`);
    }
  }

  // Mark as notified in DB to prevent re-polling
  try {
    await supabase
      .from("requests")
      .update({ notified: true })
      .eq("id", req.id);
  } catch {
    // ignore
  }
}

// ─── Bot setup ────────────────────────────────────────────────────────────────

const client = new Client({ intents: [GatewayIntentBits.Guilds] });

client.once(Events.ClientReady, async () => {
  log("info", `Logged in as ${client.user.tag} (ID: ${client.user.id})`);
  await registerCommands(client.user.id);
  client.user.setActivity("your manga panels 🎨", { type: ActivityType.Watching });
  startRealtimePoller(client);
});

// ─── Interaction handler ──────────────────────────────────────────────────────

client.on(Events.InteractionCreate, async interaction => {
  try {
    if (interaction.isChatInputCommand()) {
      await handleCommand(interaction);
    } else if (interaction.isModalSubmit()) {
      await handleModal(interaction);
    }
  } catch (err) {
    log("error", `Unhandled interaction error [${interaction.commandName ?? interaction.customId}]:`, err.message);
    const reply = { content: "❌ An unexpected error occurred.", ephemeral: true };
    if (interaction.replied || interaction.deferred) await interaction.followUp(reply).catch(() => {});
    else await interaction.reply(reply).catch(() => {});
  }
});

// ─── Command handlers ─────────────────────────────────────────────────────────

async function handleCommand(interaction) {
  const { commandName } = interaction;

  // /ping
  if (commandName === "ping") {
    const latency = Math.round(client.ws.ping);
    const embed = new EmbedBuilder()
      .setTitle("🏓 Pong!")
      .addFields({ name: "⚡ Latency", value: `\`${latency}ms\``, inline: true })
      .setColor(COLORS.primary)
      .setFooter({ text: "CLEAN-BOT" });
    return interaction.reply({ embeds: [embed] });
  }

  // /balance
  if (commandName === "balance") {
    const targetUser = interaction.options.getUser("user");
    if (targetUser && !isAdmin(interaction.user.id)) {
      return interaction.reply({ embeds: [new EmbedBuilder().setTitle("🚫 Access Denied").setDescription("You can only check your own balance.").setColor(COLORS.error)], ephemeral: true });
    }
    await interaction.deferReply({ ephemeral: true });
    const targetId = targetUser ? targetUser.id : interaction.user.id;
    const user = await ensureUser(targetId);
    const embed = new EmbedBuilder()
      .setTitle("💳 Credit Balance")
      .addFields({ name: "Available Credits", value: `**${user.credits}** credits`, inline: true })
      .setColor(COLORS.primary)
      .setFooter({ text: "Use /profile for full details" });
    return interaction.followUp({ embeds: [embed], ephemeral: true });
  }

  // /profile
  if (commandName === "profile") {
    const targetUser = interaction.options.getUser("user");
    if (targetUser && !isAdmin(interaction.user.id)) {
      return interaction.reply({ embeds: [new EmbedBuilder().setTitle("🚫 Access Denied").setDescription("You can only check your own profile.").setColor(COLORS.error)], ephemeral: true });
    }
    await interaction.deferReply({ ephemeral: true });
    const targetId = targetUser ? targetUser.id : interaction.user.id;
    const displayName = targetUser ? targetUser.displayName : interaction.user.displayName;
    const avatarURL = targetUser ? targetUser.displayAvatarURL() : interaction.user.displayAvatarURL();

    const [userData, { data: requestsData }] = await Promise.all([
      ensureUser(targetId),
      supabase.from("requests").select("id, status").eq("discord_user_id", targetId),
    ]);
    const total = requestsData?.length ?? 0;
    const completed = requestsData?.filter(r => r.status === "completed").length ?? 0;
    const memberSince = userData.created_at?.slice(0, 10) ?? "N/A";

    const embed = new EmbedBuilder()
      .setTitle(`👤 ${displayName}'s Profile`)
      .setThumbnail(avatarURL)
      .setColor(COLORS.primary)
      .addFields(
        { name: "💳 Credits", value: `**${userData.credits}**`, inline: true },
        { name: "📊 Total Credits Used", value: `**${userData.total_credits_used}**`, inline: true },
        { name: "\u200b", value: "\u200b", inline: true },
        { name: "📁 Total Requests", value: `**${total}**`, inline: true },
        { name: "✅ Completed", value: `**${completed}**`, inline: true },
        { name: "\u200b", value: "\u200b", inline: true },
      )
      .setFooter({ text: `Member since ${memberSince} • CLEAN-BOT` });
    return interaction.followUp({ embeds: [embed] });
  }

  // /history
  if (commandName === "history") {
    const targetUser = interaction.options.getUser("user");
    if (targetUser && !isAdmin(interaction.user.id)) {
      return interaction.reply({ embeds: [new EmbedBuilder().setTitle("🚫 Access Denied").setDescription("You can only check your own history.").setColor(COLORS.error)], ephemeral: true });
    }
    await interaction.deferReply({ ephemeral: true });
    const targetId = targetUser ? targetUser.id : interaction.user.id;

    const { data: txs } = await supabase
      .from("transaction_log")
      .select("*")
      .eq("discord_user_id", targetId)
      .order("created_at", { ascending: false })
      .limit(10);

    const typeEmoji = { deduct: "➖", refund: "🔄", bonus: "➕" };
    const embed = new EmbedBuilder()
      .setTitle("📜 Transaction History")
      .setDescription(txs?.length ? `Last ${txs.length} transactions` : "No transactions yet.")
      .setColor(COLORS.primary)
      .setFooter({ text: "CLEAN-BOT" });

    for (const tx of txs ?? []) {
      const emoji = typeEmoji[tx.transaction_type] ?? "•";
      const sign = tx.transaction_type !== "deduct" ? "+" : "-";
      const date = tx.created_at?.slice(0, 10) ?? "";
      embed.addFields({
        name: `${emoji} ${tx.description || tx.transaction_type} — ${date}`,
        value: `\`${tx.balance_before}\` → \`${tx.balance_after}\` credits  |  Amount: **${sign}${tx.amount}**`,
        inline: false,
      });
    }
    return interaction.followUp({ embeds: [embed], ephemeral: true });
  }

  // /clean — show modal
  if (commandName === "clean") {
    const modal = new ModalBuilder()
      .setCustomId("clean_modal")
      .setTitle("🧹 Clean Manga Panels");

    const urlInput = new TextInputBuilder()
      .setCustomId("folder_url")
      .setLabel("Google Drive Folder URL")
      .setPlaceholder("https://drive.google.com/drive/folders/...")
      .setStyle(TextInputStyle.Short)
      .setRequired(true);

    modal.addComponents(new ActionRowBuilder().addComponents(urlInput));
    return interaction.showModal(modal);
  }

  // /add_points (admin)
  if (commandName === "add_points") {
    if (!isAdmin(interaction.user.id)) {
      return interaction.reply({
        embeds: [new EmbedBuilder().setTitle("🚫 Access Denied").setDescription("This command is restricted to the bot administrator only.").setColor(COLORS.error)],
        ephemeral: true,
      });
    }
    await interaction.deferReply({ ephemeral: true });
    const target = interaction.options.getUser("user");
    const amount = interaction.options.getInteger("amount");
    const reason = interaction.options.getString("reason") ?? "Admin Love You";

    const userData = await ensureUser(target.id);
    const balanceBefore = userData.credits;
    const balanceAfter = balanceBefore + amount;

    await supabase.from("users").update({ credits: balanceAfter }).eq("discord_user_id", target.id);
    await supabase.from("transaction_log").insert({
      discord_user_id: target.id,
      transaction_type: "bonus",
      amount,
      balance_before: balanceBefore,
      balance_after: balanceAfter,
      description: reason,
    });

    const embed = new EmbedBuilder()
      .setTitle("✅ Credits Added")
      .setColor(COLORS.success)
      .addFields(
        { name: "👤 User", value: `<@${target.id}>`, inline: true },
        { name: "➕ Added", value: `**${amount}** credits`, inline: true },
        { name: "💳 New Balance", value: `**${balanceAfter}** credits`, inline: true },
        { name: "📝 Reason", value: reason, inline: false },
      )
      .setFooter({ text: `Action by ${interaction.user.tag} • CLEAN-BOT` });

    await interaction.followUp({ embeds: [embed], ephemeral: true });

    try {
      const notify = new EmbedBuilder()
        .setTitle("💳 Credits Added to Your Account")
        .setDescription(`**${amount}** credits have been added by an admin.`)
        .setColor(COLORS.success)
        .addFields(
          { name: "New Balance", value: `**${balanceAfter}** credits`, inline: true },
          { name: "Reason", value: reason, inline: true },
        );
      await target.send({ embeds: [notify] });
    } catch { /* DMs closed */ }
    return;
  }

  // /deduct (admin)
  if (commandName === "deduct") {
    if (!isAdmin(interaction.user.id)) {
      return interaction.reply({
        embeds: [new EmbedBuilder().setTitle("🚫 Access Denied").setDescription("This command is restricted to the bot administrator only.").setColor(COLORS.error)],
        ephemeral: true,
      });
    }
    await interaction.deferReply({ ephemeral: true });
    const target = interaction.options.getUser("user");
    const amount = interaction.options.getInteger("amount");
    const reason = interaction.options.getString("reason") ?? "Admin Hate You";

    const userData = await ensureUser(target.id);
    const balanceBefore = userData.credits;
    const balanceAfter = Math.max(0, balanceBefore - amount);

    await supabase.from("users").update({ credits: balanceAfter }).eq("discord_user_id", target.id);
    await supabase.from("transaction_log").insert({
      discord_user_id: target.id,
      transaction_type: "deduct",
      amount,
      balance_before: balanceBefore,
      balance_after: balanceAfter,
      description: reason,
    });

    const embed = new EmbedBuilder()
      .setTitle("✅ Credits Deducted")
      .setColor(COLORS.warning)
      .addFields(
        { name: "👤 User", value: `<@${target.id}>`, inline: true },
        { name: "➖ Deducted", value: `**${amount}** credits`, inline: true },
        { name: "💳 New Balance", value: `**${balanceAfter}** credits`, inline: true },
        { name: "📝 Reason", value: reason, inline: false },
      )
      .setFooter({ text: `Action by ${interaction.user.tag} • CLEAN-BOT` });

    await interaction.followUp({ embeds: [embed], ephemeral: true });

    try {
      const notify = new EmbedBuilder()
        .setTitle("💳 Credits Deducted from Your Account")
        .setDescription(`**${amount}** credits have been deducted by an admin.`)
        .setColor(COLORS.warning)
        .addFields(
          { name: "New Balance", value: `**${balanceAfter}** credits`, inline: true },
          { name: "Reason", value: reason, inline: true },
        );
      await target.send({ embeds: [notify] });
    } catch { /* DMs closed */ }
    return;
  }
}

// ─── Modal handler (/clean) ───────────────────────────────────────────────────

async function handleModal(interaction) {
  if (interaction.customId !== "clean_modal") return;

  await interaction.deferReply({ thinking: true }).catch(() => {});
  const url = interaction.fields.getTextInputValue("folder_url").trim();

  if (!url.includes("drive.google.com")) {
    const embed = new EmbedBuilder()
      .setTitle("❌ Invalid URL")
      .setDescription("Please provide a valid Google Drive folder link.")
      .setColor(COLORS.error);
    return interaction.editReply({ embeds: [embed] }).catch(() => {});
  }

  const userData = await ensureUser(interaction.user.id);
  if (userData.credits <= 0) {
    const embed = new EmbedBuilder()
      .setTitle("❌ Insufficient Credits")
      .setDescription("You don't have enough credits to submit a request.\nContact an admin to add more credits.")
      .setColor(COLORS.error);
    return interaction.editReply({ embeds: [embed] }).catch(() => {});
  }

  // Determine channel_id for result delivery
  const channelId = interaction.channel?.id || null;

  let data;
  try {
    const res = await fetch(`${PANEL_CLEANER_API_URL}/api/process`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${PANEL_CLEANER_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_id: interaction.user.id,
        google_drive_folder_url: url,
        channel_id: channelId,
      }),
      signal: AbortSignal.timeout(15_000),
    });

    if (!res.ok) {
      const errBody = await res.text();
      const embed = new EmbedBuilder()
        .setTitle("❌ Server Rejected Request")
        .setDescription(`The server responded with status **${res.status}**.\n\`${errBody}\``)
        .setColor(COLORS.error);
      return interaction.editReply({ embeds: [embed] }).catch(() => {});
    }

    data = await res.json();

  } catch (err) {
    const isTimeout = err.name === "TimeoutError" || err.name === "AbortError";

    if (isTimeout) {
      const embed = new EmbedBuilder()
        .setTitle("⏱️ Request Timed Out")
        .setDescription("The server took too long to respond. Please try again later.")
        .setColor(COLORS.error);
      return interaction.editReply({ embeds: [embed] }).catch(() => {});
    }

    const embed = new EmbedBuilder()
      .setTitle("❌ Connection Error")
      .setDescription(
        "The processing server is currently unreachable.\n" +
        "This is likely a server-side issue. Please try again later."
      )
      .addFields(
        { name: "Technical Details", value: `\`\`\`\n${err.message}\n\`\`\``, inline: false },
        { name: "📢 Note", value: "If this issue persists, please send the error details above to the developer.", inline: false }
      )
      .setColor(COLORS.error);
    return interaction.editReply({ embeds: [embed] }).catch(() => {});
  }

  if (!data?.success) {
    const embed = new EmbedBuilder()
      .setTitle("❌ Request Failed")
      .setDescription(data?.message || "Unknown error from processing server.")
      .setColor(COLORS.error);
    return interaction.editReply({ embeds: [embed] }).catch(() => {});
  }

  // Build confirmation embed with estimated time
  const requestId = data.request_id ?? "N/A";
  const queuePosition = data.queue_position ?? 0;

  // Read estimated processing time from settings table
  let estimatedMinutes = null;
  try {
    const { data: setting } = await supabase
      .from("settings")
      .select("value")
      .eq("key", "estimated_processing_time_per_chapter_seconds")
      .maybeSingle();

    if (setting?.value) {
      const secondsPerChapter = parseInt(setting.value, 10);
      if (!isNaN(secondsPerChapter) && secondsPerChapter > 0) {
        const totalSeconds = (queuePosition + 1) * secondsPerChapter;
        estimatedMinutes = Math.ceil(totalSeconds / 60);
      }
    }
  } catch {
    // settings table may not exist; ignore
  }

  // Check if result can be delivered in-channel
  const botInGuild = interaction.guild?.members.me != null;
  const isDM = !interaction.guild;
  const canDeliverInChannel = botInGuild && channelId && !isDM;

  const deliveryMethod = canDeliverInChannel
    ? "in this channel"
    : "via DM";

  const embedFields = [
    { name: "🆔 Request ID", value: `\`${requestId}\``, inline: true },
    { name: "📁 Folder", value: `[View Folder](${url})`, inline: true },
  ];

  if (estimatedMinutes !== null) {
    embedFields.push({
      name: "⏱️ Estimated Time",
      value: `~**${estimatedMinutes}** minute(s)`,
      inline: false,
    });
  }

  embedFields.push({
    name: "📬 Delivery",
    value: `Results will be sent **${deliveryMethod}** when processing completes.`,
    inline: false,
  });

  const embed = new EmbedBuilder()
    .setTitle("✅ Request Queued!")
    .setDescription("Your images have been queued for processing.")
    .setColor(COLORS.success)
    .addFields(embedFields)
    .setFooter({ text: `CLEAN-BOT • Results will be sent ${deliveryMethod}` });

  if (canDeliverInChannel) {
    return interaction.editReply({ embeds: [embed] }).catch(() => {});
  } else {
    try {
      await interaction.user.send({ embeds: [embed] });
      const dmEmbed = new EmbedBuilder()
        .setTitle("📬 Check your DMs!")
        .setDescription("Since I'm not in this server, I've sent your confirmation to your DMs.")
        .setColor(COLORS.primary);
      return interaction.editReply({ embeds: [dmEmbed] }).catch(() => {});
    } catch {
      return interaction.editReply({ embeds: [embed] }).catch(() => {});
    }
  }
}

// ─── Start ────────────────────────────────────────────────────────────────────

client.login(DISCORD_TOKEN);