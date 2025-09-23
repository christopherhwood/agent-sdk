/**
 * AnthropicProvider - Handles interactions with Anthropic's Claude API
 */

// NOTE: While this file keeps its original name (AnthropicProvider) and
// exported factory (`createAnthropicProvider`) to avoid any downstream code
// changes, the implementation below no longer talks to the Anthropic Claude
// API.  Instead, it transparently converts the Anthropic-shaped request into
// an OpenAI Chat Completions request, forwards that to the OpenAI REST
// endpoint, and then converts the response back into Claude-compatible
// message objects.

// We still import the official Anthropic SDK purely for its *types*.  The
// actual client instance is no longer used for the `create` call – doing so
// would require an Anthropic API key, which we no longer need once the request
// is proxied to OpenAI.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import type {
  ChatCompletionMessageParam,
  ChatCompletionCreateParams,
  ChatCompletion,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from 'openai/resources/chat/completions';
import { z } from 'zod';

import type {
  LLM,
  LLMConfig,
  LLMProvider,
  ContentBlockWithCache,
  ToolWithCache,
  SystemWithCache,
} from '../types/llm.js';
import type { ImageCaptioner } from '../types/captioning.js';
import { LogCategory } from '../types/logger.js';
import type { ModelProviderRequest } from '../types/model.js';
import type { RemoteModelInfo, ModelInfo } from '../types/provider.js';
// Official OpenAI SDK
import type { Logger } from '../utils/logger.js';
import { tokenManager as defaultTokenManager } from '../utils/TokenManager.js';

dotenv.config();

const LIST_MODELS_URL = process.env.LIST_MODELS_URL!;

const CAPTION_CACHE_TTL_MS = Number.parseInt(process.env.CAPTION_CACHE_TTL_MS || '900000', 10);
const CAPTION_CACHE_MAX_ENTRIES = Number.parseInt(
  process.env.CAPTION_CACHE_MAX_ENTRIES || '500',
  10,
);

type CaptionCacheEntry = {
  caption: string;
  updatedAt: number;
};

const captionCache: Map<string, CaptionCacheEntry> = new Map();

/**
 *
 */
function pruneCaptionCache(): void {
  const now = Date.now();
  for (const [key, entry] of captionCache.entries()) {
    if (now - entry.updatedAt > CAPTION_CACHE_TTL_MS) {
      captionCache.delete(key);
    }
  }
  if (captionCache.size <= CAPTION_CACHE_MAX_ENTRIES) return;
  const excess = captionCache.size - CAPTION_CACHE_MAX_ENTRIES;
  const keys = captionCache.keys();
  for (let i = 0; i < excess; i++) {
    const next = keys.next();
    if (next.done) break;
    captionCache.delete(next.value);
  }
}

/**
 *
 * @param url
 */
function signatureForUrl(url: string): string {
  if (url.startsWith('data:image/')) {
    const comma = url.indexOf(',');
    const payload = comma >= 0 ? url.slice(comma + 1) : url;
    return createHash('sha256').update(payload).digest('hex');
  }
  return createHash('sha256').update(url).digest('hex');
}

// ---------------------------------------------------------------------------
// OpenAI interop helpers
// ---------------------------------------------------------------------------

/**
 * Convert Anthropic tool format (name/description/input_schema) → OpenAI
 * function tools format.
 */

/**
 *
 * @param tools
 */
function convertToolsToOpenAI(tools?: LLM.Tool[]): ChatCompletionTool[] | undefined {
  if (!tools) return undefined;
  return tools.map(t => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description,
      parameters: t.input_schema,
    },
  }));
}

/**
 * Convert a single Anthropic message → OpenAI chat message object.
 * @param msg
 */
function convertMessageToOpenAI(msg: LLM.Messages.MessageParam): ChatCompletionMessageParam {
  const role = msg.role;

  // Helper to collapse text blocks into a single string
  const collapseText = (content: LLM.Messages.ContentBlock[] | string): string => {
    if (typeof content === 'string') return content;

    return content
      .filter((c): c is LLM.Messages.TextBlock => (c as any).type === 'text')
      .map(c => (c as any).text)
      .join('\n');
  };

  if (role === 'assistant') {
    // Check if this is a tool_use block
    if (
      Array.isArray(msg.content) &&
      msg.content.length > 0 &&
      msg.content[0].type === 'tool_use'
    ) {
      const tu = msg.content[0] as unknown as {
        id: string;
        name: string;
        input: Record<string, unknown>;
      };
      return {
        role: 'assistant',
        // When tool calls are present, `content` must be null (per API spec)
        content: null,
        tool_calls: [
          {
            id: tu.id,
            type: 'function',
            function: {
              name: tu.name,
              arguments: JSON.stringify(tu.input ?? {}),
            },
          },
        ],
      } as unknown as ChatCompletionMessageParam; // Cast to satisfy structural match
    }
  }

  if (role === 'user') {
    // Tool results in Anthropic are stored as user role with tool_result type,
    // but by the time they reach here they should be in the conversation with
    // role 'tool'.  We'll simply stringify the content.
    const contentStr = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    // Try to locate the tool_use_id for linkage
    let tool_call_id: string | undefined;
    if (
      Array.isArray(msg.content) &&
      msg.content.length > 0 &&
      (msg.content[0] as any).tool_use_id
    ) {
      tool_call_id = (msg.content[0] as any).tool_use_id;
    }
    if (tool_call_id) {
      // Proper tool response message
      return {
        role: 'tool',
        content: contentStr,
        tool_call_id,
      } as unknown as ChatCompletionMessageParam;
    }

    // Multimodal user message: pass through an array of text/image_url parts
    if (Array.isArray(msg.content)) {
      const parts: any[] = [];
      for (const c of msg.content as any[]) {
        if (c?.type === 'text' && typeof c.text === 'string') {
          parts.push({ type: 'text', text: c.text });
        } else if (c?.type === 'image_url' && c.image_url && typeof c.image_url.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: c.image_url.url } });
        }
      }
      if (parts.length > 0) {
        return { role: 'user', content: parts } as unknown as ChatCompletionMessageParam;
      }
    }

    // If we cannot identify the originating tool call, treat it as a plain user
    // message so that we do not violate the OpenAI schema (which requires
    // tool_call_id for `role:"tool"`).
    return {
      role: 'user',
      content: contentStr,
    } as unknown as ChatCompletionMessageParam;
  }

  // Default: treat as a plain text message
  return {
    role: role as any, // 'user' | 'assistant' | 'system'
    content: collapseText(msg.content as any),
  } as unknown as ChatCompletionMessageParam;
}

/**
 * Convert Anthropic API params to an OpenAI chat/completions request body.
 * @param apiParams
 */
function convertAnthropicRequestToOpenAI(apiParams: any): ChatCompletionCreateParams {
  const messages: ChatCompletionMessageParam[] = [];

  // Handle system prompt if provided
  if ('system' in apiParams && apiParams.system) {
    if (typeof (apiParams as any).system === 'string') {
      messages.push({
        role: 'system',
        content: (apiParams as any).system,
      } as ChatCompletionMessageParam);
    } else if (Array.isArray((apiParams as any).system)) {
      const blocks = (apiParams as any).system as Array<{ text: string }>;
      const systemText = blocks.map(b => b.text).join('\n');
      messages.push({ role: 'system', content: systemText } as ChatCompletionMessageParam);
    }
  }

  // Convert conversation history
  if (apiParams.messages) {
    for (const m of apiParams.messages) {
      messages.push(convertMessageToOpenAI(m));
    }
  }

  // Compose final request
  return {
    model: apiParams.model,
    temperature: apiParams.temperature ?? 0.7,
    messages,
    tools: convertToolsToOpenAI(apiParams.tools as LLM.Tool[]),
    // The OpenAI type expects ChatCompletionToolChoiceOption or undefined.
    tool_choice: apiParams.tool_choice?.type as ChatCompletionToolChoiceOption | undefined,
  } satisfies ChatCompletionCreateParams;
}

// ---------------------------------------------------------------------------
// Vision attachment injection helpers
// ---------------------------------------------------------------------------

type ToolFeedback = {
  text?: string;
  attachments?: Array<{
    kind: 'image';
    path?: string;
    dataBase64?: string;
    url?: string;
    mime?: 'image/png' | 'image/jpeg' | 'image/webp';
    width?: number;
    height?: number;
    alt?: string;
  }>;
};

/**
 *
 * @param original
 * @param model
 * @param logger
 * @param captioner
 * @param sessionId
 */
async function injectVisionMessages(
  original: LLM.Messages.MessageParam[],
  model: string,
  logger?: Logger,
  captioner?: ImageCaptioner,
  sessionId?: string,
): Promise<LLM.Messages.MessageParam[]> {
  // Inject attachments as images if supported, otherwise as text captions

  const MAX_BYTES = 10 * 1024 * 1024; // 10MB
  const augmented: LLM.Messages.MessageParam[] = [];

  for (const msg of original) {
    augmented.push(msg);
    if (msg.role !== 'user') continue;
    const contentArr = Array.isArray(msg.content) ? (msg.content as any[]) : null;
    if (!contentArr || contentArr.length === 0) continue;
    const first = contentArr[0] as any;
    if (first?.type !== 'tool_result') continue;

    let parsed: any = undefined;
    try {
      parsed = JSON.parse(first.content ?? '{}');
    } catch {
      continue;
    }

    const feedback: ToolFeedback | undefined = parsed?.additionalInformation as
      | ToolFeedback
      | undefined;
    const images = Array.isArray(feedback?.attachments)
      ? feedback!.attachments!.filter(a => a && a.kind === 'image')
      : [];
    if (images.length === 0) continue;

    const supportsImages = (process.env.LLM_SUPPORTS_IMAGES || 'true').toLowerCase() !== 'false';
    const urls: string[] = [];
    const alts: string[] = [];
    for (const att of images) {
      let url: string | undefined = att.url;
      if (!url && att.dataBase64) {
        const mime = att.mime || 'image/png';
        url = `data:${mime};base64,${att.dataBase64}`;
      }
      if (!url && att.path) {
        try {
          const abs = path.resolve(process.cwd(), att.path);
          const buf = await fs.readFile(abs);
          if (buf.byteLength > MAX_BYTES) {
            logger?.warn('Skipping image: file too large', LogCategory.MODEL, {
              path: att.path,
              bytes: buf.byteLength,
            });
            continue;
          }
          const mime =
            att.mime ||
            (att.path.toLowerCase().endsWith('.jpg') || att.path.toLowerCase().endsWith('.jpeg')
              ? 'image/jpeg'
              : att.path.toLowerCase().endsWith('.webp')
                ? 'image/webp'
                : 'image/png');
          url = `data:${mime};base64,${buf.toString('base64')}`;
        } catch (e) {
          logger?.warn('Failed to read image from path for vision injection', LogCategory.MODEL, {
            path: att.path,
            error: e instanceof Error ? e.message : String(e),
          });
          continue;
        }
      }
      if (!url) continue;
      urls.push(url);
      if (att.alt && typeof att.alt === 'string' && att.alt.trim().length > 0) {
        alts.push(att.alt.trim());
      }
    }

    if (urls.length > 0) {
      if (supportsImages) {
        const parts: any[] = [];
        for (const alt of alts) parts.push({ type: 'text', text: alt });
        for (const url of urls) parts.push({ type: 'image_url', image_url: { url } });
        augmented.push({ role: 'user', content: parts } as unknown as LLM.Messages.MessageParam);
        logger?.info('Injected vision user message after tool_result', LogCategory.MODEL, {
          images: urls.length,
        });
      } else {
        try {
          const captions = await captionImages(urls, logger, captioner, sessionId);
          const parts: any[] = [];
          for (const alt of alts) parts.push({ type: 'text', text: alt });
          for (const cap of captions)
            if (cap && cap.trim()) parts.push({ type: 'text', text: `Image: ${cap.trim()}` });
          if (parts.length > 0) {
            augmented.push({
              role: 'user',
              content: parts,
            } as unknown as LLM.Messages.MessageParam);
            logger?.info(
              'Injected caption text in place of images (vision unsupported)',
              LogCategory.MODEL,
              { captions: captions.length },
            );
          }
        } catch (e) {
          logger?.warn('Captioning failed; skipping vision injection', LogCategory.MODEL, {
            error: e instanceof Error ? e.message : String(e),
          });
        }
      }
    }
  }

  return augmented;
}

// Remove image_url parts from messages when vision is not supported
/**
 *
 * @param original
 * @param logger
 * @param captioner
 * @param sessionId
 */
async function sanitizeNoVision(
  original: LLM.Messages.MessageParam[],
  logger?: Logger,
  captioner?: ImageCaptioner,
  sessionId?: string,
): Promise<LLM.Messages.MessageParam[]> {
  const cleaned: LLM.Messages.MessageParam[] = [];
  for (const msg of original) {
    if (!Array.isArray(msg.content)) {
      cleaned.push(msg);
      continue;
    }
    const contentArr = (msg.content as any[]) || [];
    const urls: string[] = [];
    const keepBlocks: any[] = [];
    for (const c of contentArr) {
      if (c && c.type === 'image_url' && c.image_url && typeof c.image_url.url === 'string') {
        urls.push(c.image_url.url);
      } else if (c) {
        keepBlocks.push(c);
      }
    }

    if (urls.length > 0) {
      try {
        const caps = await captionImages(urls, logger, captioner, sessionId);
        for (const cap of caps) {
          if (cap && cap.trim()) keepBlocks.push({ type: 'text', text: `Image: ${cap.trim()}` });
        }
      } catch (e) {
        logger?.warn('sanitizeNoVision: captionImages failed', LogCategory.MODEL, {
          error: e instanceof Error ? e.message : String(e),
        });
      }
    }

    if (keepBlocks.length === 0) {
      // If we still have nothing, add a minimal neutral fallback
      keepBlocks.push({ type: 'text', text: '(image content converted to text unavailable)' });
    }
    cleaned.push({ ...(msg as any), content: keepBlocks } as LLM.Messages.MessageParam);
  }
  return cleaned;
}

// Caption a list of images via an OpenAI-compatible endpoint
/**
 *
 * @param urls
 * @param logger
 * @param captioner
 * @param sessionId
 */
async function captionImages(
  urls: string[],
  logger?: Logger,
  captioner?: ImageCaptioner,
  sessionId?: string,
): Promise<string[]> {
  if (!urls || urls.length === 0) return [];

  const now = Date.now();
  pruneCaptionCache();

  const order: Array<{ index: number; signature: string }> = urls.map((url, index) => ({
    index,
    signature: signatureForUrl(url),
  }));
  const hits: Array<{ index: number; caption: string }> = [];
  const misses: Array<{ index: number; url: string; signature: string }> = [];

  for (const { index, signature } of order) {
    const entry = captionCache.get(signature);
    if (entry && now - entry.updatedAt <= CAPTION_CACHE_TTL_MS) {
      hits.push({ index, caption: entry.caption });
    } else {
      misses.push({ index, url: urls[index], signature });
    }
  }

  if (misses.length === 0) {
    const captions = order
      .sort((a, b) => a.index - b.index)
      .map(({ signature }) => captionCache.get(signature)!.caption);
    if (logger) {
      try {
        logger.info('captionImages cache hit', LogCategory.MODEL, {
          sessionId,
          requested: urls.length,
        });
      } catch {
        // Intentionally empty - suppress logging errors
      }
    } else {
      try {
        console.info('[captionImages] cache hit', { sessionId, requested: urls.length });
      } catch {
        // Intentionally empty - suppress logging errors
      }
    }
    return captions;
  }

  if (logger) {
    try {
      logger.info('captionImages cache status', LogCategory.MODEL, {
        sessionId,
        requested: urls.length,
        hits: hits.length,
        misses: misses.length,
      });
    } catch {
      // Intentionally empty - suppress logging errors
    }
  } else {
    try {
      console.debug('[captionImages] cache status', {
        sessionId,
        requested: urls.length,
        hits: hits.length,
        misses: misses.length,
      });
    } catch {
      // Intentionally empty - suppress logging errors
    }
  }

  let freshCaptions: string[] | null = null;

  if (captioner) {
    try {
      const inputs = misses.map(miss => ({ urlOrData: miss.url, sessionId }));
      const custom = await captioner.describeImages(inputs);
      if (Array.isArray(custom) && custom.length >= misses.length) {
        freshCaptions = custom;
      }
    } catch (error) {
      logger?.warn(
        'Custom captioner failed; falling back to default implementation',
        LogCategory.MODEL,
        {
          error: error instanceof Error ? error.message : String(error),
        },
      );
    }
  }
  if (!freshCaptions) {
    const apiKey = process.env.CAPTION_LLM_API_KEY || process.env.LLM_API_KEY;
    const baseURL = (
      process.env.CAPTION_LLM_BASE_URL ||
      process.env.LLM_BASE_URL ||
      'https://api.openai.com/v1'
    ).replace(/\/$/, '');
    const model = process.env.CAPTION_MODEL || 'gpt-4.1-mini';
    const openai = new OpenAI({ apiKey, baseURL });
    const parts: any[] = [
      {
        type: 'text',
        text: 'Provide a concise caption (1–2 sentences) for each image. Return a JSON array of strings in order.',
      },
    ];
    for (const miss of misses) parts.push({ type: 'image_url', image_url: { url: miss.url } });
    const resp = await openai.chat.completions.create({
      model,
      temperature: 0.2,
      messages: [
        { role: 'system', content: 'You describe images succinctly for text-only models.' },
        { role: 'user', content: parts as any },
      ],
    });
    const text = resp.choices?.[0]?.message?.content?.toString?.() || '';
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        freshCaptions = parsed.map(v => (typeof v === 'string' ? v : JSON.stringify(v)));
      } else if (parsed && Array.isArray((parsed as any).descriptions)) {
        freshCaptions = (parsed as any).descriptions as string[];
      } else {
        freshCaptions = text
          .split(/\r?\n/)
          .map(s => s.trim())
          .filter(Boolean);
      }
    } catch {
      freshCaptions = text
        .split(/\r?\n/)
        .map(s => s.trim())
        .filter(Boolean);
    }
  }

  const results: string[] = new Array(urls.length);
  for (const hit of hits) {
    results[hit.index] = hit.caption;
  }

  if (freshCaptions && freshCaptions.length > 0) {
    for (let i = 0; i < misses.length; i++) {
      const miss = misses[i];
      const caption = freshCaptions[i] ?? '';
      results[miss.index] = caption;
      captionCache.set(miss.signature, { caption, updatedAt: Date.now() });
    }
  }

  return results;
}

/**
 * Perform the Chat Completions request via the official OpenAI SDK instead of a
 * raw `fetch`.  This returns the strongly-typed `ChatCompletion` object.
 * @param sessionLlmApiKey
 */
function getOpenAIClient(sessionLlmApiKey?: string): OpenAI {
  // Use the session API key if provided, otherwise fall back to environment variable
  const apiKey = sessionLlmApiKey || process.env.LLM_API_KEY;

  // Get base URL from environment
  const baseURL = (process.env.LLM_BASE_URL || 'https://api.openai.com/v1').replace(/\/$/, '');

  // Create a new client with the API key
  return new OpenAI({ apiKey, baseURL });
}

/**
 *
 * @param requestBody
 * @param logger
 * @param llmApiKey
 */
async function callOpenAI(
  requestBody: ChatCompletionCreateParams,
  logger?: Logger,
  llmApiKey?: string,
): Promise<ChatCompletion> {
  const openai = getOpenAIClient(llmApiKey);

  logger?.debug('Dispatching request to OpenAI (SDK)', LogCategory.MODEL, {
    model: requestBody.model,
    messageCount: requestBody.messages?.length ?? 0,
    usingSessionApiKey: !!llmApiKey,
  });

  try {
    const response = await openai.chat.completions.create(requestBody);
    return response as ChatCompletion;
  } catch (error: unknown) {
    // The OpenAI SDK throws rich errors.  Normalise a subset of their fields so
    // that the retry logic in `withRetryAndBackoff` continues to work.
    if (error && typeof error === 'object') {
      const err = error as { status?: number; message?: string };
      const normalised = new Error(err.message || 'OpenAI API error');
      if (err.status) (normalised as any).status = err.status;
      throw normalised;
    }
    throw error;
  }
}

/**
 * Convert an OpenAI ChatCompletion response back into Anthropic's Message
 * shape so that the rest of the codebase remains unchanged.
 * @param openaiResp
 */
function convertOpenAIResponseToAnthropic(openaiResp: ChatCompletion): LLM.Messages.Message {
  const choice = openaiResp.choices?.[0];
  const msg = choice?.message ?? {};

  // Build content blocks
  const contentBlocks: any[] = [];

  if (msg.tool_calls && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
    // Each tool call becomes a tool_use block
    for (const tc of msg.tool_calls) {
      const input = (() => {
        try {
          return JSON.parse(tc.function.arguments || '{}');
        } catch {
          return {};
        }
      })();
      contentBlocks.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
      });
    }
  }

  if (msg.content) {
    contentBlocks.push({
      type: 'text',
      text: msg.content,
    });
  }

  const usage = openaiResp.usage || {};

  const anthropicMessage: LLM.Messages.Message = {
    id: openaiResp.id,
    role: 'assistant',
    content: contentBlocks,
    model: openaiResp.model,
    stop_reason: choice?.finish_reason ?? 'stop',
    usage: {
      input_tokens: 0,
      output_tokens: 0,
    },
  } as any; // Cast to satisfy structural typing

  return anthropicMessage;
}

// Re-export provider type for convenience so callers can still import from
// this module (will be cleaned up in a later pass).
export type { LLMProvider };

/**
 * Creates a model list fetcher that provides methods to fetch and retrieve available models
 * @returns Object with fetchModelList and getAvailableModels methods
 */
function createModelListFetcher() {
  // Define Zod schema for model response
  const ModelInfoSchema = z.object({
    model_name: z.string(),
    model_info: z.object({
      litellm_provider: z.string().optional(),
      key: z.string().optional(),
      max_input_tokens: z.number().optional(),
      max_tokens: z.number().optional(),
    }),
    litellm_params: z
      .object({
        model: z.string().optional(),
      })
      .optional(),
  });

  const ModelListResponseSchema = z.object({
    data: z.array(ModelInfoSchema),
  });

  let cache: RemoteModelInfo[] | null = null;
  let inflight: Promise<RemoteModelInfo[]> | null = null;

  /**
   * Fetches the list of available models from the remote API
   * @param llmKey
   * @param logger
   * @returns Promise with array of model information
   */
  async function fetchModelList(llmKey?: string, logger?: Logger): Promise<RemoteModelInfo[]> {
    // If we already have an inflight request, return it
    if (inflight) return inflight;

    // If we have cached results, return them
    if (cache) return cache;

    // Check if we have an API URL configured
    if (!LIST_MODELS_URL) {
      logger?.warn('LIST_MODELS_URL not configured', LogCategory.MODEL);

      // Check if we have a default model configured
      const defaultModel = process.env.LLM_DEFAULT_MODEL;
      if (defaultModel) {
        // Create a cache with the default model
        cache = [
          {
            model_name: defaultModel,
            litellm_provider: 'default',
            max_input_tokens: 100000, // fallback default
          },
        ];
        return cache;
      }

      logger?.warn(
        'No LLM_DEFAULT_MODEL configured, returning empty model list',
        LogCategory.MODEL,
      );
      return [];
    }

    // Create a new fetch request
    inflight = (async () => {
      try {
        // Fetch the models list
        const response = await fetch(LIST_MODELS_URL, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': llmKey || process.env.LLM_API_KEY || '',
          },
        });
        if (!response.ok) {
          console.warn(`Failed to fetch models: ${response.status} ${response.statusText}`);

          // Check if we have a default model configured
          const defaultModel = process.env.LLM_DEFAULT_MODEL;
          if (defaultModel) {
            // Create a cache with the default model
            return [
              {
                model_name: defaultModel,
                litellm_provider: 'default',
                max_input_tokens: 100000, // fallback default
              },
            ];
          }

          return [];
        }

        // Parse the response
        const data = await response.json();

        // Parse with Zod schema
        const result = ModelListResponseSchema.safeParse(data);

        if (!result.success) {
          logger?.warn(`Failed to parse model list: ${result.error.message}`, LogCategory.MODEL);
          logger?.warn('Full error:', JSON.stringify(result.error.format(), null, 2));

          // Check if we have a default model configured
          const defaultModel = process.env.LLM_DEFAULT_MODEL;
          if (defaultModel) {
            // Create a response with the default model
            return [
              {
                model_name: defaultModel,
                litellm_provider: 'default',
                max_input_tokens: 100000, // fallback default
              },
            ];
          }

          return [];
        }

        // Transform the models to our internal format
        const modelInfo = result.data.data.map(model => ({
          model_name: model.model_name,
          litellm_provider:
            model.model_info.litellm_provider ||
            model.litellm_params?.model?.split('/')[0] ||
            'unknown',
          max_input_tokens:
            model.model_info.max_input_tokens || model.model_info.max_tokens || 100000, // fallback default
        }));

        // If the model list is empty, check for default model
        if (modelInfo.length === 0) {
          const defaultModel = process.env.LLM_DEFAULT_MODEL;
          if (defaultModel) {
            modelInfo.push({
              model_name: defaultModel,
              litellm_provider: 'default',
              max_input_tokens: 100000, // fallback default
            });
          }
        }

        // Store in cache and return
        cache = modelInfo;
        return modelInfo;
      } catch (error) {
        console.warn(
          `Error fetching model list: ${error instanceof Error ? error.message : String(error)}`,
        );

        // Check if we have a default model configured
        const defaultModel = process.env.LLM_DEFAULT_MODEL;
        if (defaultModel) {
          return [
            {
              model_name: defaultModel,
              litellm_provider: 'default',
              max_input_tokens: 100000, // fallback default
            },
          ];
        }

        return [];
      } finally {
        // Clear the inflight request
        inflight = null;
      }
    })();

    return inflight;
  }

  /**
   * Returns the list of available models with their providers
   * @param llmKey
   * @param logger
   * @returns Promise with array of model names and providers
   */
  async function getAvailableModels(llmKey?: string, logger?: Logger): Promise<ModelInfo[]> {
    try {
      const models = await fetchModelList(llmKey, logger);

      // If models list is empty, fallback to default model
      if (models.length === 0) {
        const defaultModel = process.env.LLM_DEFAULT_MODEL;
        if (defaultModel) {
          return [
            {
              model_name: defaultModel,
              provider: 'default',
            },
          ];
        }
      }

      return models.map(model => ({
        model_name: model.model_name,
        provider: model.litellm_provider,
      }));
    } catch (error) {
      // In case of any error, fallback to default model from env
      const defaultModel = process.env.LLM_DEFAULT_MODEL;
      if (defaultModel) {
        return [
          {
            model_name: defaultModel,
            provider: 'default',
          },
        ];
      }

      // If no default model is configured, return empty array
      return [];
    }
  }

  return { fetchModelList, getAvailableModels };
}

// Default token limits - these will be overridden per model if available
const DEFAULT_MAX_TOKEN_LIMIT = 200000;
// Target token limit after compression (half of max to provide ample buffer)
const DEFAULT_TARGET_TOKEN_LIMIT = DEFAULT_MAX_TOKEN_LIMIT / 2;

/**
 * Exponential backoff implementation for rate limit handling
 * @param fn - Function to call with retry logic
 * @param maxRetries - Maximum number of retry attempts
 * @param initialDelay - Initial delay in milliseconds
 * @param maxDelay - Maximum delay cap in milliseconds
 * @param logger - Logger instance
 * @returns Result of the function call
 */
async function withRetryAndBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 5,
  initialDelay = 1000,
  maxDelay = 30000,
  logger?: Logger,
): Promise<T> {
  let retries = 0;
  let delay = initialDelay;

  // Using retries with defined maxRetries, so no infinite loop
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error: unknown) {
      // Cast to a type that includes typical API error properties
      const apiError = error as {
        status?: number;
        response?: { status?: number };
        message?: string;
      };

      // Check if it's a rate limit error (HTTP 429)
      const isRateLimit =
        apiError.status === 429 ||
        apiError.response?.status === 429 ||
        (apiError.message && apiError.message.includes('rate_limit_error'));

      // If max retries reached or not a rate limit error, rethrow
      if (retries >= maxRetries || !isRateLimit) {
        throw error;
      }

      // Increment retry count and calculate next delay
      retries++;

      // Apply exponential backoff with jitter
      const jitter = Math.random() * 0.3 * delay;
      delay = Math.min(delay * 1.5 + jitter, maxDelay);

      // Log the retry attempt
      if (logger) {
        logger.warn(
          `Rate limit hit, retrying in ${Math.round(delay)}ms (attempt ${retries}/${maxRetries})`,
          LogCategory.MODEL,
        );
      } else {
        console.warn(
          `Rate limit hit, retrying in ${Math.round(delay)}ms (attempt ${retries}/${maxRetries})`,
        );
      }

      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  // This line will never be reached due to the for loop and return/throw,
  // but TypeScript requires it for compile-time checking
  throw new Error('Maximum retries exceeded');
}

/**
 * Creates a provider for Anthropic's Claude API
 * @param config - Configuration options
 * @returns The provider function
 */
function createAnthropicProvider(config: LLMConfig): LLMProvider {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  const baseURL = (process.env.LLM_BASE_URL || 'https://api.anthropic.com/v1').replace(/\/$/, '');

  const model = config.model || 'claude-3-7-sonnet';
  const maxTokens = config.maxTokens || 4096;
  const logger = config.logger;
  const captioner = config.captioner;

  // Use the provided tokenManager or fall back to the default
  const tokenManager = config.tokenManager || defaultTokenManager;

  // By default, enable caching unless explicitly disabled
  const cachingEnabled = config.cachingEnabled !== undefined ? config.cachingEnabled : true;

  const anthropic: any = null;

  // Create the model list fetcher
  const modelFetcher = createModelListFetcher();

  /**
   * Provider function that handles API calls to Claude
   * @param prompt - The prompt object
   * @returns The API response
   */
  const provider = async (prompt: ModelProviderRequest): Promise<LLM.Messages.Message> => {
    try {
      // Use the model from the prompt, which is now required
      const modelToUse = prompt.model!;

      // Get dynamic max input tokens for the chosen model
      let dynamicMaxInputTokens: number | undefined;

      try {
        // Attempt to fetch model list to determine model-specific token limits
        const list = await modelFetcher.fetchModelList(prompt.sessionState?.llmApiKey, logger);
        const info = list.find((m: RemoteModelInfo) => m.model_name === modelToUse);
        dynamicMaxInputTokens = info?.max_input_tokens;

        if (info) {
          logger?.debug('Using model-specific token limits', LogCategory.MODEL, {
            model: modelToUse,
            max_input_tokens: info.max_input_tokens,
          });
        }
      } catch (error) {
        logger?.warn('Failed to fetch model-specific token limits', LogCategory.MODEL, {
          error: error instanceof Error ? error.message : String(error),
        });
      }

      // Use dynamic token limits if available, otherwise fall back to defaults
      const MAX_TOKEN_LIMIT = dynamicMaxInputTokens ?? DEFAULT_MAX_TOKEN_LIMIT;
      const TARGET_TOKEN_LIMIT = MAX_TOKEN_LIMIT / 2;

      // Check if caching is enabled either at the provider level or in the prompt
      const shouldUseCache =
        prompt.cachingEnabled !== undefined ? prompt.cachingEnabled : cachingEnabled;

      if (prompt.sessionState?.contextWindow && prompt.sessionState.contextWindow.getLength() > 0) {
        logger?.debug('Calling Anthropic API', LogCategory.MODEL, {
          model,
          messageCount: prompt.sessionState.contextWindow.getLength(),
        });
      } else {
        logger?.debug('Calling Anthropic API', LogCategory.MODEL, {
          model,
          prompt: 'No messages provided',
        });
      }

      const conversationHistory = prompt.sessionState?.contextWindow?.getMessages() || [];

      // Proactively check token count if conversation history is getting long (> 8 messages)
      if (prompt.sessionState && conversationHistory.length > 2) {
        try {
          // Count tokens
          // token counting temporarily disabled – will be reintroduced once we
          // have a generic implementation not tied to the Anthropic SDK.
        } catch (error) {
          // If token counting fails, just log and continue
          logger?.warn(
            'Token counting failed, continuing with uncompressed conversation',
            LogCategory.MODEL,
            error instanceof Error ? error : String(error),
          );
        }
      }

      logger?.debug('Preparing API call with caching configuration', LogCategory.MODEL, {
        cachingEnabled: shouldUseCache,
      });

      // If caching is enabled and tools are provided, add cache_control to the last tool
      let modifiedTools = prompt.tools;
      if (shouldUseCache && prompt.tools && prompt.tools.length > 0) {
        // Create a deep copy to avoid modifying the original tools
        modifiedTools = JSON.parse(JSON.stringify(prompt.tools)) as LLM.Tool[];
        const lastToolIndex = modifiedTools.length - 1;

        // Add cache_control to the last tool using our extended type
        const toolWithCache = modifiedTools[lastToolIndex] as ToolWithCache;
        toolWithCache.cache_control = { type: 'ephemeral' };

        logger?.debug(
          `AnthropicProvider: Added cache_control to the last tool: ${toolWithCache.name}`,
          LogCategory.MODEL,
        );
      }

      // Format system message with cache_control if caching is enabled
      let systemContent: string | SystemWithCache = prompt.systemMessage || '';
      if (shouldUseCache && prompt.systemMessage) {
        // Convert system message to array of content blocks for caching
        systemContent = [
          {
            type: 'text',
            text: prompt.systemMessage,
            cache_control: { type: 'ephemeral' },
          },
        ];

        logger?.debug(
          'AnthropicProvider: Added cache_control to system message',
          LogCategory.MODEL,
        );
      }

      // Add cache_control to the last message in conversation history if available
      let modifiedMessages = prompt.sessionState?.contextWindow?.getMessages() || [];
      if (
        shouldUseCache &&
        prompt.sessionState?.contextWindow &&
        prompt.sessionState.contextWindow.getLength() > 0
      ) {
        // Create a deep copy to avoid modifying the original conversation history
        modifiedMessages = JSON.parse(
          JSON.stringify(modifiedMessages),
        ) as LLM.Messages.MessageParam[];

        // Find the last user message to add cache_control
        for (let i = modifiedMessages.length - 1; i >= 0; i--) {
          if (modifiedMessages[i].role === 'user') {
            // Get the content array from the last user message
            const content = modifiedMessages[i].content;

            if (Array.isArray(content) && content.length > 0) {
              // Add cache_control to the last content block when it's text-like
              const lastContentIndex = content.length - 1;
              const lastBlock = content[lastContentIndex] as any;
              if (lastBlock && typeof lastBlock === 'object' && 'type' in lastBlock) {
                if (
                  lastBlock.type === 'text' ||
                  lastBlock.type === 'tool_use' ||
                  lastBlock.type === 'tool_result'
                ) {
                  lastBlock.cache_control = { type: 'ephemeral' };
                }
              }

              logger?.debug(
                `AnthropicProvider: Added cache_control to last user message at index ${i} with type ${content[lastContentIndex].type}`,
                LogCategory.MODEL,
              );
            } else if (typeof content === 'string') {
              // If content is a string, convert to content block array with cache_control
              modifiedMessages[i].content = [
                {
                  type: 'text',
                  text: content,
                  cache_control: { type: 'ephemeral' },
                },
              ];

              logger?.debug(
                `AnthropicProvider: Converted string content to block with cache_control in last user message at index ${i}`,
                LogCategory.MODEL,
              );
            }
            break;
          }
        }
      }

      // Prepare API call parameters
      const apiParams: any = {
        model: modelToUse,
        max_tokens: maxTokens,
        // System will be set based on caching configuration
        messages: modifiedMessages,
        temperature: prompt.temperature,
      };

      logger?.debug('preparing system messages', LogCategory.MODEL);
      // Handle system messages based on new multi-message support
      if (prompt.systemMessages && prompt.systemMessages.length > 0) {
        // If systemMessages array is provided and has content, use that
        // For Claude API compatibility, only use the first system message
        // in the standard system parameter, and include the rest in the messages array

        // Make sure we have at least one system message
        if (prompt.systemMessages[0]) {
          // Set the first one as the official system message
          if (shouldUseCache) {
            // For cached requests, use an array of content blocks
            (
              apiParams as unknown as {
                system: Array<{ type: string; text: string; cache_control?: { type: string } }>;
              }
            ).system = [
              {
                type: 'text',
                text: prompt.systemMessages[0],
                cache_control: { type: 'ephemeral' },
              },
            ];
          } else {
            // For non-cached requests, use a simple string
            apiParams.system = prompt.systemMessages[0];
          }
        }

        // If there are additional system messages (beyond the first),
        // add them as assistant role messages at the beginning of the conversation
        if (prompt.systemMessages.length > 1) {
          const additionalSystemMessages = prompt.systemMessages.slice(1).map((msg, index) => {
            // We only want to add cache_control to the directory structure message (index 0 of the additional messages)
            const addCacheControlToThisMessage = shouldUseCache && index === 0;

            const systemMsg: LLM.Messages.MessageParam = {
              role: 'assistant', // Using 'assistant' role instead of 'system'
              content: addCacheControlToThisMessage
                ? [
                    {
                      type: 'text',
                      text: msg,
                      cache_control: { type: 'ephemeral' },
                    },
                  ]
                : msg,
            };
            return systemMsg;
          });

          // Add the additional system messages at the beginning of the messages array
          apiParams.messages = [...additionalSystemMessages, ...apiParams.messages];
        }
      } else if (shouldUseCache && Array.isArray(systemContent)) {
        // For cached requests with older style, system must be an array of content blocks
        (
          apiParams as unknown as {
            system: Array<{ type: string; text: string; cache_control?: { type: string } }>;
          }
        ).system = systemContent;
      } else {
        // For non-cached requests with older style, system is a simple string
        apiParams.system = prompt.systemMessage;
      }

      // Inject vision messages (images from tool_result attachments)
      try {
        apiParams.messages = await injectVisionMessages(
          apiParams.messages as LLM.Messages.MessageParam[],
          modelToUse,
          logger,
          captioner,
          prompt.sessionState?.id,
        );
      } catch (e) {
        logger?.warn('Vision injection failed; continuing without images', LogCategory.MODEL, {
          error: e instanceof Error ? e.message : String(e),
        });
      }

      // If images are not supported by the target model, strip any residual
      // image_url parts that may exist in historical messages.
      try {
        const supportsImages =
          (process.env.LLM_SUPPORTS_IMAGES || 'true').toLowerCase() !== 'false';
        if (!supportsImages) {
          apiParams.messages = await sanitizeNoVision(
            apiParams.messages as LLM.Messages.MessageParam[],
            logger,
            captioner,
            prompt.sessionState?.id,
          );
        }
      } catch (e) {
        logger?.warn('sanitizeNoVision failed; proceeding', LogCategory.MODEL, {
          error: e instanceof Error ? e.message : String(e),
        });
      }

      // Add tools if provided (for tool use mode)
      if (modifiedTools) {
        apiParams.tools = modifiedTools as LLM.Tool[];
      }

      // Add tool_choice if provided
      if (prompt.tool_choice) {
        apiParams.tool_choice = prompt.tool_choice as LLM.ToolChoice;
      }

      try {
        // ------------------------------------------------------------------
        // 1. Convert Anthropic-style request → OpenAI Chat Completions format
        // ------------------------------------------------------------------

        const openaiRequest = convertAnthropicRequestToOpenAI(apiParams);

        // ------------------------------------------------------------------
        // 2. Perform the network request to OpenAI with retry / back-off
        // ------------------------------------------------------------------

        // Get the LLM API key from session state if available
        const llmApiKey = prompt.sessionState?.llmApiKey;

        const openaiResponse = await withRetryAndBackoff(
          () => callOpenAI(openaiRequest, logger, llmApiKey),
          5,
          1000,
          30000,
          logger,
        );

        // ------------------------------------------------------------------
        // 3. Convert OpenAI response → Anthropic-compatible shape
        // ------------------------------------------------------------------

        const messageResponse = convertOpenAIResponseToAnthropic(openaiResponse);

        // Make sure token usage information is available for tracking
        if (!messageResponse.usage) {
          logger?.warn('Token usage information not provided in the response', LogCategory.MODEL);
        }

        // Log cache metrics if available
        if (
          messageResponse &&
          messageResponse.usage &&
          (messageResponse.usage.cache_creation_input_tokens ||
            messageResponse.usage.cache_read_input_tokens)
        ) {
          logger?.info('Cache metrics', LogCategory.MODEL, {
            cache_creation_input_tokens: messageResponse.usage.cache_creation_input_tokens || 0,
            cache_read_input_tokens: messageResponse.usage.cache_read_input_tokens || 0,
            input_tokens: messageResponse.usage.input_tokens,
            output_tokens: messageResponse.usage.output_tokens,
            cache_hit: messageResponse.usage.cache_read_input_tokens ? true : false,
          });

          // Calculate savings from caching if applicable
          if (messageResponse.usage.cache_read_input_tokens) {
            const cacheSavings = {
              tokens: messageResponse.usage.cache_read_input_tokens,
              percentage: Math.round(
                (messageResponse.usage.cache_read_input_tokens /
                  (messageResponse.usage.input_tokens +
                    messageResponse.usage.cache_read_input_tokens)) *
                  100,
              ),
            };

            logger?.info('Cache performance', LogCategory.MODEL, {
              saved_tokens: cacheSavings.tokens,
              savings_percentage: `${cacheSavings.percentage}%`,
            });
          }
        }

        logger?.debug('Anthropic API response', LogCategory.MODEL, {
          id: messageResponse.id,
          usage: messageResponse.usage,
          contentTypes: Array.isArray(messageResponse.content)
            ? messageResponse.content.map((c: LLM.Messages.ContentBlock) => c.type)
            : [],
        });

        // Handle empty content array by providing a fallback message
        if (!messageResponse.content || messageResponse.content.length === 0) {
          // Create a fallback content that matches Anthropic's expected format
          const fallbackContent: LLM.Messages.TextBlock = {
            type: 'text',
            text: "I just wanted to check in that everything looks okay with you, please let me know if you'd like to me change anything or continue on.",
            citations: [],
          };
          messageResponse.content = [fallbackContent];
          logger?.debug('Added fallback content for empty response', LogCategory.MODEL, {
            content: messageResponse.content,
          });
        }

        return messageResponse;
      } catch (error: unknown) {
        // Cast to a type that includes typical API error properties
        const apiError = error as {
          status?: number;
          message?: string;
          body?: unknown;
          response?: { body?: unknown };
        };

        // Check for token limit error
        const isTokenLimitError =
          apiError.status === 400 &&
          (apiError.message?.includes('prompt is too long') ||
            (apiError.message?.includes('token') && apiError.message?.includes('maximum')));

        // Log detailed error information for troubleshooting
        logger?.error('API error details', LogCategory.MODEL, {
          errorStatus: apiError.status,
          errorMessage: apiError.message,
          errorBody: apiError.body || apiError.response?.body || null,
          isTokenLimitError,
        });

        // If it's a token limit error and we have a session state and token manager, try to compress
        if (
          isTokenLimitError &&
          prompt.sessionState &&
          prompt.sessionState.contextWindow.getLength() > 0
        ) {
          logger?.warn(
            `Token limit exceeded ${apiError.message ? `(${apiError.message})` : ''}. Attempting to compress conversation history.`,
            LogCategory.MODEL,
          );

          // Use token manager to compress conversation to target limit (half of max)
          // Ensure we pass the logger that matches the expected interface
          tokenManager.manageConversationSize(prompt.sessionState, TARGET_TOKEN_LIMIT, logger);

          logger?.info(
            `Compressed conversation history to ${prompt.sessionState.contextWindow.getLength()} messages. Retrying API call.`,
            LogCategory.MODEL,
          );

          // Update API params with compressed conversation
          apiParams.messages = prompt.sessionState.contextWindow.getMessages();

          // Retry the API call with compressed conversation
          // If we're using the OpenAI proxy, we need to convert the request and call OpenAI
          const llmApiKey = prompt.sessionState?.llmApiKey;

          // Convert the request and call OpenAI
          const openaiRetryRequest = convertAnthropicRequestToOpenAI(apiParams);
          const openaiRetryResponse = await withRetryAndBackoff(
            () => callOpenAI(openaiRetryRequest, logger, llmApiKey),
            3, // fewer retries for the second attempt
            1000,
            30000,
            logger,
          );

          // Convert OpenAI response back to Anthropic format
          const messageRetryResponse = convertOpenAIResponseToAnthropic(openaiRetryResponse);

          // Handle empty content array
          if (!messageRetryResponse.content || messageRetryResponse.content.length === 0) {
            const fallbackContent: LLM.Messages.TextBlock = {
              type: 'text',
              text: "I just wanted to check in that everything looks okay with you, please let me know if you'd like to me change anything or continue on.",
              citations: [],
            };
            messageRetryResponse.content = [fallbackContent];
          }

          return messageRetryResponse;
        }

        // If not a token limit error or compression didn't help, re-throw
        logger?.error('Error calling Anthropic API', LogCategory.MODEL, error);
        throw error;
      }
    } catch (error) {
      logger?.error('Error in Anthropic provider', LogCategory.SYSTEM, {
        error,
        errorType: error instanceof Error ? error.constructor.name : typeof error,
        errorMessage: error instanceof Error ? error.message : String(error),
        errorStack: error instanceof Error ? error.stack : undefined,
      });
      throw error;
    }
  };

  return provider;
}

// Create and export the LLMFactory
export const LLMFactory = {
  createProvider: createAnthropicProvider,
  getAvailableModels: createModelListFetcher().getAvailableModels,
};
