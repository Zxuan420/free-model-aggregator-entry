// The client wires and upstream wire translations (OpenAI, Anthropic, Gemini).

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** Anthropic upstream response to OpenAI client request */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = body.content;
  let text = textOf(content);
  const toolCalls: Json[] = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : `call_${Math.random().toString(36).slice(2, 9)}`,
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      },
      finish_reason: stopReasonOf(body.stop_reason),
    }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** OpenAI client request to Anthropic upstream request (same body structure essentially, but ensure model field) */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      system += (system ? '\n' : '') + textOf(raw.content);
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const toolCalls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of toolCalls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    if (raw.role === 'tool') {
      blocks.push({
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: toolResultText(raw.content),
      });
    } else {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          blocks.push({ type: 'text', text: block.text });
        } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          if (url.startsWith('data:')) {
            const parts = url.split(';base64,', 2);
            if (parts.length === 2) {
              const media = parts[0].slice(5);
              const data = parts[1];
              blocks.push({ type: 'image', source: { type: 'base64', media_type: media, data } });
            } else {
              blocks.push({ type: 'image', source: { type: 'url', url } });
            }
          } else {
            blocks.push({ type: 'image', source: { type: 'url', url } });
          }
        }
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop)) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.map((t) => {
      if (!isRecord(t) || !isRecord(t.function)) return t;
      const fn = t.function;
      return {
        name: fn.name,
        description: fn.description,
        input_schema: fn.parameters,
      };
    });
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** OpenAI client request -> Gemini upstream request */
export function openAItoGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  let systemText = '';
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = m.role;
    if (role === 'system') {
      systemText += (systemText ? '\n' : '') + textOf(m.content);
      continue;
    }
    const geminiRole = role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    if (typeof m.content === 'string' && m.content !== '') {
      parts.push({ text: m.content });
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') {
          parts.push({ text: part.text });
        } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          if (url.startsWith('data:')) {
            const match = /^data:([^;]+);base64,(.+)$/.exec(url);
            if (match) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
          } else {
            parts.push({ fileData: { fileUri: url, mimeType: 'image/jpeg' } });
          }
        }
      }
    }
    if (role === 'tool') {
      parts.push({ functionResponse: { name: 'tool', response: { output: toolResultText(m.content) } } });
    }
    if (parts.length > 0) {
      contents.push({ role: geminiRole, parts });
    }
  }

  const out: Json = { contents };
  if (systemText !== '') {
    out.systemInstruction = { parts: [{ text: systemText }] };
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (Array.isArray(body.stop)) config.stopSequences = body.stop;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) {
    out.generationConfig = config;
  }
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      const fn = raw.function;
      return [{
        name: fn.name,
        description: typeof fn.description === 'string' ? fn.description : '',
        parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} },
      }];
    });
    if (decls.length > 0) {
      out.tools = [{ functionDeclarations: decls }];
    }
  }
  return out;
}

/** Anthropic client request -> Gemini upstream request */
export function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = m.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const content = m.content;
    if (typeof content === 'string' && content !== '') {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
            parts.push({ inlineData: { mimeType: media, data: source.data } });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            parts.push({ fileData: { fileUri: source.url, mimeType: 'image/jpeg' } });
          }
        } else if (block.type === 'tool_use') {
          parts.push({ functionCall: { name: block.name, args: isRecord(block.input) ? block.input : {} } });
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: 'tool', response: { output: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length > 0) {
      contents.push({ role, parts });
    }
  }

  const out: Json = { contents };
  if (systemText !== '') {
    out.systemInstruction = { parts: [{ text: systemText }] };
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (Array.isArray(body.stop_sequences)) config.stopSequences = body.stop_sequences;
  if (Object.keys(config).length > 0) {
    out.generationConfig = config;
  }
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      }];
    });
    if (decls.length > 0) {
      out.tools = [{ functionDeclarations: decls }];
    }
  }
  return out;
}

/** Gemini response -> OpenAI client response */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const p of parts) {
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string') text += p.text;
    if (isRecord(p.functionCall)) {
      const fc = p.functionCall;
      toolCalls.push({
        id: `call_${Math.random().toString(36).slice(2, 9)}`,
        type: 'function',
        function: {
          name: typeof fc.name === 'string' ? fc.name : '',
          arguments: JSON.stringify(isRecord(fc.args) ? fc.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const finishReason = candidate.finishReason === 'MAX_TOKENS' ? 'length' : toolCalls.length > 0 ? 'tool_calls' : 'stop';
  return {
    id: `chatcmpl-${typeof body.modelVersion === 'string' ? body.modelVersion : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      },
      finish_reason: finishReason,
    }],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/** Gemini response -> Anthropic client response */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const outContent: Json[] = [];
  let text = '';
  for (const p of parts) {
    if (!isRecord(p)) continue;
    if (typeof p.text === 'string') text += p.text;
    if (isRecord(p.functionCall)) {
      const fc = p.functionCall;
      outContent.push({
        type: 'tool_use',
        id: `toolu_${Math.random().toString(36).slice(2, 9)}`,
        name: typeof fc.name === 'string' ? fc.name : '',
        input: isRecord(fc.args) ? fc.args : {},
      });
    }
  }
  if (text !== '') outContent.unshift({ type: 'text', text });
  if (outContent.length === 0) outContent.push({ type: 'text', text: '' });
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const finish = candidate.finishReason;
  const stopReason = finish === 'MAX_TOKENS' ? 'max_tokens' : outContent.some(b => b.type === 'tool_use') ? 'tool_use' : 'end_turn';
  return {
    id: `msg_gemini`,
    type: 'message',
    role: 'assistant',
    model,
    content: outContent,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: numberOrZero(usage.promptTokenCount),
      output_tokens: numberOrZero(usage.candidatesTokenCount),
    },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Read a whole upstream SSE body to text (TextDecoder streaming). */
export async function collectStreamText(upstream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let text = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return text;
}

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a raw SSE byte stream (or text) into {event: string|null, data: string} frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const normalized = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  const blocks = normalized.split(/\n\s*\n/);
  const frames: SseFrame[] = [];

  for (const block of blocks) {
    if (!block.trim()) continue;
    let currentEvent: string | null = null;
    const dataLines: string[] = [];

    for (const line of block.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith(':')) {
        continue;
      }
      if (trimmed.startsWith('event:')) {
        currentEvent = trimmed.slice(6).trim();
      } else if (trimmed.startsWith('data:')) {
        let data = trimmed.slice(5);
        if (data.startsWith(' ')) {
          data = data.slice(1);
        }
        dataLines.push(data);
      }
    }

    if (dataLines.length > 0) {
      frames.push({
        event: currentEvent,
        data: dataLines.join('\n'),
      });
    }
  }

  return frames;
}

/** OpenAI events from Anthropic upstream stream */
export async function* openaiEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await collectStreamText(upstream);
  const frames = parseSseFrames(text);
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'stop';
  let hasFinished = false;

  const chatCmplId = `chatcmpl-${Math.random().toString(36).slice(2, 9)}`;

  const processFrame = (frame: SseFrame): string[] => {
    const out: string[] = [];
    const dataPayload = frame.data;
    if (dataPayload === '[DONE]') return out;

    let chunk: unknown;
    try {
      chunk = JSON.parse(dataPayload);
    } catch {
      return out;
    }
    if (!isRecord(chunk)) return out;

    if (frame.event === 'message_start' || chunk.type === 'message_start') {
      const msg = isRecord(chunk.message) ? chunk.message : {};
      const usage = isRecord(msg.usage) ? msg.usage : {};
      inputTokens = numberOrZero(usage.input_tokens);
    } else if (frame.event === 'content_block_delta' || chunk.type === 'content_block_delta') {
      const delta = isRecord(chunk.delta) ? chunk.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        const openaiChunk = {
          id: chatCmplId,
          object: 'chat.completion.chunk',
          created: FIXED_OPENAI_CREATED,
          model,
          choices: [{ index: 0, delta: { content: delta.text }, finish_reason: null }],
        };
        out.push(`data: ${JSON.stringify(openaiChunk)}\n\n`);
      }
    } else if (frame.event === 'message_delta' || chunk.type === 'message_delta') {
      const delta = isRecord(chunk.delta) ? chunk.delta : {};
      const usage = isRecord(chunk.usage) ? chunk.usage : {};
      outputTokens = numberOrZero(usage.output_tokens);
      if (typeof delta.stop_reason === 'string') {
        stopReason = stopReasonOf(delta.stop_reason);
      }
    }

    return out;
  };

  for (const frame of frames) {
    for (const event of processFrame(frame)) yield event;
  }

  if (!hasFinished) {
    hasFinished = true;
    const finishChunk = {
      id: chatCmplId,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: stopReason }],
    };
    yield `data: ${JSON.stringify(finishChunk)}\n\n`;

    const usageChunk = {
      id: chatCmplId,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: inputTokens,
        completion_tokens: outputTokens,
        total_tokens: inputTokens + outputTokens,
      },
    };
    yield `data: ${JSON.stringify(usageChunk)}\n\n`;

    yield 'data: [DONE]\n\n';
  }
}

/** OpenAI events from Gemini upstream stream */
export async function* openaiEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await collectStreamText(upstream);
  const frames = parseSseFrames(text);
  const textPieces: string[] = [];
  let promptTokens = 0;
  let candidatesTokenCount = 0;
  let totalTokenCount = 0;
  let finishReasonGemini = 'STOP';

  for (const frame of frames) {
    const payload = frame.data;
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;

    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    if (candidates.length > 0) {
      const cand = isRecord(candidates[0]) ? candidates[0] : {};
      if (typeof cand.finishReason === 'string') finishReasonGemini = cand.finishReason;
      const content = isRecord(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) {
        if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
          textPieces.push(p.text);
        }
      }
    }

    if (isRecord(chunk.usageMetadata)) {
      const um = chunk.usageMetadata;
      promptTokens = numberOrZero(um.promptTokenCount);
      candidatesTokenCount = numberOrZero(um.candidatesTokenCount);
      totalTokenCount = numberOrZero(um.totalTokenCount);
    }
  }

  const chatCmplId = `chatcmpl-${Math.random().toString(36).slice(2, 9)}`;

  for (const textPiece of textPieces) {
    const chunk = {
      id: chatCmplId,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta: { content: textPiece }, finish_reason: null }],
    };
    yield `data: ${JSON.stringify(chunk)}\n\n`;
  }

  const finishReason = finishReasonGemini === 'MAX_TOKENS' ? 'length' : 'stop';
  const finishChunk = {
    id: chatCmplId,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
  };
  yield `data: ${JSON.stringify(finishChunk)}\n\n`;

  const usageChunk = {
    id: chatCmplId,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: candidatesTokenCount,
      total_tokens: totalTokenCount > 0 ? totalTokenCount : promptTokens + candidatesTokenCount,
    },
  };
  yield `data: ${JSON.stringify(usageChunk)}\n\n`;

  yield 'data: [DONE]\n\n';
}

/** Anthropic events from Gemini upstream stream */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await collectStreamText(upstream);
  const frames = parseSseFrames(text);
  const textPieces: string[] = [];
  let promptTokens = 0;
  let candidatesTokenCount = 0;
  let finishReasonGemini = 'STOP';

  for (const frame of frames) {
    const payload = frame.data;
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;

    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    if (candidates.length > 0) {
      const cand = isRecord(candidates[0]) ? candidates[0] : {};
      if (typeof cand.finishReason === 'string') finishReasonGemini = cand.finishReason;
      const content = isRecord(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) {
        if (isRecord(p) && typeof p.text === 'string' && p.text !== '') {
          textPieces.push(p.text);
        }
      }
    }

    if (isRecord(chunk.usageMetadata)) {
      const um = chunk.usageMetadata;
      promptTokens = numberOrZero(um.promptTokenCount);
      candidatesTokenCount = numberOrZero(um.candidatesTokenCount);
    }
  }

  yield sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_gemini_stream',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: promptTokens, output_tokens: 0 },
    },
  });

  yield sseEvent('content_block_start', {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  });

  for (const textPiece of textPieces) {
    yield sseEvent('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: textPiece },
    });
  }

  yield sseEvent('content_block_stop', {
    type: 'content_block_stop',
    index: 0,
  });

  const stopReason = finishReasonGemini === 'MAX_TOKENS' ? 'max_tokens' : 'end_turn';
  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: candidatesTokenCount },
  });

  yield sseEvent('message_stop', {
    type: 'message_stop',
  });
}

/** Anthropic events from OpenAI upstream stream */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await collectStreamText(upstream);
  const frames = parseSseFrames(text);
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const processFrame = (frame: SseFrame): string[] => {
    const out: string[] = [];
    const payload = frame.data;
    if (payload === '[DONE]') {
      out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
      return out;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return out;
    }
    if (!isRecord(chunk)) return out;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return out;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    out.push(...startEvents());
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    return out;
  };

  for (const frame of frames) {
    for (const event of processFrame(frame)) yield event;
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** OpenAI events from OpenAI upstream stream */
export async function* openaiEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await collectStreamText(upstream);
  const frames = parseSseFrames(text);
  for (const frame of frames) {
    const payload = frame.data;
    if (payload === '[DONE]') {
      yield `data: [DONE]\n\n`;
      continue;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (isRecord(chunk)) {
      chunk.model = model;
      yield `data: ${JSON.stringify(chunk)}\n\n`;
    }
  }
}
