// free-model-aggregator-01 competitive aggregator server: quality routing, budget spreading, 3-wire support, failover, catalog.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  anthropicToGemini,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAItoGemini,
  openaiEventsFromAnthropic,
  openaiEventsFromGemini,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const PROVIDER_TIMEOUT_MS = 10000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** Tracking calls and failures per provider for budget spreading and failover */
const providerCallCounts = new Map<string, number>();
const providerFailures = new Map<string, number>();
const providerLocks = new Map<string, Promise<void>>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface RouteCandidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Find all valid candidate routes meeting requirements, ranked by quality rating with budget spreading. */
function findCandidates(requires: Requirement, jsonMode: boolean): RouteCandidate[] {
  const candidates: RouteCandidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    // If jsonMode is required, wire must be openai
    if (jsonMode && probe.provider.wire !== 'openai') continue;

    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      candidates.push({ probe, model, rating });
    }
  }

  // Sort by rating descending. For ties or close rating bands (within 50 milli-points), prefer providers with fewer calls so far.
  candidates.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const callsA = providerCallCounts.get(a.probe.provider.id) || 0;
    const callsB = providerCallCounts.get(b.probe.provider.id) || 0;
    if (callsA !== callsB) return callsA - callsB;
    return a.probe.provider.id.localeCompare(b.probe.provider.id);
  });

  return candidates;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Execute a call to a provider ensuring serial execution per provider and timeout */
async function callProviderWithLock(providerId: string, fn: () => Promise<Response>): Promise<Response> {
  let prev = providerLocks.get(providerId) || Promise.resolve();
  let release: () => void = () => {};
  const next = new Promise<void>((resolve) => {
    release = resolve;
  });
  providerLocks.set(providerId, next);

  await prev;
  try {
    return await fn();
  } finally {
    release();
  }
}

/** Call provider depending on its wire and client wire */
async function callProvider(
  candidate: RouteCandidate,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
): Promise<Response> {
  const { probe, model } = candidate;
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  const isStream = clientBody.stream === true;
  if (isStream) headers.accept = 'text/event-stream';

  let url = '';
  let bodyJson: Json = {};

  if (probe.provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    if (clientApi === 'openai') {
      bodyJson = { ...clientBody, model: model.id };
    } else {
      bodyJson = anthropicToOpenAI(clientBody, model.id);
    }
    if (bodyJson.stream === true) {
      bodyJson.stream_options = { ...(isRecord(bodyJson.stream_options) ? bodyJson.stream_options : {}), include_usage: true };
    }
  } else if (probe.provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    if (clientApi === 'anthropic') {
      bodyJson = { ...clientBody, model: model.id };
    } else {
      bodyJson = openAIToAnthropicRequest(clientBody, model.id);
    }
  } else if (probe.provider.wire === 'gemini') {
    const action = isStream ? 'streamGenerateContent?alt=sse' : 'generateContent';
    url = `${base}/v1beta/models/${model.id}:${action}`;
    if (clientApi === 'openai') {
      bodyJson = openAItoGemini(clientBody, model.id);
    } else {
      bodyJson = anthropicToGemini(clientBody, model.id);
    }
  }

  return callProviderWithLock(probe.provider.id, async () => {
    try {
      return await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(bodyJson),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (err) {
      throw new Error(`fetch failed: ${(err as Error).message}`);
    }
  });
}

/** Handle client chat/messages request with failover across candidates */
async function handleInference(
  req: IncomingMessage,
  res: ServerResponse,
  clientApi: 'openai' | 'anthropic',
): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, clientApi === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error', 'the request body is not one JSON object');
  }

  const reqRequirements = requirementsFor(body);
  const jsonMode = reqRequirements.json || (isRecord(body.response_format) && !!body.response_format.type);
  const candidates = findCandidates(reqRequirements, jsonMode);

  if (candidates.length === 0) {
    return sendError(
      res,
      clientApi,
      503,
      clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model on the directory meets this request',
    );
  }

  const triedProviders = new Set<string>();
  let lastError = 'no candidates tried';

  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    if (triedProviders.has(providerId)) continue;
    const failures = providerFailures.get(providerId) || 0;
    if (failures >= 2) continue; // deprioritise providers that failed twice

    let upstream: Response;
    try {
      upstream = await callProvider(candidate, clientApi, body);
    } catch (err) {
      lastError = (err as Error).message;
      providerFailures.set(providerId, failures + 1);
      triedProviders.add(providerId);
      continue;
    }

    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
      providerFailures.set(providerId, failures + 1);
      triedProviders.add(providerId);
      continue;
    }

    if (!upstream.ok) {
      providerFailures.set(providerId, failures + 1);
      triedProviders.add(providerId);
      continue;
    }

    // Success! Record call count
    providerCallCounts.set(providerId, (providerCallCounts.get(providerId) || 0) + 1);

    const isStream = body.stream === true;
    const providerWire = candidate.probe.provider.wire;
    const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;

    if (isStream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (upstream.body !== null) {
        if (clientApi === 'anthropic') {
          if (providerWire === 'openai') {
            for await (const event of anthropicEventsFromOpenAI(upstream.body, requestedModel)) res.write(event);
          } else if (providerWire === 'anthropic') {
            const reader = upstream.body.getReader();
            try {
              for (;;) {
                const { value, done } = await reader.read();
                if (done) break;
                res.write(value);
              }
            } finally {
              reader.releaseLock();
            }
          } else if (providerWire === 'gemini') {
            for await (const event of anthropicEventsFromGemini(upstream.body, requestedModel)) res.write(event);
          }
        } else {
          // OpenAI client request
          if (providerWire === 'openai') {
            const reader = upstream.body.getReader();
            try {
              for (;;) {
                const { value, done } = await reader.read();
                if (done) break;
                res.write(Buffer.from(value));
              }
            } finally {
              reader.releaseLock();
            }
          } else if (providerWire === 'anthropic') {
            for await (const event of openaiEventsFromAnthropic(upstream.body, requestedModel)) res.write(event);
          } else if (providerWire === 'gemini') {
            for await (const event of openaiEventsFromGemini(upstream.body, requestedModel)) res.write(event);
          }
        }
      }
      res.end();
      return;
    }

    // Non-stream response
    const upstreamJson: unknown = await upstream.json();
    const jsonRecord = isRecord(upstreamJson) ? upstreamJson : {};

    let clientResponse: Json = {};
    if (clientApi === 'anthropic') {
      if (providerWire === 'openai') {
        clientResponse = openAIToAnthropic(jsonRecord, requestedModel);
      } else if (providerWire === 'anthropic') {
        clientResponse = jsonRecord;
      } else {
        clientResponse = geminiToAnthropic(jsonRecord, requestedModel);
      }
    } else {
      // OpenAI client
      if (providerWire === 'openai') {
        clientResponse = { ...jsonRecord, model: requestedModel };
      } else if (providerWire === 'anthropic') {
        clientResponse = anthropicToOpenAIResponse(jsonRecord, requestedModel);
      } else {
        clientResponse = geminiToOpenAI(jsonRecord, requestedModel);
      }
    }

    sendJson(res, 200, clientResponse);
    return;
  }

  // All candidates exhausted or failed
  return sendError(
    res,
    clientApi,
    503,
    clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model',
    `all candidate providers failed or refused. Last error: ${lastError}`,
  );
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleInference(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleInference(req, res, 'anthropic');
    return;
  }
  sendError(
    res,
    url.pathname === '/v1/messages' ? 'anthropic' : 'openai',
    404,
    'not_found',
    `no route ${req.method ?? ''} ${url.pathname}`,
  );
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Retry on next tick
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (process.argv.includes('--list')) {
    if (ENV_URL === '' && PROMPT_FILE === '') {
      const text = `${JSON.stringify({ providers: [] })}\n`;
      await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
      process.exit(0);
    }
  }
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
