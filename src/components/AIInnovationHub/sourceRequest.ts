class SourceHTTPError extends Error {
  constructor(source: string, public status: number) {
    super(status === 403 || status === 429
      ? `${source} rejected the request or reached its rate limit. Check API settings or try again later.`
      : `${source} returned ${status}.`);
  }
}

/** Bound each request and retry transient failures once, including HTTP 503. */
export async function sourceJSON<T>(url: string, source: string, signal?: AbortSignal, headers?: HeadersInit): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    signal?.throwIfAborted();
    const controller = new AbortController();
    const abort = () => controller.abort(signal?.reason);
    signal?.addEventListener('abort', abort, {once: true});
    const timeout = setTimeout(() => controller.abort(new DOMException('Request timed out', 'TimeoutError')), 8000);
    try {
      const response = await fetch(url, {signal: controller.signal, headers, cache: 'no-store'});
      if (!response.ok) throw new SourceHTTPError(source, response.status);
      return await response.json() as T;
    } catch (error) {
      signal?.throwIfAborted();
      const transient = error instanceof TypeError
        || (error as Error).name === 'TimeoutError'
        || (error instanceof SourceHTTPError && error.status >= 500);
      if (attempt === 0 && transient) continue;
      if ((error as Error).name === 'TimeoutError') throw new Error(`${source} timed out. Please try again later.`);
      if (error instanceof TypeError) throw new Error(`${source} could not be reached.`);
      throw error;
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener('abort', abort);
    }
  }
}
