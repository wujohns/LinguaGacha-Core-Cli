import type { ApiJsonValue } from "./api-types";

export type ApiStreamPayload = Record<string, ApiJsonValue>;

export interface ApiStreamMessage {
  topic: string;
  payload: ApiStreamPayload;
}

export type ApiStreamListener = (message: ApiStreamMessage) => void;

/**
 * Pure in-process stream hub used by the standalone CLI to wait for task snapshots.
 */
export class ApiStreamHub {
  private readonly local_subscribers = new Map<string, Set<ApiStreamListener>>();
  private started = false;

  public start(): void {
    this.started = true;
  }

  public stop(): void {
    this.started = false;
    this.local_subscribers.clear();
  }

  public publish(topic: string, payload: ApiStreamPayload): void {
    if (!this.started) {
      return;
    }
    const listeners = this.local_subscribers.get(topic);
    if (listeners === undefined) {
      return;
    }
    const message: ApiStreamMessage = { topic, payload };
    for (const listener of Array.from(listeners)) {
      listener(message);
    }
  }

  public subscribe(topic: string, listener: ApiStreamListener): () => void {
    let listeners = this.local_subscribers.get(topic);
    if (listeners === undefined) {
      listeners = new Set<ApiStreamListener>();
      this.local_subscribers.set(topic, listeners);
    }
    listeners.add(listener);
    return () => {
      const current_listeners = this.local_subscribers.get(topic);
      if (current_listeners === undefined) {
        return;
      }
      current_listeners.delete(listener);
      if (current_listeners.size === 0) {
        this.local_subscribers.delete(topic);
      }
    };
  }
}
